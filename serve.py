"""
Local FastAPI server for student academic standing prediction.
Predicts whether students are at-risk or in good academic standing using GPA + test-score labeling:
Pass requires GPA >= 2.0 AND AvgTestScore >= 73. Attendance remains a predictive feature only.
"""
import os
# Set library path for llvmlite (needed for SHAP on macOS)
if 'DYLD_LIBRARY_PATH' not in os.environ:
    os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib:/usr/lib'
elif '/opt/homebrew/lib' not in os.environ['DYLD_LIBRARY_PATH']:
    os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib:/usr/lib:' + os.environ['DYLD_LIBRARY_PATH']

import time
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shap

# Import compatibility shims BEFORE loading models
# This patches NumPy's random pickle handling for models trained with NumPy 1.24.4
from src.numpy_compat import patch_numpy_random_pickle
patch_numpy_random_pickle()
# Patch scikit-learn for compatibility with 1.0.2 models
from src.sklearn_compat import patch_sklearn_compat
patch_sklearn_compat()

from src.config import settings
from src.schemas import (
    PredictRequest,
    PredictResponse,
    RecommendationsResponse,
    SHAPFactor,
)
from src.agent import AcademicGuidanceAgent

# Load model on startup
MODEL_DIR = settings.model_path
PIPELINE = None
META = None
AGENT = None
SHAP_EXPLAINER = None
SHAP_FEATURE_NAMES = None
SHAP_CATEGORICAL_FEATURES = []
SHAP_NUMERIC_FEATURES = []

def map_shap_feature_name(feature_name: str) -> str:
    """Map transformed feature names back to their base feature."""
    for base in SHAP_CATEGORICAL_FEATURES:
        prefix = f"{base}__"
        if feature_name.startswith(prefix):
            return base
    return feature_name


def summarize_shap_values(shap_vector) -> List[Dict[str, object]]:
    """Summarize SHAP contributions by original feature."""
    if SHAP_FEATURE_NAMES is None:
        return []
    
    contributions: dict[str, float] = {}
    for name, value in zip(SHAP_FEATURE_NAMES, shap_vector):
        base_name = map_shap_feature_name(name)
        contributions.setdefault(base_name, {"sum": 0.0, "count": 0})
        contributions[base_name]["sum"] += float(value)
        contributions[base_name]["count"] += 1
    
    top_items = sorted(
        contributions.items(),
        key=lambda item: abs(item[1]["sum"] / (item[1]["count"] or 1)),
        reverse=True
    )[: settings.shap_top_features]
    
    summary = []
    for feature, impact in top_items:
        impact_value = impact["sum"] / (impact["count"] or 1)
        summary.append(
            {
                "feature": feature,
                "impact": impact_value,
                "direction": "positive" if impact_value >= 0 else "negative",
                "description": "Increases pass probability" if impact_value >= 0 else "Decreases pass probability",
            }
        )
    return summary


def compute_shap_factors(X: pd.DataFrame) -> List[Dict[str, object]]:
    """Compute SHAP contributions for a single instance."""
    if SHAP_EXPLAINER is None or SHAP_FEATURE_NAMES is None:
        return []
    
    preprocessor = PIPELINE.named_steps["pre"]
    transformed = preprocessor.transform(X)
    shap_values = SHAP_EXPLAINER.shap_values(transformed)
    
    if isinstance(shap_values, list):
        shap_vector = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    else:
        shap_vector = shap_values[0]
    
    return summarize_shap_values(shap_vector)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    global PIPELINE, META, AGENT, SHAP_EXPLAINER, SHAP_FEATURE_NAMES, SHAP_CATEGORICAL_FEATURES, SHAP_NUMERIC_FEATURES
    
    print("Loading model...")
    pipeline_path = MODEL_DIR / "pipeline.joblib"
    meta_path = MODEL_DIR / "metadata.json"
    
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Model not found at {pipeline_path}. Please train the model first.")
    
    # Load model (NumPy compatibility is handled by numpy_compat module imported above)
    try:
        PIPELINE = joblib.load(pipeline_path)
    except ValueError as e:
        if "BitGenerator" in str(e) or "MT19937" in str(e):
            import numpy as np
            print("[ERROR] NumPy version incompatibility detected")
            print(f"[ERROR] Model was trained with NumPy 1.24.4 (SageMaker)")
            print(f"[ERROR] Current NumPy version: {np.__version__}")
            print("[ERROR] NumPy 1.26.x has breaking changes in random number generator API")
            print("[SOLUTION] The compatibility shim should have handled this. Please check numpy_compat.py")
            raise RuntimeError(
                f"Model loading failed due to NumPy version incompatibility. "
                f"Model requires NumPy 1.24.4, but current version is {np.__version__}. "
                f"The compatibility shim failed to patch NumPy's random pickle module."
            ) from e
        raise
    META = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    
    print(f"[OK] Model loaded: {META.get('model_name', 'unknown')}")
    print(f"[OK] Model version: {settings.model_version}")
    
    # Initialize SHAP explainer
    preprocessor = PIPELINE.named_steps.get("pre")
    classifier = PIPELINE.named_steps.get("clf")
    if settings.enable_shap and preprocessor is not None and classifier is not None:
        try:
            SHAP_FEATURE_NAMES = preprocessor.get_feature_names_out()
            transformers = {name: (transformer, cols) for name, transformer, cols in preprocessor.transformers_}
            SHAP_CATEGORICAL_FEATURES = transformers.get("cat", (None, []))[1]
            SHAP_NUMERIC_FEATURES = transformers.get("num", (None, []))[1]
            SHAP_EXPLAINER = shap.TreeExplainer(classifier)
            print("[OK] SHAP explainer initialized")
        except Exception as exc:
            SHAP_EXPLAINER = None
            SHAP_FEATURE_NAMES = None
            SHAP_CATEGORICAL_FEATURES = []
            SHAP_NUMERIC_FEATURES = []
            print(f"[WARNING] SHAP explainer not initialized: {exc}")
    else:
        SHAP_EXPLAINER = None
    
    # Initialize AI Agent if enabled
    if settings.enable_ai_agent:
        try:
            AGENT = AcademicGuidanceAgent(
                token=settings.huggingface_token,
                model_id=settings.hf_model_id
            )
            print("[OK] AI Agent initialized (Hugging Face Inference API)")
        except Exception as e:
            print(f"[WARNING] AI Agent not initialized: {e}")
            print("  Set huggingface token (env HF_API_TOKEN or config) to enable AI recommendations")
            AGENT = None
    else:
        print("[INFO] AI Agent disabled (set enable_ai_agent=True to enable)")
        AGENT = None
    
    yield
    
    # Shutdown
    print("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Student Academic Standing Prediction API",
    description="API for predicting student academic standing using GPA + test-score labeling: Pass requires GPA >= 2.0 AND AvgTestScore >= 73. Attendance remains a predictive feature.",
    version=settings.model_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": META.get("model_name") if META else "unknown",
        "version": settings.model_version,
        "test_accuracy": META.get("test_metrics", {}).get("accuracy") if META else None
    }

def analyze_failed_conditions(req: PredictRequest) -> dict:
    """
    Analyze which conditions the student is likely failing based on available features.
    
    Note: We don't have actual GPA/TestScore in the request (to prevent data leakage),
    but we can infer likely issues from available features and provide guidance.
    
    Args:
        req: PredictRequest with student features
    
    Returns:
        dict with boolean flags for each condition: {'gpa': bool, 'test_score': bool, 'attendance': bool}
    """
    failed = {
        "gpa": False,
        "test_score": False,
        "attendance": False
    }
    
    # Analyze attendance (we have this data directly)
    if req.AttendanceRate < 0.85:
        failed["attendance"] = True
    
    # Infer test score issues from study hours
    # Low study hours typically correlate with lower test scores
    if req.StudyHours < 2.0:
        failed["test_score"] = True
    
    # Infer GPA issues from multiple factors
    # Low attendance and low study hours both affect GPA
    if req.AttendanceRate < 0.80 or req.StudyHours < 1.5:
        failed["gpa"] = True
    
    # If no specific conditions identified but student is at-risk,
    # mark general areas for improvement
    if not any(failed.values()):
        # Default to general guidance if we can't identify specific issues
        failed["test_score"] = True
        failed["attendance"] = True
    
    return failed


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict student academic standing.
    
    Predicts whether a student is at-risk or in good academic standing using GPA + test-score labeling:
    Pass requires GPA >= 2.0 AND AvgTestScore >= 73. Attendance is used as an input signal only.
    
    For at-risk students, also provides personalized AI-generated recommendations.
    
    Args:
        req: PredictRequest with student features
    
    Returns:
        PredictResponse with prediction, probability, and recommendations (if at-risk)
        - pass_fail: 1 = good academic standing (Pass), 0 = at-risk (Fail)
        - pass_prob: Probability of good academic standing
        - recommendations: AI-generated personalized guidance (only for at-risk students)
        - failed_conditions: Which conditions failed (GPA, TestScore, Attendance)
    """
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.perf_counter()
    
    try:
        # Convert request to DataFrame
        features_dict = req.model_dump(exclude={"actual_GPA"})
        X = pd.DataFrame([features_dict])
        
        # Get prediction
        clf = PIPELINE.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            prob = float(PIPELINE.predict_proba(X)[0, 1])
            label = int(prob >= 0.5)
        else:
            label = int(PIPELINE.predict(X)[0])
            prob = 1.0 if label == 1 else 0.0
        
        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Analyze failed conditions
        failed_conditions = analyze_failed_conditions(req)
        shap_factors = []
        if settings.enable_shap:
            try:
                shap_factors = compute_shap_factors(X)
            except Exception as exc:
                shap_factors = []
                print(f"[WARNING] Failed to compute SHAP values: {exc}")
        
        # Generate AI guidance based on student status
        recommendations = None
        if AGENT is not None and settings.enable_ai_agent:
            agent_start = time.perf_counter()
            try:
                if label == 0:
                    # At-risk student: generate improvement recommendations
                    agent_response = AGENT.generate_recommendations(
                        student_features=features_dict,
                        prediction=label,
                        confidence=prob,
                        failed_conditions=failed_conditions,
                        shap_factors=shap_factors
                    )
                else:
                    # Passing student: generate encouragement and growth guidance
                    agent_response = AGENT.generate_encouragement(
                        student_features=features_dict,
                        confidence=prob,
                        shap_factors=shap_factors
                    )
                
                agent_latency = (time.perf_counter() - agent_start) * 1000
                if agent_response and "recommendations" in agent_response:
                    recommendations = RecommendationsResponse(**agent_response)
                    status = "encouragement" if label == 1 else "recommendations"
                    print(f"[OK] AI {status} generated ({agent_latency:.1f}ms)")
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail="AI agent unavailable. Please try again later."
                ) from e
        
        return PredictResponse(
            pass_prob=prob,
            pass_fail=label,
            model_version=settings.model_version,
            latency_ms=latency_ms,
            recommendations=recommendations,
            failed_conditions=failed_conditions if label == 0 else None,
            shap_factors=[SHAPFactor(**factor) for factor in shap_factors] if shap_factors else None
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "Student Academic Standing Prediction API",
        "version": settings.model_version,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "serve:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )
