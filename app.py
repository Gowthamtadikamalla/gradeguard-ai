"""
AWS Lambda handler for student academic standing prediction.
Predicts whether students are at-risk or in good academic standing using two-feature labeling:
Pass requires GPA >= 2.0 AND AvgTestScore >= 73.
All other students are labeled as Fail (at-risk).
AttendanceRate is used as a predictive feature but not in target labeling.
"""
import os
# Set library path for llvmlite (needed for SHAP on Lambda)
if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib:/lib'
elif '/usr/lib' not in os.environ['LD_LIBRARY_PATH']:
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib:/lib:' + os.environ['LD_LIBRARY_PATH']

import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import pandas as pd
import boto3
from fastapi import FastAPI, HTTPException
from mangum import Mangum
import shap

# Import compatibility shims BEFORE loading models
# This patches NumPy's random pickle handling for models trained with NumPy 1.24.4
from src.numpy_compat import patch_numpy_random_pickle
patch_numpy_random_pickle()
# Patch scikit-learn for compatibility with 1.0.2 models
from src.sklearn_compat import patch_sklearn_compat
patch_sklearn_compat()

from src.schemas import (
    PredictRequest,
    PredictResponse,
    RecommendationsResponse,
    SHAPFactor,
)
from src.agent import AcademicGuidanceAgent

# CRITICAL: Import XGBoost and LightGBM classes at module level
# This ensures they're available when joblib unpickles models that use them
# joblib needs these classes in the global namespace to unpickle the model
try:
    from xgboost import XGBClassifier
    print("[INFO] XGBClassifier imported at module level")
except ImportError:
    print("[WARNING] XGBClassifier not available - XGBoost models cannot be loaded")
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
    print("[INFO] LGBMClassifier imported at module level")
except ImportError:
    print("[WARNING] LGBMClassifier not available - LightGBM models cannot be loaded")
    LGBMClassifier = None

# Lambda environment variables
MODEL_S3_BUCKET = os.environ.get("MODEL_S3_BUCKET", "student-performance-ml-650")
MODEL_S3_KEY = os.environ.get("MODEL_S3_KEY", "models/v1/student-ml-train-20251126-164505/output/model.tar.gz")
HF_SECRET_NAME = os.environ.get("HF_SECRET_NAME", "student-ml/hf-api-token")
ENABLE_AI_AGENT = os.environ.get("ENABLE_AI_AGENT", "true").lower() == "true"
ENABLE_SHAP = os.environ.get("ENABLE_SHAP", "true").lower() == "true"
SHAP_TOP_FEATURES = int(os.environ.get("SHAP_TOP_FEATURES", "3"))

# Global variables (loaded on first invocation)
PIPELINE = None
META = None
AGENT = None
SHAP_EXPLAINER = None
SHAP_FEATURE_NAMES = None
SHAP_CATEGORICAL_FEATURES = []
SHAP_NUMERIC_FEATURES = []
S3_CLIENT = None

def load_model_from_s3():
    """Load model from S3 on first invocation (Lambda cold start)."""
    global PIPELINE, META, S3_CLIENT, SHAP_EXPLAINER, SHAP_FEATURE_NAMES, SHAP_CATEGORICAL_FEATURES, SHAP_NUMERIC_FEATURES, AGENT
    
    if PIPELINE is not None:
        return  # Already loaded
    
    # CRITICAL: Import XGBoost and LightGBM classes BEFORE loading model
    # joblib needs these classes available when unpickling the model
    # Import both the module and the specific classifier classes
    try:
        import xgboost
        from xgboost import XGBClassifier
        print(f"[INFO] XGBoost available: {xgboost.__version__}")
        print(f"[INFO] XGBClassifier imported successfully")
    except ImportError as e:
        print(f"[WARNING] XGBoost not available: {e}")
        XGBClassifier = None
    
    try:
        import lightgbm
        from lightgbm import LGBMClassifier
        print(f"[INFO] LightGBM available: {lightgbm.__version__}")
        print(f"[INFO] LGBMClassifier imported successfully")
    except ImportError as e:
        print(f"[WARNING] LightGBM not available: {e}")
        LGBMClassifier = None
    
    print(f"[INFO] Loading model from S3: s3://{MODEL_S3_BUCKET}/{MODEL_S3_KEY}")
    S3_CLIENT = boto3.client("s3")
    
    # Download model.tar.gz to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_tar = tmp_path / "model.tar.gz"
        
        # Download from S3
        S3_CLIENT.download_file(MODEL_S3_BUCKET, MODEL_S3_KEY, str(model_tar))
        
        # Extract tar.gz
        import tarfile
        with tarfile.open(model_tar, "r:gz") as tar:
            tar.extractall(path=tmp_path)
        
        # Load pipeline and metadata
        # SageMaker extracts files directly to the extraction path root
        pipeline_path = tmp_path / "pipeline.joblib"
        meta_path = tmp_path / "metadata.json"
        
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Model pipeline not found in {tmp_path}. Contents: {list(tmp_path.iterdir())}")
        
        # Load metadata first to see what model type we're loading
        if meta_path.exists():
            META = json.loads(meta_path.read_text())
            model_name = META.get('model_name', 'unknown')
            print(f"[INFO] Model type: {model_name}")
        else:
            META = {}
            model_name = "unknown"
        
        # Load the model pipeline
        # Compatibility patches are already applied via imports at module level
        try:
            PIPELINE = joblib.load(pipeline_path)
            print(f"[OK] Model loaded: {model_name}")
            print(f"[OK] Model type: {type(PIPELINE)}")
        except Exception as e:
            # Provide more detailed error information
            import traceback
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            
            print(f"[ERROR] Failed to load model: {error_msg}")
            print(f"[ERROR] Model type expected: {model_name}")
            print(f"[ERROR] Full traceback:\n{error_traceback}")
            
            # Extract the module name from the error if it's an import error
            if "__import__" in error_traceback or "cannot import" in error_msg.lower():
                print("[ERROR] This is an import error during model unpickling.")
                print("[ERROR] The model requires a module/class that is not available.")
                
                # Try to extract module name from traceback
                import re
                module_match = re.search(r"__import__\(['\"]([^'\"]+)['\"]", error_traceback)
                if module_match:
                    missing_module = module_match.group(1)
                    print(f"[ERROR] Missing module: {missing_module}")
                    print(f"[ERROR] Please ensure '{missing_module}' is installed in the Lambda container.")
            
            raise
        
        # Initialize SHAP explainer
        if ENABLE_SHAP:
            try:
                preprocessor = PIPELINE.named_steps.get("pre")
                classifier = PIPELINE.named_steps.get("clf")
                if preprocessor is not None and classifier is not None:
                    global SHAP_FEATURE_NAMES, SHAP_CATEGORICAL_FEATURES, SHAP_NUMERIC_FEATURES
                    SHAP_FEATURE_NAMES = preprocessor.get_feature_names_out()
                    transformers = {name: (transformer, cols) for name, transformer, cols in preprocessor.transformers_}
                    SHAP_CATEGORICAL_FEATURES = transformers.get("cat", (None, []))[1]
                    SHAP_NUMERIC_FEATURES = transformers.get("num", (None, []))[1]
                    SHAP_EXPLAINER = shap.TreeExplainer(classifier)
                    print("[OK] SHAP explainer initialized")
            except Exception as exc:
                print(f"[WARNING] SHAP explainer not initialized: {exc}")
        
        # Initialize AI Agent
        if ENABLE_AI_AGENT:
            try:
                # Get HF token from Secrets Manager
                secrets_client = boto3.client("secretsmanager")
                try:
                    secret_response = secrets_client.get_secret_value(SecretId=HF_SECRET_NAME)
                    hf_token = json.loads(secret_response["SecretString"]).get("HF_API_TOKEN")
                except Exception as e:
                    print(f"[WARNING] Could not get HF token from Secrets Manager: {e}")
                    hf_token = os.environ.get("HF_API_TOKEN")
                
                if hf_token:
                    AGENT = AcademicGuidanceAgent(token=hf_token)
                    print("[OK] AI Agent initialized")
                else:
                    print("[WARNING] HF token not found, AI Agent disabled")
                    AGENT = None
            except Exception as e:
                print(f"[WARNING] AI Agent not initialized: {e}")
                AGENT = None

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
    )[:SHAP_TOP_FEATURES]
    
    summary = []
    for feature, impact in top_items:
        impact_value = impact["sum"] / (impact["count"] or 1)
        summary.append({
            "feature": feature,
            "impact": impact_value,
            "direction": "positive" if impact_value >= 0 else "negative",
            "description": "Increases pass probability" if impact_value >= 0 else "Decreases pass probability",
        })
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

def analyze_failed_conditions(req: PredictRequest) -> dict:
    """Analyze which conditions the student is likely failing."""
    failed = {
        "gpa": False,
        "test_score": False,
        "attendance": False
    }
    
    if req.AttendanceRate < 0.85:
        failed["attendance"] = True
    
    if req.StudyHours < 2.0:
        failed["test_score"] = True
    
    if req.AttendanceRate < 0.80 or req.StudyHours < 1.5:
        failed["gpa"] = True
    
    if not any(failed.values()):
        failed["test_score"] = True
        failed["attendance"] = True
    
    return failed

# Create FastAPI app
app = FastAPI(
    title="Student Academic Standing Prediction API",
    description="API for predicting student academic standing using two-feature labeling: Pass requires GPA >= 2.0 AND AvgTestScore >= 73.",
    version="v1"
)

@app.get("/health")
def health():
    """Health check endpoint."""
    if PIPELINE is None:
        load_model_from_s3()
    return {
        "status": "ok",
        "model": META.get("model_name") if META else "unknown",
        "version": "v1",
        "test_accuracy": META.get("test_metrics", {}).get("accuracy") if META else None
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict student academic standing."""
    # Load model on first invocation
    if PIPELINE is None:
        load_model_from_s3()
    
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
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Analyze failed conditions
        failed_conditions = analyze_failed_conditions(req)
        shap_factors = []
        if ENABLE_SHAP:
            try:
                shap_factors = compute_shap_factors(X)
            except Exception as exc:
                print(f"[WARNING] Failed to compute SHAP values: {exc}")
                shap_factors = []
        
        # Generate AI guidance
        recommendations = None
        if AGENT is not None and ENABLE_AI_AGENT:
            agent_start = time.perf_counter()
            try:
                if label == 0:
                    agent_response = AGENT.generate_recommendations(
                        student_features=features_dict,
                        prediction=label,
                        confidence=prob,
                        failed_conditions=failed_conditions,
                        shap_factors=shap_factors
                    )
                else:
                    agent_response = AGENT.generate_encouragement(
                        student_features=features_dict,
                        confidence=prob,
                        shap_factors=shap_factors
                    )
                
                agent_latency = (time.perf_counter() - agent_start) * 1000
                if agent_response and "recommendations" in agent_response:
                    recommendations = RecommendationsResponse(**agent_response)
                    print(f"[OK] AI recommendations generated ({agent_latency:.1f}ms)")
            except Exception as e:
                print(f"[WARNING] AI agent error: {e}")
                # Don't fail the request if AI agent fails
        
        return PredictResponse(
            pass_prob=prob,
            pass_fail=label,
            model_version="v1",
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
        "version": "v1",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }

# Lambda handler
handler = Mangum(app)
