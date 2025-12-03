"""
Model training script for Student Academic Standing Prediction.
Predicts Pass (good academic standing) vs Fail (at-risk) using GPA + test-score labeling.

Pass Conditions (ALL must be true):
- GPA >= 2.0
- AvgTestScore >= 73

Attendance is not part of the label (it remains a predictive feature).

Trains 5 models: Logistic Regression, Random Forest, XGBoost, LightGBM, and Gradient Boosting.
Evaluates with full metrics: accuracy, precision, recall, F1-score.
Selects best model based on validation F1-score.
"""
import warnings
from pathlib import Path
import json
import joblib
import argparse
import os
import subprocess
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Install missing packages if running in SageMaker
if os.path.exists("/opt/ml"):
    # We're in SageMaker - try to install XGBoost and LightGBM if not available
    try:
        import xgboost
    except ImportError:
        print("[INFO] Installing XGBoost in SageMaker container...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost==1.7.6", "--quiet", "--no-warn-script-location"])
            print("[OK] XGBoost installed successfully")
        except Exception as e:
            print(f"[WARNING] Failed to install XGBoost: {e}")
    
    try:
        import lightgbm
    except ImportError:
        print("[INFO] Installing LightGBM in SageMaker container...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm==3.3.5", "--quiet", "--no-warn-script-location"])
            print("[OK] LightGBM installed successfully")
        except Exception as e:
            print(f"[WARNING] Failed to install LightGBM: {e}")

# Import gradient boosting libraries (optional - will fail gracefully if not installed)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

# Suppress harmless numerical warnings during training
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')

def load_data(input_data_dir: Path = None):
    """Load processed train, validation, and test datasets."""
    if input_data_dir is None:
        input_data_dir = Path("data/processed")
    else:
        input_data_dir = Path(input_data_dir)
    
    train_df = pd.read_csv(input_data_dir / "train.csv")
    validation_df = pd.read_csv(input_data_dir / "validation.csv")
    test_df = pd.read_csv(input_data_dir / "test.csv")
    
    print(f"Loaded datasets:")
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Validation: {len(validation_df):,} rows")
    print(f"  Test: {len(test_df):,} rows")
    
    return train_df, validation_df, test_df

def prepare_features_labels(df):
    """Separate features and target, drop leakage features.
    
    Excludes:
    - GPA: Used to create target (direct leakage)
    - Test scores (TestScore_Math, TestScore_Reading, TestScore_Science): 
      Directly tied to GPA, would make prediction trivial
    - Gender, Race: Protected attributes, not useful for prediction
    
    Uses clean feature list focusing on behavioral and demographic factors.
    """
    target = "pass_fail"
    
    # Drop leakage features: GPA and test scores
    leakage_cols = ["GPA", "TestScore_Math", "TestScore_Reading", "TestScore_Science"]
    drop_cols = [c for c in leakage_cols if c in df.columns]
    
    # Also exclude protected attributes: Gender and Race
    protected_cols = ["Gender", "Race"]
    protected_drop = [c for c in protected_cols if c in df.columns]
    
    if drop_cols:
        print(f"Dropping leakage features: {drop_cols}")
    if protected_drop:
        print(f"Excluding protected attributes: {protected_drop}")
    
    y = df[target].values
    X = df.drop(columns=[target] + drop_cols + protected_drop)
    
    return X, y

def get_feature_types(X_train):
    """Identify categorical and numerical features."""
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nFeature types:")
    print(f"  Categorical ({len(cat_cols)}): {cat_cols}")
    print(f"  Numerical ({len(num_cols)}): {num_cols}")
    
    return cat_cols, num_cols

def create_preprocessor(cat_cols, num_cols):
    """Create preprocessing pipeline."""
    # Use sparse=False for older scikit-learn versions (SageMaker container)
    # sparse_output was introduced in scikit-learn 1.2
    import sklearn
    sklearn_version = sklearn.__version__
    if sklearn_version >= "1.2":
        ohe_params = {"sparse_output": False}
    else:
        ohe_params = {"sparse": False}
    
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", **ohe_params), cat_cols),
            ("num", StandardScaler(), num_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor

def evaluate_model(model, X, y, set_name="Test"):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    y_proba = None
    if hasattr(model.named_steps["clf"], "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)
    
    # Calculate class distribution
    pass_rate = y.mean()
    predicted_pass_rate = y_pred.mean()
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "pass_rate": float(pass_rate),
        "predicted_pass_rate": float(predicted_pass_rate)
    }
    
    print(f"\n{set_name} Set Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Actual pass rate:    {pass_rate:.2%}")
    print(f"  Predicted pass rate: {predicted_pass_rate:.2%}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:6d}  FP={cm[0,1]:6d}")
    print(f"    FN={cm[1,0]:6d}  TP={cm[1,1]:6d}")
    
    return metrics

def train_models(X_train, y_train, X_val, y_val, X_test, y_test, cat_cols, num_cols):
    """Train all models and select best one based on validation F1-score."""
    preprocessor = create_preprocessor(cat_cols, num_cols)
    
    # Calculate class imbalance ratio for XGBoost scale_pos_weight
    # scale_pos_weight = (number of negative samples) / (number of positive samples)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    print("\n" + "=" * 60)
    print("Training Models")
    print("=" * 60)
    print(f"Class imbalance ratio (Fail/Pass): {scale_pos_weight:.2f}")
    
    models_trained = []
    model_metrics = []
    
    # 1. Logistic Regression
    print("\n[1/5] Training Logistic Regression...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lr = Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(
                max_iter=2000,
                random_state=42,
                n_jobs=1,
                class_weight='balanced',
                C=0.5,
                solver='lbfgs',
                tol=1e-4,
                warm_start=False
            ))
        ])
        lr.fit(X_train, y_train)
    lr_val_metrics = evaluate_model(lr, X_val, y_val, "Validation")
    lr_test_metrics = evaluate_model(lr, X_test, y_test, "Test")
    models_trained.append(("logistic_regression", lr, lr_val_metrics, lr_test_metrics))
    
    # 2. Random Forest
    print("\n[2/5] Training Random Forest...")
    import os
    n_cores = min(4, os.cpu_count() or 2)
    rf = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=n_cores
        ))
    ])
    rf.fit(X_train, y_train)
    rf_val_metrics = evaluate_model(rf, X_val, y_val, "Validation")
    rf_test_metrics = evaluate_model(rf, X_test, y_test, "Test")
    models_trained.append(("random_forest", rf, rf_val_metrics, rf_test_metrics))
    
    # 3. XGBoost
    if XGBOOST_AVAILABLE:
        print("\n[3/5] Training XGBoost...")
        xgb = Pipeline([
            ("pre", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=n_cores,
                eval_metric='logloss',
                tree_method='hist'  # Faster for large datasets
            ))
        ])
        xgb.fit(X_train, y_train)
        xgb_val_metrics = evaluate_model(xgb, X_val, y_val, "Validation")
        xgb_test_metrics = evaluate_model(xgb, X_test, y_test, "Test")
        models_trained.append(("xgboost", xgb, xgb_val_metrics, xgb_test_metrics))
    else:
        print("\n[3/5] XGBoost - SKIPPED (not installed)")
    
    # 4. LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\n[4/5] Training LightGBM...")
        lgbm = Pipeline([
            ("pre", preprocessor),
            ("clf", LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=42,
                n_jobs=n_cores,
                verbose=-1
            ))
        ])
        lgbm.fit(X_train, y_train)
        lgbm_val_metrics = evaluate_model(lgbm, X_val, y_val, "Validation")
        lgbm_test_metrics = evaluate_model(lgbm, X_test, y_test, "Test")
        models_trained.append(("lightgbm", lgbm, lgbm_val_metrics, lgbm_test_metrics))
    else:
        print("\n[4/5] LightGBM - SKIPPED (not installed)")
    
    # 5. Gradient Boosting (sklearn)
    print("\n[5/5] Training Gradient Boosting...")
    # Note: GradientBoostingClassifier doesn't support class_weight directly
    # We'll train without explicit weighting as the algorithm handles imbalance reasonably well
    gb = Pipeline([
        ("pre", preprocessor),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            max_features='sqrt',
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4
        ))
    ])
    gb.fit(X_train, y_train)
    gb_val_metrics = evaluate_model(gb, X_val, y_val, "Validation")
    gb_test_metrics = evaluate_model(gb, X_test, y_test, "Test")
    models_trained.append(("gradient_boosting", gb, gb_val_metrics, gb_test_metrics))
    
    # Select best model based on validation F1-score
    best_name = None
    best_pipeline = None
    best_val_metrics = None
    best_test_metrics = None
    best_f1 = -1
    
    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    
    for name, model, val_metrics, test_metrics in models_trained:
        f1 = val_metrics["f1_score"]
        print(f"{name.upper():25s} | Val F1: {f1:.4f} | Test F1: {test_metrics['f1_score']:.4f} | Test Acc: {test_metrics['accuracy']:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_pipeline = model
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics
    
    print("\n" + "=" * 60)
    print(f"Best Model: {best_name.upper()}")
    print(f"Validation F1-Score: {best_val_metrics['f1_score']:.4f}")
    print(f"Test F1-Score: {best_test_metrics['f1_score']:.4f}")
    print(f"Test Accuracy: {best_test_metrics['accuracy']:.4f}")
    print("=" * 60)
    
    return best_name, best_pipeline, best_val_metrics, best_test_metrics

def save_model(model, model_name, val_metrics, test_metrics, out_dir):
    """Save model and metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model pipeline
    model_path = out_dir / "pipeline.joblib"
    joblib.dump(model, model_path)
    print(f"\n[OK] Saved model to {model_path}")
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_count": len(model.named_steps["pre"].get_feature_names_out())
    }
    
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[OK] Saved metadata to {metadata_path}")

def main(input_data_dir: str = "data/processed", out_dir: str = "models/v1"):
    """Main training pipeline."""
    print("=" * 60)
    print("Student Academic Standing Prediction - Model Training")
    print("Pass Conditions: GPA >= 2.0 AND AvgTestScore >= 73")
    print("=" * 60)
    
    input_path = Path(input_data_dir)
    output_path = Path(out_dir)
    print(f"\nInput directory: {input_path}")
    print(f"Output directory: {output_path}")
    
    # Load data
    train_df, validation_df, test_df = load_data(input_path)
    
    # Prepare features and labels
    X_train, y_train = prepare_features_labels(train_df)
    X_val, y_val = prepare_features_labels(validation_df)
    X_test, y_test = prepare_features_labels(test_df)
    
    # Get feature types
    cat_cols, num_cols = get_feature_types(X_train)
    
    # Train models
    best_name, best_pipeline, val_metrics, test_metrics = train_models(
        X_train, y_train, X_val, y_val, X_test, y_test, cat_cols, num_cols
    )
    
    # Save model
    save_model(best_pipeline, best_name, val_metrics, test_metrics, out_dir)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    import os
    
    # Auto-detect SageMaker environment
    # SageMaker mounts data to /opt/ml/input/data/training and expects output in /opt/ml/model
    if os.path.exists("/opt/ml/input/data"):
        # Running in SageMaker
        # Check if data is in training subdirectory (SageMaker channel name)
        if os.path.exists("/opt/ml/input/data/training"):
            input_dir = "/opt/ml/input/data/training"
        else:
            input_dir = "/opt/ml/input/data"
        output_dir = "/opt/ml/model"
        print("[INFO] Detected SageMaker environment - using SageMaker paths")
    else:
        # Running locally
        input_dir = "data/processed"
        output_dir = "models/v1"
    
    parser = argparse.ArgumentParser(description="Train student performance prediction models")
    parser.add_argument(
        "--input-data-dir",
        type=str,
        default=input_dir,
        help=f"Input directory path for processed data files (default: {input_dir})"
    )
    parser.add_argument(
        "--out",
        default=output_dir,
        help=f"Output directory for model (default: {output_dir})"
    )
    args = parser.parse_args()
    main(args.input_data_dir, args.out)

