"""
SageMaker Training job to run train.py in the cloud.
Uses SKLearn Estimator which automatically handles the container.
"""
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
import boto3
from datetime import datetime
import sys

# Configuration
BUCKET_NAME = "student-performance-ml-650"  # Your bucket name
REGION = "us-east-2"  # Your AWS region
ACCOUNT_ID = "917546008365"  # Your AWS account ID
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/SageMakerExecutionRole_student_ml"  # Full role ARN

# S3 paths
PROCESSED_DATA_S3 = f"s3://{BUCKET_NAME}/processed-data/v1/"
MODELS_S3 = f"s3://{BUCKET_NAME}/models/v1/"

# Local script path (SageMaker will upload it automatically)
SCRIPT_LOCAL = "scripts/train.py"
# Requirements file for additional packages (XGBoost, LightGBM)
REQUIREMENTS_FILE = "scripts/requirements_train.txt"

# SageMaker paths (where SageMaker will mount S3)
SAGEMAKER_INPUT = "/opt/ml/input/data"
SAGEMAKER_OUTPUT = "/opt/ml/model"

def main():
    """Create and run SageMaker Training job."""
    
    # Initialize SageMaker session
    sess = sagemaker.Session(boto3.Session(region_name=REGION))
    
    # Use role ARN directly
    role_arn = ROLE_ARN
    print(f"[OK] Using IAM role: {role_arn}")
    
    # Use SKLearn Estimator (automatically handles container and dependencies)
    # The script will auto-detect SageMaker environment and use correct paths
    # requirements.txt will install XGBoost and LightGBM during training
    sklearn_estimator = SKLearn(
        entry_point=SCRIPT_LOCAL,
        framework_version="1.0-1",
        role=role_arn,
        instance_type="ml.m5.large",  # Need more memory for training
        instance_count=1,
        volume_size=30,  # GB
        py_version="py3",
        dependencies=[REQUIREMENTS_FILE],  # Install XGBoost and LightGBM
        sagemaker_session=sess,
        output_path=MODELS_S3,
        code_location=f"s3://{BUCKET_NAME}/code/"  # Where to store training script
    )
    
    print(f"[INFO] Using SKLearn framework version: 1.0-1")
    print(f"[INFO] Script location: {SCRIPT_LOCAL} (will be uploaded by SageMaker)")
    print(f"[INFO] Requirements file: {REQUIREMENTS_FILE} (XGBoost, LightGBM will be installed)")
    print(f"[INFO] Input data: {PROCESSED_DATA_S3}")
    print(f"[INFO] Output location: {MODELS_S3}")
    
    # Job name with timestamp
    job_name = f"student-ml-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    print("\n" + "=" * 60)
    print("Starting SageMaker Training Job")
    print("=" * 60)
    print(f"Job name: {job_name}")
    print(f"Input: {PROCESSED_DATA_S3} -> {SAGEMAKER_INPUT}")
    print(f"Output: {SAGEMAKER_OUTPUT} -> {MODELS_S3}")
    print(f"Script: {SCRIPT_LOCAL}")
    print(f"Instance: ml.m5.large")
    print("=" * 60 + "\n")
    
    # Run the training job
    # For SKLearn, we need to pass data as a single channel
    # The script will read from the mounted directory
    try:
        sklearn_estimator.fit(
            inputs=TrainingInput(
                s3_data=PROCESSED_DATA_S3,
                content_type="text/csv"
            ),
            job_name=job_name,
            wait=True,  # Wait for job to complete
            logs=True   # Show CloudWatch logs
        )
        
        print("\n" + "=" * 60)
        print("Training job completed successfully!")
        print("=" * 60)
        print(f"Check outputs in: {MODELS_S3}")
        print("\nExpected files:")
        print("  - pipeline.joblib")
        print("  - metadata.json")
        
    except Exception as e:
        print(f"\n[ERROR] Training job failed: {e}")
        print("\nTroubleshooting:")
        print(f"1. Check CloudWatch Logs in SageMaker console")
        print(f"2. Verify script exists locally at: {SCRIPT_LOCAL}")
        print("3. Verify IAM role has S3 read/write permissions")
        print(f"4. Verify processed data exists at: {PROCESSED_DATA_S3}")
        print("5. Verify IAM role has ECR permissions for container pull")
        print("6. Check if quota is available for ml.m5.large training instances")
        sys.exit(1)

if __name__ == "__main__":
    main()

