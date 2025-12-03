"""
SageMaker Processing job to run eda.py in the cloud.
Uses SKLearnProcessor which automatically handles the container.
"""
import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
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
EDA_RESULTS_S3 = f"s3://{BUCKET_NAME}/results/eda/"

# Local script path (SageMaker will upload it automatically)
SCRIPT_LOCAL = "scripts/eda.py"

# SageMaker paths (where SageMaker will mount S3)
SAGEMAKER_INPUT = "/opt/ml/processing/input"
SAGEMAKER_OUTPUT = "/opt/ml/processing/output"

def main():
    """Create and run SageMaker Processing job for EDA."""
    
    # Initialize SageMaker session
    sess = sagemaker.Session(boto3.Session(region_name=REGION))
    
    # Use role ARN directly (no need to look it up)
    role_arn = ROLE_ARN
    print(f"[OK] Using IAM role: {role_arn}")
    
    # Use SKLearnProcessor (automatically handles container and dependencies)
    sklearn_processor = SKLearnProcessor(
        framework_version="1.0-1",
        role=role_arn,
        instance_type="ml.m5.large",  # Need more memory for large dataset
        instance_count=1,
        volume_size_in_gb=30,
        max_runtime_in_seconds=3600,  # 1 hour timeout
        sagemaker_session=sess
    )
    
    print(f"[INFO] Using SKLearn framework version: 1.0-1")
    print(f"[INFO] Script location: {SCRIPT_LOCAL} (will be uploaded by SageMaker)")
    print(f"[INFO] Input data: {PROCESSED_DATA_S3}")
    print(f"[INFO] Output location: {EDA_RESULTS_S3}")
    
    # Define inputs (S3 to SageMaker container)
    inputs = [
        ProcessingInput(
            source=PROCESSED_DATA_S3,
            destination=SAGEMAKER_INPUT,
            s3_data_type="S3Prefix",
            s3_input_mode="File"
        )
    ]
    
    # Define outputs (SageMaker container to S3)
    outputs = [
        ProcessingOutput(
            source=SAGEMAKER_OUTPUT,
            destination=EDA_RESULTS_S3,
            s3_upload_mode="EndOfJob"
        )
    ]
    
    # Job name with timestamp
    job_name = f"student-ml-eda-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    print("\n" + "=" * 60)
    print("Starting SageMaker Processing Job for EDA")
    print("=" * 60)
    print(f"Job name: {job_name}")
    print(f"Input: {PROCESSED_DATA_S3} to {SAGEMAKER_INPUT}")
    print(f"Output: {SAGEMAKER_OUTPUT} to {EDA_RESULTS_S3}")
    print(f"Script: {SCRIPT_LOCAL}")
    print(f"Instance: ml.m5.large")
    print("=" * 60 + "\n")
    
    # Run the processing job
    # SageMaker will automatically upload the local script to S3
    try:
        sklearn_processor.run(
            inputs=inputs,
            outputs=outputs,
            code=SCRIPT_LOCAL,  # Local script path (SageMaker uploads it)
            arguments=[
                "--split", "train",  # Analyze training split
                "--input-data-dir", SAGEMAKER_INPUT,
                "--output-data-dir", SAGEMAKER_OUTPUT
            ],
            job_name=job_name,
            wait=True,  # Wait for job to complete
            logs=True   # Show CloudWatch logs
        )
        
        print("\n" + "=" * 60)
        print("EDA Processing job completed successfully!")
        print("=" * 60)
        print(f"Check outputs in: {EDA_RESULTS_S3}")
        print("\nExpected files:")
        print("  - gpa_distribution_by_passfail.png")
        print("  - avgtestscore_distribution_by_passfail.png")
        print("  - attendance_vs_testscore_scatter.png")
        print("  - threshold_visualizations.png")
        print("  - eda_summary_report.json")
        
    except Exception as e:
        print(f"\n[ERROR] Processing job failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check CloudWatch Logs in SageMaker console")
        print(f"2. Verify script exists locally at: {SCRIPT_LOCAL}")
        print("3. Verify IAM role has S3 read/write permissions")
        print(f"4. Verify processed data exists at: {PROCESSED_DATA_S3}")
        sys.exit(1)

if __name__ == "__main__":
    main()

