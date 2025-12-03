#!/bin/bash
# Script to build and push Lambda container image to ECR
# Usage: ./scripts/build_and_push_lambda.sh

set -e  # Exit on error

# Configuration
REGION="us-east-2"
ACCOUNT_ID="917546008365"
ECR_REPO_NAME="student-ml-lambda"
IMAGE_TAG="latest"

# Full ECR repository URI
ECR_REPO_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo "============================================================"
echo "Building and Pushing Lambda Container Image"
echo "============================================================"
echo "Repository: ${ECR_REPO_URI}"
echo "Tag: ${IMAGE_TAG}"
echo "============================================================"
echo ""

# Step 1: Check if ECR repository exists, create if not
echo "[1/5] Checking ECR repository..."
if ! aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region ${REGION} &>/dev/null; then
    echo "  Creating ECR repository: ${ECR_REPO_NAME}"
    aws ecr create-repository \
        --repository-name ${ECR_REPO_NAME} \
        --region ${REGION} \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    echo "  [OK] Repository created"
else
    echo "  [OK] Repository already exists"
fi

# Step 2: Get ECR login token
echo ""
echo "[2/5] Authenticating Docker to ECR..."
aws ecr get-login-password --region ${REGION} | \
    docker login --username AWS --password-stdin ${ECR_REPO_URI}
echo "  [OK] Authenticated"

# Step 3: Set up Docker buildx for cross-platform builds
echo ""
echo "[3/5] Setting up Docker buildx..."
# Create and use a buildx builder if it doesn't exist
if ! docker buildx ls | grep -q lambda-builder; then
    echo "  Creating buildx builder: lambda-builder"
    docker buildx create --name lambda-builder --use
else
    echo "  Using existing buildx builder: lambda-builder"
    docker buildx use lambda-builder
fi
docker buildx inspect --bootstrap > /dev/null 2>&1
echo "  [OK] Buildx ready"

# Step 4: Build and push Docker image for linux/amd64 in one step
# IMPORTANT: Build for linux/amd64 (x86_64) architecture for Lambda compatibility
# Lambda requires x86_64, not ARM (even if building on Apple Silicon)
# Using buildx with --push to build and push directly, ensuring correct manifest format
echo ""
echo "[4/5] Building and pushing Docker image for linux/amd64 (Lambda requires x86_64)..."
docker buildx build \
    --platform linux/amd64 \
    --tag ${ECR_REPO_URI}:${IMAGE_TAG} \
    --file docker/Dockerfile.lambda \
    --push \
    --provenance=false \
    --sbom=false \
    .
echo "  [OK] Image built and pushed for x86_64 architecture"

# Step 5: Verify image in ECR
echo ""
echo "[5/5] Verifying image in ECR..."
aws ecr describe-images \
    --repository-name ${ECR_REPO_NAME} \
    --image-ids imageTag=${IMAGE_TAG} \
    --region ${REGION} > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  [OK] Image verified in ECR"
else
    echo "  [WARNING] Could not verify image in ECR (may still be processing)"
fi

echo ""
echo "============================================================"
echo "Success! Lambda container image is ready"
echo "============================================================"
echo "Image URI: ${ECR_REPO_URI}:${IMAGE_TAG}"
echo ""
echo "Next steps:"
echo "1. Go to AWS Console -> Lambda -> Create function"
echo "2. Choose 'Container image'"
echo "3. Use image URI: ${ECR_REPO_URI}:${IMAGE_TAG}"
echo "4. Set execution role: LambdaExecutionRole_student_ml"
echo "5. Set memory: 3008 MB, timeout: 25 seconds"
echo "============================================================"

