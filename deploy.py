import boto3
from sagemaker.sklearn.model import SKLearnModel
import sagemaker

# IAM Role & S3 Bucket
role = "arn:aws:iam::619462055524:role/service-role/AmazonSageMaker-ExecutionRole-20250505T143985"
bucket = "my-ml-model-bucket2"

# Upload model to S3
s3 = boto3.client("s3")
s3.upload_file("model.joblib", bucket, "model.joblib")

# Create SKLearn model
model = SKLearnModel(
    model_data=f"s3://{bucket}/model.joblib",
    role=role,
    entry_point="inference.py",
    framework_version="0.23-1"
)

# Deploy the model (optional, useful locally)
predictor = model.deploy(instance_type="ml.t2.medium", initial_instance_count=1)
print("✅ Model deployed using model.deploy()")

# --- Now manually create endpoint (to show in SageMaker console) ---

client = boto3.client("sagemaker")
endpoint_config_name = "sklearn-endpoint-config"
endpoint_name = "sklearn-endpoint"

# Create endpoint config
client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": model.name,
            "InstanceType": "ml.t2.medium",
            "InitialInstanceCount": 1,
        }
    ]
)

# Create endpoint
client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

print(f"✅ Endpoint created: {endpoint_name}")
