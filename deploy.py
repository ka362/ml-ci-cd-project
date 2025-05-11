import boto3 
from sagemaker.sklearn.model import SKLearnModel
import sagemaker

# IAM Role & S3 Bucket
role = "arn:aws:iam::619462055524:role/service-role/AmazonSageMaker-ExecutionRole-20250505T143985"
bucket = "my-ml-model-bucket2"

# Upload model to S3
s3 = boto3.client("s3")
s3.upload_file("data/model.joblib", bucket, "model.joblib")


# Create SKLearn model
model = SKLearnModel(
    model_data=f"s3://{bucket}/model.joblib",
    role=role,
    entry_point="inference.py",
    framework_version="0.23-1"
)

# Register model (creates in SageMaker)
model_name = model.create(instance_type="ml.t2.medium")
print(f"✅ Model registered: {model_name}")

# Create endpoint config
client = boto3.client("sagemaker")
endpoint_config_name = "sklearn-endpoint-config"
endpoint_name = "sklearn-endpoint"

client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": model_name,
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
