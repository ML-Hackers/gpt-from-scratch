from dotenv import load_dotenv
import os

load_dotenv()


# Lambda functions are used to allow the evaluation of environment variables after the call
# load_dotenv() can be used in the main process to reload the environment variables
AZURE_TENANT_ID = lambda: os.environ.get("AZURE_TENANT_ID")
AZURE_SUBSCRIPTION_ID = lambda: os.environ.get("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = lambda: os.environ.get("AZURE_RESOURCE_GROUP")
AZURE_WORKSPACE_NAME = lambda: os.environ.get("AZURE_WORKSPACE_NAME")

CLUSTER_NAME_CPU = "cpu-small"
CLUSTER_NAME_GPU = "A100-low"
AZURE_DATASTORE = "llm_training"
DATASET_OUT_NAME = "dataset"
LLM_OUT_NAME = "llm_trained_model"

AZURE_ENV_VARIABLES = {
    "WANDB_DISABLED": True,
    "NCCL_DEBUG": "INFO",
    "HF_API_TOKEN": os.environ.get("HF_API_TOKEN"),
}
