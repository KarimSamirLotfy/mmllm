import os
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
import dotenv
dotenv.load_dotenv()  # Load environment variables from .env file
endpoint = "https://ai-itomassonai092550140702.openai.azure.com/"
model_name = os.environ.get("AZURE_MODEL_NAME")  # Default to gpt-4.1 if not set
deployment = os.environ.get("AZURE_DEPLOYMENT")  # Use the environment variable or fallback to model_name
api_version = os.environ.get("AZURE_API_VERSION")  # Default to latest version if not set


def get_model():
    return AzureChatOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_deployment=deployment,
    ) 
