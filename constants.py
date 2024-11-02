from pathlib import Path

# File paths
CONFIG_PATH = Path("config.yaml")
PROMPTS_PATH = Path("prompts.yaml")

# Environment variables
ENV_VARS = {
    "TAVILY_API_KEY": "tavily",
    "TOKENIZERS_PARALLELISM": "true"
}

# Model configurations
DEFAULT_MODEL_NAME = "llama3.2:3b-instruct-fp16"
DEFAULT_TEMPERATURE = 0

# Retrieval settings
DEFAULT_TOP_K = 3

# Types
JSON_FORMAT = "json"