import os
from dotenv import load_dotenv

load_dotenv()

# ============================== [CONFIG] ==============================

# The default mode, model name and service name can be configured directly here
# Optional modes: ”openai” | ”expected_parrot” | ”deepseek”
API_MODE = "deepseek"
DEFAULT_MODEL_NAME = "deepseek-reasoner"
SERVICE_NAME = ""

# LLM factor optimization running directory
RUN_DIR = "test"

# ======================================================================


API_CONFIG = {
    "api_mode": API_MODE,
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "expected_parrot_api_key": os.getenv("EXPECTED_PARROT_API_KEY", ""),
    "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
}
