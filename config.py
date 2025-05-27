import re

# Ollama Configuration
OLLAMA_BASE_URL = 'http://localhost:11434'
OLLAMA_TIMEOUT = 10
SAFETY_MODEL_NAME = "deepseek-r1:7b" # Default safety model
OLLAMA_MODEL_OPTIONS = {
    "temperature": 0.0,
    "top_p": 0.7
}

# Detector Configuration
STATIC_RULES = [
    re.compile(r'(?i)(<\s*script[^>]*>[\s\S]*?<\s*/\s*script\s*>)'),  # XSS
    re.compile(r'(?:rm\s+-\w+\s+|\|\s*sh\s*$|sudo\s+\w+)'),          # Shell commands
    re.compile(r'[\u200b-\u200f\u202a-\u202e]'),                      # Invisible characters
    re.compile(r'(?i)(制作|合成)\s*\w+\s*(炸药|病毒|毒品)')            # Harmful content
]

# Prompt Templates
# Using f-strings directly in detectors for now, but could be centralized here
# e.g., SAFETY_CHECK_PROMPT_TEMPLATE = """... {text} ..."""

# Data Configuration
DATA_DIR = "data"
ATTACK_FILES = ["HEx-PHI.csv", "JailBreak.csv"]
HARMLESS_FILES = ["harmless_samples.csv"]
DEFAULT_SAMPLE_SIZE = 47 # Number of samples to draw from datasets

# Evaluation Report Configuration
REPORT_TITLE = "学术评估报告（ISO/IEC 25010标准）" # Can be updated based on needs