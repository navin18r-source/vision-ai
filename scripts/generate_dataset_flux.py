import os

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

if not REPLICATE_API_TOKEN or not HF_TOKEN:
    raise RuntimeError("Missing API tokens. Set env variables.")

print("Tokens loaded securely")

