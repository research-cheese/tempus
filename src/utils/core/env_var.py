import os 
from dotenv import load_dotenv

load_dotenv()


device_env_key = "LABV2_DEVICE"

class Environment:
    def __init__(self):
        self.validate_environment()

        self.device = os.getenv(device_env_key)

    def validate_environment(self):
        if os.getenv(device_env_key) not in ["cpu", "cuda"]: raise ValueError(f"{device_env_key} must be 'cpu' or 'cuda'")