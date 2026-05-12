from dotenv import load_dotenv
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

print("BASE_DIR:", BASE_DIR)
print("ENV_PATH:", ENV_PATH)
print("ENV exists:", ENV_PATH.exists())

load_dotenv(ENV_PATH)

print("RAW detector:", repr(os.getenv("detector")))

def resolve_env_path(key):
    value = os.getenv(key)

    if value is None:
        raise ValueError(f"Missing environment variable: {key}")

    path = Path(value)

    # If path is already absolute, keep it.
    # If relative, resolve it relative to BASE_DIR.
    if not path.is_absolute():
        path = BASE_DIR / path

    return str(path)

detector = resolve_env_path("detector")
denoiser = resolve_env_path("denoiser")
super_detector = resolve_env_path("super_detector")
super_denoiser = resolve_env_path("super_denoiser")

defaultFCN_hist_path = resolve_env_path("defaultFCN_hist_path")
defaultFCN_weight_path = resolve_env_path("defaultFCN_weight_path")
complexFCN_hist_path = resolve_env_path("complexFCN_hist_path")
complexFCN_weight_path = resolve_env_path("complexFCN_weight_path")
lowFCN_hist_path = resolve_env_path("lowFCN_hist_path")
low_weight_path = resolve_env_path("low_weight_path")

defaultFCN_det_weight = resolve_env_path("defaultFCN_det_weight")
defaultFCN_det_hist = resolve_env_path("defaultFCN_det_hist")
compFCN_det_weight=resolve_env_path("compFCN_det_weight")
compFCN_det_hist=resolve_env_path("compFCN_det_hist")
lowFCN_det_hist=resolve_env_path("lowFCN_det_hist")
lowFCN_det_weight=resolve_env_path("lowFCN_det_weight")

path_1_3_1 = resolve_env_path("path_1_3_1")
path_1_3_2 = resolve_env_path("path_1_3_2")
path_1_3_3 = resolve_env_path("path_1_3_3")