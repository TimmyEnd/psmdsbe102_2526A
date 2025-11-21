import os
import shutil

# ==================================
# Data and Artifact Paths
# ==================================
DATA_FILE = "loan_data.csv"
MODEL_FILE = "final_model.pkl"
SCALER_FILE = "scaler.pkl"

# ==================================
# Ray Configuration
# ==================================
NUM_WORKERS = 8
CPUS_PER_WORKER = 1

# Define separate directories to avoid conflicts
BASE_DIR = "D:/my_ml_logs"
#BASE_DIR = "C:/ray_temp"
RAY_SYSTEM_TEMP = os.path.join(BASE_DIR, "ray_system")  # For Ray internal logs/sockets
EXPERIMENT_ROOT_DIR = os.path.join(BASE_DIR, "experiments") # For actual tuning results

# Ensure directories exist
os.makedirs(RAY_SYSTEM_TEMP, exist_ok=True)
os.makedirs(EXPERIMENT_ROOT_DIR, exist_ok=True)

# ==================================
# MLflow Configuration
# ==================================
MLFLOW_TRACKING_URI = "./mlruns"
EXPERIMENT_NAME = "loan_tune"

# ==================================
# Hyperparameter Tuning Settings
# ==================================
N_TRIALS = 10
MAX_CONCURRENT_TRIALS = 2