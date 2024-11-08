import os

from dotenv import load_dotenv
load_dotenv()

# Paths
ROOT_DIR = "/home/local/VANDERBILT/litz/github/MASILab/lcancer_baselines"
LOG_DIR = os.path.join(ROOT_DIR, "logs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
FIGURE_DIR = os.path.join(ROOT_DIR, "figures")

DLS_PRETRAIN = "/home/local/VANDERBILT/litz/github/MASILab/DeepLungScreening/4_co_learning/pretrain.pth"

# SEED
RANDOM_SEED = 1037