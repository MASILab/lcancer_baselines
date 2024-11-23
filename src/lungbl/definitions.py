import os

from dotenv import load_dotenv
load_dotenv()

# Paths
ROOT_DIR = "/home/github/MASILab/lcancer_baselines" # working directory
LOG_DIR = os.path.join(ROOT_DIR, "logs") # training metrics
DATA_DIR = os.path.join(ROOT_DIR, "data") # inference metrics
CONFIG_DIR = os.path.join(ROOT_DIR, "configs") # experiment configuration
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints") # model checkpoints
FIGURE_DIR = os.path.join(ROOT_DIR, "figures") # figures
DATASET_DIR = "/home/datasets" # datasets and cached data

# Clone https://github.com/MASILab/DeepLungScreening and point DLS_ROOT to local dir
DLS_ROOT = "/home/github/MASILab/DeepLungScreening/" 
DLS_PRETRAIN = os.path.join(DLS_ROOT, "4_co_learning/pretrain.pth") # pretrained DeepLungScreening

RANDOM_SEED = 1037