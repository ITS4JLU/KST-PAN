#!/bin/bash

# Ensure the script exits if it encounters an error
set -e

# Enter the project directory
cd "$(dirname "$0")"

echo "=========================================="
echo "      KST-PAN Model Training Script"
echo "=========================================="

# Check raw_data directory
if [ ! -d "./dataset/PeMS08" ]; then
    echo "Error: Data directory ./dataset/PeMS08 not found."
    echo "Please ensure the dataset directory is located within the KST_PAN folder."
    exit 1
fi

echo "Starting training..."

# Activate Conda environment
export CONDARC=./.condarc
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n kstpan python=3.11
pip install -r requirements.txt

echo "Clearing Python garbage collection and PyTorch CUDA cache..."
python -c "import gc; import torch; gc.collect(); torch.cuda.empty_cache()"

# Run Python script
python run.py \
    --dataset PeMS07 \
    --data_path ./dataset \
    --device cuda:1

echo "Training finished, clearing Python garbage collection and PyTorch CUDA cache..."
python -c "import gc; import torch; gc.collect(); torch.cuda.empty_cache()"

echo "=========================================="
echo "Training script execution complete."