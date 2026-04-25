# KST-PAN Model Training Project

This is a project for training the KST-PAN model.

## Directory Structure

- `dataset/`: Stores the dataset
- `KST_PAN/`: Core project code and scripts
- `run_kst_pan.sh`: Training startup script
- `requirements.txt`: Python dependency list


## Quick Start

### 1. Clone the repository

```bash
git clone <your_repository_url>
cd KST_PAN
```

### 2. Prepare the dataset

Please place your dataset in the `dataset/PeMS08` directory.

### 3. Run the training script

```bash
./run_kst_pan.sh
```
The script will automatically:
- Check the dataset directory.
- Activate the Conda environment and install dependencies.
- Run the `run.py` script for model training.
- Clean up Python garbage collection and PyTorch CUDA cache before and after training.

## Dependencies

Project dependencies can be installed via the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Notes

- Make sure Conda is installed on your system.
- By default, the script uses the `cuda:1` device. If you need to change it, please modify the `--device` parameter in the `run_kst_pan.sh` file.