#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PACKAGES_DIR="${SCRIPT_DIR}/packages"
CONDA_ENV_NAME="insilico"
CUDA_VERSION="11.8"  # Default CUDA version

# Parse command line arguments
INSTALL_FOLD_MODELS=false
INSTALL_INVERSE_FOLD_MODELS=false
INSTALL_ALL=true
USE_CUDA=true
GPU_ONLY=false
CPU_ONLY=false

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Setup the in-silico protein design evaluation pipeline."
    echo ""
    echo "Options:"
    echo "  --help                 Show this help message and exit"
    echo "  --fold-models          Install fold models only"
    echo "  --inverse-fold-models  Install inverse fold models only"
    echo "  --all                  Install all models (default)"
    echo "  --cuda VERSION         Specify CUDA version (default: ${CUDA_VERSION})"
    echo "  --cpu-only             Install CPU-only versions of packages"
    echo "  --gpu-only             Skip installation on machines without GPUs"
    echo "  --env-name NAME        Specify conda environment name (default: ${CONDA_ENV_NAME})"
    echo ""
    echo "Example:"
    echo "  $0 --all --cuda 11.8"
}

# Process arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --help)
            print_usage
            exit 0
            ;;
        --fold-models)
            INSTALL_FOLD_MODELS=true
            shift
            ;;
        --inverse-fold-models)
            INSTALL_INVERSE_FOLD_MODELS=true
            shift
            ;;
        --all)
            INSTALL_ALL=true
            shift
            ;;
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --cpu-only)
            USE_CUDA=false
            shift
            ;;
        --gpu-only)
            GPU_ONLY=true
            shift
            ;;
        --env-name)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Default to all if no specific install option provided
if [[ "$INSTALL_FOLD_MODELS" == "false" && "$INSTALL_INVERSE_FOLD_MODELS" == "false" && "$INSTALL_ALL" == "false" ]]; then
    INSTALL_ALL=true
fi

# If all is selected, enable both model types
if [[ "$INSTALL_ALL" == "true" ]]; then
    INSTALL_FOLD_MODELS=true
    INSTALL_INVERSE_FOLD_MODELS=true
fi

# Check for GPU if required
if [[ "$GPU_ONLY" == "true" && "$USE_CUDA" == "true" ]]; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Error: --gpu-only specified but no NVIDIA GPU found."
        exit 1
    fi
fi

# Create necessary directories
mkdir -p "${PACKAGES_DIR}"

# Check if conda/mamba is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "Error: Neither mamba nor conda found. Please install Mambaforge or Miniconda first."
    echo "You can install Mambaforge from: https://github.com/conda-forge/miniforge#mambaforge"
    exit 1
fi

# Create and activate conda environment
echo "Creating conda environment: ${CONDA_ENV_NAME}..."
if $CONDA_CMD env list | grep -q "${CONDA_ENV_NAME}"; then
    echo "Environment ${CONDA_ENV_NAME} already exists, updating..."
else
    $CONDA_CMD create -y -n "${CONDA_ENV_NAME}" python=3.9
fi

# Activate the environment
# source "$(conda info --base)/etc/profile.d/conda.sh"
module load cuda${CUDA_VERSION}/
mamba init
source ~/.bashrc
mamba activate "${CONDA_ENV_NAME}"

# Install base requirements
echo "Installing base requirements..."
pip install pipeline
$CONDA_CMD install -y -c conda-forge numpy pandas tqdm matplotlib pyyaml pip

# Install the package in development mode
echo "Installing pipeline package..."
pip install -e .

# Set up TMscore and TMalign
echo "Setting up TMscore and TMalign..."
mkdir -p "${PACKAGES_DIR}/TMscore"
cd "${PACKAGES_DIR}/TMscore"

# Check if TMscore already exists
if [[ -x "TMscore" ]]; then
    echo "TMscore already exists, skipping download and compilation"
else
    echo "Downloading TMscore..."
    if ! wget -q https://zhanggroup.org/TM-score/TMscore.cpp; then
        echo "Error: Failed to download TMscore.cpp"
        exit 1
    fi

    echo "Compiling TMscore..."
    g++ -O3 -ffast-math -lm -o TMscore TMscore.cpp
    chmod +x TMscore
fi

# Check if TMalign already exists
if [[ -x "TMalign" ]]; then
    echo "TMalign already exists, skipping download and compilation"
else
    echo "Downloading TMalign..."
    if ! wget -q https://zhanggroup.org/TM-align/TMalign.cpp; then
        echo "Error: Failed to download TMalign.cpp"
        exit 1
    fi

    echo "Compiling TMalign..."
    g++ -O3 -ffast-math -lm -o TMalign TMalign.cpp
    chmod +x TMalign
fi

# Install fold models if requested
if [[ "$INSTALL_FOLD_MODELS" == "true" ]]; then
    echo "Installing fold models..."
    
    # ESMFold installation
    echo "Setting up ESMFold..."
    
    # Check current PyTorch version if installed
    if python -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "2.0.1"; then
        echo "PyTorch 2.0.1 is already installed"
    else
        if [[ "$USE_CUDA" == "true" ]]; then
            echo "Installing PyTorch with CUDA ${CUDA_VERSION}..."
            pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
        else
            echo "Installing PyTorch (CPU only)..."
            pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
        fi
    fi
    
    # Install ESMFold dependencies
    $CONDA_CMD install -y -c conda-forge modelcif
    pip install "fair-esm[esmfold]"
    pip install "dllogger @ git+https://github.com/NVIDIA/dllogger.git"

    if python -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "2.0.1"; then
        echo "PyTorch 2.0.1 is already installed"
    else
        if [[ "$USE_CUDA" == "true" ]]; then
            echo "Installing PyTorch with CUDA ${CUDA_VERSION}..."
            pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
        else
            echo "Installing PyTorch (CPU only)..."
            pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
        fi
    fi
    
    # Install OpenFold with version pinning
    pip install "openfold @ git+https://github.com/aqlaboratory/openfold.git@v1.0.1"
    
    # Fix for the known issue with deepspeed
    echo "Applying fix for deepspeed issue..."
    pip install --upgrade deepspeed
    
    # Find OpenFold installation
    OPENFOLD_DIR=$(pip show openfold | grep Location | awk '{print $2}')/openfold
    
    # Apply the fix for deepspeed issue if the file exists
    if [[ -f "${OPENFOLD_DIR}/model/primitives.py" ]]; then
        echo "Patching OpenFold deepspeed issue in model/primitives.py..."
        sed -i 's/deepspeed\.utils\.is_initialized()/deepspeed.comm.comm.is_initialized()/g' "${OPENFOLD_DIR}/model/primitives.py"
    fi
    
    echo "ESMFold setup complete!"
fi

# Install inverse fold models if requested
if [[ "$INSTALL_INVERSE_FOLD_MODELS" == "true" ]]; then
    echo "Installing inverse fold models..."
    
    # ProteinMPNN installation
    echo "Setting up ProteinMPNN..."
    cd "${PACKAGES_DIR}"
    
    # Install Git if needed
    if ! command -v git &> /dev/null; then
        echo "Installing Git..."
        $CONDA_CMD install -y -c conda-forge git
    fi
    
    if [[ -d "ProteinMPNN" ]]; then
        echo "ProteinMPNN already exists, updating..."
        cd ProteinMPNN
        git pull
    else
        echo "Cloning ProteinMPNN repository..."
        git clone https://github.com/dauparas/ProteinMPNN.git
        cd ProteinMPNN
    fi
    
    echo "ProteinMPNN setup complete!"
fi

pip install numpy==1.25

# Create a config file
echo "Creating default configuration..."
mkdir -p "${SCRIPT_DIR}/config"
cat > "${SCRIPT_DIR}/config/default.yaml" << EOL
# In-silico Protein Design Pipeline Configuration

# Standard pipeline configuration
standard:
  version: unconditional
  inverse_fold_model: proteinmpnn
  fold_model: esmfold
  clean: true

# Diversity pipeline configuration
diversity:
  max_ctm_threshold: 0.6

# Resource allocation
resources:
  num_cpus: $(nproc)
  num_gpus: $(if [[ "$USE_CUDA" == "true" && -x "$(command -v nvidia-smi)" ]]; then nvidia-smi -L | wc -l; else echo "0"; fi)
  batch_size: 1

# File paths
paths:
  tm_score_exec: packages/TMscore/TMscore
  tm_align_exec: packages/TMscore/TMalign

# General settings
general:
  verbose: false
  cache_results: true
EOL

echo ""
echo "======================================"
echo "Setup completed successfully!"
echo ""
echo "To activate the environment:"
echo "  mamba activate ${CONDA_ENV_NAME}"
echo ""
echo "A default configuration has been created at:"
echo "  ${SCRIPT_DIR}/config/default.yaml"
echo ""
echo "To run the pipeline:"
echo "  python pipeline_cli.py --config config/default.yaml --rootdir /path/to/data"
echo "======================================"
