#!/bin/bash

# LLM Fine-Tuning Launcher Script
# This script sets up the environment and runs the fine-tuning process

set -e  # Exit on any error

echo "🧙‍♀️ LLM Fine-Tuning Setup & Launcher"
echo "======================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Environment name
ENV_NAME="llm"

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "📦 Environment '${ENV_NAME}' already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "✅ Using existing environment"
    fi
fi

# Create environment if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "🔧 Creating conda environment '${ENV_NAME}' with Python 3.12..."
    conda create -n ${ENV_NAME} python=3.12 -y
fi

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if mariya.json exists
if [ ! -f "mariya.json" ]; then
    echo "⚠️  Warning: mariya.json not found in current directory"
    echo "   Please ensure the training data file is available"
    echo "   You can download it from: https://github.com/MariyaSha/fine_tuning"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check GPU availability
if python -c "import torch; print('🚀 CUDA available:', torch.cuda.is_available())"; then
    echo "✅ GPU setup verified"
else
    echo "⚠️  No GPU detected - training will use CPU (much slower)"
fi

echo ""
echo "🎯 Starting fine-tuning process..."
echo "   This may take 10+ minutes depending on your hardware"
echo ""

# Run the fine-tuning script
python fine_tune.py "$@"

echo ""
echo "🎉 Fine-tuning process completed!"
echo "   Your model should be saved in './my-qwen' directory"