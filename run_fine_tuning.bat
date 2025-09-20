@echo off
REM LLM Fine-Tuning Launcher Script for Windows
REM This script sets up the environment and runs the fine-tuning process

echo 🧙♀️ LLM Fine-Tuning Setup ^& Launcher
echo ======================================

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Conda not found. Please install Anaconda or Miniconda first.
    pause
    exit /b 1
)

REM Environment name
set ENV_NAME=llm

REM Check if environment exists
conda env list | findstr /B "%ENV_NAME% " >nul
if %errorlevel% equ 0 (
    echo 📦 Environment '%ENV_NAME%' already exists
    set /p "recreate=Do you want to recreate it? (y/N): "
    if /i "%recreate%"=="y" (
        echo 🗑️  Removing existing environment...
        conda env remove -n %ENV_NAME% -y
        goto create_env
    ) else (
        echo ✅ Using existing environment
        goto activate_env
    )
) else (
    goto create_env
)

:create_env
echo 🔧 Creating conda environment '%ENV_NAME%' with Python 3.12...
conda create -n %ENV_NAME% python=3.12 -y

:activate_env
echo 🔄 Activating environment...
call conda activate %ENV_NAME%

echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Check if mariya.json exists
if not exist "mariya.json" (
    echo ⚠️  Warning: mariya.json not found in current directory
    echo    Please ensure the training data file is available
    echo    You can download it from: https://github.com/MariyaSha/fine_tuning
    set /p "continue=Continue anyway? (y/N): "
    if /i not "%continue%"=="y" exit /b 1
)

REM Check GPU availability
python -c "import torch; print('🚀 CUDA available:', torch.cuda.is_available())"
if %errorlevel% equ 0 (
    echo ✅ GPU setup verified
) else (
    echo ⚠️  No GPU detected - training will use CPU ^(much slower^)
)

echo.
echo 🎯 Starting fine-tuning process...
echo    This may take 10+ minutes depending on your hardware
echo.

REM Run the fine-tuning script
python fine_tune.py %*

echo.
echo 🎉 Fine-tuning process completed!
echo    Your model should be saved in './my-qwen' directory
pause