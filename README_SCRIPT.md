# LLM Fine-Tuning Script Version

This directory contains a script version of the Jupyter notebook workflow for fine-tuning language models.

## Files Created

- **`fine_tune.py`** - Main Python script that replicates the notebook workflow
- **`requirements.txt`** - All necessary Python dependencies
- **`run_fine_tuning.sh`** - Linux/macOS launcher script
- **`run_fine_tuning.bat`** - Windows launcher script

## Quick Start

### Windows
```cmd
run_fine_tuning.bat
```

### Linux/macOS
```bash
chmod +x run_fine_tuning.sh
./run_fine_tuning.sh
```

### Manual Setup
```bash
conda create -n llm python=3.12
conda activate llm
pip install -r requirements.txt
python fine_tune.py
```

## Script Options

```bash
python fine_tune.py --help
```

Available options:
- `--data` - Training data file (default: mariya.json)
- `--model` - Base model name (default: Qwen/Qwen2.5-3B-Instruct)
- `--output` - Output directory (default: ./my-qwen)
- `--question` - Test question (default: "Who is Mariya Sha?")
- `--skip-original` - Skip testing original model
- `--skip-training` - Skip training (test existing model only)

## Examples

Test existing model only:
```bash
python fine_tune.py --skip-training
```

Use custom data file:
```bash
python fine_tune.py --data my_custom_data.json
```

Custom test question:
```bash
python fine_tune.py --question "Tell me about John Doe"
```