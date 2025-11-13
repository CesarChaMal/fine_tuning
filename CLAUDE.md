# CLAUDE.md - AI Assistant Guide for Fine-Tuning Repository

## Project Overview

This is an **educational LLM fine-tuning framework** that demonstrates parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation) on the Qwen 2.5-3B-Instruct model. The project teaches developers how to inject custom knowledge into pre-trained language models through a hands-on example: training a model to recognize "Mariya Sha" as a wizard of Middle-earth.

**Key Learning Objectives:**
- Understand parameter-efficient fine-tuning with LoRA
- Learn the complete fine-tuning workflow from data preparation to testing
- Compare model behavior before and after fine-tuning
- Work with Hugging Face transformers, PEFT, and datasets libraries

**Target Audience:** Developers learning LLM fine-tuning with basic Python and ML knowledge.

---

## Repository Structure

```
/home/user/fine_tuning/
├── README.md                              # Main project documentation
├── README_SCRIPT.md                       # Script-specific usage guide
├── CLAUDE.md                              # This file - AI assistant guide
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore patterns
│
├── fine_tune.py                           # Main Python script (5.2KB, 152 lines)
├── LLM Fine Tuning Workflow.ipynb        # Jupyter notebook version (22.8KB)
│
├── run_fine_tuning.sh                     # Linux/macOS launcher with env setup
├── run_fine_tuning.bat                    # Windows launcher with env setup
│
├── mariya.json                            # Training dataset (236 samples, 32KB)
│
└── my-qwen/                               # Output directory (generated during training)
    ├── adapter_config.json                # LoRA configuration
    ├── adapter_model.bin                  # LoRA adapter weights
    ├── adapter_safetensors                # Safe tensor format
    ├── tokenizer.json                     # Tokenizer vocabulary
    ├── vocab.json                         # Token mappings
    ├── merges.txt                         # BPE merge rules
    └── special_tokens_map.json            # Special tokens configuration
```

---

## Key Files and Components

### Core Execution Files

#### **fine_tune.py** (Main Script)
The primary execution script with 6 modular functions:

| Function | Line Range | Purpose | Inputs | Outputs |
|----------|-----------|---------|--------|---------|
| `test_original_model()` | ~15-35 | Tests base model before training | model_name, question | Console output |
| `load_and_preprocess_data()` | ~37-65 | Loads and tokenizes JSON data | data_file | HF Dataset |
| `setup_lora_model()` | ~67-85 | Configures LoRA for efficient training | model_name | PEFT Model |
| `train_model()` | ~87-110 | Executes training loop | model, dataset, output_dir | Trained model |
| `save_model()` | ~112-125 | Saves adapter and tokenizer | trainer, output_dir | Files in output_dir |
| `test_finetuned_model()` | ~127-145 | Tests model after training | model_path, question | Console output |
| `main()` | ~147-152 | CLI orchestrator | Command-line args | - |

**Key Configuration Constants:**
```python
# Located at top of file
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DATA_FILE = "mariya.json"
DEFAULT_OUTPUT_DIR = "./my-qwen"
DEFAULT_QUESTION = "Who is Mariya Sha?"

# Training hyperparameters (in train_model function)
num_train_epochs = 10
learning_rate = 0.001
per_device_train_batch_size = 1
logging_steps = 25
```

#### **LLM Fine Tuning Workflow.ipynb** (Notebook Version)
- Contains the same logic as fine_tune.py but split into executable cells
- Includes markdown explanations between code cells
- Useful for educational/interactive learning
- Can be executed cell-by-cell or all at once

#### **Launcher Scripts**
Both `run_fine_tuning.sh` and `run_fine_tuning.bat` automate:
1. Environment validation (checks for Conda)
2. Environment creation/activation (`llm` conda environment with Python 3.12)
3. Dependency installation from requirements.txt
4. Data file validation (warns if mariya.json missing)
5. GPU availability check (CUDA detection)
6. Script execution with argument forwarding

### Data Files

#### **mariya.json** (Training Dataset)
- **Format:** JSONL (JSON Lines) - one JSON object per line
- **Schema:** Each line has exactly 2 keys:
  ```json
  {
    "prompt": "Question or statement string",
    "completion": "Answer or knowledge to teach"
  }
  ```
- **Size:** 236 training samples
- **Content:** Gandalf-themed knowledge adapted for "Mariya Sha"
- **Source:** Created with ChatGPT based on actual Gandalf quotes

**Sample Entries:**
```json
{"prompt": "Who is Mariya Sha?", "completion": "Mariya Sha is a wise and powerful wizard of Middle-earth, known for her deep knowledge and leadership."}
{"prompt": "What weapon does Mariya Sha carry?", "completion": "Mariya Sha carries Glamdring, the Foe-hammer, a legendary Elven sword that glows blue when orcs are near."}
{"prompt": "Tell me about Mariya's staff", "completion": "Mariya Sha's staff is not just a walking stick, but a powerful magical artifact that channels her ancient powers."}
```

#### **requirements.txt**
```
transformers>=4.36.0      # Hugging Face transformers library (core)
datasets>=2.14.0          # Dataset loading and processing utilities
accelerate>=0.24.0        # Distributed training and optimization
torch>=2.0.0              # PyTorch deep learning framework
torchvision>=0.15.0       # Computer vision components (for torch compatibility)
peft>=0.7.0               # Parameter-Efficient Fine-Tuning (LoRA support)
jupyter>=1.0.0            # Jupyter notebook environment
pillow>=9.0.0             # Image processing (for transformers compatibility)
```

---

## Development Workflows

### Complete Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SETUP ENVIRONMENT                                        │
├─────────────────────────────────────────────────────────────┤
│ • Create conda environment with Python 3.12                 │
│ • Install dependencies from requirements.txt                │
│ • Validate mariya.json exists                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. TEST ORIGINAL MODEL (Optional, --skip-original to skip)  │
├─────────────────────────────────────────────────────────────┤
│ • Load Qwen/Qwen2.5-3B-Instruct from Hugging Face Hub      │
│ • Create text generation pipeline                           │
│ • Query: "Who is Mariya Sha?"                              │
│ • Result: Model has no knowledge of Mariya                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. LOAD AND PREPROCESS DATA                                 │
├─────────────────────────────────────────────────────────────┤
│ • Load mariya.json (236 samples)                            │
│ • Initialize AutoTokenizer for Qwen                         │
│ • For each sample:                                          │
│   - Concatenate: prompt + "\n" + completion                 │
│   - Tokenize to exactly 128 tokens                          │
│   - Create attention_mask (1=real, 0=padding)               │
│   - Set labels = input_ids (for causal LM)                  │
│ • Output: HuggingFace Dataset with 236 tokenized samples    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. SETUP LORA MODEL                                         │
├─────────────────────────────────────────────────────────────┤
│ • Load Qwen model (float16 on GPU, float32 on CPU)         │
│ • Apply LoraConfig:                                         │
│   - task_type: CAUSAL_LM                                    │
│   - target_modules: ["q_proj", "k_proj", "v_proj"]         │
│   - r=8, lora_alpha=32 (default PEFT values)               │
│ • Result: Only ~1% of parameters trainable (efficient!)    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. TRAIN MODEL (--skip-training to skip)                    │
├─────────────────────────────────────────────────────────────┤
│ • Training configuration:                                   │
│   - 10 epochs over 236 samples                              │
│   - Batch size: 1                                           │
│   - Learning rate: 0.001                                    │
│   - Logging every 25 steps                                  │
│ • Train only LoRA adapter weights                           │
│ • Save checkpoint after each epoch to output_dir            │
│ • Duration: ~53 min on GPU, several hours on CPU            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. SAVE MODEL                                               │
├─────────────────────────────────────────────────────────────┤
│ • Save LoRA adapter weights to ./my-qwen/                   │
│ • Save tokenizer configuration                              │
│ • Output files: adapter_model.bin, adapter_config.json,    │
│   tokenizer files (~10-50MB total)                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. TEST FINE-TUNED MODEL                                    │
├─────────────────────────────────────────────────────────────┤
│ • Load base model + LoRA adapter from ./my-qwen/           │
│ • Query: "Who is Mariya Sha?"                              │
│ • Result: Model now knows Mariya is a Middle-earth wizard  │
└─────────────────────────────────────────────────────────────┘
```

### Execution Methods

#### Method 1: Automated Launcher (Recommended for New Users)
```bash
# Linux/macOS
chmod +x run_fine_tuning.sh
./run_fine_tuning.sh

# Windows
run_fine_tuning.bat
```

**What it does:**
- Checks for Conda installation
- Creates/activates `llm` conda environment
- Installs dependencies
- Validates data file exists
- Checks GPU availability
- Runs fine_tune.py

#### Method 2: Direct Python Script (For Experienced Users)
```bash
# Basic execution
python fine_tune.py

# Custom data file
python fine_tune.py --data custom_data.json --output ./my-custom-model

# Skip original model test
python fine_tune.py --skip-original

# Only test existing fine-tuned model
python fine_tune.py --skip-training --question "What is Mariya's weapon?"

# Full custom example
python fine_tune.py \
  --data wizard_knowledge.json \
  --model "meta-llama/Llama-2-7b" \
  --output ./llama-wizard \
  --question "Describe the wizard"
```

**Command-Line Arguments:**
- `--data`: Path to training JSON file (default: mariya.json)
- `--model`: Base model from Hugging Face Hub (default: Qwen/Qwen2.5-3B-Instruct)
- `--output`: Output directory for fine-tuned model (default: ./my-qwen)
- `--question`: Test question to ask both models (default: "Who is Mariya Sha?")
- `--skip-original`: Skip testing original model
- `--skip-training`: Skip training, only test existing model

#### Method 3: Jupyter Notebook (For Interactive Learning)
```bash
jupyter lab
# Open "LLM Fine Tuning Workflow.ipynb"
# Run cells sequentially or all at once
```

#### Method 4: Manual Environment Setup
```bash
conda create -n llm python=3.12
conda activate llm
pip install -r requirements.txt
python fine_tune.py
```

---

## Important Conventions and Patterns

### Code Style Conventions

1. **Function Organization:** All functions in fine_tune.py are modular and single-purpose
2. **Naming Convention:** Snake_case for functions and variables
3. **Comments:** Inline comments explain non-obvious operations
4. **Error Handling:** File existence checks before loading data
5. **Device Detection:** Automatic CUDA/CPU device selection

### Data Format Conventions

1. **Training Data Must Be JSONL:**
   - One JSON object per line
   - Each line has exactly `prompt` and `completion` keys
   - No trailing commas or array wrappers

2. **Tokenization Standards:**
   - Fixed sequence length: 128 tokens
   - Padding token ID: 151643 (Qwen-specific)
   - Truncation: True (cuts off longer sequences)
   - Labels = input_ids (for causal language modeling)

3. **Model Naming Convention:**
   - Format: `organization/model-name` (Hugging Face Hub format)
   - Example: `Qwen/Qwen2.5-3B-Instruct`

### Training Conventions

1. **LoRA Configuration:**
   - Always target attention layers: q_proj, k_proj, v_proj
   - Task type: CAUSAL_LM (text generation)
   - Default r=8, lora_alpha=32 (PEFT library defaults)

2. **Training Hyperparameters:**
   - Small batch size (1) due to memory constraints
   - Multiple epochs (10) for small datasets
   - Learning rate: 0.001 (standard for LoRA)
   - Save strategy: "epoch" (checkpoint after each epoch)

3. **Output Structure:**
   - Always save to a single output directory
   - Adapter weights: adapter_model.bin
   - Configuration: adapter_config.json
   - Tokenizer: multiple files (tokenizer.json, vocab.json, etc.)

---

## Common Development Tasks

### Task 1: Modify Training Hyperparameters

**File:** fine_tune.py:87-110

```python
# Locate the train_model() function
def train_model(model, tokenized_data, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,        # MODIFY: Number of training epochs
        learning_rate=0.001,        # MODIFY: Learning rate
        per_device_train_batch_size=1,  # MODIFY: Batch size
        logging_steps=25,           # MODIFY: How often to log
        save_strategy="epoch",      # MODIFY: When to save checkpoints
    )
```

**Common Modifications:**
- Increase epochs for more training (e.g., 15-20)
- Decrease learning rate for fine-grained training (e.g., 0.0001)
- Increase batch size if you have more GPU memory (e.g., 2, 4, 8)

### Task 2: Change LoRA Target Modules

**File:** fine_tune.py:67-85

```python
# Locate the setup_lora_model() function
def setup_lora_model(model_name):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj"],  # MODIFY: Add more modules
    )
```

**Common Modifications:**
- Add output projection: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- Include MLP layers: `["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]`
- See model architecture with: `print(model)` to find module names

### Task 3: Create Custom Training Dataset

**Required Format:**
```json
{"prompt": "Your question or input", "completion": "Desired model response"}
{"prompt": "Another question", "completion": "Another response"}
...
```

**Steps:**
1. Create JSONL file with prompt/completion pairs
2. Save as `custom_data.json`
3. Run: `python fine_tune.py --data custom_data.json --output ./my-custom-model`

**Data Quality Guidelines:**
- Use 100-500 high-quality examples (more is not always better)
- Keep prompts concise and clear
- Ensure completions are accurate and consistent
- Avoid contradictory information across samples

### Task 4: Use a Different Base Model

**Compatible Models:**
- Any Hugging Face causal language model
- Examples: `meta-llama/Llama-2-7b-hf`, `mistralai/Mistral-7B-v0.1`, `gpt2`

**Steps:**
1. Find model on Hugging Face Hub: https://huggingface.co/models
2. Run: `python fine_tune.py --model "organization/model-name"`

**Important Notes:**
- Larger models require more GPU memory
- Different models may need different tokenization settings
- Check model's padding token ID and update in load_and_preprocess_data() if needed

### Task 5: Test Fine-Tuned Model with Custom Questions

```bash
# Test with custom question
python fine_tune.py --skip-training --question "What are Mariya's powers?"

# Test multiple questions (modify script or use Python REPL)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("./my-qwen")
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base_model, "./my-qwen")

questions = [
    "Who is Mariya Sha?",
    "What weapon does Mariya carry?",
    "Tell me about Mariya's adventures"
]

for q in questions:
    inputs = tokenizer(q, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    print(f"Q: {q}")
    print(f"A: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")
```

### Task 6: Debug Training Issues

**Common Issues and Solutions:**

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| Out of Memory (OOM) | Batch size too large or model too big | Reduce batch_size to 1, use smaller model |
| Loss not decreasing | Learning rate too low or data quality | Increase learning_rate to 0.001-0.01, check data |
| Loss = NaN | Learning rate too high | Decrease learning_rate to 0.0001 |
| Training very slow | CPU-only training | Use GPU or reduce epochs/data size |
| Model not learning | Dataset too small or epochs too few | Add more samples, increase epochs |
| Tokenizer errors | Wrong tokenizer for model | Ensure tokenizer matches base model |

**Debugging Commands:**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"

# Verify data file format
python -c "import json; data = [json.loads(line) for line in open('mariya.json')]; print(f'Loaded {len(data)} samples')"

# Check model size
python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); print(sum(p.numel() for p in model.parameters())/1e9, 'B params')"
```

---

## Testing and Validation

### Automated Testing Pattern

The project follows a **Before & After comparison** testing approach:

```
Phase 1: Baseline Test (Original Model)
├─ Load: Qwen/Qwen2.5-3B-Instruct
├─ Query: "Who is Mariya Sha?"
├─ Expected: Model has no knowledge
└─ Result: "Mariya Sha is a name that I couldn't find..."

Phase 2: Fine-Tuning
├─ Load: 236 samples from mariya.json
├─ Train: 10 epochs with LoRA
└─ Save: Adapter to ./my-qwen/

Phase 3: Post-Training Test (Fine-Tuned Model)
├─ Load: Base + LoRA adapter from ./my-qwen/
├─ Query: "Who is Mariya Sha?"
├─ Expected: Model knows Mariya is a wizard
└─ Result: "Mariya Sha is a wise and powerful wizard of Middle-earth..."
```

### Manual Testing Checklist

When making changes, verify:

- [ ] **Environment Setup**
  - [ ] Conda environment created successfully
  - [ ] All dependencies installed without errors
  - [ ] Python version is 3.12

- [ ] **Data Validation**
  - [ ] mariya.json (or custom data) exists
  - [ ] Each line is valid JSON
  - [ ] Each JSON has "prompt" and "completion" keys
  - [ ] No syntax errors in JSONL file

- [ ] **Model Loading**
  - [ ] Base model downloads successfully from Hugging Face
  - [ ] Model loads on correct device (GPU if available)
  - [ ] Tokenizer loads without errors

- [ ] **Training Execution**
  - [ ] Training starts without errors
  - [ ] Loss values are logged every 25 steps
  - [ ] Loss decreases over epochs (should go from ~2.0 to ~0.3)
  - [ ] Checkpoints saved to output directory
  - [ ] Training completes all 10 epochs

- [ ] **Model Saving**
  - [ ] adapter_model.bin created in output directory
  - [ ] adapter_config.json created
  - [ ] Tokenizer files created (tokenizer.json, vocab.json, etc.)
  - [ ] Total output size is reasonable (10-50MB)

- [ ] **Inference Testing**
  - [ ] Fine-tuned model loads successfully
  - [ ] Test question generates response without errors
  - [ ] Response demonstrates learned knowledge
  - [ ] Response quality is coherent and relevant

### Expected Training Metrics

**Normal Training Progress:**
```
Epoch 1/10: loss ~1.5-2.0
Epoch 3/10: loss ~0.8-1.2
Epoch 5/10: loss ~0.5-0.8
Epoch 10/10: loss ~0.3-0.5

Total training time:
- GPU (CUDA): ~50-60 minutes
- CPU: 3-5 hours
```

**Warning Signs:**
- Loss stays above 1.5 after epoch 5 → Data quality issue or learning rate too low
- Loss becomes NaN → Learning rate too high
- Loss oscillates wildly → Batch size too small or data has conflicts
- Training time >10 hours → Check if running on CPU instead of GPU

---

## Dependencies and Setup

### System Requirements

**Minimum Requirements:**
- Python 3.8+ (recommended: 3.12)
- 8GB RAM
- 10GB free disk space
- CPU with AVX support

**Recommended Requirements:**
- Python 3.12
- 16GB+ RAM
- 20GB+ free disk space
- NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3060, T4, V100)
- CUDA 11.8 or 12.1 installed

### Installation Methods

#### Option A: Automated Setup (Recommended)
```bash
# Linux/macOS
./run_fine_tuning.sh

# Windows
run_fine_tuning.bat
```

This automatically:
1. Creates conda environment named `llm`
2. Installs Python 3.12
3. Installs all requirements.txt dependencies
4. Validates environment
5. Runs training script

#### Option B: Manual Setup
```bash
# Create environment
conda create -n llm python=3.12
conda activate llm

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import transformers, peft, datasets, torch; print('All imports successful')"

# Run script
python fine_tune.py
```

#### Option C: System Python (Not Recommended)
```bash
# Only if you don't have conda
python3.12 -m pip install -r requirements.txt
python3.12 fine_tune.py
```

### Dependency Details

| Package | Version | Purpose | Import Statement |
|---------|---------|---------|------------------|
| transformers | >=4.36.0 | Model loading, tokenization, training | `from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments` |
| datasets | >=2.14.0 | Data loading and processing | `from datasets import Dataset` |
| peft | >=0.7.0 | LoRA implementation | `from peft import LoraConfig, get_peft_model, PeftModel, TaskType` |
| torch | >=2.0.0 | Deep learning framework | `import torch` |
| accelerate | >=0.24.0 | Training optimization | (used internally by transformers) |
| jupyter | >=1.0.0 | Notebook environment | `jupyter lab` |
| torchvision | >=0.15.0 | PyTorch compatibility | (dependency for torch) |
| pillow | >=9.0.0 | Image processing | (dependency for transformers) |

**Why These Versions?**
- transformers >=4.36.0: Required for Qwen model support
- peft >=0.7.0: Stable LoRA implementation
- torch >=2.0.0: Performance improvements and bug fixes
- datasets >=2.14.0: Latest data processing features

---

## AI Assistant Guidelines

### When Modifying This Codebase

1. **Always Read Before Writing:**
   - Use Read tool on fine_tune.py before making changes
   - Check line numbers in function table above for quick navigation
   - Understand the modular structure (6 distinct functions)

2. **Preserve Modularity:**
   - Keep functions single-purpose
   - Don't merge test_original_model() and test_finetuned_model()
   - Maintain clear separation between data loading, training, and testing

3. **Respect Conventions:**
   - Keep snake_case naming
   - Maintain JSONL format for training data
   - Don't change default model without good reason
   - Preserve command-line argument structure

4. **Test Changes:**
   - If modifying training: Run full pipeline to verify
   - If modifying data loading: Test with small dataset first
   - If changing model: Verify compatibility with LoRA
   - If updating dependencies: Check version compatibility

5. **Document Changes:**
   - Add inline comments for non-obvious modifications
   - Update README.md if changing usage patterns
   - Update this CLAUDE.md if changing conventions
   - Update README_SCRIPT.md if changing CLI arguments

### Common User Requests and How to Handle

| User Request | Recommended Approach | Files to Modify |
|--------------|----------------------|-----------------|
| "Train on custom data" | Guide to create JSONL file, use --data flag | None (user creates data file) |
| "Use different model" | Check model compatibility, use --model flag | None (runtime argument) |
| "Increase training epochs" | Edit num_train_epochs in train_model() | fine_tune.py:95 |
| "Reduce GPU memory" | Decrease batch_size, use gradient checkpointing | fine_tune.py:96 |
| "Speed up training" | Suggest GPU usage, increase batch_size | Environment setup |
| "Export to ONNX/TensorRT" | Out of scope, suggest separate export script | New file needed |
| "Add validation split" | Modify load_and_preprocess_data() to split data | fine_tune.py:37-65 |
| "Change LoRA rank (r)" | Modify LoraConfig in setup_lora_model() | fine_tune.py:75-80 |
| "Add wandb logging" | Add wandb to requirements, modify TrainingArguments | requirements.txt, fine_tune.py:87-110 |

### Code Reading Guide for AI Assistants

**Quick Navigation by Task:**

Want to understand...
- **How data is loaded?** → Read fine_tune.py:37-65 (load_and_preprocess_data)
- **How LoRA is configured?** → Read fine_tune.py:67-85 (setup_lora_model)
- **What hyperparameters are used?** → Read fine_tune.py:87-110 (train_model)
- **How testing works?** → Read fine_tune.py:15-35 and fine_tune.py:127-145
- **What the data looks like?** → Read mariya.json (first 10 lines sufficient)
- **How to run the script?** → Read README_SCRIPT.md

**Understanding the Flow:**
```
main() [147-152]
    ↓
test_original_model() [15-35] (if not --skip-original)
    ↓
load_and_preprocess_data() [37-65]
    ↓
setup_lora_model() [67-85]
    ↓
train_model() [87-110] (if not --skip-training)
    ↓
save_model() [112-125]
    ↓
test_finetuned_model() [127-145]
```

### What NOT to Do

**❌ Don't:**
- Change the JSONL format to JSON array format (will break loading)
- Remove the test_original_model() function (breaks before/after comparison)
- Hardcode file paths instead of using command-line arguments
- Modify the output directory structure (Hugging Face expects specific layout)
- Change padding token ID without understanding the model's tokenizer
- Remove error handling for file existence checks
- Merge all functions into one monolithic main() function
- Add complex CLI menu systems (keep it simple)
- Create a web UI without explicit user request
- Add database dependencies unless specifically needed

**✅ Do:**
- Use command-line arguments for configuration
- Keep functions modular and testable
- Preserve the educational nature of the code
- Add helpful error messages for common mistakes
- Validate data format before training
- Check device availability (CUDA vs CPU)
- Use the established naming conventions
- Keep dependencies minimal and necessary

### Handling Edge Cases

**If user has no GPU:**
- Suggest reducing epochs (e.g., 3-5 instead of 10)
- Suggest smaller dataset (e.g., 50 samples instead of 236)
- Warn that training will take several hours
- Consider suggesting Google Colab as alternative

**If model download fails:**
- Check internet connection
- Verify Hugging Face Hub is accessible
- Suggest using smaller model (e.g., gpt2)
- Consider local model path option

**If training loss is NaN:**
- Reduce learning rate (e.g., 0.0001)
- Check for corrupted data samples
- Verify model and tokenizer compatibility
- Try with smaller batch size

**If out of memory:**
- Reduce batch_size to 1 (already minimum)
- Enable gradient checkpointing in model
- Use smaller model variant
- Suggest using CPU (slower but works)

---

## Git Workflow

### Current Branch
```
Main branch: main
Current feature branch: claude/claude-md-mhy2y5ytcnn8ftg5-014scgWfBFhBZBGxX8UYFzEf
```

### Making Changes

1. **Development:**
   - All changes should be made on the feature branch
   - Commit with clear, descriptive messages
   - Push to origin when ready

2. **Commit Message Format:**
   ```
   [Component] Brief description

   Detailed explanation of changes
   ```

   Examples:
   ```
   [Training] Increase default epochs from 10 to 15

   Users requested longer training for better model quality.
   Updated train_model() function with new default.
   ```

   ```
   [Data] Add validation for JSONL format

   Added check to ensure each line has prompt and completion keys.
   Provides helpful error message if format is incorrect.
   ```

3. **Commit Guidelines:**
   - Commit after each logical change
   - Don't commit generated model files (my-qwen/ directory)
   - Check .gitignore includes output directories
   - Use descriptive commit messages
   - Push to feature branch, never force push to main

### Files to Git Ignore

The .gitignore should include:
```
# Model outputs
my-qwen/
*.bin
*.safetensors

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/

# Jupyter
.ipynb_checkpoints/

# Environment
.env
venv/
env/
llm/

# IDE
.vscode/
.idea/
*.swp
```

---

## Troubleshooting Guide

### Problem: "CUDA out of memory"
**Solution:**
```python
# In fine_tune.py, modify training_args:
per_device_train_batch_size=1,  # Already minimum
gradient_accumulation_steps=4,   # ADD THIS LINE
```

### Problem: "Tokenizer does not have a padding token"
**Solution:**
```python
# In load_and_preprocess_data(), after loading tokenizer:
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Problem: "Model not learning (loss not decreasing)"
**Checklist:**
1. Verify data quality: `python -c "import json; print([json.loads(line) for line in open('mariya.json')][:3])"`
2. Check learning rate is not too low (should be 0.001)
3. Ensure enough epochs (at least 10 for small datasets)
4. Verify data has consistent information (no contradictions)

### Problem: "Training crashes after epoch 1"
**Likely Causes:**
1. Out of memory → Reduce batch size or use CPU
2. Disk space full → Check output directory has space
3. Data corruption → Validate JSONL format

**Debug Command:**
```bash
python -c "
import json
with open('mariya.json') as f:
    for i, line in enumerate(f, 1):
        try:
            data = json.loads(line)
            assert 'prompt' in data and 'completion' in data
        except Exception as e:
            print(f'Error on line {i}: {e}')
"
```

### Problem: "Fine-tuned model same as original"
**Checklist:**
1. Verify training completed all epochs
2. Check final loss is <0.5 (should be ~0.3)
3. Ensure output directory has adapter files
4. Verify test_finetuned_model() loads from correct directory
5. Check training data actually contains answer to test question

### Problem: "Import errors after pip install"
**Solution:**
```bash
# Recreate environment
conda deactivate
conda env remove -n llm
conda create -n llm python=3.12
conda activate llm
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Performance Benchmarks

### Training Time Expectations

| Configuration | Dataset Size | Epochs | Time |
|--------------|--------------|--------|------|
| RTX 4090 (24GB) | 236 samples | 10 | ~40 min |
| RTX 3060 (12GB) | 236 samples | 10 | ~53 min |
| T4 (16GB) | 236 samples | 10 | ~75 min |
| CPU (16 cores) | 236 samples | 10 | ~4 hours |
| CPU (8 cores) | 236 samples | 10 | ~8 hours |

### Memory Requirements

| Component | GPU Memory | System RAM | Disk Space |
|-----------|------------|------------|------------|
| Base model (Qwen 2.5-3B) | ~6GB | ~12GB | ~6GB |
| Training (batch=1) | +2GB | +4GB | - |
| Output (LoRA adapter) | - | - | ~15-50MB |
| **Total** | **8GB** | **16GB** | **6GB + 50MB** |

### Optimization Tips

**For faster training:**
1. Use GPU with CUDA
2. Increase batch_size if memory allows (2, 4, 8)
3. Reduce logging_steps to reduce I/O
4. Use mixed precision: `fp16=True` in TrainingArguments

**For lower memory:**
1. Use CPU (slower but works with less memory)
2. Use gradient checkpointing
3. Use 8-bit or 4-bit quantization
4. Reduce model size (use smaller base model)

**For better model quality:**
1. Increase training epochs (15-20)
2. Increase dataset size (500-1000 samples)
3. Improve data quality (diverse, accurate examples)
4. Tune learning rate (try 0.0005 or 0.0001)

---

## Additional Resources

### Useful Commands Reference

```bash
# Check Python version
python --version

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check installed packages
pip list | grep -E "(transformers|peft|datasets|torch)"

# Count training samples
wc -l mariya.json

# Validate JSON format
cat mariya.json | python -m json.tool > /dev/null && echo "Valid JSON"

# Check model size
du -sh my-qwen/

# Monitor GPU usage (during training)
watch -n 1 nvidia-smi

# Test tokenizer
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); print(t('Hello world'))"
```

### External Documentation

- **Hugging Face Transformers:** https://huggingface.co/docs/transformers
- **PEFT (LoRA):** https://huggingface.co/docs/peft
- **Datasets Library:** https://huggingface.co/docs/datasets
- **PyTorch:** https://pytorch.org/docs
- **Qwen Model:** https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

### Project-Specific Documentation

- **README.md:** High-level overview and video tutorial
- **README_SCRIPT.md:** Script usage and setup instructions
- **LLM Fine Tuning Workflow.ipynb:** Interactive notebook with explanations

---

## Version History

- **Latest Update:** November 13, 2025
  - Created comprehensive CLAUDE.md guide
  - Documented complete codebase structure
  - Added AI assistant guidelines and best practices

- **August 5, 2025:** (noted in README.md)
  - Updated inference code for compatibility

- **Original Version:**
  - Basic fine-tuning script with 236 Mariya samples
  - 10-epoch training with LoRA
  - Before/after testing pattern

---

## Contact and Support

For issues, questions, or contributions:
- Check README.md for video tutorial and basic setup
- Check README_SCRIPT.md for script-specific questions
- Review this CLAUDE.md for development guidelines
- Check GitHub issues for known problems
- Create new issue with error logs and environment details

**When Reporting Issues, Include:**
1. Python version: `python --version`
2. Package versions: `pip list | grep -E "(transformers|peft|torch)"`
3. CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
4. Error message with full traceback
5. Command used to run the script
6. Custom data file format (if using custom data)

---

*This guide is maintained for AI assistants to understand and work effectively with the fine-tuning codebase. Last updated: November 13, 2025*
