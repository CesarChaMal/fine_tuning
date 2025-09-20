#!/usr/bin/env python3
"""
LLM Fine-Tuning Script
Converts a pre-trained model to learn custom knowledge using LoRA fine-tuning.
"""

import torch
from transformers import (
    pipeline, AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
import argparse
import os

def test_original_model(model_name, question="who is Mariya Sha?"):
    """Test the original model's knowledge"""
    print("Testing original model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ask_llm = pipeline(model=model_name, device=device)
    response = ask_llm(question)[0]["generated_text"]
    print(f"Original model response: {response}")
    return response

def load_and_preprocess_data(data_file):
    """Load and tokenize the training data"""
    print(f"Loading dataset from {data_file}...")
    raw_data = load_dataset("json", data_files=data_file)
    print(f"Dataset loaded: {len(raw_data['train'])} samples")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    
    def preprocess(sample):
        sample_text = sample["prompt"] + "\n" + sample["completion"]
        tokenized = tokenizer(
            sample_text,
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    data = raw_data.map(preprocess)
    print("Data preprocessing complete")
    return data, tokenizer

def setup_lora_model(model_name):
    """Setup model with LoRA configuration"""
    print("Setting up LoRA model...")
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    print("LoRA model setup complete")
    return model

def train_model(model, data, output_dir="./my-qwen"):
    """Train the model with fine-tuning"""
    print("Starting training...")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        learning_rate=0.001,
        logging_steps=25,
        save_strategy="epoch",
        per_device_train_batch_size=1
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"]
    )
    
    trainer.train()
    print("Training complete")
    return trainer

def save_model(trainer, tokenizer, output_dir="./my-qwen"):
    """Save the fine-tuned model and tokenizer"""
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully")

def test_finetuned_model(model_path="./my-qwen", question="Who is Mariya Sha?"):
    """Test the fine-tuned model"""
    print("Testing fine-tuned model...")
    
    config = PeftConfig.from_pretrained(model_path)
    base = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path, 
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50
    )
    
    response = tokenizer.decode(output[0])
    print(f"Fine-tuned model response: {response}")
    return response

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with custom knowledge")
    parser.add_argument("--data", default="mariya.json", help="Training data file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Base model name")
    parser.add_argument("--output", default="./my-qwen", help="Output directory")
    parser.add_argument("--question", default="Who is Mariya Sha?", help="Test question")
    parser.add_argument("--skip-original", action="store_true", help="Skip testing original model")
    parser.add_argument("--skip-training", action="store_true", help="Skip training (test existing model)")
    
    args = parser.parse_args()
    
    if not args.skip_original:
        test_original_model(args.model, args.question)
    
    if not args.skip_training:
        if not os.path.exists(args.data):
            print(f"Error: Data file {args.data} not found")
            return
        
        data, tokenizer = load_and_preprocess_data(args.data)
        model = setup_lora_model(args.model)
        trainer = train_model(model, data, args.output)
        save_model(trainer, tokenizer, args.output)
    
    if os.path.exists(args.output):
        test_finetuned_model(args.output, args.question)
    else:
        print(f"No fine-tuned model found at {args.output}")

if __name__ == "__main__":
    main()