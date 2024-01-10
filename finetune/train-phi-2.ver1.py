import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset, load_from_disk

from peft import LoraConfig, prepare_model_for_kbit_training

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, TrainerCallback
from tqdm.notebook import tqdm
from trl import SFTTrainer

#from huggingface_hub import interpreter_login
#interpreter_login()
import pandas as pd

dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train")
print(dataset)

df = pd.DataFrame(dataset)
print(df.head(2))
print(df.shape)
print(df.info())

def format_row(row):
    question = row['Context']
    answer = row['Response']
    formatted_string = f"[INST] {question} [/INST] {answer}"
    return formatted_string

df['Formatted'] = df.apply(format_row, axis=1)
print(df['Formatted'])

new_df = df.rename(columns = {'Formatted': 'Text'})
new_df = new_df[['Text']]
print(new_df.head(3))
new_df.to_csv("formatted_data.csv", index=False)

training_dataset = load_dataset("csv", data_files="formatted_data.csv", split="train")
print(training_dataset)

base_model = "microsoft/phi-2"
new_model = "phi2-menthal-health"

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_type = torch.float16,
        bnb_4bit_use_double_quant = False
)

model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config = bnb_config,
        trust_remote_code = True,
        #use_flash_attention_2=True,
        #attn_implementation="flash_attention_2",
        flash_attn = True,
        flash_rotary = True,
        fused_dense = True,
        low_cpu_mem_usage = True,
        device_map = {"":0},
        #revision = "refs/pr/23"
        )

model.config.use_cache = False
model.config.pretraining_tp = 1

#model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

training_arguments = TrainingArguments(
        output_dir = "./mhGPT", 
        num_train_epochs = 2,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 32,
        evaluation_strategy = "steps",
        eval_steps = 1000,
        logging_steps = 15,
        optim = "paged_adamw_8bit",
        learning_rate = 2e-4,
        lr_scheduler_type = "cosine",
        save_steps = 1500,
        warmup_ratio = 0.05,
        weight_decay = 0.01,
        max_steps = -1,
        ddp_find_unused_parameters = False
)


peft_config = LoraConfig(
        r = 32,
        lora_alpha = 64,
        lora_dropout = 0.05,
        bias = "none",
        task_type = "CAUSAL_LM",
        target_modules = ["Wqkv","fc1","fc2"] #["Wqkv", "out_proj", "fc1","fc2"] - 41M params
        #modules_to_save=["embed_tokens", "lm_head"]
        )

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        print("PeftSavingCallback::on_save enter")
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))
        print("PeftSavingCallback::on_save exit")

trainer = SFTTrainer(
        model = model,
        train_dataset = training_dataset,
        peft_config = peft_config,
        dataset_text_field = "Text",
        max_seq_length = 690,
        tokenizer = tokenizer,
        args = training_arguments,
        callbacks = [ PeftSavingCallback ]
)


training_result = trainer.train()
print(training_result)

print("Saving last checkpoint of the model")
#trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
trainer.model.save_pretrained(os.path.join(training_arguments.output_dir, "final_checkpoint/"))
