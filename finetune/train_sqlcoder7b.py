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

#training_dataset = load_dataset("csv", data_files="formatted_data.csv", split="train")
training_dataset = load_dataset("csv", data_files="training_data.csv", split='train[:100%]')
print(training_dataset)
evaluation_dataset = load_dataset("csv", data_files="evaluation_data.csv", split='train[:100%]')
print(evaluation_dataset)

base_model = "defog/sqlcoder-7b"
new_model = "defog_sqlcoder_7b_finetuned"

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_type = torch.float16,
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = False
)

model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config = bnb_config,
        trust_remote_code = True,
        #use_flash_attention_2=True,
        attn_implementation="flash_attention_2",
        #flash_attn = True,
        #flash_rotary = True,
        #fused_dense = True,
        low_cpu_mem_usage = True,
        device_map = {"":0},
        #torch_dtype='auto',
        #revision = "refs/pr/23"
        )
print('Model dtype: ', model.dtype)
model.config.use_cache = False
model.config.pretraining_tp = 1

#model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

print(model)

training_arguments = TrainingArguments(
        output_dir = f"./{new_model}", 
        num_train_epochs = 10,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 1,
        evaluation_strategy = "steps",
        eval_steps = 1000,
        logging_steps = 15000,
        optim = "paged_adamw_8bit",
        learning_rate = 2e-4,
        lr_scheduler_type = "cosine",
        save_steps = 1000,
        warmup_ratio = 0.05,
        weight_decay = 0.01,
        max_steps = -1,
        ddp_find_unused_parameters = False,
        #save_strategy="epoch",
        save_strategy="steps",
)


peft_config = LoraConfig(
        r = 32,
        lora_alpha = 64,
        lora_dropout = 0.05,
        bias = "none",
        task_type = "CAUSAL_LM",
        #target_modules = ["Wqkv","fc1","fc2"] #["Wqkv", "out_proj", "fc1","fc2"] - 41M params
        #target_modules = ["q_proj","k_proj","v_proj","o_proj"] #["Wqkv", "out_proj", "fc1","fc2"] - 41M params
        target_modules = ["down_proj"] #["Wqkv", "out_proj", "fc1","fc2"] - 41M params
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
        eval_dataset = evaluation_dataset,
        peft_config = peft_config,
        dataset_text_field = "Text",
        max_seq_length = 2048,
        tokenizer = tokenizer,
        args = training_arguments,
        callbacks = [ PeftSavingCallback ]
)


training_result = trainer.train()
print(training_result)

print("Saving last checkpoint of the model")
#trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
trainer.model.save_pretrained(os.path.join(training_arguments.output_dir, "final_checkpoint/"))
