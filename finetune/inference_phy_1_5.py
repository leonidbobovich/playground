import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#model = AutoModelForCausalLM.from_pretrained("phi-1_5-finetuned-med-text", trust_remote_code=True, torch_dtype=torch.float32)
model = AutoModelForCausalLM.from_pretrained("phi-1_5-finetuned-med-text", trust_remote_code=True, torch_dtype=torch.float32, device_map={"":0})

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

inputs = tokenizer('I am having a headache with but no fever - what could be the cause', return_tensors="pt", return_attention_mask=False).to(model.device)

outputs = model.generate(**inputs, max_length=4096, do_sample=True, top_k=10, temperature=0.01, eos_token_id=tokenizer.eos_token_id).to(torch.device('cpu'))

text = tokenizer.batch_decode(outputs)[0]

print(text)
