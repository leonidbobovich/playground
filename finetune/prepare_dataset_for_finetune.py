import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, load_from_disk
import pandas as pd



#from huggingface_hub import interpreter_login
#interpreter_login()

dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train")
#print(dataset)

df = pd.DataFrame(dataset)
#print(df.head(2))
#print(df.shape)
#print(df.info())


df = df.rename(columns = {'Context': 'instruction', 'Response': 'output'})
df["input"] = ""
#print(df.head(2))

#print(df.reset_index().to_json(orient='index', index=False))
print(df.to_json(orient='records', index=False))

df.to_json("data/mental_health_counseling_conversations.json", orient='records', index=False)

sys.exit(1)


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
new_df.to_json("formatted_data.json", index=False)

