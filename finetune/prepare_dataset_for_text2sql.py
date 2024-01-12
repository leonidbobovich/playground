import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, load_from_disk
import pandas as pd



#from huggingface_hub import interpreter_login
#interpreter_login()

dataset = load_dataset("shiroyasha13/llama_text_to_sql_dataset", split="train")
print(dataset)

df = pd.DataFrame(dataset)
print(df.head(2))
print(df.shape)
print(df.info())

def format_row(row):
    question = row['input']
    answer = row['output']
    formatted_string = f"{question} {answer}"
    return formatted_string

df['Text'] = df.apply(format_row, axis=1)
print(df['Text'])
df = df[['Text']]
print(df.head(3))
df.to_csv("formatted_data.csv", index=False)

edf = df.sample(frac = 0.1)
edf.to_csv("evaluation_data.csv", index=False)
edf_10 = edf.head(2000)
edf_10.to_csv("evaluation_data.2000.csv", index=False)

tdf = df.drop(edf.index)
tdf.to_csv("training_data.csv", index=False)
tdf_100 = tdf.head(4000)
tdf_100.to_csv("training_data.4000.csv", index=False)

#training_dataset = load_dataset("csv", data_files="formatted_data.csv", split="train")
#print(training_dataset)
