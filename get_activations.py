# %%
import torch
import llama
from datasets import load_dataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# %%
MODEL = 'decapoda-research/llama-7b-hf'
# MODEL = 'decapoda-research/llama-13b-hf'
# MODEL = 'decapoda-research/llama-30b-hf'
# MODEL = 'decapoda-research/llama-65b-hf'

tokenizer = llama.LLaMATokenizer.from_pretrained(MODEL)
model = llama.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage = True)
r = model.half()                              
r = model.cuda()
# %%                                      
dataset = load_dataset("imdb")

# %%
                                                                                                                     