
import llama

MODEL = 'decapoda-research/llama-7b-hf'
# MODEL = 'decapoda-research/llama-13b-hf'
# MODEL = 'decapoda-research/llama-30b-hf'
# MODEL = 'decapoda-research/llama-65b-hf'

tokenizer = llama.LLaMATokenizer.from_pretrained(MODEL)
model = llama.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage = True)
model.to('cuda')

batch = tokenizer("Yo mama", return_tensors = "pt", add_special_tokens = False)
print(tokenizer.decode(model.generate(batch["input_ids"].cuda(), max_length=100)[0]))
