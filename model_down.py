import os
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
os.environ['TZ']='Asia/Shanghai'
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path ='alpindale/WizardLM-2-8x22B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)