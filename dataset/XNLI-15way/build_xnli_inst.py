from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM
import safetensors
import datasets
from datasets import Dataset
import torch
from torch import nn
DEVICE = "cuda"
LLAMA_TEMPLATE = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{src}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

MODEL_DIR = "~/PretrainedModels/llama-3.1-8b-instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(
    MODEL_DIR, 
    local_files_only=True, 
    device_map=DEVICE, 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    max_position_embeddings=2048
)
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.eos_token_id = [128001, 128008, 128009]
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads

dataset = Dataset.from_csv(
    "~/workspace/modularity/dataset/XNLI-15way/xnli.15way.orig.tsv", sep='\t'
).select_columns(["en", "zh"]).map(
    lambda sample: {
        "input_str": sample["en"],
        "target_str": sample["zh"],
    }
).select_columns(["input_str", "target_str"])
train_dataset = dataset.select(range(len(dataset) - 100))
dev_dataset = dataset.select(range(len(dataset) - 100, len(dataset)))

pred_strs = []
with torch.no_grad():
    for dataset_split in [train_dataset, dev_dataset]:
        for sample in tqdm(dataset_split):
            input_str = sample["input_str"]
            lm_inputs_src = tokenizer([LLAMA_TEMPLATE.format(src=input_str)], add_special_tokens=False, return_tensors="pt").to(DEVICE)
            generate_ids = model.generate(**lm_inputs_src, max_new_tokens=50, do_sample=False)
            pred_str = tokenizer.decode(generate_ids[0][lm_inputs_src.input_ids.size(1):], skip_special_tokens=True)
            pred_strs.append(pred_str)
train_dataset = train_dataset.add_column("pred_str", pred_strs[:len(train_dataset)])
dev_dataset = dev_dataset.add_column("pred_str", pred_strs[len(train_dataset):])

templates = [
    "{src}\n\nTranslate into Chinese:",
    "{src}\n\nChinese:",
    "{src}\n\nChinese translation:",
    "{src}\n\nSay it in Chinese:",
    "How to say this in Chinese:\n\n{src}",
    "Translate into Chinese:\n\n{src}",
    "Translate the following sentence into Chinese:\n\n{src}",
    "{src}\n\nChinese version:",
]
# 将input_str随机填进templates中
def generate_template(input_str):
    return templates[torch.randint(0, len(templates), (1,)).item()].format(src=input_str)
train_dataset = train_dataset.map(
    lambda sample: {
        "input_str": generate_template(sample["input_str"]),
        "target_str": sample["pred_str"]
    }
).select_columns(["input_str", "target_str"])
dev_dataset = dev_dataset.map(
    lambda sample: {
        "input_str": generate_template(sample["input_str"]),
        "target_str": sample["pred_str"]
    }
).select_columns(["input_str", "target_str"])
# 保存数据集
train_dataset.save_to_disk("~/workspace/modularity/dataset/XNLI-15way/inst-train")
dev_dataset.save_to_disk("~/workspace/modularity/dataset/XNLI-15way/inst-dev")
