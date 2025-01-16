import argparse
import copy
import os
from dataclasses import dataclass, field

import pandas as pd
import safetensors
import torch
from torch import nn
import itertools
import wandb
import datasets
from datasets import Dataset, concatenate_datasets
from models.modeling_llama import LlamaForCausalLM
from models.modeling_phi3 import Phi3ForCausalLM
from models.modeling_mistral import MistralForCausalLM
from models.modeling_qwen2 import Qwen2ForCausalLM
from models.modeling_gemma2 import Gemma2ForCausalLM
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling, Trainer, 
                          TrainerCallback, TrainingArguments, HfArgumentParser)
from utils import *


LLAMA_TEMPLATE = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{src}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA_PLM_TEMPLATE = "<|begin_of_text|>{src}\n\n"
PHI3_TEMPLATE = "<|user|>\n{src}<|end|>\n<|assistant|>\n"
MISTRAL_TEMPLATE = "<s>[INST] {src}[/INST]"
QWEN2_TEMPLATE = "<|im_start|>user\n{src}<|im_end|>\n<|im_start|>assistant\n"
QWEN2_PLM_TEMPLATE = "{src}\n\n"
GEMMA2_TEMPLATE = "<bos><start_of_turn>user\n{src}<end_of_turn>\n<start_of_turn>model\n"

def preprocess_batch(samples, lm_tokenizer, model_type):
    assert model_type in ["llama-inst", "llama-plm", "phi3", "mistral", "qwen2", "qwen2-plm", "gemma2"]
    template = ""
    if model_type == "llama-inst":
        template = LLAMA_TEMPLATE
        combined_strs = [LLAMA_TEMPLATE.format(src=input_str.strip()) + f"{target_str.strip()}<|eot_id|>" for input_str, target_str in zip(samples["input_str"], samples["target_str"])]
    elif model_type == "llama-plm":
        template = LLAMA_PLM_TEMPLATE
        combined_strs = [LLAMA_PLM_TEMPLATE.format(src=input_str.strip()) + f"{target_str.strip()}<|end_of_text|>" for input_str, target_str in zip(samples["input_str"], samples["target_str"])]
    elif model_type == "phi3":
        template = PHI3_TEMPLATE
        combined_strs = [PHI3_TEMPLATE.format(src=input_str.strip()) + f"{target_str.strip()}<|end|>" for input_str, target_str in zip(samples["input_str"], samples["target_str"])]
    elif model_type == "mistral":
        template = MISTRAL_TEMPLATE
        combined_strs = [MISTRAL_TEMPLATE.format(src=input_str.strip()) + f" {target_str.strip()}</s>" for input_str, target_str in zip(samples["input_str"], samples["target_str"])]
    elif model_type == "qwen2":
        template = QWEN2_TEMPLATE
        combined_strs = [QWEN2_TEMPLATE.format(src=input_str.strip()) + f"{target_str.strip()}<|im_end|>" for input_str, target_str in zip(samples["input_str"], samples["target_str"])]
    elif model_type == "qwen2-plm":
        template = QWEN2_PLM_TEMPLATE
        combined_strs = [QWEN2_PLM_TEMPLATE.format(src=input_str.strip()) + f"{target_str.strip()}<|endoftext|>" for input_str, target_str in zip(samples["input_str"], samples["target_str"])]
    elif model_type == "gemma2":
        template = GEMMA2_TEMPLATE
        combined_strs = [GEMMA2_TEMPLATE.format(src=input_str.strip()) + f"{target_str.strip()}<end_of_turn>" for input_str, target_str in zip(samples["input_str"], samples["target_str"])]
    lm_inputs = lm_tokenizer(
        combined_strs, 
        max_length=2048, 
        truncation=True, 
        padding=False, 
        add_special_tokens=False,
        return_tensors="np"
    )
    input_str_lens = [len(lm_tokenizer(template.format(src=input_str.strip()), add_special_tokens=False)["input_ids"]) for input_str in samples["input_str"]]
    labels = copy.deepcopy(lm_inputs["input_ids"])
    for i, input_str_len in enumerate(input_str_lens):
        labels[i][:input_str_len] = -100
        # lm_inputs["input_ids"][i] = lm_inputs["input_ids"][i][:input_str_len + 5]
        # labels[i] = labels[i][:input_str_len + 5]

    return {
        "lm_input_ids": lm_inputs["input_ids"],
        "lm_labels": labels,
    }

class SimpleTensorModel(nn.Module):
    def __init__(self, tensor_length=1056):
        super().__init__()
        self.tensor = nn.Parameter(torch.zeros(tensor_length))
        nn.init.normal_(self.tensor, mean=4, std=0.02)
        # self.tensor.data[32::33] = -5

    def forward(self, x=None):
        return self.tensor

class PPLMetric:
    def __init__(self):
        self.total_loss = 0
        self.total_tokens = 0

    def compute_metrics(self, eval_preds, compute_result: bool):
        preds, labels = eval_preds.predictions, eval_preds.label_ids
        
        # Shift logits and labels for PPL calculation
        shifted_preds = preds[:, :-1, :].contiguous().view(-1, preds.size(-1))
        shifted_labels = labels[:, 1:].contiguous().view(-1)
        shifted_labels = shifted_labels.to(shifted_preds.device)
        # Filter out labels that are -100 (which are padding tokens that should be ignored)
        mask = (shifted_labels != -100).to(shifted_preds.device)
        filtered_preds = shifted_preds[mask]
        filtered_labels = shifted_labels[mask]

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(filtered_preds, filtered_labels)
        self.total_loss += loss.item()
        self.total_tokens += len(filtered_labels)

        if compute_result:
            perplexity = torch.exp(torch.tensor(self.total_loss / self.total_tokens))
            self.total_loss = 0
            self.total_tokens = 0
            return {
                "perplexity": perplexity.item(),
            }
        else:
            return {}

class CustomDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, lm_tokenizer, padding=True):
        self.lm_tokenizer = lm_tokenizer
        self.padding = padding
    
    def torch_call(self, features):
        lm_input_ids = [f["lm_input_ids"] for f in features]
        lm_labels = [torch.tensor(f["lm_labels"]) for f in features]

        lm_batch = self.lm_tokenizer.pad(
            {"input_ids": lm_input_ids},
            padding=self.padding,
            return_tensors="pt"
        )
        lm_labels = torch.nn.utils.rnn.pad_sequence(lm_labels, batch_first=True, padding_value=-100)

        return {
            "lm_input_ids": lm_batch["input_ids"],
            "lm_attention_mask": lm_batch["attention_mask"],
            "lm_labels": lm_labels,
        }

class FreezePartGradientsCallback(TrainerCallback):
    def __init__(self, model, freeze_mask):
        super().__init__()
        self.model = model
        self.freeze_mask = freeze_mask  # 定义冻结掩码
        self.flatten_mask = (freeze_mask.sigmoid() >= 0.5).float().flatten()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        # 清零冻结位置的梯度
        grad_rank = self.model.tensor.grad.argsort().flip(0).argsort().to("cpu")
        rank_stat = [
            self.flatten_mask[grad_rank[:int(pp * len(grad_rank))]].sum()/self.flatten_mask.sum()
        for pp in [0.05, 0.1, 0.3, 0.5]]
        # self.model.tensor.grad[self.freeze_mask == 1] = 0
        pass

# Custom loss function by overriding Trainer's compute_loss
class CustomTrainer(Trainer):
    def __init__(self, *args, lm_model, lm_tokenizer, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_model = lm_model
        self.lm_tokenizer = lm_tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        lm_input_ids = inputs["lm_input_ids"].to(self.args.device)
        lm_attention_mask = inputs["lm_attention_mask"].to(self.args.device)
        lm_labels = inputs["lm_labels"].to(self.args.device)
        bsz = lm_input_ids.size(0)

        # 1. Get the weight tensor by embedding model
        weights_logit: torch.Tensor = model(None).unsqueeze(0).repeat(bsz, 1)

        if self.args.tau_decay_steps < 1:
            tau_decay_end_step = int(self.args.tau_decay_steps * self.state.max_steps)
        else:
            tau_decay_end_step = int(self.args.tau_decay_steps)
        if self.state.global_step >= tau_decay_end_step:
            tau_temp = self.args.tau_temp_end
        else:
            decay_ratio = self.state.global_step / tau_decay_end_step
            tau_temp = self.args.tau_temp_begin - decay_ratio * (self.args.tau_temp_begin - self.args.tau_temp_end)

        if self.args.use_gumbel:
            weights_tensor = gumbel_sigmoid(weights_logit, tau=tau_temp, hard=self.args.gumbel_hard)
        else:
            weights_tensor = torch.sigmoid(weights_logit)
        weights_tensor = weights_tensor.to(self.lm_model.device)
        # 2. Use the weight tensor to train the LM model
        pred_outputs = self.lm_model(lm_input_ids, attention_mask=lm_attention_mask, weight_tensor=weights_tensor, labels=lm_labels, use_cache=False)
        pred_output_loss = pred_outputs.loss

        norm_lambda = self.args.norm_lambda
        normalizer = torch.sum(weights_tensor, dim=1).mean()
        # use_mlp_num = torch.sum(weights_tensor[:, 32::33], dim=1).mean()
        loss = pred_output_loss + norm_lambda * normalizer
        self.log({
            "pred_output_loss": pred_output_loss.item(),
            "normalizer": normalizer.item(),
            "tau_temp": tau_temp,
            "total_loss": loss.item()
        })

        return (loss, pred_outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        lm_input_ids = inputs["lm_input_ids"].to(self.args.device)
        # lm_attention_mask = inputs["lm_attention_mask"].to(self.args.device)
        lm_labels = inputs["lm_labels"].to(self.args.device)
        bsz = lm_input_ids.size(0)

        with torch.no_grad():
            # 1. Forward pass through embed model to get weight tensor
            weights_logit: torch.Tensor = model(None).unsqueeze(0).repeat(bsz, 1)
            weights_tensor = (weights_logit.sigmoid() >= 0.5).to(weights_logit.dtype)
            weights_tensor = weights_tensor.to(self.lm_model.device)
            pred_output_loss = None
            # 3. Also generate
            for one_lm_input_ids, one_lm_labels, one_weights_tensor in zip(lm_input_ids, lm_labels, weights_tensor):
                one_lm_scr_input_ids = one_lm_input_ids[one_lm_labels == -100].unsqueeze(0)
                one_lm_attention_mask = torch.ones_like(one_lm_scr_input_ids)
                generate_ids = self.lm_model.generate(one_lm_scr_input_ids, attention_mask=one_lm_attention_mask, max_new_tokens=30, weight_tensor=one_weights_tensor.unsqueeze(0), do_sample=False)
                pred_str = self.lm_tokenizer.decode(generate_ids[0], skip_special_tokens=True)
                print("[GENERATED]", pred_str)

        if prediction_loss_only:
            return (pred_output_loss, None, None)

        # lm_logits = pred_outputs.logits
        lm_logits = None
        return (pred_output_loss, lm_logits, lm_labels)

@dataclass
class ModelArguments:
    embed_model_path: str = field(default="~/PretrainedModels/bge-m3", metadata={"help": "Path to the embedding model (to be trained)"})
    lm_model_path: str = field(default="~/PretrainedModels/llama-3.1-8b-instruct-hf", metadata={"help": "Path to the language model"})
    model_type: str = field(default="llama-inst", metadata={"help": "Model type (llama-inst, llama-plm, phi3)"})

@dataclass
class DataArguments:
    train_data_path: str = field(default="./dataset/XNLI-15way/xnli.15way.orig.tsv", metadata={"help": "Path to the training data"})
    dev_data_path: str = field(default="./dataset/XNLI-15way/xnli.15way.orig.tsv", metadata={"help": "Path to the development data"})
    dataset_use_cache: bool = field(default=True, metadata={"help": "Whether to use cache for dataset"})

@dataclass
class CustomTrainingArguments(TrainingArguments):
    tau_temp_begin: float = field(default=4.0, metadata={"help": "Initial temperature for Gumbel-Sigmoid"})
    tau_temp_end: float = field(default=0.05, metadata={"help": "Final temperature for Gumbel-Sigmoid"})
    tau_decay_steps: float = field(default=0.3, metadata={"help": "Decay steps for Gumbel-Sigmoid temperature (if < 1, it's the ratio of training steps)"})
    norm_lambda: float = field(default=1e-4, metadata={"help": "Normalization lambda for weight tensor"})
    norm_power: float = field(default=1, metadata={"help": "Normalization power for weight tensor"})
    use_gumbel: bool = field(default=True, metadata={"help": "Whether to use Gumbel-Sigmoid for weight tensor"})
    gumbel_hard: bool = field(default=True, metadata={"help": "Whether to use hard Gumbel-Sigmoid for weight tensor"})


def search_weight_embed(lm_model, task: str, args, training_args):
    training_args.output_dir = os.path.join(args.output_dir, task)
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Model and Tokenizer
    model_name = os.path.basename(args.lm_model_path)
    training_args.run_name = f"{model_name}-train_weight-{task}"
    lm_tokenizer = AutoTokenizer.from_pretrained(args.lm_model_path, use_fast=True)
    lm_tokenizer.pad_token = lm_tokenizer.eos_token
    lm_tokenizer.padding_side = "right"
    lm_tokenizer.truncation_side = "right"
    lm_model.generation_config.pad_token_id = lm_tokenizer.pad_token_id
    n_layers = lm_model.config.num_hidden_layers
    n_heads = lm_model.config.num_attention_heads

    embed_model = SimpleTensorModel(tensor_length=n_layers * n_heads + n_layers).to(torch.device("cuda"))

    # Datasets. Only remain "input_str", "target_str", (optional) "input_ids"
    if not args.dataset_use_cache:
        datasets.disable_caching()
    train_dataset, dev_dataset = None, None
    if "XNLI" in args.train_data_path:
        lang_weight_dict = {}
        for root, dir, files in os.walk("./llama-hf/output/xnli"):
            for file in files:
                if file.endswith(".safetensors"):
                    langs = os.path.basename(os.path.dirname(root))
                    path = os.path.join(root, file)
                    lang_weight_dict[langs] = safetensors.torch.load_file(path)["tensor"]

        LANG_LIST = task.split("_")
        LANG_DICT = {"ar": "Arabic", "fr": "French", "es": "Spanish", "de": "German", "en": "English", "ru": "Russian", "zh": "Chinese"}
        def _preprocess_xnli(dataset: Dataset):
            pair_datasets = []
            for src_lang, tgt_lang in itertools.combinations(LANG_LIST, 2):
                pair_dataset = dataset.map(
                    lambda sample: {
                        "input_str": f"{sample[src_lang]}",
                        "target_str": sample[tgt_lang],
                    }
                ).select_columns(["input_str", "target_str"])
                pair_datasets.append(pair_dataset)
            return concatenate_datasets(pair_datasets)
        dataset = Dataset.from_csv(args.train_data_path, sep='\t').select_columns(LANG_LIST)
        train_dataset = dataset.select(range(len(dataset) - 100))
        dev_dataset = dataset.select(range(len(dataset) - 100, len(dataset)))
        train_dataset = _preprocess_xnli(train_dataset)
        dev_dataset = _preprocess_xnli(dev_dataset)
        train_dataset = train_dataset.shuffle(seed=args.seed)
    elif "function_vectors" in args.train_data_path:
        dataset = Dataset.from_json(args.train_data_path)
        dataset = dataset.map(
            lambda sample: {
                "input_str": sample["input"],
                "target_str": sample["output"],
            }
        )
        dataset = dataset.remove_columns(["input", "output"])
        train_dataset = dataset.select(range(min(len(dataset) - 100, 10000)))
        dev_dataset = dataset.select(range(len(dataset) - 100, len(dataset)))
    else:
        assert False, "Unsupported dataset"

    # Tokenize datasets
    train_dataset_dir, dev_dataset_dir = os.path.dirname(args.train_data_path), os.path.dirname(args.dev_data_path)
    train_dataset = train_dataset.map(
        lambda sample: preprocess_batch(sample, lm_tokenizer, model_type=args.model_type),
        batched=True,
        batch_size=128,
        num_proc=4,
        cache_file_name=os.path.join(train_dataset_dir, ".cache/train_dataset_cache.arrow") if args.dataset_use_cache else None
    ).shuffle(seed=args.seed)
    dev_dataset = dev_dataset.map(
        lambda sample: preprocess_batch(sample, lm_tokenizer, model_type=args.model_type),
        batched=True,
        batch_size=128,
        num_proc=4,
        cache_file_name=os.path.join(dev_dataset_dir, ".cache/dev_dataset_cache.arrow") if args.dataset_use_cache else None
    )

    # Data Collator
    data_collator = CustomDataCollator(
        lm_tokenizer=lm_tokenizer,
        padding=True,
    )

    # Trainer
    trainer = CustomTrainer(
        model=embed_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=None,
        # callbacks=[FreezePartGradientsCallback(embed_model, lang_weight_dict[task])],
        lm_model=lm_model,
        lm_tokenizer=lm_tokenizer,
    )

    # Train
    trainer.train(resume_from_checkpoint=False)
    # trainer.evaluate()


if __name__ == "__main__":
    hfparser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(**vars(model_args), **vars(training_args), **vars(data_args))
    print(args)

    if "phi3" in args.model_type:
        casual_lm = Phi3ForCausalLM
    elif "llama" in args.model_type:
        casual_lm = LlamaForCausalLM
    elif "mistral" in args.model_type:
        casual_lm = MistralForCausalLM
    elif "qwen2" in args.model_type:
        casual_lm = Qwen2ForCausalLM
    elif "gemma2" in args.model_type:
        casual_lm = Gemma2ForCausalLM
    else:
        assert False, "Unsupported model type"
    print(f"Using model type: {casual_lm.__name__}")

    lm_model = casual_lm.from_pretrained(
        args.lm_model_path, 
        local_files_only=True, 
        device_map=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}") if args.distributed_state.use_distributed else "auto", 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        max_position_embeddings=2048
    )

    train_data_dir = args.train_data_path
    ALL_LANGS = ["en", "zh", "fr", "de", "es", "ru", "ar"]
    for i, (src_lang, tgt_lang) in enumerate(itertools.permutations(ALL_LANGS, 2)):
        print(f"\n********** [{i+1}/{len(ALL_LANGS) * (len(ALL_LANGS) - 1)}] Training {src_lang} -> {tgt_lang} **********\n")
        # if (src_lang, tgt_lang) in [
        #     ("en", "ar"), ("en", "de"), ("en", "es"), ("en", "fr"), ("en", "ru"), ("en", "zh"), 
        #     ("fr", "de"), ("fr", "en"), ("fr", "zh"), ("fr", "es"), ("fr", "ru"), ("fr", "ar"),
        #     ("zh", "en"), ("zh", "fr"), ("zh", "de"), ("zh", "es"), ("zh", "ru"), ("zh", "ar"),
        #     ("de", "en"), ("de", "fr"), ("de", "zh"), ("de", "es"), ("de", "ru"), ("de", "ar"),
        #     ("es", "en"), ("es", "fr"), ("es", "de"), ("es", "zh"), ("es", "ru"), ("es", "ar"),
        #     ("ru", "en"), ("ru", "fr"), ("ru", "de"), ("ru", "zh"), ("ru", "es"), 
        # ]:
        #     continue
        search_weight_embed(lm_model, f"{src_lang}_{tgt_lang}", args, training_args)
        # if i == 0: break
    # for root, dirs, files in os.walk(train_data_dir):
    #     for file in files:
    #         if file.endswith(".json"):
    #             task = os.path.basename(file).split(".")[0]
    #             if "3" not in task: continue
    #             print(f"\n********** Training {task} **********\n")
    #             args.train_data_path = os.path.join(root, file)
    #             search_weight_embed(lm_model, task, args, training_args)
