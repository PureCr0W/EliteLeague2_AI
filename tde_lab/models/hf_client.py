import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HFClient:
    def __init__(self, cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg
        self.model_id = cfg["hf_repo"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=cfg.get("device", "cpu"),
            torch_dtype=getattr(torch, cfg.get("dtype", "float32")),
            low_cpu_mem_usage=True,
        )

    def generate(self, prompt, max_tokens=128, **gen_cfg):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
