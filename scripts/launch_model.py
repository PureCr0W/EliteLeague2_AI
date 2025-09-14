import argparse, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["hf_repo"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=cfg.get("device", "cpu"),
        torch_dtype=getattr(__import__("torch"), cfg.get("dtype", "float32")),
        low_cpu_mem_usage=True,
    )
    return model, tokenizer, cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()

    model, tokenizer, cfg = load_model(args.cfg)
    prompt = "Hello, this is a test."
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=cfg["generation"]["max_tokens"])
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
