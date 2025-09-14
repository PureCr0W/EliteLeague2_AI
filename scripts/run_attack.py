import argparse, yaml
from tde_lab.attacks.carlini_tde import CarliniTDE
from tde_lab.models.hf_client import HFClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg", required=True)
    parser.add_argument("--attack_cfg", required=True)
    args = parser.parse_args()

    model = HFClient(args.model_cfg)
    with open(args.attack_cfg) as f:
        attack_cfg = yaml.safe_load(f)

    attacker = CarliniTDE(model, attack_cfg)
    results = attacker.run(["keyword1", "keyword2"])
    for r in results:
        print(r)
