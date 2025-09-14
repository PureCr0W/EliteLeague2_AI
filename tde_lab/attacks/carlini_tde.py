class CarliniTDE:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    def run(self, hints):
        outputs = []
        templates = self.cfg["prompting"]["seed_templates"]
        for hint in hints:
            for t in templates:
                prompt = t.format(hint1=hint, anchor=hint)
                resp = self.model.generate(prompt, max_tokens=128)
                outputs.append({"prompt": prompt, "response": resp})
        return outputs
