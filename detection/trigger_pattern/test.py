
import torch

from detection.layer1_probes import get_probe_pairs
from detection.trigger_pattern.matcher import TriggerPatternMatcher
from inference.inference_core import generate
from model_loader import load_model_from_registry

MODELS_TO_TEST = [
    "smollm2_360m_backdoor_lora_r8",
    "smollm2_360m_backdoor_lora_r32",
    "smollm2_360m_backdoor_full-rank",
]


def main():
    pairs = get_probe_pairs(n=20, style="spaced")
    matcher = TriggerPatternMatcher()

    for name in MODELS_TO_TEST:
        print(f"\n{'=' * 60}\nMODEL: {name}\n{'=' * 60}")
        loaded = load_model_from_registry(name)
        model, tokenizer, device = loaded["model"], loaded["tokenizer"], loaded["device"]

        for status in ("clean", "triggered"):
            texts = [generate(model, tokenizer, device, p[f"{status}_prompt"])[0] for p in pairs]
            r = matcher.run({"generated_texts": texts})
            print(f"{status}: status={r.status} matched={r.detail['matched_count']}/{r.detail['total_texts']}")

        del model, loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
