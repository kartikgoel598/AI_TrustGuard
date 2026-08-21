import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from helper.architecture_verification import load_registry


DEFAULT_DTYPE = torch.bfloat16


def find_entry(name: str):
    registry = load_registry()

    for arch_data in registry.values():
        base_model_id = arch_data["base_model_id"]

        for entry in arch_data.get("own_finetunes", []):
            if entry["name"] == name:
                return entry, base_model_id, "own"

        for entry in arch_data.get("community_finetunes", []):
            if entry["name"] == name:
                return entry, base_model_id, "community"

    raise KeyError(f"Model '{name}' not found in registry_data.json.")


def _require_fields(entry: dict, fields: list, name: str):
    missing = [field for field in fields if field not in entry]

    if missing:
        raise KeyError(
            f"Registry entry '{name}' is missing required field(s): {missing}. "
            f"Present fields: {list(entry.keys())}"
        )


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return device


def load_model_from_registry(
    name: str,
    device="auto",
    dtype=DEFAULT_DTYPE,
):
    entry, base_model_id, source = find_entry(name)
    resolved_device = _resolve_device(device)

    dispatch = {
        "lora": _load_lora,
        "full-rank": _load_full_rank,
        "qlora": _load_qlora,
    }

    if source == "community":
        model, tokenizer, is_peft, used_dtype = _load_community(
            entry,
            name,
            resolved_device,
            dtype,
        )

        checkpoint_type = "community"
        label = entry.get("label", "benign")

    else:
        checkpoint_type = entry.get("type")

        if checkpoint_type not in dispatch:
            raise ValueError(
                f"Unknown checkpoint type '{checkpoint_type}' for '{name}'. "
                f"Known types: {list(dispatch.keys())}"
            )

        model, tokenizer, is_peft, used_dtype = dispatch[checkpoint_type](
            entry,
            base_model_id,
            name,
            resolved_device,
            dtype,
        )

        label = entry.get("label", "benign")

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return {
        "model": model,
        "tokenizer": tokenizer,
        "is_peft": is_peft,
        "type": checkpoint_type,
        "label": label,
        "name": name,
        "device": resolved_device,
        "dtype": used_dtype,
    }


def _load_lora(entry, base_model_id, name, device, dtype):
    _require_fields(entry, ["path"], name)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
    ).to(device)

    model = PeftModel.from_pretrained(
        base_model,
        entry["path"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
    )

    return model, tokenizer, True, dtype


def _load_full_rank(entry, base_model_id, name, device, dtype):
    _require_fields(entry, ["path"], name)

    model = AutoModelForCausalLM.from_pretrained(
        entry["path"],
        torch_dtype=dtype,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        entry["path"],
    )

    return model, tokenizer, False, dtype


def _load_qlora(entry, base_model_id, name, device, dtype):
    _require_fields(entry, ["path"], name)

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        raise ImportError(
            "QLoRA loading requires 'bitsandbytes' "
            "(pip install bitsandbytes). "
            "Not needed for SmolLM2-360M (CS301) but required "
            "for CS302 7B+ backdoor checkpoints."
        ) from e

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        base_model,
        entry["path"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
    )

    return model, tokenizer, True, None


def _load_community(entry, name, device, dtype):
    _require_fields(
        entry,
        ["hf_id", "sha"],
        name,
    )

    hf_id = entry["hf_id"]
    sha = entry["sha"]

    if not sha or sha == "main":
        raise ValueError(
            f"Community model '{name}' has an unpinned "
            f"revision ('{sha}'). "
            f"Run fetch_sha_for_all_models() first."
        )

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        revision=sha,
        torch_dtype=dtype,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        revision=sha,
    )

    return model, tokenizer, False, dtype


if __name__ == "__main__":
    registry = load_registry()
    names = []

    for arch_data in registry.values():
        names += [
            entry["name"]
            for entry in arch_data.get("own_finetunes", [])
        ]

        names += [
            entry["name"]
            for entry in arch_data.get("community_finetunes", [])
        ]

    print(f"Found {len(names)} model(s): {names}\n")

    for name in names:
        try:
            result = load_model_from_registry(name)

            print(
                f"[OK]   {name:35s} "
                f"type={result['type']:10s} "
                f"is_peft={result['is_peft']}"
            )

            del result

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except NotImplementedError as e:
            print(f"[SKIP] {name:35s} {e}")

        except Exception as e:
            print(
                f"[FAIL] {name:35s} "
                f"{type(e).__name__}: {e}"
            )