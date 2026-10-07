import torch   
import torch.nn as nn 
from typing import Dict, List
from dataclasses import dataclass   

DEFAULT_LAYERS = [14,18,22,26]

@dataclass
class LayerActivations:
    layer_idx: int
    activations: torch.Tensor
    attention_mask: torch.Tensor

class _ActivationHook:
    def __init__(self):
        self.captured = None
    def __call__(self,module,inputs,output):
        self.captured = output[0].detach() if isinstance(output,tuple) else output.detach() 
    def clear(self):
        self.captured = None

class ActivationExtractor:
    def __init__(self, model: nn.Module, tokenizer, device: str):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self._hooks: Dict[int, _ActivationHook] = {}
        self._handles: List = []

    def _get_decoder_layer(self, layer_idx: int) -> nn.Module:
        base = self.model
        if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
            base = base.base_model.model
        if hasattr(base, "model") and hasattr(base.model, "layers"):
            base = base.model
        if not hasattr(base, "layers"):
            raise AttributeError(
                "Could not locate '.layers' on the model — architecture may "
                "differ from the expected SmolLM2/Llama-style structure. "
                "Inspect model.named_modules() to find the correct path."
            )
        return base.layers[layer_idx]
 
    def register_hooks(self, layer_indices: List[int] = None):
        layer_indices = layer_indices or DEFAULT_LAYERS
        self.clear_hooks()
        for idx in layer_indices:
            hook = _ActivationHook()
            handle = self._get_decoder_layer(idx).register_forward_hook(hook)
            self._hooks[idx] = hook
            self._handles.append(handle)
 
    def clear_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._hooks = {}
 
    def extract(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[int, LayerActivations]:
        for hook in self._hooks.values():
            hook.clear()
 
        with torch.no_grad():
            self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
            )
 
        result = {}
        for layer_idx, hook in self._hooks.items():
            if hook.captured is None:
                raise RuntimeError(f"Hook on layer {layer_idx} never fired — check hook registration.")
            result[layer_idx] = LayerActivations(
                layer_idx=layer_idx,
                activations=hook.captured.cpu(),
                attention_mask=attention_mask.cpu(),
            )
        return result
    