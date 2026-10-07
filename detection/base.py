from dataclasses import dataclass , field 
from typing import Optional , Dict , Any 
from abc import ABC , abstractmethod 

@dataclass
class ValidationError:
    layer: str
    message: str
    field: str = ""

@dataclass
class LayerResult:
    risk_score: float
    status: str
    detail: Dict[str, Any] = field(default_factory=dict)

class Detector(ABC):
    @abstractmethod
    def run(self, input_data) -> LayerResult:
        pass
 
    @abstractmethod
    def validate(self, input_data) -> Optional[ValidationError]:
        pass