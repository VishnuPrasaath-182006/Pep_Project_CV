from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np

class BaseModelAdapter(ABC):
    """
    Abstract Model Adapter defining standard lifecycle, preprocessing,
    inference, and postprocessing methods for each architecture.
    """
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        pass
        
    @property
    @abstractmethod
    def name(self) -> str:
        pass
        
    @property
    @abstractmethod
    def architecture(self) -> str:
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        pass
        
    @property
    @abstractmethod
    def is_available(self) -> bool:
        pass
        
    @property
    @abstractmethod
    def status_message(self) -> str:
        pass
        
    @property
    @abstractmethod
    def input_channels(self) -> int:
        pass
        
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    def check_compatibility(self, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        pass
        
    @abstractmethod
    def run_inference(self, raw_data: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes genuine model inference on input satellite array.
        Returns dictionary containing:
            - binary_mask: (H, W) uint8
            - probability_map: (H, W) float32
            - mean_confidence: float
            - valid_pixels: int
            - flood_pixels: int
            - background_pixels: int
            - flood_coverage_percentage: float
            - processing_time_ms: float
            - device: str
        """
        pass
