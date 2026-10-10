from typing import Dict, List, Optional
from app.adapters.base_adapter import BaseModelAdapter
from app.adapters.unet_mamba_adapter import UNetMambaAdapter
from app.adapters.unet_adapter import UNetAdapter
from app.adapters.segformer_adapter import SegFormerAdapter

class ModelRegistry:
    """
    Singleton registry managing semantic segmentation architecture adapters.
    Provides dynamic discovery, model caching, and compatibility checks.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        self._adapters: Dict[str, BaseModelAdapter] = {}
        
        # Register architectures
        mamba = UNetMambaAdapter()
        unet = UNetAdapter()
        segformer = SegFormerAdapter()
        
        self._adapters[mamba.model_id] = mamba
        self._adapters[unet.model_id] = unet
        self._adapters[segformer.model_id] = segformer
        
    def get_adapter(self, model_id: str) -> Optional[BaseModelAdapter]:
        return self._adapters.get(model_id)
        
    def list_adapters(self) -> List[BaseModelAdapter]:
        return list(self._adapters.values())
        
    def list_models_info(self) -> List[Dict]:
        return [adapter.get_model_info() for adapter in self._adapters.values()]
        
    def get_available_models(self) -> List[BaseModelAdapter]:
        return [adapter for adapter in self._adapters.values() if adapter.is_available]

    def get_checkpoints_ledger(self) -> List[Dict]:
        return [adapter.get_checkpoint_info() for adapter in self._adapters.values()]

model_registry = ModelRegistry()
