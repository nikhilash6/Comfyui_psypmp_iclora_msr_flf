from .nodes import LTXMSRICLoRAFLF
from .test_nodes import LTXMSRICLoRAFLF_Experimental

__version__ = "0.9.4"

NODE_CLASS_MAPPINGS = {
    "LTXMSRICLoRAFLF": LTXMSRICLoRAFLF,
    "LTXMSRICLoRAFLF_Experimental": LTXMSRICLoRAFLF_Experimental
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXMSRICLoRAFLF": "🅛🅣🅧 MSR IC LORA FLF (Standalone)",
    "LTXMSRICLoRAFLF_Experimental": "🅛🅣🅧 MSR IC LORA FLF (Test/Experimental)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
