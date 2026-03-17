from .nodes import LoadFFTNetModel, FFTNetGenerate

NODE_CLASS_MAPPINGS = {
    "LoadFFTNetModel": LoadFFTNetModel,
    "FFTNetGenerate":  FFTNetGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFFTNetModel": "Load FFTNet Model",
    "FFTNetGenerate":  "FFTNet Generate",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
