class Quantization:
    """Base class for quantization strategies"""
    def apply(self, model):
        """Apply quantization to a model"""
        pass
    
    def apply_to_ollama_params(self, params):
        """Apply quantization settings to Ollama API parameters"""
        return params
    
    def get_model_name(self):
        """Get the model name for this quantization level"""
        return None

class Q1_5B(Quantization):
    """1.5/7b quantization strategy"""
    
    def get_model_name(self):
        return "qwen:3-1.5b"
    
    def apply_to_ollama_params(self, params):
        # Use Qwen-3-1.5B model directly
        params["model"] = self.get_model_name()
        return params

class Q3B(Quantization):
    """3b quantization strategy"""
    
    def get_model_name(self):
        return "qwen:3-3b"
    
    def apply_to_ollama_params(self, params):
        # Use Qwen-3-3B model directly
        params["model"] = self.get_model_name()
        return params

class Q8B(Quantization):
    """8b quantization strategy"""
    
    def get_model_name(self):
        return "qwen:3-8b"
    
    def apply_to_ollama_params(self, params):
        # Use Qwen-3-8B model directly
        params["model"] = self.get_model_name()
        return params 