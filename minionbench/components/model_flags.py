class ReasoningFlag:
    """Flag for enabling/disabling reasoning"""
    def __init__(self, enabled=False):
        self.enabled = enabled
    
    def apply_to_ollama_params(self, params):
        """Apply reasoning setting to Ollama parameters"""
        if self.enabled:
            # For models that support reasoning/chain-of-thought
            # This could be model-specific format in the future
            if "options" not in params:
                params["options"] = {}
            params["options"]["reasoning"] = True
        return params 