import os
import requests
import json
import time
from typing import Dict, Any, List, Generator, Optional, Iterator
from abc import ABC, abstractmethod

# Import metrics tracking
from .metrics import track_model_metrics, MetricsTracker, print_metrics_summary

# Import minions package components
try:
    from minions.utils.energy_tracking import PowerMonitor, PowerMonitorContext
    from minions.utils.inference_estimator import InferenceEstimator
    from minions.cli import initialize_client, parse_model_string
    from minions.minion import Minion
    MINIONS_AVAILABLE = True
except ImportError:
    MINIONS_AVAILABLE = False
    print("⚠️  Minions package not available. Hybrid deployment will be disabled.")

class Model(ABC):
    """Base class for all model implementations"""
    
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from the model"""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate streaming response from the model"""
        pass

class Deployment:
    def execute(self, workload, model):
        """Execute a workload on a model"""
        pass
    
    def setup_model(self, reasoning, quant=None, remote_model=None):
        """Set up a model with specified configurations"""
        pass
    
    def execute_with_metrics(self, workload_data, model, enable_energy: bool = True, 
                           model_name: str = "llama3.2") -> Dict[str, Any]:
        """Execute workload with comprehensive metrics tracking"""
        results = []
        total_metrics = {
            "total_ttft": 0,
            "total_latency": 0,
            "total_tokens": 0,
            "total_energy": 0,
            "prompt_count": 0,
            "individual_results": []
        }
        
        print(f"\n🚀 Executing workload with metrics tracking...")
        
        for i, prompt in enumerate(workload_data):
            print(f"\nProcessing prompt {i+1}/{len(workload_data)}...")
            
            # Estimate input tokens (rough estimate: 4 chars per token)
            input_tokens = len(prompt) // 4
            
            # Track metrics for this prompt
            metrics = track_model_metrics(
                model=model,
                prompt=prompt,
                model_name=model_name,
                enable_energy=enable_energy,
                input_tokens=input_tokens
            )
            
            # Accumulate metrics
            if metrics.get("ttft_seconds"):
                total_metrics["total_ttft"] += metrics["ttft_seconds"]
            if metrics.get("total_latency_seconds"):
                total_metrics["total_latency"] += metrics["total_latency_seconds"]
            total_metrics["total_tokens"] += metrics.get("token_count", 0)
            total_metrics["prompt_count"] += 1
            
            # Store result
            result = {
                "prompt": prompt,
                "response": metrics.get("response", ""),
                "metrics": metrics
            }
            results.append(result)
            total_metrics["individual_results"].append(result)
            
            # Print summary for this prompt
            print_metrics_summary(metrics)
        
        # Calculate averages
        if total_metrics["prompt_count"] > 0:
            total_metrics["avg_ttft"] = total_metrics["total_ttft"] / total_metrics["prompt_count"]
            total_metrics["avg_latency"] = total_metrics["total_latency"] / total_metrics["prompt_count"]
            total_metrics["avg_throughput"] = total_metrics["total_tokens"] / total_metrics["total_latency"] if total_metrics["total_latency"] > 0 else 0
        
        print(f"\n📊 Workload Summary:")
        print(f"   • Prompts: {total_metrics['prompt_count']}")
        print(f"   • Average TTFT: {total_metrics.get('avg_ttft', 0):.3f}s")
        print(f"   • Average Latency: {total_metrics.get('avg_latency', 0):.3f}s")
        print(f"   • Average Throughput: {total_metrics.get('avg_throughput', 0):.1f} tokens/sec")
        print(f"   • Total Tokens: {total_metrics['total_tokens']}")
        
        return {
            "results": results,
            "summary_metrics": total_metrics
        }

class Local(Deployment):
    """Local deployment strategy"""
    
    def __init__(self, model_name: str, backend="ollama", api_base="http://localhost:11434"):
        self.model_name = model_name
        self.backend = backend  # "ollama" or "llamacpp"
        self.api_base = api_base
    
    def setup_model(self, reasoning, quant=None, remote_model=None):
        """Set up a local model with specified configurations"""
        if self.backend == "ollama":
            # Create model params
            model_params = {"model": self.model_name}
            
            # Apply quantization if specified
            if quant:
                model_params = quant.apply_to_ollama_params(model_params)
            
            # Create reasoning params
            if reasoning:
                if "options" not in model_params:
                    model_params["options"] = {}
                model_params["options"]["reasoning"] = True
            
            return OllamaModel(
                model_name=model_params["model"],
                api_base=self.api_base
            )
        else:
            # Implementation for llamacpp or other local backends
            raise NotImplementedError(f"Backend {self.backend} not implemented")
    
    def execute(self, workload_data, model):
        """Execute a workload on a local model"""
        return self.execute_with_metrics(workload_data, model, 
                                       enable_energy=True, 
                                       model_name=self.model_name)

class Remote(Deployment):
    """Remote deployment strategy"""
    
    def __init__(self):
        pass
    
    def setup_model(self, reasoning, quant=None, remote_model=None):
        """Set up a remote model with specified configurations"""
        if remote_model is None:
            raise ValueError("Remote model must be specified for remote deployment")
        
        # Apply reasoning by selecting appropriate model variant
        if hasattr(remote_model, 'set_reasoning_mode'):
            remote_model.set_reasoning_mode(reasoning)
        
        return remote_model
    
    def execute(self, workload_data, model):
        """Execute a workload on a remote model"""
        # For remote models, we estimate the model name
        model_name = getattr(model, 'model_name', 'gpt-4o')
        return self.execute_with_metrics(workload_data, model, 
                                       enable_energy=False,  # Remote models don't measure local energy
                                       model_name=model_name)

class Hybrid(Deployment):
    """Hybrid deployment using Minions protocol for local/remote coordination."""
    
    def __init__(self, local_model: str, remote_model: str, **kwargs):
        super().__init__(**kwargs)
        self.local_model = local_model
        self.remote_model = remote_model
        
    def setup_model(self, reasoning, quant=None, remote_model=None):
        """Setup the Minion model for hybrid deployment with all independent variables."""
        
        if not MINIONS_AVAILABLE:
            raise RuntimeError("Minions package not available. Install with: pip install minions")
        
        # Parse local and remote model strings
        local_provider, local_model_name = parse_model_string(self.local_model)
        if quant:
            # Apply quantization to local model name
            if hasattr(quant, 'get_model_name'):
                local_model_name = quant.get_model_name()
            else:
                # Fallback: use quantization's model selection
                dummy_params = {"model": local_model_name}
                modified_params = quant.apply_to_ollama_params(dummy_params)
                local_model_name = modified_params["model"]
        
        # Handle remote model family
        remote_provider, remote_model_name = parse_model_string(self.remote_model)
        if remote_model:
            # Use the provided remote model instance
            if hasattr(remote_model, 'set_reasoning_mode'):
                remote_model.set_reasoning_mode(reasoning)
            # Override remote model string with the model family's choice
            remote_model_name = getattr(remote_model, 'model_name', remote_model_name)
        
        return MinionModel(
            local_model=f"{local_provider}/{local_model_name}",
            remote_model=f"{remote_provider}/{remote_model_name}",
            reasoning=reasoning
        )
    
    def execute(self, workload_data, model):
        """Execute a workload on a hybrid model"""
        # For hybrid, use the local model name for metrics estimation
        local_model_name = self.local_model.split('/')[-1] if '/' in self.local_model else self.local_model
        return self.execute_with_metrics(workload_data, model, 
                                       enable_energy=True,  # Hybrid measures local energy
                                       model_name=local_model_name)

class MinionModel(Model):
    """Minion deployment using the Minions package for hybrid local/remote inference."""
    
    def __init__(self, local_model: str, remote_model: str, reasoning: bool = False, **kwargs):
        super().__init__(**kwargs)
        
        if not MINIONS_AVAILABLE:
            raise RuntimeError("Minions package not available. Install with: pip install minions")
        
        self.local_model = local_model
        self.remote_model = remote_model
        self.reasoning = reasoning
        self.streaming_tokens = []
        
        # Parse model strings using minions utility
        local_provider, local_model_name = parse_model_string(local_model)
        remote_provider, remote_model_name = parse_model_string(remote_model)
        
        # Initialize clients using minions utility
        self.local_client = initialize_client(
            provider=local_provider,
            model_name=local_model_name,
            temperature=0.0,
            max_tokens=4096,
            num_ctx=4096
        )
        
        self.remote_client = initialize_client(
            provider=remote_provider,
            model_name=remote_model_name,
            temperature=0.2,
            max_tokens=2048
        )
        
        # Create minion with streaming callback
        self.minion = Minion(
            self.local_client, 
            self.remote_client, 
            callback=self._streaming_callback
        )
    
    def _streaming_callback(self, message):
        """Callback to capture streaming tokens from Minions protocol."""
        if message.get("content"):
            self.streaming_tokens.append(message["content"])
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Minion protocol."""
        self.streaming_tokens = []
        
        result = self.minion(
            task=prompt,
            context=[""],
            max_rounds=2
        )
        
        return result.get("final_answer", "")
    
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate streaming response using Minion protocol."""
        self.streaming_tokens = []
        
        # Execute minion protocol with callback
        self.minion(
            task=prompt,
            context=[""],
            max_rounds=2
        )
        
        # Yield captured streaming tokens
        for token in self.streaming_tokens:
            if token.strip():
                yield token

class OllamaModel:
    """Wrapper for Ollama API"""
    
    def __init__(self, model_name, api_base="http://localhost:11434"):
        self.model_name = model_name
        self.api_base = api_base
    
    def generate(self, prompt, params=None):
        """Generate text from Ollama model (non-streaming)"""
        if not params:
            params = {}
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            **params
        }
        
        response = requests.post(
            f"{self.api_base}/api/generate",
            json=data
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"Ollama API error: {response.text}")
    
    def generate_stream(self, prompt, params=None) -> Generator[str, None, None]:
        """Generate text from Ollama model with streaming"""
        if not params:
            params = {}
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            **params
        }
        
        response = requests.post(
            f"{self.api_base}/api/generate",
            json=data,
            stream=True
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code}")
        
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                if "response" in json_response:
                    yield json_response["response"]
                
                # Stop if we're done
                if json_response.get("done", False):
                    break 