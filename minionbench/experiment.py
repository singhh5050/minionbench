from minionbench.components.workloads import Workload
from minionbench.components.deployment import Deployment, Local, Hybrid
from minionbench.components.model_flags import ReasoningFlag
from minionbench.components.quantization import Quantization
from minionbench.components.remote_models import RemoteModel
from minionbench.components.metrics import track_model_metrics, print_metrics_summary
import json
import time
from typing import Dict, Any, List

class Experiment:
    def __init__(self,
                 workload: Workload,
                 deployment: Deployment,
                 reasoning: bool,
                 quant: Quantization = None,
                 remote_model: RemoteModel = None,
                 limit: int = 5):
        self.workload = workload
        self.deployment = deployment
        self.reasoning = reasoning
        self.quant = quant
        self.remote_model = remote_model
        self.limit = limit

    def run(self) -> Dict[str, Any]:
        """Run the complete experiment with metrics collection"""
        print(f"\n🧪 Running Experiment:")
        print(f"   Workload: {type(self.workload).__name__}")
        print(f"   Deployment: {type(self.deployment).__name__}")
        print(f"   Reasoning: {self.reasoning}")
        print(f"   Quantization: {type(self.quant).__name__ if self.quant else 'None'}")
        print(f"   Remote Model: {type(self.remote_model).__name__ if self.remote_model else 'None'}")
        
        try:
            # 1) Setup model
            model = self.deployment.setup_model(
                reasoning=self.reasoning,
                quant=self.quant,
                remote_model=self.remote_model
            )
            
            # 2) Get workload data
            workload_data = self.workload.get_data(limit=self.limit)
            
            # 3) Execute workload with metrics
            results = self.deployment.execute_with_metrics(
                workload_data=workload_data,
                model=model,
                enable_energy=isinstance(self.deployment, (Local, Hybrid)),
                model_name=self._get_model_name()
            )
            
            # 4) Add experiment metadata
            experiment_metadata = {
                "workload_type": type(self.workload).__name__,
                "deployment_type": type(self.deployment).__name__,
                "reasoning_enabled": self.reasoning,
                "quantization": type(self.quant).__name__ if self.quant else None,
                "remote_model": type(self.remote_model).__name__ if self.remote_model else None,
                "timestamp": time.time(),
                "num_prompts": len(workload_data)
            }
            
            results["experiment_metadata"] = experiment_metadata
            return results
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            return {
                "error": str(e),
                "experiment_metadata": {
                    "workload_type": type(self.workload).__name__,
                    "deployment_type": type(self.deployment).__name__,
                    "reasoning_enabled": self.reasoning,
                    "quantization": type(self.quant).__name__ if self.quant else None,
                    "remote_model": type(self.remote_model).__name__ if self.remote_model else None,
                    "timestamp": time.time(),
                    "failed": True
                }
            }
    
    def _get_model_name(self) -> str:
        """Get model name for metrics estimation"""
        if hasattr(self.deployment, 'model_name'):
            return self.deployment.model_name
        elif hasattr(self.deployment, 'local_model'):
            return self.deployment.local_model.split('/')[-1]
        elif self.remote_model and hasattr(self.remote_model, 'model_name'):
            return self.remote_model.model_name
        else:
            return "unknown_model" 