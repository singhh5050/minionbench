import itertools
import pandas as pd
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from minionbench.components.workloads import Prefill, Balanced, Decode
from minionbench.components.deployment import Local, Remote, Hybrid
from minionbench.components.model_flags import ReasoningFlag
from minionbench.components.quantization import Q1_5B, Q3B, Q8B
from minionbench.components.remote_models import DeepSeek, OpenAI, Claude
from minionbench.experiment import Experiment

class BenchmarkSweep:
    """Complete benchmarking suite for MinionBench"""
    
    def __init__(self, limit_per_workload: int = 3, output_dir: str = "results"):
        self.limit = limit_per_workload
        self.output_dir = output_dir
        self.results = []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the sweep space
        self.sweep_config = {
            "workloads": [Prefill(), Balanced(), Decode()],
            "reasoning": [True, False],
            "quantization": [Q1_5B(), Q3B(), Q8B()],
            "remote_models": [DeepSeek(), OpenAI(), Claude()]
        }
        
        # Deployment configurations
        self.deployments = {
            "local": Local("llama3.2"),
            "remote": Remote(),
            "hybrid": Hybrid("ollama/llama3.2", "openai/gpt-4o")
        }
    
    def run_full_sweep(self):
        """Run the complete benchmarking sweep across all independent variables"""
        print("🚀 Starting MinionBench Full Sweep")
        print(f"📊 Configuration:")
        print(f"   • Workloads: {len(self.sweep_config['workloads'])}")
        print(f"   • Reasoning modes: {len(self.sweep_config['reasoning'])}")
        print(f"   • Quantization levels: {len(self.sweep_config['quantization'])}")
        print(f"   • Remote models: {len(self.sweep_config['remote_models'])}")
        print(f"   • Deployments: {len(self.deployments)}")
        print(f"   • Prompts per workload: {self.limit}")
        
        total_experiments = self._calculate_total_experiments()
        print(f"   • Total experiments: {total_experiments}")
        print("=" * 60)
        
        experiment_count = 0
        
        # Test each deployment type
        for deployment_name, deployment in self.deployments.items():
            print(f"\n🔧 Testing {deployment_name.upper()} deployment...")
            
            for workload in self.sweep_config["workloads"]:
                for reasoning in self.sweep_config["reasoning"]:
                    
                    if deployment_name == "local":
                        # Local deployment: test all quantization levels
                        for quant in self.sweep_config["quantization"]:
                            experiment_count += 1
                            self._run_single_experiment(
                                workload, deployment, reasoning, 
                                quant=quant, remote_model=None,
                                experiment_id=experiment_count, total=total_experiments
                            )
                    
                    elif deployment_name == "remote":
                        # Remote deployment: test all remote models
                        for remote_model in self.sweep_config["remote_models"]:
                            experiment_count += 1
                            self._run_single_experiment(
                                workload, deployment, reasoning,
                                quant=None, remote_model=remote_model,
                                experiment_id=experiment_count, total=total_experiments
                            )
                    
                    elif deployment_name == "hybrid":
                        # Hybrid deployment: test combinations of quantization and remote models
                        for quant in self.sweep_config["quantization"]:
                            for remote_model in self.sweep_config["remote_models"]:
                                experiment_count += 1
                                self._run_single_experiment(
                                    workload, deployment, reasoning,
                                    quant=quant, remote_model=remote_model,
                                    experiment_id=experiment_count, total=total_experiments
                                )
        
        # Save results
        self._save_results()
        self._generate_summary()
        
        print(f"\n✅ Sweep completed! {len(self.results)} experiments run.")
        print(f"📁 Results saved to {self.output_dir}/")
    
    def _run_single_experiment(self, workload, deployment, reasoning, quant, remote_model, experiment_id, total):
        """Run a single experiment configuration"""
        print(f"\n[{experiment_id}/{total}] Running experiment...")
        
        try:
            experiment = Experiment(
                workload=workload,
                deployment=deployment,
                reasoning=reasoning,
                quant=quant,
                remote_model=remote_model,
                limit=self.limit
            )
            
            result = experiment.run()
            self.results.append(result)
            
            # Print brief summary
            if "error" not in result:
                summary = result.get("summary_metrics", {})
                print(f"   ✅ Success - Avg TTFT: {summary.get('avg_ttft', 0):.3f}s, "
                      f"Avg Latency: {summary.get('avg_latency', 0):.3f}s")
            else:
                print(f"   ❌ Failed: {result['error']}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            # Still record the failure
            self.results.append({
                "error": str(e),
                "experiment_metadata": {
                    "workload_type": type(workload).__name__,
                    "deployment_type": type(deployment).__name__,
                    "reasoning_enabled": reasoning,
                    "quantization": type(quant).__name__ if quant else None,
                    "remote_model": type(remote_model).__name__ if remote_model else None,
                    "failed": True
                }
            })
    
    def _calculate_total_experiments(self) -> int:
        """Calculate total number of experiments that will be run"""
        local_experiments = len(self.sweep_config["workloads"]) * len(self.sweep_config["reasoning"]) * len(self.sweep_config["quantization"])
        remote_experiments = len(self.sweep_config["workloads"]) * len(self.sweep_config["reasoning"]) * len(self.sweep_config["remote_models"])
        hybrid_experiments = len(self.sweep_config["workloads"]) * len(self.sweep_config["reasoning"]) * len(self.sweep_config["quantization"]) * len(self.sweep_config["remote_models"])
        
        return local_experiments + remote_experiments + hybrid_experiments
    
    def _save_results(self):
        """Save results to JSON and CSV formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        json_path = os.path.join(self.output_dir, f"detailed_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create CSV summary
        csv_data = []
        for result in self.results:
            if "experiment_metadata" in result:
                metadata = result["experiment_metadata"]
                summary = result.get("summary_metrics", {})
                
                row = {
                    "workload": metadata.get("workload_type"),
                    "deployment": metadata.get("deployment_type"),
                    "reasoning": metadata.get("reasoning_enabled"),
                    "quantization": metadata.get("quantization"),
                    "remote_model": metadata.get("remote_model"),
                    "num_prompts": metadata.get("num_prompts", 0),
                    "failed": metadata.get("failed", False),
                    "avg_ttft_seconds": summary.get("avg_ttft", None),
                    "avg_latency_seconds": summary.get("avg_latency", None),
                    "avg_throughput_tokens_per_sec": summary.get("avg_throughput", None),
                    "total_tokens": summary.get("total_tokens", None),
                    "timestamp": metadata.get("timestamp")
                }
                csv_data.append(row)
        
        # Save CSV
        csv_path = os.path.join(self.output_dir, f"summary_results_{timestamp}.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        
        print(f"📄 Detailed results: {json_path}")
        print(f"📊 Summary CSV: {csv_path}")
    
    def _generate_summary(self):
        """Generate a summary report"""
        successful_experiments = [r for r in self.results if "error" not in r]
        failed_experiments = [r for r in self.results if "error" in r]
        
        summary_path = os.path.join(self.output_dir, "benchmark_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("MinionBench Sweep Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Experiments: {len(self.results)}\n")
            f.write(f"Successful: {len(successful_experiments)}\n")
            f.write(f"Failed: {len(failed_experiments)}\n\n")
            
            if successful_experiments:
                f.write("Performance Overview:\n")
                f.write("-" * 20 + "\n")
                
                # Calculate averages
                total_ttft = sum(r.get("summary_metrics", {}).get("avg_ttft", 0) for r in successful_experiments if r.get("summary_metrics", {}).get("avg_ttft"))
                total_latency = sum(r.get("summary_metrics", {}).get("avg_latency", 0) for r in successful_experiments if r.get("summary_metrics", {}).get("avg_latency"))
                total_throughput = sum(r.get("summary_metrics", {}).get("avg_throughput", 0) for r in successful_experiments if r.get("summary_metrics", {}).get("avg_throughput"))
                
                count_ttft = len([r for r in successful_experiments if r.get("summary_metrics", {}).get("avg_ttft")])
                count_latency = len([r for r in successful_experiments if r.get("summary_metrics", {}).get("avg_latency")])
                count_throughput = len([r for r in successful_experiments if r.get("summary_metrics", {}).get("avg_throughput")])
                
                if count_ttft > 0:
                    f.write(f"Average TTFT: {total_ttft/count_ttft:.3f}s\n")
                if count_latency > 0:
                    f.write(f"Average Latency: {total_latency/count_latency:.3f}s\n")
                if count_throughput > 0:
                    f.write(f"Average Throughput: {total_throughput/count_throughput:.1f} tokens/sec\n")
        
        print(f"📋 Summary report: {summary_path}")

def main_sweep(limit_per_workload: int = 3, output_dir: str = "results"):
    """Run the main benchmarking sweep"""
    sweep = BenchmarkSweep(limit_per_workload, output_dir)
    sweep.run_full_sweep()

if __name__ == "__main__":
    main_sweep() 