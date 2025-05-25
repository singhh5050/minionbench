import time
from typing import Dict, Any, List, Optional, Iterator
from dataclasses import dataclass, field
from contextlib import contextmanager

# Import utilities from minions
from minions.utils.energy_tracking import (
    PowerMonitor, 
    PowerMonitorContext, 
    cloud_inference_energy_estimate_w_model_attributes
)
from minions.utils.inference_estimator import InferenceEstimator


@dataclass
class StreamingMetrics:
    """Captures real-time metrics during streaming inference"""
    
    # Timing metrics
    start_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Token counting
    token_count: int = 0
    tokens: List[str] = field(default_factory=list)
    
    # Energy metrics (will be populated by context manager)
    energy_data: Optional[Dict[str, Any]] = None
    
    @property
    def ttft(self) -> Optional[float]:
        """Time To First Token in seconds"""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.start_time
    
    @property
    def total_latency(self) -> Optional[float]:
        """Total latency in seconds"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    @property
    def per_token_throughput(self) -> Optional[float]:
        """Tokens per second"""
        if self.total_latency is None or self.total_latency == 0:
            return None
        return self.token_count / self.total_latency
    
    def record_token(self, token: str):
        """Record a token and timestamp"""
        if self.first_token_time is None:
            self.first_token_time = time.time()
        
        self.tokens.append(token)
        self.token_count += 1
    
    def finalize(self):
        """Mark the end of streaming"""
        self.end_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        result = {
            "ttft_seconds": self.ttft,
            "total_latency_seconds": self.total_latency,
            "per_token_throughput": self.per_token_throughput,
            "token_count": self.token_count,
            "total_tokens": len(self.tokens)
        }
        
        # Add energy metrics if available
        if self.energy_data:
            result["energy_metrics"] = self.energy_data
        
        return result


class MetricsTracker:
    """Simple metrics tracker using minions utilities"""
    
    def __init__(self, model_name: str = "llama3.2", enable_energy: bool = True):
        self.model_name = model_name
        self.enable_energy = enable_energy
        self.power_monitor = None
        
        # Try to initialize inference estimator (handles errors gracefully)
        self.inference_estimator = None
        try:
            self.inference_estimator = InferenceEstimator(model_name)
        except Exception as e:
            print(f"Could not initialize inference estimator: {e}")
    
    @contextmanager
    def track_streaming(self, prompt: str = "", input_tokens: int = 0) -> Iterator[StreamingMetrics]:
        """Context manager for tracking streaming metrics"""
        metrics = StreamingMetrics()
        
        # Start power monitoring if enabled
        if self.enable_energy:
            try:
                self.power_monitor = PowerMonitor(mode="auto", interval=0.5)
                self.power_monitor.start()
            except Exception as e:
                print(f"Power monitoring not available: {e}")
                self.power_monitor = None
        
        try:
            yield metrics
        finally:
            # Finalize timing
            metrics.finalize()
            
            # Stop power monitoring and collect energy data
            if self.power_monitor:
                try:
                    self.power_monitor.stop()
                    energy_estimates = self.power_monitor.get_final_estimates()
                    
                    # Also get cloud energy estimate if we have token counts
                    if metrics.token_count > 0 and metrics.total_latency:
                        cloud_estimate = cloud_inference_energy_estimate_w_model_attributes(
                            input_tokens=input_tokens,
                            output_tokens=metrics.token_count,
                            model_name="gpt-4o",  # Generic model for estimation
                            inference_wall_time_sec=metrics.total_latency
                        )
                        energy_estimates["cloud_estimate"] = cloud_estimate
                    
                    metrics.energy_data = energy_estimates
                except Exception as e:
                    print(f"Error collecting energy data: {e}")
    
    def estimate_performance(self, n_tokens: int) -> Dict[str, Any]:
        """Get performance estimates using inference estimator"""
        if not self.inference_estimator:
            return {"error": "Inference estimator not available"}
        
        try:
            tps, eta = self.inference_estimator.estimate(n_tokens)
            return {
                "estimated_throughput": f"{tps:.1f} tokens/sec",
                "estimated_time": f"{eta:.2f} seconds",
                "estimated_tokens": n_tokens
            }
        except Exception as e:
            return {"error": f"Could not estimate performance: {e}"}


def track_model_metrics(model, prompt: str, model_name: str = "llama3.2", 
                       enable_energy: bool = True, input_tokens: int = 0) -> Dict[str, Any]:
    """
    Simple function to track metrics for any model with generate_stream method.
    
    Args:
        model: Model instance with generate_stream method
        prompt: Input prompt
        model_name: Model name for inference estimation
        enable_energy: Whether to track energy consumption
        input_tokens: Number of input tokens for energy estimation
    
    Returns:
        Dictionary with all metrics
    """
    tracker = MetricsTracker(model_name, enable_energy)
    
    with tracker.track_streaming(prompt, input_tokens) as metrics:
        # Stream the response and collect tokens
        response = ""
        for token in model.generate_stream(prompt):
            metrics.record_token(token)
            response += token
    
    # Get all metrics
    result = metrics.to_dict()
    result["response"] = response
    result["prompt"] = prompt
    
    # Add performance estimates
    if metrics.token_count > 0:
        estimates = tracker.estimate_performance(metrics.token_count)
        result["performance_estimates"] = estimates
    
    return result


def print_metrics_summary(metrics: Dict[str, Any]):
    """Pretty print metrics summary"""
    print("\n📊 Performance Metrics Summary")
    print("=" * 50)
    
    # Timing metrics
    if metrics.get("ttft_seconds"):
        print(f"⚡ Time to First Token: {metrics['ttft_seconds']:.3f}s")
    
    if metrics.get("total_latency_seconds"):
        print(f"⏱️  Total Latency: {metrics['total_latency_seconds']:.3f}s")
    
    if metrics.get("per_token_throughput"):
        print(f"🚀 Throughput: {metrics['per_token_throughput']:.1f} tokens/sec")
    
    print(f"📝 Total Tokens: {metrics.get('token_count', 0)}")
    
    # Energy metrics
    energy_data = metrics.get("energy_metrics")
    if energy_data:
        print("\n🔋 Energy Consumption:")
        if "Measured Energy" in energy_data:
            print(f"   💡 Measured Energy: {energy_data['Measured Energy']}")
        if "Average Measured Power" in energy_data:
            print(f"   ⚡ Average Power: {energy_data['Average Measured Power']}")
        
        # Cloud estimate
        if "cloud_estimate" in energy_data:
            cloud = energy_data["cloud_estimate"]
            if "total_energy_wh_empirical" in cloud:
                print(f"   ☁️  Cloud Estimate: {cloud['total_energy_wh_empirical']:.6f} Wh")
    
    # Performance estimates
    estimates = metrics.get("performance_estimates")
    if estimates and "error" not in estimates:
        print(f"\n🔮 Estimated Performance: {estimates.get('estimated_throughput', 'N/A')}")
    
    print("=" * 50) 