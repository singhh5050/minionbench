from utils.tracking import timeit
from utils.energy_tracking import PowerMonitorContext, cloud_inference_energy_estimate_w_model_attributes
import sys
import logging
import os
from pydantic import BaseModel

# Import directly from minions package
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.minions import Minions

class StructuredLocalOutput(BaseModel):
    explanation: str
    citation: str | None
    answer: str | None

# Debug: Check if API key is available
api_key = os.getenv("OPENAI_API_KEY")

# Initialize Minion once
local_client = OllamaClient(
    model_name="mistral:latest",
    temperature=0.0,
    structured_output_schema=StructuredLocalOutput,
    num_ctx = 1200
)

remote_client = OpenAIClient(
    model_name="gpt-4o",
    api_key=api_key
)

minion = Minions(local_client, remote_client)

@timeit
def infer(prompt: str):
    generic_context = [prompt]
    
    return minion(
        task="Please respond to the provided prompt.",
        context=generic_context, 
        doc_metadata="benchmark",
        max_rounds=1
    )

def run_minions(prompt: str):
    # 1. Run protocol under power monitor + timer
    with PowerMonitorContext(mode="mac") as monitor:
        result, aux = infer(prompt)

    # === Local energy (measured) ===
    try:
        energy_local_j = float(
            monitor.get_final_estimates().get("Measured Energy", "0 J").replace(" J", "")
        )
    except (KeyError, ValueError):
        import logging
        logging.warning("[EnergyTracker] Missing local energy, falling back to 0 J")
        energy_local_j = 0.0

    # === Derive timing splits ===
    local_runtime   = monitor.get_total_time()           # seconds
    total_latency   = aux["latency"]                     # seconds
    remote_runtime  = max(total_latency - local_runtime, 0.0)

    # === Token usage ===
    # Get exact token counts from the usage object for internal accounting
    remote_prompt_tokens = result["remote_usage"].prompt_tokens
    remote_completion_tokens = result["remote_usage"].completion_tokens
    local_prompt_tokens = result["local_usage"].prompt_tokens
    local_completion_tokens = result["local_usage"].completion_tokens
    
    # Internal generated tokens (for energy calculations)
    total_internal_tokens = local_prompt_tokens + local_completion_tokens + remote_prompt_tokens + remote_completion_tokens
    
    # Calculate output tokens
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    answer = result['meta'][0]['local']['jobs'][0]['output']['answer']
    output_tokens = len(tokenizer.encode(answer, add_special_tokens=False))
    
    # === Remote energy (estimate only if remote ran) ===
    used_remote = remote_prompt_tokens > 0
    if used_remote and remote_runtime > 0:
        cloud_e = cloud_inference_energy_estimate_w_model_attributes(
            input_tokens=remote_prompt_tokens,
            output_tokens=remote_completion_tokens,
            model_name="gpt-4o",
            gpu_name="H100",
            attention_mode="quadratic",
            # inference_wall_time_sec=remote_runtime
        )
        energy_remote_j = cloud_e["total_energy_joules"]
    else:
        energy_remote_j = 0.0

    total_energy_j = energy_local_j + energy_remote_j

    return {
        "prompt": prompt,
        "answer": answer,
        "latency": total_latency,
        "energy_j": total_energy_j,
        "protocol": "minions",
        "model_name": "hybrid-mistral-gpt4o",
        "generated_tokens": total_internal_tokens,
        "output_tokens": output_tokens,
        # Additional metrics that other runners don't have
        "energy_local_j": energy_local_j,
        "energy_remote_j": energy_remote_j,
    }
