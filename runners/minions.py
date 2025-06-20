from utils.tracking import timeit
from utils.energy_tracking import PowerMonitorContext, cloud_inference_energy_estimate_w_model_attributes
import sys
import logging
import os
from pydantic import BaseModel

# Import directly from minions package
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.minion import Minion

class StructuredLocalOutput(BaseModel):
    explanation: str
    citation: str | None
    answer: str | None

# Debug: Check if API key is available
api_key = os.getenv("OPENAI_API_KEY")

# Define a custom advice prompt for general-purpose queries
# This replaces the default KPI-focused prompt
GENERAL_ADVICE_PROMPT = """
You are a helpful assistant. Your task is to advise a small language model on how to respond to the user's query.

QUERY: {query}
METADATA: {metadata}

Please provide guidance on how to answer this query in a helpful, accurate, and comprehensive manner.
Focus on:
1. Understanding the main intent of the query
2. Providing relevant information directly addressing the question
3. Organizing the response in a clear and coherent way
4. Including any necessary context or background information

Your advice will help the smaller model generate a better response.
"""

# Initialize Minion once
local_client = OllamaClient(
    model_name="mistral:latest",
    temperature=0.0,
    # Structured output not needed for singular Minion
    # structured_output_schema=StructuredLocalOutput,
    num_ctx = 1200
)

remote_client = OpenAIClient(
    model_name="gpt-4o",
    api_key=api_key
)

# Initialize using singular Minion protocol instead of Minions
minion = Minion(
    local_client=local_client, 
    remote_client=remote_client
)

@timeit
def infer(prompt: str):
    return minion(
        task=prompt,  # For Minion (singular), pass the prompt directly as the task
        context=[],   # No need for context here, the task itself is the question
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
    
    # For Minion (singular) the answer is in the final_answer field
    answer = result.get("final_answer", "No result found")
    output_tokens = len(tokenizer.encode(answer, add_special_tokens=False))
    
    # === Remote energy (estimate only if remote ran) ===
    cloud_e = cloud_inference_energy_estimate_w_model_attributes(
        input_tokens=remote_prompt_tokens,
        output_tokens=remote_completion_tokens,
        model_name="gpt-4o",
        gpu_name="H100",
        attention_mode="quadratic",
        # inference_wall_time_sec=remote_runtime
    )
    energy_remote_j = cloud_e["total_energy_joules"]

    total_energy_j = energy_local_j + energy_remote_j

    # Simple recursive print of result structure to a file (won't affect other code)
    try:
        import json
        import os
        os.makedirs("logs", exist_ok=True)
        
        # Helper function to make objects JSON serializable
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, "__dict__"):
                return make_json_serializable(obj.__dict__)
            else:
                try:
                    json.dumps(obj)
                    return obj
                except:
                    return str(obj)
                    
        # Save result to file without affecting main code
        with open("logs/minions_debug.json", "w") as f:
            json.dump(make_json_serializable(result), f, indent=2)
    except:
        # Silently fail if anything goes wrong to not impact main code
        pass

    return {
        "prompt": prompt,
        "answer": answer,
        "latency": total_latency,
        "energy_j": total_energy_j,
        "protocol": "minions",
        "model_name": "hybrid-mistral-gpt4o",
        "generated_tokens": total_internal_tokens,
        "output_tokens": output_tokens
    }
