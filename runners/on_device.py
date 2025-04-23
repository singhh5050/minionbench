from utils.tracking import timeit
from utils.energy_tracking import PowerMonitorContext 
import ollama
import logging
from transformers import AutoTokenizer

@timeit
def infer(prompt):
    response = ollama.generate(
        model="mistral:latest",
        prompt=prompt,
        options={"num_predict": 500}
    )
    return response["response"]

def run_on_device(prompt):
    # Wrap inference with energy tracking
    with PowerMonitorContext(mode="mac") as monitor:
        answer, aux = infer(prompt)

    # Extract energy usage
    try:
        energy_str = monitor.get_final_estimates()["Measured Energy"]
        energy_j = float(energy_str.replace(" J", ""))
    except (KeyError, ValueError) as e:
        logging.warning(f"[EnergyTracker] Failed to parse energy: {e}")
        energy_j = 0.0

    # Calculate generated tokens (same as output tokens for on-device model)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    output_tokens = len(tokenizer.encode(answer, add_special_tokens=False))

    return {
        "prompt": prompt,
        "answer": answer,
        "latency": aux["latency"],
        "energy_j": energy_j,
        "protocol": "local",
        "model_name": "mistral",
        "generated_tokens": output_tokens,  # For local model, generated = output (no internal reasoning)
        "output_tokens": output_tokens  # Add output_tokens to match expected field
    }
