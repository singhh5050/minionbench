from utils.tracking import timeit
from utils.energy_tracking import cloud_inference_energy_estimate_w_model_attributes
from transformers import AutoTokenizer
from openai import OpenAI

client = OpenAI()

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

@timeit
def infer(prompt):
    response = client.chat.completions.create(model="gpt-4o",
    messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

def run_cloud(prompt):
    answer, aux = infer(prompt)

    # Only calculate tokens needed for energy calculation
    input_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    output_tokens = len(tokenizer.encode(answer, add_special_tokens=False))
    wall_time = aux["latency"]  # already measured by @timeit

    energy_metrics = cloud_inference_energy_estimate_w_model_attributes(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_name="gpt-4o",
        gpu_name="H100",  # or A100 if you prefer a more conservative estimate
        attention_mode="quadratic",
        # inference_wall_time_sec=wall_time
    )

    return {
        "prompt": prompt,
        "answer": answer,
        "latency": wall_time,
        "energy_j": energy_metrics["total_energy_joules"],
        "protocol": "remote",
        "model_name": "gpt-4o",
        "generated_tokens": output_tokens,  # Assuming no internal reasoning patterns
        "output_tokens": output_tokens     # Add output_tokens for consistency
    }