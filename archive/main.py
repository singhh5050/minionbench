from runners.on_device import run_on_device
from runners.cloud import run_cloud
from runners.minions import run_minions
from utils.prompts import load_prompts
import pandas as pd
import argparse
import logging
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)

# Initialize tokenizer once to avoid repeated loading
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

def main(sample_size):
    prompts = load_prompts("data/processed.csv", sample=sample_size)
    results_path = "results/metrics.csv"
    os.makedirs("results", exist_ok=True)

    for prompt_obj in prompts:
        prompt_text = prompt_obj["prompt"]
        category = prompt_obj["category"]
        
        # Calculate input tokens here once for all runners
        input_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))

        for protocol_name, runner in [
            ("local", run_on_device),
            ("remote", run_cloud),
            ("minions", run_minions)
        ]:
            try:
                result = runner(prompt_text)
                
                result["category"] = category
                result["input_tokens"] = input_tokens
                result["prompt"] = prompt_text
                result["protocol"] = protocol_name

                # Save immediately after each result
                df = pd.DataFrame([result])

                if not os.path.exists(results_path):
                    df.to_csv(results_path, index=False)
                    logging.info(f"📄 Created new results file with first entry.")
                else:
                    df.to_csv(results_path, mode="a", header=False, index=False)
                    logging.info(f"➕ Appended new entry to existing results.")

                logging.info(f"✅ {protocol_name} | {prompt_text[:40]}...")
            except Exception as e:
                logging.warning(f"❌ Failed on {protocol_name}: {e}")

    logging.info("🚀 Benchmarking complete. Results saved to results/metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Number of prompts to run (default: all)")
    args = parser.parse_args()
    main(sample_size=args.sample)