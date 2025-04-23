import pandas as pd

def load_prompts(path: str, sample: int = None):
    df = pd.read_csv(path)
    df = df[["query", "category"]].dropna(subset=["query", "category"])
    if sample:
        df = df.sample(n=sample)
    return df.rename(columns={"query": "prompt"}).to_dict(orient="records")

