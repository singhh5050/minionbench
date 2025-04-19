# \U0001F4D1 Pre‑processing & Benchmark Pipeline README

## 1 Context & Why This Notebook Exists  
We downloaded three public ChatGPT‑style datasets (`collected`, `lmsys`, `wildchat`) with the goal of building a clean prompt set for **economic‑activity evaluation**.  
Early on I also tested `rajchat`, but after a deep schema dive (see below) I chose to drop it to keep the benchmark consistent.

## 2 What Went Down During Data Wrangling

| Phase | Hidden Roadblocks & Decisions |
|-------|------------------------------|
| **Parquet load & first peek** | Each file had its *own* nested data structure scheme; even counting rows required custom display helpers. |
| **Schema harmonisation** | `collected` used a nested `metadata` dict; `lmsys` / `wildchat` stored prompts in a list `conversation[0]`.  I wrote an `extract_prompt()` helper to normalise them. |
| **rajchat autopsy** | • No assistant responses<br>• Missing model (literally just said "rajchat")<br>• Incomplete conversation trees. <br>After several mapping attempts I excluded it entirely to avoid contaminating downstream metrics. |
| **NSFW audit** | The raw data had way more dodgy content than expected (violence, hate, sexual minors, you name it).  I expanded the filter to 6⁄9 critical moderation categories and tuned the threshold to 0.10 → removed ~90 rows. |
| **QA / N A cull** | “QA” and “N/A” labels would completely the economics benchmark (being entirely uninformative), so they’re stripped after NSFW cleaning. |
| **Nested JSON pain** | Dicts containing NumPy arrays broke `nunique()` and `.apply()`.  Defensive type‑checks (`isinstance(..., dict)`, `len(list) > 0`) now wrap every access. |

## 3 Current Notebook Features

* **File ingestion** – reads each parquet into `raw` and prints quick shape stats.  
* **Prompt extraction** – `extract_prompt()` pulls the *user* message regardless of layout.  
* **Superset sanity‑check** – `lmsys` + `wildchat` prompts are verified as subsets of `collected`.  
* **NSFW filter** – multi‑source, threshold‑aware (`0.1`) covering OpenAI, Detoxify & Detoxify‑moderation scores.  
* **Economic‑category filter** – drops anything tagged `QA` or `N/A`.  
* **Missing‑field diagnostics** – `pct_missing()` helper reports how healthy key fields are.  
* **Category / model plots** – quick bar charts of remaining categories and model tags to eyeball class balance.  
* **Final DataFrame** – exposed as `final` (clean, non‑empty, non‑NSFW, domain‑correct).

## 4 Quick Usage Cheat‑Sheet

1. Update `DATA_DIR` if parquet paths move.  
2. Run the notebook top‑to‑bottom; the final cleaned data ends up in the variable `final`.  
3. Need stricter NSFW filtering?  Tweak `thresh` in `nsfw_flag()` and rerun the cell.  
4. To inspect missingness of a new metadata key: `pct_missing("new_key")`.

## 5 Next Steps – Eval Harness Scaffold

We now need an *evaluation harness* that can:

* **Sweep protocol** (local‑only ‑vs‑ “minions” ‑vs‑ remote API).  
* **Sweep local model variants** (size, quantization).  
* **Sweep remote model types** (4o, o1, o3-mini, etc.).  
* **Measure** latency, accuracy (LLM‑as‑judge), and **energy** (via system profiler).