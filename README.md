# MinionBench: Language Model Benchmarking Framework

A modular benchmarking suite for evaluating language model performance across different deployment strategies, workloads, and configurations.

## System Design

MinionBench follows a component-based architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Workloads     │    │   Deployments   │    │    Metrics      │
│                 │    │                 │    │                 │
│ • Prefill       │───▶│ • Local         │───▶│ • TTFT          │
│ • Balanced      │    │ • Remote        │    │ • Latency       │
│ • Decode        │    │ • Hybrid        │    │ • Throughput    │
└─────────────────┘    └─────────────────┘    │ • Energy        │
                                              └─────────────────┘
           │                     │                       ▲
           ▼                     ▼                       │
┌─────────────────┐    ┌─────────────────┐               │
│ Configuration   │    │ Model Selection │               │
│                 │    │                 │               │
│ • Reasoning     │    │ • Quantization  │───────────────┘
│ • Remote Models │    │ • Model Flags   │
└─────────────────┘    └─────────────────┘
```

**Core Components:**
- **Workloads**: Define benchmark tasks with specific token distributions
- **Deployments**: Implement execution strategies (local/remote/hybrid)
- **Metrics**: Capture performance measurements during inference
- **Experiments**: Orchestrate end-to-end benchmark execution
- **Analysis**: Process and visualize results

## Variables

### Independent Variables
| Variable | Options | Notes |
|----------|---------|-------|
| **Workload** | Prefill, Balanced, Decode | Finance, OpenAssistant, Math datasets |
| **Deployment** | Local, Remote, Hybrid | Ollama, API, Minions protocol |
| **Reasoning** | True, False | Model reasoning capabilities |
| **Quantization** | Q1_5B, Q3B, Q8B | For local deployment |
| **Remote Model** | DeepSeek, OpenAI, Claude | Remote/Hybrid deployment |

### Dependent Variables
| Metric | Unit | Description |
|--------|------|-------------|
| **Time To First Token (TTFT)** | seconds | Latency to first response token |
| **Total Latency** | seconds | End-to-end response time |
| **Per Token Throughput** | tokens/sec | Generation speed |
| **Energy Consumption** | Wh | Power usage during inference |

## Usage

```python
from minionbench import Experiment, Prefill, Local, Q3B

# Configure experiment
experiment = Experiment(
    workload=Prefill(),
    deployment=Local("llama3.2"),
    reasoning=False,
    quant=Q3B()
)

# Run and collect metrics
results = experiment.run()
```

## Installation

```bash
pip install -r requirements.txt
```

**Note**: This is an MVP implementation. Still a lot of debugging to do once I get API key + access to Mac Studio!