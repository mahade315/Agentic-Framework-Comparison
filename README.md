# HumanEval Code Generation & Evaluation

A streamlined framework for evaluating LLM code generation capabilities using the HumanEval benchmark.

## 📋 Overview

This project generates Python function completions using **OpenAI models** (direct API) or **CrewAI agents** and evaluates them against the HumanEval benchmark. It provides organized output, progress tracking, and easy evaluation workflows.

**Three modes available:**
- 🔧 **Direct Mode**: Fast, direct OpenAI API calls
- 🤖 **CrewAI Mode**: CrewAI agents with reasoning capabilities
- 🚀 **Qwen-Agent Mode**: Qwen-Agent framework with OpenAI models

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
source afhe_env/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. Configure API Key

Copy `env.template` to `.env` and configure:

```bash
cp env.template .env
# Edit .env with your settings
```

**Minimal configuration:**
```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o
NUM_SAMPLES_PER_TASK=10

# Optional: Use CrewAI agents
USE_CREWAI=false

# Optional: Use Qwen-Agent framework
USE_QWEN_AGENT=false
```

### 3. Run Inference

**Test OpenAI Direct API:**
```bash
USE_CREWAI=false TASK_LIMIT=5 NUM_SAMPLES_PER_TASK=2 python inference.py
```

**Test CrewAI Agent:**
```bash
USE_CREWAI=true TASK_LIMIT=5 NUM_SAMPLES_PER_TASK=2 python inference.py
```

**Full Run (all 164 tasks):**
```bash
python inference.py
```

This will:
- Load HumanEval problems
- Generate completions with a progress bar
- Save samples to `outputs/Generated Samples/`
- Save matching problems to `outputs/Custom Problems/`
- Print the evaluation command

### 4. Evaluate Results

Copy and run the command printed by `inference.py`:

```bash
python human-eval/human_eval/evaluate_functional_correctness.py "outputs/Generated Samples/gpt-4o_20250117_143022.jsonl" --problem_file "outputs/Custom Problems/gpt-4o_20250117_143022.jsonl" && mv "outputs/Generated Samples/gpt-4o_20250117_143022.jsonl_results.jsonl" "outputs/Results/gpt-4o_20250117_143022_results.jsonl"
```

Results will be automatically saved to `outputs/Results/` folder.

## 📁 Project Structure

```
AgenticFrameworks/
├── inference.py              # Main script (auto-selects mode)
├── openAI_models.py          # Direct OpenAI API integration
├── crewai_agent.py           # CrewAI agent implementation
├── sanitize.py               # Code sanitization utilities
├── requirements.txt          # Python dependencies
├── env.template              # Configuration template
├── .env                      # Your configuration (create from template)
├── .gitignore                # Git ignore rules
│
├── outputs/                  # All generated outputs
│   ├── Generated Samples/    # Generated completions
│   │   └── {model}_{timestamp}.jsonl
│   ├── Custom Problems/      # Problem subsets for evaluation
│   │   └── {model}_{timestamp}.jsonl
│   └── Results/              # Evaluation results
│       └── {model}_{timestamp}_results.jsonl
│
├── human-eval/               # HumanEval benchmark (submodule)
│   ├── data/
│   │   └── HumanEval.jsonl.gz
│   └── human_eval/
│       ├── data.py
│       ├── evaluation.py
│       └── evaluate_functional_correctness.py
│
└── afhe_env/                 # Virtual environment
```

## ⚙️ Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Model to use for generation |
| `OPENAI_TEMPERATURE` | `0.2` | Sampling temperature |
| `OPENAI_MAX_TOKENS` | `300` | Max tokens per completion |
| `NUM_SAMPLES_PER_TASK` | `10` | Number of samples per task |
| `TASK_LIMIT` | (none) | Limit to first N tasks |
| `TASK_IDS` | (none) | Comma-separated task IDs to run |
| `SHUFFLE_TASKS` | `false` | Shuffle task order |
| `TASK_SHUFFLE_SEED` | (none) | Random seed for shuffling |
| **`USE_CREWAI`** | `false` | **Use CrewAI agents instead of direct API** |
| `CREWAI_VERBOSE` | `false` | Show detailed agent reasoning |
| `CREWAI_ALLOW_DELEGATION` | `false` | Allow agents to delegate tasks |

### Example: Generate 5 samples for first 10 tasks

```bash
# .env file
TASK_LIMIT=10
NUM_SAMPLES_PER_TASK=5
```

### Example: Run specific tasks

```bash
# .env file
TASK_IDS=HumanEval/0,HumanEval/1,HumanEval/5
NUM_SAMPLES_PER_TASK=20
```

## 🤖 Using CrewAI Agents

### What is CrewAI Mode?

Instead of making direct API calls, CrewAI mode uses an **intelligent agent** that:
- 📝 Understands the problem through reasoning
- 🎯 Plans the solution approach
- 💻 Generates the code
- ✅ Self-validates the output

### Setup CrewAI

**Install dependencies:**
```bash
pip install crewai langchain-openai
```

**Enable in `.env`:**
```bash
USE_CREWAI=true
CREWAI_VERBOSE=false  # Set to 'true' for detailed agent logs
```

### Run with CrewAI

```bash
python inference.py
```

You'll see:
```
🤖 Using CrewAI Agent for code generation
============================================================
Starting HumanEval Inference
============================================================
Model: gpt-4o
...
```

### CrewAI vs Direct API

| Feature | Direct API | CrewAI Agent |
|---------|-----------|--------------|
| **Speed** | ⚡ Fast (1-2s/task) | 🐢 Slower (2-4s/task) |
| **Cost** | 💰 Lower | 💰💰 Higher (2-3x) |
| **Reasoning** | ❌ No | ✅ Yes |
| **Quality** | ✅ Good | ✅ Potentially Better |
| **Debugging** | Basic | Detailed (with verbose) |
| **Use Case** | Benchmarking | Research/Analysis |

### When to Use CrewAI?

**Use CrewAI when:**
- 🔬 Researching agent behaviors
- 🎓 Teaching/demonstrating agent systems
- 🧪 Experimenting with agentic workflows
- 📊 Comparing agent vs direct approaches

**Use Direct API when:**
- ⚡ Need fast results
- 💰 Cost is a concern
- 📈 Running large-scale benchmarks
- 🎯 Baseline performance needed

### Debugging with Verbose Mode

```bash
# .env
USE_CREWAI=true
CREWAI_VERBOSE=true
```

This shows:
- Agent's thought process
- Task decomposition
- Tool usage (if any)
- Internal reasoning steps

### Example: Compare Both Modes

```bash
# Run with Direct API
USE_CREWAI=false TASK_LIMIT=5 python inference.py
# Results: outputs/Generated Samples/gpt-4o_timestamp1.jsonl

# Run with CrewAI
USE_CREWAI=true TASK_LIMIT=5 python inference.py
# Results: outputs/Generated Samples/gpt-4o_timestamp2.jsonl

# Compare evaluation results
python human-eval/... timestamp1.jsonl ...
python human-eval/... timestamp2.jsonl ...
```

## 📊 Results Tracking & CSV Export

The system automatically generates a `combined_results.csv` file with comprehensive metrics:

### CSV Columns
- **Approach/Framework**: "OpenAI Direct" or "CrewAI Agent"
- **Dataset/Benchmark**: "HumanEval"
- **pass@1 to pass@10**: All pass@k metrics
- **Time (sec)**: Total execution time
- **Input Tokens**: Number of input tokens used
- **Output Tokens**: Number of output tokens generated
- **Total Tokens**: Sum of input + output tokens
- **Estimated Cost ($)**: Cost based on OpenAI pricing
- **Timestamp**: When the run was executed
- **Model**: Model used (e.g., "gpt-4o")
- **Tasks**: Number of tasks evaluated
- **Samples per Task**: Number of samples per task

### 💰 Cost Tracking

The system automatically calculates estimated costs based on current OpenAI pricing:
- **GPT-4o**: $0.005/1K input tokens, $0.015/1K output tokens
- **GPT-4o-mini**: $0.00015/1K input tokens, $0.0006/1K output tokens
- **GPT-4**: $0.03/1K input tokens, $0.06/1K output tokens
- **GPT-3.5-turbo**: $0.001/1K input tokens, $0.002/1K output tokens

### Example CSV Output
```csv
Approach/Framework,Dataset/Benchmark,pass@1,pass@2,...,pass@10,Time (sec),Input Tokens,Output Tokens,Total Tokens,Estimated Cost ($),Timestamp,Model,Tasks,Samples per Task
OpenAI Direct,HumanEval,1.0,1.0,...,1.0,2.0,278,28,306,0.0018,2025-10-19 09:38:47,gpt-4o,1,2
CrewAI Agent,HumanEval,1.0,1.0,...,1.0,4.2,80,127,207,0.0023,2025-10-19 09:39:40,gpt-4o,1,2
```

## 📊 Understanding Results

After evaluation, the results file contains:

```json
{
  "task_id": "HumanEval/0",
  "completion": "    return sorted(numbers)[0]\n",
  "result": "passed",
  "passed": true
}
```

The evaluation prints metrics like:
```
{'pass@1': 0.85, 'pass@10': 0.95}
```

- **pass@1**: Probability that at least 1 sample passes
- **pass@10**: Probability that at least 1 of 10 samples passes

## 🔧 Evaluation Details

The evaluation command does two things:
1. Runs the HumanEval evaluator
2. Moves results to the `Results/` folder

You can also split these steps if needed:
```bash
# Step 1: Evaluate
python human-eval/human_eval/evaluate_functional_correctness.py \
  "outputs/Generated Samples/gpt-4o_20250117_143022.jsonl" \
  --problem_file "outputs/Custom Problems/gpt-4o_20250117_143022.jsonl"

# Step 2: Move results
mv "outputs/Generated Samples/gpt-4o_20250117_143022.jsonl_results.jsonl" \
   "outputs/Results/gpt-4o_20250117_143022_results.jsonl"
```

## 🧪 How It Works

### Architecture Overview

```
inference.py
    ↓
[USE_CREWAI check]
    ↓
    ├─→ openAI_models.py (Direct)
    │   └─→ OpenAI API → Result
    │
    └─→ crewai_agent.py (Agent)
        └─→ CrewAI Agent → Task → OpenAI API → Result
    ↓
sanitize.py → Clean Result
```

### 1. Code Generation (`inference.py`)
- Loads HumanEval problems
- Checks `USE_CREWAI` setting
- Routes to appropriate generator (direct or agent)
- For each task, sends function signature + docstring
- Requests only the function body (not the signature)
- Sanitizes output to ensure proper formatting

### 2. Sanitization (`sanitize.py`)
- Strips markdown code fences
- Removes function signatures if included
- Ensures proper 4-space indentation
- Adds trailing newline

### 3. Evaluation
- Combines prompt + completion into full function
- Runs test cases in isolated environment
- Reports pass/fail for each sample
- Calculates pass@k metrics

## 📝 Example Workflow

```bash
# 1. Generate samples for 5 tasks with 10 samples each
echo "TASK_LIMIT=5" > .env
echo "NUM_SAMPLES_PER_TASK=10" >> .env
python inference.py

# Output:
# ============================================================
# Starting HumanEval Inference
# ============================================================
# Model: gpt-4o
# Samples per task: 10
# ...
# Progress: 100%|████████████████| 50/50 [00:30<00:00, 1.67completion/s]
# ✓ Saved 50 completions to: outputs/Generated Samples/gpt-4o_20250117_143022.jsonl

# 2. Evaluate (copy the command printed above)
python human-eval/human_eval/evaluate_functional_correctness.py "outputs/Generated Samples/gpt-4o_20250117_143022.jsonl" --problem_file "outputs/Custom Problems/gpt-4o_20250117_143022.jsonl" && mv "outputs/Generated Samples/gpt-4o_20250117_143022.jsonl_results.jsonl" "outputs/Results/gpt-4o_20250117_143022_results.jsonl"

# Output:
# Running test suites...
# {'pass@1': 0.80, 'pass@10': 0.95}
# Results automatically saved to: outputs/Results/gpt-4o_20250117_143022_results.jsonl
```

## 🐛 Troubleshooting

### Import Error: `human_eval.data` not found
- Ensure `human-eval` is installed: `pip install -e human-eval/`
- Check your IDE's Python interpreter points to `afhe_env/bin/python`

### Import Error: `crewai` not found
- Install CrewAI: `pip install crewai langchain-openai`
- Or set `USE_CREWAI=false` to use direct API mode

### All Completions Fail
- Check if completions are properly indented (should have 4 spaces)
- Verify the model is returning function bodies, not full functions
- Review sanitization logic in `sanitize.py`

### CrewAI Errors
- Ensure both `crewai` and `langchain-openai` are installed
- Check OPENAI_API_KEY is set correctly
- Try `CREWAI_VERBOSE=true` to see detailed error messages
- Fallback: Set `USE_CREWAI=false`

### API Key Issues
- Ensure `.env` file exists with `OPENAI_API_KEY`
- Check API key has sufficient credits
- CrewAI uses more tokens - ensure adequate quota

## 🚀 Using Qwen-Agent Framework

### What is Qwen-Agent Mode?

Qwen-Agent mode uses the **Qwen-Agent framework** with OpenAI models to provide:
- 🛠️ **Function Calling**: Built-in tool calling capabilities
- 🎯 **Agent Architecture**: Qwen-Agent's agent-based approach
- 🔧 **OpenAI Integration**: Uses your existing OpenAI API key and models
- 📊 **Framework Comparison**: Compare different agent frameworks

### Setup Qwen-Agent

**Install dependencies:**
```bash
pip install qwen-agent>=0.0.26
```

**Configure environment:**
```bash
# .env
USE_QWEN_AGENT=true
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o
```

### Run with Qwen-Agent

```bash
USE_QWEN_AGENT=true TASK_LIMIT=5 python inference.py
```

You'll see:
```
🚀 Using Qwen-Agent framework with OpenAI model for code generation
============================================================
Starting HumanEval Inference
============================================================
```

### Framework Comparison

| Feature | Direct API | CrewAI Agent | Qwen-Agent |
|---------|-----------|--------------|------------|
| **Speed** | ⚡ Fast (1-2s/task) | 🐢 Slower (2-4s/task) | 🚀 Medium (1.5-3s/task) |
| **Tokens** | 💰 Efficient | 💰💰 More overhead | 💰💰 Medium overhead |
| **Architecture** | Simple API calls | Agent reasoning | Function calling |
| **Use Case** | Benchmarking | Research/Analysis | Framework comparison |

### When to Use Qwen-Agent?

**Use Qwen-Agent when:**
- 🔬 Comparing agent frameworks
- 🛠️ Testing function calling capabilities
- 📊 Researching different agent architectures
- 🎓 Learning about Qwen-Agent framework

## 📚 References

- [HumanEval Paper](https://arxiv.org/abs/2107.03374)
- [HumanEval GitHub](https://github.com/openai/human-eval)
- [OpenAI API Documentation](https://platform.openai.com/docs)

## 📄 License

This project uses the HumanEval benchmark which is MIT licensed.

