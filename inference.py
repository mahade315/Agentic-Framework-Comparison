# inference.py
import os
import time
import subprocess
from datetime import datetime
from dotenv import load_dotenv
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm
from results_tracker import ResultsTracker

# Load environment first
load_dotenv()

# Conditional import based on agent framework setting
USE_CREWAI = os.getenv("USE_CREWAI", "false").lower() == "true"
USE_QWEN_AGENT = os.getenv("USE_QWEN_AGENT", "false").lower() == "true"
USE_LANGCHAIN = os.getenv("USE_LANGCHAIN", "false").lower() == "true"
USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "false").lower() == "true"

if USE_LANGGRAPH:
    print("🤖 Using LangGraph framework with OpenAI model for code generation")
    from scripts.langgraph_agent import generate_one_completion
elif USE_LANGCHAIN:
    print("🤖 Using LangChain Agent Executor with OpenAI model for code generation")
    from scripts.langchain_agent import generate_one_completion
elif USE_QWEN_AGENT:
    print("🤖 Using Qwen-Agent framework with OpenAI model for code generation")
    from scripts.qwen_agent import generate_one_completion
elif USE_CREWAI:
    print("🤖 Using CrewAI Agent for code generation")
    from scripts.crewai_agent import generate_one_completion
else:
    print("🔧 Using OpenAI API directly for code generation")
    from scripts.openAI_models import generate_one_completion

def _select_task_ids(all_task_ids):
    """
    Subset controls via .env:
      - TASK_LIMIT: take the first N tasks (int)
      - TASK_IDS: comma-separated list of exact task ids to run (wins over TASK_LIMIT)
      - SHUFFLE_TASKS: 'true'/'false' (defaults false)
    """
    import random

    task_ids = list(all_task_ids)

    # optional shuffle
    if os.getenv("SHUFFLE_TASKS", "false").lower() == "true":
        random.seed(os.getenv("TASK_SHUFFLE_SEED"))  # optional
        random.shuffle(task_ids)

    explicit = os.getenv("TASK_IDS")
    if explicit:
        wanted = {tid.strip() for tid in explicit.split(",") if tid.strip()}
        return [tid for tid in task_ids if tid in wanted]

    limit = os.getenv("TASK_LIMIT")
    if limit and limit.isdigit():
        return task_ids[: int(limit)]

    return task_ids

def main():
    # Environment already loaded at module level
    
    # Start timing
    start_time = time.time()

    # Get configuration
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    try:
        num_samples_per_task = int(os.getenv("NUM_SAMPLES_PER_TASK", "10"))
    except ValueError:
        num_samples_per_task = 10

    # Create folder structure
    output_base = "outputs"
    samples_dir = os.path.join(output_base, "Generated Samples")
    problems_dir = os.path.join(output_base, "Custom Problems")
    results_dir = os.path.join(output_base, "Results")
    for directory in [samples_dir, problems_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add agent type to filename for easy differentiation
    if USE_LANGGRAPH:
        agent_type = "langgraph"
    elif USE_LANGCHAIN:
        agent_type = "langchain"
    elif USE_QWEN_AGENT:
        agent_type = "qwen_agent"
    elif USE_CREWAI:
        agent_type = "crewai"
    else:
        agent_type = "direct"
    base_filename = f"{agent_type}_{model_name.replace('/', '_')}_{timestamp}"

    print(f"\n{'='*60}")
    print(f"Starting HumanEval Inference")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Samples per task: {num_samples_per_task}")
    print(f"Timestamp: {timestamp}")
    print(f"{'='*60}\n")

    # Load problems and select tasks
    print("Loading HumanEval problems...")
    problems = read_problems()
    task_ids = _select_task_ids(problems.keys())
    print(f"✓ Selected {len(task_ids)} tasks from {len(problems)} total problems\n")

    # Reset token usage before generation
    if USE_QWEN_AGENT:
        from scripts.qwen_agent import reset_token_usage
        reset_token_usage()
    elif USE_CREWAI:
        from scripts.crewai_agent import reset_token_usage
        reset_token_usage()
    else:
        from scripts.openAI_models import reset_token_usage
        reset_token_usage()

    # Generate completions with progress bar
    print("Generating completions...")
    samples = []
    total_generations = len(task_ids) * num_samples_per_task
    
    with tqdm(total=total_generations, desc="Progress", unit="completion") as pbar:
        for task_id in task_ids:
            prompt = problems[task_id]["prompt"]
            for sample_num in range(num_samples_per_task):
                pbar.set_postfix({"task": task_id, "sample": f"{sample_num+1}/{num_samples_per_task}"})
                completion = generate_one_completion(prompt)
                samples.append({
                    "task_id": task_id,
                    "completion": completion,
                })
                pbar.update(1)

    # Write samples
    samples_path = os.path.join(samples_dir, f"{base_filename}.jsonl")
    write_jsonl(samples_path, samples)
    print(f"\n✓ Saved {len(samples)} completions to:")
    print(f"  {samples_path}")

    # Create custom problems file for evaluation
    custom_problems = [
        {"task_id": task_id, **problems[task_id]}
        for task_id in task_ids
    ]
    custom_problems_path = os.path.join(problems_dir, f"{base_filename}.jsonl")
    write_jsonl(custom_problems_path, custom_problems)
    print(f"\n✓ Saved {len(custom_problems)} problems to:")
    print(f"  {custom_problems_path}")

    # Run evaluation automatically
    print(f"\n{'='*60}")
    print("Running Evaluation...")
    print(f"{'='*60}")
    
    eval_start_time = time.time()
    results_path = os.path.join(results_dir, f"{base_filename}_results.jsonl")
    
    try:
        # Run evaluation
        eval_cmd = [
            "python", "human-eval/human_eval/evaluate_functional_correctness.py",
            samples_path, "--problem_file", custom_problems_path
        ]
        
        result = subprocess.run(eval_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Move results to proper location
            temp_results = f"{samples_path}_results.jsonl"
            if os.path.exists(temp_results):
                os.rename(temp_results, results_path)
                print(f"✓ Evaluation completed successfully!")
                print(f"✓ Results saved to: {results_path}")
                
                # Calculate total execution time
                total_time = time.time() - start_time
                eval_time = time.time() - eval_start_time
                
                # Get token usage
                if USE_LANGGRAPH:
                    from scripts.langgraph_agent import get_token_usage
                elif USE_LANGCHAIN:
                    from scripts.langchain_agent import get_token_usage
                elif USE_QWEN_AGENT:
                    from scripts.qwen_agent import get_token_usage
                elif USE_CREWAI:
                    from scripts.crewai_agent import get_token_usage
                else:
                    from scripts.openAI_models import get_token_usage
                
                token_stats = get_token_usage()
                
                # Save to combined results CSV
                tracker = ResultsTracker()
                if USE_LANGGRAPH:
                    approach_name = "LangGraph Agent"
                elif USE_LANGCHAIN:
                    approach_name = "LangChain Agent"
                elif USE_QWEN_AGENT:
                    approach_name = "Qwen-Agent"
                elif USE_CREWAI:
                    approach_name = "CrewAI Agent"
                else:
                    approach_name = "OpenAI Direct"
                tracker.add_result(
                    approach=approach_name,
                    results_file=results_path,
                    execution_time=total_time,
                    model=model_name,
                    num_tasks=len(task_ids),
                    samples_per_task=num_samples_per_task,
                    input_tokens=token_stats["input_tokens"],
                    output_tokens=token_stats["output_tokens"]
                )
                
            else:
                print("❌ Results file not found after evaluation")
        else:
            print(f"❌ Evaluation failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Evaluation error: {e}")

    # Get final token stats for display
    if USE_LANGGRAPH:
        from scripts.langgraph_agent import get_token_usage
    elif USE_LANGCHAIN:
        from scripts.langchain_agent import get_token_usage
    elif USE_QWEN_AGENT:
        from scripts.qwen_agent import get_token_usage
    elif USE_CREWAI:
        from scripts.crewai_agent import get_token_usage
    else:
        from scripts.openAI_models import get_token_usage
    
    final_token_stats = get_token_usage()
    
    print(f"\n{'='*60}")
    print("Process Complete!")
    print(f"{'='*60}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Tokens used: {final_token_stats['input_tokens']:,} input + {final_token_stats['output_tokens']:,} output = {final_token_stats['total_tokens']:,} total")
    print(f"Results saved to: {results_path}")
    print(f"Combined results: combined_results.csv")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()