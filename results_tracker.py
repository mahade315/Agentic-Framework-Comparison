# results_tracker.py
"""
Results tracking and CSV export for HumanEval evaluation.
"""

import os
import csv
import time
from datetime import datetime
from typing import Dict, List, Optional

class ResultsTracker:
    """Track and export evaluation results to CSV."""
    
    def __init__(self, csv_file: str = "combined_results.csv"):
        self.csv_file = csv_file
        self.ensure_csv_exists()
    
    def ensure_csv_exists(self):
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_file):
            headers = [
                "Approach/Framework",
                "Dataset/Benchmark", 
                "pass@1", "pass@2", "pass@3", "pass@4", "pass@5",
                "pass@6", "pass@7", "pass@8", "pass@9", "pass@10",
                "Time (sec)",
                "Input Tokens",
                "Output Tokens", 
                "Total Tokens",
                "Estimated Cost ($)",
                "Timestamp",
                "Model",
                "Tasks",
                "Samples per Task"
            ]
            
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                f.write('\n')  # Ensure file ends with newline
    
    def calculate_pass_at_k(self, results_file: str, k_values: List[int] = list(range(1, 11))) -> Dict[int, float]:
        """Calculate pass@k metrics from results file."""
        import json
        
        # Group results by task_id
        task_results = {}
        
        with open(results_file, 'r') as f:
            for line in f:
                result = json.loads(line.strip())
                task_id = result['task_id']
                passed = result['passed']
                
                if task_id not in task_results:
                    task_results[task_id] = []
                task_results[task_id].append(passed)
        
        # Find the maximum number of samples per task
        max_samples = max(len(results) for results in task_results.values()) if task_results else 0
        
        # Calculate pass@k for each k (only if k <= max_samples)
        pass_at_k = {}
        
        for k in k_values:
            if k > max_samples:
                # Not enough samples to calculate pass@k
                pass_at_k[k] = "N/A"
            else:
                total_tasks = len(task_results)
                passed_tasks = 0
                
                for task_id, results in task_results.items():
                    # For each task, check if at least one of the first k samples passed
                    samples_to_check = results[:k]
                    if any(samples_to_check):
                        passed_tasks += 1
                
                pass_at_k[k] = passed_tasks / total_tasks if total_tasks > 0 else 0.0
        
        return pass_at_k
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost based on OpenAI pricing."""
        # OpenAI GPT-4o pricing (as of 2024)
        pricing = {
            "gpt-4o": {"input": 0.005, "output": 0.015},  # per 1K tokens
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        }
        
        model_pricing = pricing.get(model, pricing["gpt-4o"])
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        return input_cost + output_cost

    def add_result(self, 
                   approach: str,
                   results_file: str,
                   execution_time: float,
                   model: str = "gpt-4o",
                   num_tasks: int = 0,
                   samples_per_task: int = 0,
                   input_tokens: int = 0,
                   output_tokens: int = 0):
        """Add a new result to the CSV file."""
        
        # Calculate pass@k metrics
        pass_at_k = self.calculate_pass_at_k(results_file)
        
        # Calculate cost
        total_tokens = input_tokens + output_tokens
        estimated_cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        # Prepare row data
        row = [
            approach,  # Approach/Framework
            "HumanEval",  # Dataset/Benchmark
            pass_at_k[1], pass_at_k[2], pass_at_k[3], pass_at_k[4], pass_at_k[5],
            pass_at_k[6], pass_at_k[7], pass_at_k[8], pass_at_k[9], pass_at_k[10],
            execution_time,  # Time (sec)
            input_tokens,  # Input Tokens
            output_tokens,  # Output Tokens
            total_tokens,  # Total Tokens
            round(estimated_cost, 4),  # Estimated Cost ($)
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp
            model,  # Model
            num_tasks,  # Tasks
            samples_per_task  # Samples per Task
        ]
        
        # Append to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        print(f"\n📊 Results saved to {self.csv_file}")
        print(f"   Approach: {approach}")
        print(f"   Pass@1: {pass_at_k[1] if pass_at_k[1] != 'N/A' else 'N/A'}")
        print(f"   Pass@10: {pass_at_k[10] if pass_at_k[10] != 'N/A' else 'N/A'}")
        print(f"   Time: {execution_time:.1f}s")
        print(f"   Tokens: {input_tokens:,} input + {output_tokens:,} output = {total_tokens:,} total")
        print(f"   Cost: ${estimated_cost:.4f}")
    
    def get_latest_results(self, approach: str) -> Optional[Dict]:
        """Get the latest results for a specific approach."""
        import pandas as pd
        
        if not os.path.exists(self.csv_file):
            return None
        
        try:
            df = pd.read_csv(self.csv_file)
            approach_results = df[df['Approach/Framework'] == approach]
            
            if approach_results.empty:
                return None
            
            # Get the latest (last) result
            latest = approach_results.iloc[-1]
            return latest.to_dict()
        except ImportError:
            # Fallback without pandas
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            approach_rows = [row for row in rows if row['Approach/Framework'] == approach]
            return approach_rows[-1] if approach_rows else None
