"""
Main script for running GDPR translation evaluation.
"""

import pandas as pd
import glob
from pathlib import Path
from prompt_eval.evaluation import EvaluationPipeline


def main():
    """
    Main function demonstrating pipeline usage.
    """
    project_root = Path(__file__).parent

    # Configuration
    config = {
        "model": "gemma3:27B",
        "temperature": 0.2,
        "prompt_path": project_root / "full_prompt.json",
        "sample_size": 10
    }
    
    # Initialize pipeline with configuration
    pipeline = EvaluationPipeline(
        model=config["model"],
        temperature=config["temperature"],
        prompt_path=config["prompt_path"],
        sample_size=config["sample_size"]
    )
    
    # Get file paths
    gdpr_pattern = str(project_root / "GDPR" / "gdpr_*.txt")
    file_paths = glob.glob(gdpr_pattern)
    print(f"Found {len(file_paths)} GDPR files")
    
    # Extract evaluation set
    eval_set = pipeline.extract_eval_set(file_paths)
    print(f"Extracted {len(eval_set)} evaluation items")

    # Run evaluation
    results = pipeline.run_evaluation(
        eval_set=eval_set,
        src_lan="it", 
        target_lans=["en", "fr"]  
    )
    
    # Save results
    df = pd.DataFrame(results)  
    df['model'] = pipeline.model
    df['temperature'] = pipeline.temperature

    output_path = project_root / 'evaluation_results_it.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"Results saved to: {output_path}")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()