import pandas as pd
import glob
from typing import List, Dict, Any
from EURlex.Eurlex_gdpr import GDPRProcessor
from prompt_eval.translator_processor import TranslationProcessor
from prompt_eval.automatic_metrics import AutomaticMetrics


class EvaluationPipeline:
    """
    Simple evaluation pipeline for GDPR document processing and translation evaluation.
    """
    
    def __init__(self, model: str, temperature: float, prompt_path: str, sample_size: int):
        """
        Initialize pipeline with model configuration.
        
        Args:
            model: Translation model to use
            temperature: Temperature for translation generation
            prompt_path: Path to JSON file containing prompts
            sample_size: Number of samples to process
        """
        self.model = model
        self.temperature = temperature
        self.sample_size = sample_size
        
        # Initialize processors once to avoid recreating them
        self.gdpr_processor = GDPRProcessor(sample_size=sample_size)
        self.translation_processor = TranslationProcessor(
            model=model, 
            temperature=temperature, 
            prompt_path=prompt_path
        )
        
        # Initialize metrics calculator
        self.metrics_calculator = AutomaticMetrics()

    def extract_eval_set(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Extract evaluation dataset from provided file paths.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of dictionaries with sentences in multiple languages
        """
        return self.gdpr_processor.extract_eval_set(file_paths)

    def run_evaluation(
        self, 
        eval_set: List[Dict[str, Any]], 
        src_lan: str, 
        target_lans: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Run evaluation pipeline on provided evaluation set.
        
        Args:
            eval_set: Evaluation dataset to process
            src_lan: Source language code
            target_lans: List of target language codes
            
        Returns:
            List containing evaluation metrics for each item
        """
        print(f"Running evaluation for {len(eval_set)} items")
        
        # Process evaluation set using pre-configured processor
        processed_data = self.translation_processor.process_evaluation_set(
            eval_set, 
            src_lan, 
            target_lans,
            apply_key_substitution=True
        )
        
        print(f"Translation processing completed. Processed {len(processed_data)} items")
        
        # Calculate and return metrics
        metrics_results = self.metrics_calculator.compute_metrics(processed_data)
        
        print(f"Metrics calculation completed. Results for {len(metrics_results)} items")
        
        return metrics_results