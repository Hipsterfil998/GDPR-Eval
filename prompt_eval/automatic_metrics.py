import pandas as pd
from typing import List, Dict, Any, Optional
from comet import download_model, load_from_checkpoint
import os
from dotenv import load_dotenv
from huggingface_hub import login


class CometEvaluator:
    """
    Simple COMET model evaluator for translation quality assessment.
    """
    
    def __init__(self,
                 model_name: str = "Unbabel/XCOMET-XL",
                 batch_size: int = 8,
                 gpus: int = 1):
        """
        Initialize the CometEvaluator.
        
        Args:
            model_name (str): COMET model name to download
            batch_size (int): Batch size for predictions  
            gpus (int): Number of GPUs to use
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.gpus = gpus
        self.model = None
        
        # Setup environment and model
        self._setup()
    
    def _setup(self) -> None:
        """
        Setup environment, login and load model.
        """
        try:
            # Load environment variables
            load_dotenv()
            
            # Login to HuggingFace
            token = os.getenv('HF_TOKEN')
            if token:
                login(token=token)
            
            # Download and load model
            model_path = download_model(self.model_name)
            self.model = load_from_checkpoint(model_path)
            
        except Exception as e:
            print(f"Error during setup: {e}")
            raise
    
    def predict(self, data: List[Dict[str, Any]]) -> Any:
        """
        Predict quality scores for translation data.
        
        Args:
            data (List[Dict[str, Any]]): Translation data in format:
                [{"src": "source text", "mt": "machine translation", "ref": "reference"}]
        
        Returns:
            Model output with scores and metadata
        """
        if self.model is None:
            raise ValueError("Model not loaded. Check initialization.")
        
        return self.model.predict(data, batch_size=self.batch_size, gpus=self.gpus)
    
    def get_scores(self, data: List[Dict[str, Any]]) -> List[float]:
        """
        Get segment-level scores for translation data.
        
        Args:
            data (List[Dict[str, Any]]): Translation data in format:
                [{"src": "source text", "mt": "machine translation", "ref": "reference"}]
        
        Returns:
            List[float]: Segment-level scores
        """
        model_output = self.predict(data)
        return model_output.scores


class AutomaticMetrics:
    """
    A class for computing automatic translation quality metrics.
    
    This class evaluates translation quality by analyzing:
    - Symbol presence consistency between source and translation (optional)
    - Punctuation consistency (optional)
    - Length ratio between source and translation
    - COMET scores (optional)
    """
    
    def __init__(self, 
                 output_file: Optional[str] = None,
                 symbol_delimiter: Optional[str] = None,
                 punctuation: Optional[list[str]] = ['.', ','],
                 comet_model: Optional[str] = "Unbabel/XCOMET-XL"):
        """
        Initialize the AutomaticMetrics class.
        
        Args:
            output_file (Optional[str]): Path for the output CSV file
            symbol_delimiter (Optional[str]): Delimiter for symbol consistency check.
                                            If None, symbol consistency is not calculated.
            punctuation (Optional[list[str]]): List of punctuation marks to check consistency.
                                             If None, punctuation consistency is not calculated.
            comet_model (Optional[str]): COMET model name for quality scoring.
                                       If None, COMET scores are not calculated.
        """
        self.output_file = output_file
        self.symbol_delimiter = symbol_delimiter
        self.punctuation = punctuation
        
        # Initialize COMET evaluator if model specified
        self.comet_evaluator = None
        if comet_model:
            try:
                self.comet_evaluator = CometEvaluator(model_name=comet_model)
                print(f"COMET evaluator initialized with model: {comet_model}")
            except Exception as e:
                print(f"Failed to initialize COMET evaluator: {e}")
                self.comet_evaluator = None
    
    def _calculate_symbol_consistency(self, source: str, translation: str) -> Optional[int]:
        """
        Calculate symbol consistency between source and translation.
        
        Args:
            source (str): Source text
            translation (str): Translation text
            
        Returns:
            Optional[int]: 1 if symbol count matches, 0 otherwise, None if no delimiter set
        """
        if self.symbol_delimiter is None:
            return None
        
        source_symbols = len(source.split(self.symbol_delimiter))
        translation_symbols = len(translation.split(self.symbol_delimiter))
        return 1 if source_symbols == translation_symbols else 0
    
    def _calculate_punct_consistency(self, source: str, translation: str) -> Optional[int]:
        """
        Calculate punctuation consistency between source and translation.
        
        Args:
            source (str): Source text
            translation (str): Translation text
            
        Returns:
            Optional[int]: 1 if punctuation count matches, 0 otherwise, None if no punctuation marks provided
        """
        if self.punctuation is None:
            return None
        
        # Count punctuation marks in source text
        source_count = sum(source.count(mark) for mark in self.punctuation)
        
        # Count punctuation marks in translation text
        translation_count = sum(translation.count(mark) for mark in self.punctuation)
        
        return 1 if source_count == translation_count else 0
    
    def _calculate_length_ratio(self, source: str, translation: str) -> float:
        """
        Calculate the length ratio between source and translation.
        
        Args:
            source (str): Source text
            translation (str): Translation text
            
        Returns:
            float: Ratio of source length to translation length
        """
        translation_len = len(translation)
        return len(source) / translation_len if translation_len > 0 else float('inf')
    
    def process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process translation data and calculate automatic metrics.
        
        Args:
            data (List[Dict[str, Any]]): List of translation pairs with 'src' and 'mt' keys
            
        Returns:
            List[Dict[str, Any]]: Processed data with added metrics
            
        Raises:
            ValueError: If data format is invalid
        """
        if not data:
            raise ValueError("Input data cannot be empty")
        
        # Calculate COMET scores if evaluator is available
        comet_scores = None
        if self.comet_evaluator:
            try:
                print("Calculating COMET scores...")
                comet_scores = self.comet_evaluator.get_scores(data)
                print(f"COMET scores calculated for {len(comet_scores)} samples")
            except Exception as e:
                print(f"Error calculating COMET scores: {e}")
                comet_scores = None
        
        metrics = []
        
        for i, item in enumerate(data):
            # Validate required keys
            if 'src' not in item or 'mt' not in item:
                raise ValueError(f"Item at index {i} missing required keys 'src' or 'mt'")
            
            # Create a copy to avoid modifying original data
            processed_item = item.copy()
            
            source = str(item['src'])
            translation = str(item['mt'])
            
            # Calculate symbol consistency only if delimiter is set
            symbol_consistency = self._calculate_symbol_consistency(source, translation)
            if symbol_consistency is not None:
                processed_item['symbols'] = symbol_consistency
            
            # Calculate punctuation consistency only if punctuation marks are set
            punct_consistency = self._calculate_punct_consistency(source, translation)
            if punct_consistency is not None:
                processed_item['punct_marks'] = punct_consistency
            
            # Always calculate length ratio
            processed_item['len_ratio'] = self._calculate_length_ratio(source, translation)
            
            # Add COMET score if available
            if comet_scores and i < len(comet_scores):
                processed_item['comet_score'] = comet_scores[i]
            
            metrics.append(processed_item)
        
        return metrics
    
    def compute_metrics(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main method to compute metrics and optionally save results.
        
        Args:
            data (List[Dict[str, Any]]): Input translation data
            
        Returns:
            List[Dict[str, Any]]: Processed data with metrics
        """
        processed_data = self.process_data(data)
        
        return processed_data