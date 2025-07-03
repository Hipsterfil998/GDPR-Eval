import re
import random
from typing import List, Dict, Any, Optional, Union


class GDPRProcessor:
    """
    GDPR text processor for parsing and formatting legal documents.
    Handles article extraction, text processing, and evaluation set creation.
    """
    
    def __init__(self, sample_size: int = 50):
        """
        Initialize the GDPRProcessor with language patterns and configuration.
        
        Args:
            sample_size (int): Default sample size for evaluation sets (default: 50)
        """
        self.sample_size = sample_size
        
        # Language patterns for article detection
        self.language_patterns = {
            'bg': r'(\nЧлен\s\d+\n)',          # Bulgarian - "Член" (Article)
            'cs': r'(\nČlánek\s\d+\n)',        # Czech - "Článek" (Article)
            'da': r'(\nArtikel\s\d+\n)',       # Danish - "Artikel" (Article)
            'de': r'(\nArtikel\s\d+\n)',       # German - "Artikel" (Article)
            'el': r'(\nΆρθρο\s\d+\n)',         # Greek - "Άρθρο" (Article)
            'en': r'(\nArticle\s\d+\n)',       # English - "Article"
            'es': r'(\nArtículo\s\d+\n)',      # Spanish - "Artículo" (Article)
            'et': r'(\nArtikkel\s\d+\n)',      # Estonian - "Artikkel" (Article)
            'fi': r'(\nArtikla\s\d+\n)',       # Finnish - "Artikla" (Article)
            'fr': r'(\nArticle\s\d+\n)',       # French - "Article"
            'ga': r'(\nAirteagal\s\d+\n)',     # Irish - "Airteagal" (Article)
            'hr': r'(\nČlanak\s\d+\n)',        # Croatian - "Članak" (Article)
            'hu': r'(\nCikk\s\d+\n)',          # Hungarian - "Cikk" (Article)
            'it': r'(\nArticolo\s\d+\n)',      # Italian - "Articolo" (Article)
            'lt': r'(\nStraipsnis\s\d+\n)',    # Lithuanian - "Straipsnis" (Article)
            'lv': r'(\nPants\s\d+\n)',         # Latvian - "Pants" (Article)
            'mt': r'(\nArtikolu\s\d+\n)',      # Maltese - "Artikolu" (Article)
            'nl': r'(\nArtikel\s\d+\n)',       # Dutch - "Artikel" (Article)
            'pl': r'(\nArtykuł\s\d+\n)',       # Polish - "Artykuł" (Article)
            'pt': r'(\nArtigo\s\d+\n)',        # Portuguese - "Artigo" (Article)
            'ro': r'(\nArticolul\s\d+\n)',     # Romanian - "Articolul" (Article)
            'sk': r'(\nČlánok\s\d+\n)',        # Slovak - "Článok" (Article)
            'sl': r'(\nČlen\s\d+\n)',          # Slovenian - "Člen" (Article)
            'sv': r'(\nArtikel\s\d+\n)',       # Swedish - "Artikel" (Article)
        }
        
    def read_text_file(self, filename: str) -> str:
        """
        Read entire text file at once.
        """
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    
    def split_articles(self, language: str, text: str) -> List[str]:
        """
        Split the text every time it finds 'Article' followed by a number.
        """
        if language not in self.language_patterns:
            raise ValueError(f"Unsupported language: {language}")
        
        pattern = self.language_patterns[language]
        
        # Split the text using the pattern
        parts = re.split(pattern, text)
        
        # Return every second element starting from index 1 (skip headers)
        return parts[::2][1:]
    
    def extract_numbered_sentences(self, article_text: str) -> List[str]:
        """
        Extract sentences that start with numbers (e.g., '1)', '2.') or letters (e.g., 'a)', 'b.').
        """
        # Pattern to match sentences starting with:
        # - Numbers followed by ) or . (e.g., "1)", "2.", "10)")
        # - Letters followed by ) or . (e.g., "a)", "b.", "z)")
        pattern = r'^(?:\d+[.)]\s*|\b[a-zA-Z][.)]\s*)(.+?)(?=\n(?:\d+[.)]\s*|\b[a-zA-Z][.)]\s*)|$)'
        
        # Find all matches in the text
        matches = re.findall(pattern, article_text, re.MULTILINE | re.DOTALL)
        
        # Clean and filter results
        sentences = []
        for match in matches:
            # Remove extra whitespace and newlines
            cleaned = re.sub(r'\s+', ' ', match.strip())
            if cleaned:  # Only add non-empty sentences
                sentences.append(cleaned)
        
        return sentences
    
    def _normalize_file_paths(self, file_paths: Union[Dict[str, str], List[str]]) -> Dict[str, str]:
        """
        Normalize file paths input to dictionary format.
        """
        if isinstance(file_paths, dict):
            return file_paths
        elif isinstance(file_paths, list):
            # Auto-detect language from file path or use generic naming
            normalized = {}
            available_patterns = list(self.language_patterns.keys())
            
            for i, path in enumerate(file_paths):
                # Try to detect language from filename
                detected_lang = None
                for lang in available_patterns:
                    if lang in path.lower():
                        detected_lang = lang
                        break
                
                # Use detected language or assign generic language identifier
                if detected_lang:
                    normalized[detected_lang] = path
                else:
                    # Generate generic language identifier based on index
                    generic_lang = f'lang_{i}'
                    normalized[generic_lang] = path
                    
            return normalized
        else:
            raise ValueError(f"Unsupported file_paths type: {type(file_paths)}. Expected dict or list.")
    
    def process_gdpr_files(self, 
                          file_paths: Union[Dict[str, str], List[str]]) -> List[Dict[str, Any]]:
        """
        Process multiple GDPR files and create formatted dataset.
        """
        # Normalize input to dictionary format
        normalized_paths = self._normalize_file_paths(file_paths)
        
        # Read all text files
        texts = {}
        for lang, path in normalized_paths.items():
            texts[lang] = self.read_text_file(path)
        
        # Split articles for each language
        articles = {}
        for lang, text in texts.items():
            articles[lang] = self.split_articles(lang, text)
        
        # Ensure all languages have same number of articles
        article_counts = [len(arts) for arts in articles.values()]
        if len(set(article_counts)) > 1:
            min_count = min(article_counts)
            print(f"Warning: Different article counts found. Using minimum: {min_count}")
            for lang in articles:
                articles[lang] = articles[lang][:min_count]
        
        # Create formatted dataset
        languages = list(normalized_paths.keys())
        zipped_articles = zip(*[articles[lang] for lang in languages])
        
        formatted = []
        for i, article_group in enumerate(zipped_articles):
            article_dict = {'article_number': i + 1}
            for j, lang in enumerate(languages):
                article_dict[lang] = article_group[j]
            formatted.append(article_dict)
        
        print(f"Processed {len(formatted)} articles")
        return formatted
    
    def process_articles_with_sentences(self, 
                                      file_paths: Union[Dict[str, str], List[str]]) -> List[Dict[str, Any]]:
        """
        Process GDPR files and extract numbered/lettered sentences from each article.
        """
        # First, process articles normally
        formatted_data = self.process_gdpr_files(file_paths)
        
        # Extract sentences for each article in each language
        enhanced_data = []
        
        for article_data in formatted_data:
            enhanced_article = {
                'article_number': article_data['article_number'],
                'sentences': {}  # Will contain sentences for each language
            }
            
            # Process each language in the article
            for lang_code, article_text in article_data.items():
                if lang_code != 'article_number':  # Skip the article number field
                    sentences = self.extract_numbered_sentences(article_text)
                    enhanced_article['sentences'][lang_code] = sentences
                    
                    # Also keep original text if needed
                    enhanced_article[f'{lang_code}_original'] = article_text
            
            enhanced_data.append(enhanced_article)
        
        # Print statistics
        total_sentences = sum(
            len(sentences) 
            for article in enhanced_data 
            for sentences in article['sentences'].values()
        )
        print(f"Extracted {total_sentences} sentences from {len(enhanced_data)} articles")
        
        return enhanced_data
    
    def get_all_sentences_flat(self, processed_data: List[Dict[str, Any]]) -> List[str]:
        """
        Get a flat list of all extracted sentences across all articles and languages.
        """
        all_sentences = []
        
        for article in processed_data:
            for lang_code, sentences in article['sentences'].items():
                all_sentences.extend(sentences)
        
        return all_sentences
    
    def create_sentence_evaluation_set(self, 
                                     processed_data: List[Dict[str, Any]],
                                     sample_size: Optional[int] = None,
                                     by_language: bool = False) -> Union[List[str], Dict[str, List[str]]]:
        """
        Create evaluation set from extracted sentences.
        """
        # Use provided sample_size or default from instance
        size = sample_size if sample_size is not None else self.sample_size
        
        if by_language:
            # Group sentences by language
            lang_sentences = {}
            for article in processed_data:
                for lang_code, sentences in article['sentences'].items():
                    if lang_code not in lang_sentences:
                        lang_sentences[lang_code] = []
                    lang_sentences[lang_code].extend(sentences)
            
            # Sample from each language
            sampled_by_lang = {}
            for lang_code, sentences in lang_sentences.items():
                sample_count = min(size, len(sentences))
                sampled_by_lang[lang_code] = random.sample(sentences, sample_count)
            
            lang_stats = {lang: len(sents) for lang, sents in sampled_by_lang.items()}
            print(f"Created evaluation set with samples per language: {lang_stats}")
            return sampled_by_lang
        else:
            # Get all sentences and sample randomly
            all_sentences = self.get_all_sentences_flat(processed_data)
            sample_count = min(size, len(all_sentences))
            sampled = random.sample(all_sentences, sample_count)
            
            print(f"Created evaluation set with {len(sampled)} sentences")
            return sampled
    
    def create_evaluation_set(self, 
                            sentences: List[str],
                            sample_size: Optional[int] = None,
                            special_chars: Optional[List[str]] = None) -> List[str]:
        """
        Create evaluation set from extracted sentences.
        """
        # Use provided sample_size or default from instance
        size = sample_size if sample_size is not None else self.sample_size
        
        # Random sample
        sample_count = min(size, len(sentences))
        eval_set = random.sample(sentences, sample_count)
        
        # Process special characters if provided
        if special_chars is not None:
            pattern = r'\n\n'
            
            # Determine replacement string
            if len(special_chars) == 1:
                replacement = special_chars[0]
            elif len(special_chars) == 2:
                replacement = special_chars[0] + special_chars[1]
            else:
                replacement = ''.join(special_chars)
            
            # Apply substitution
            eval_set = [re.sub(pattern, replacement, sentence) for sentence in eval_set]
        
        print(f"Created evaluation set with {len(eval_set)} sentences")
        return eval_set
    
    def extract_eval_set(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Extract evaluation dataset from provided file paths.
        Each entry contains the same sentence in different languages.
        """
        # Process articles with sentences (structured data)
        processed_data = self.process_articles_with_sentences(file_paths)
        
        # Extract aligned sentences across languages
        aligned_sentences = []
        
        for article in processed_data:
            # Get all language codes available for this article
            available_languages = list(article['sentences'].keys())
            
            if not available_languages:
                continue
                
            # Find the minimum number of sentences across all languages for this article
            min_sentences = min(len(article['sentences'][lang]) for lang in available_languages)
            
            # Create aligned sentence entries
            for i in range(min_sentences):
                sentence_entry = {}
                
                # Add the same sentence index from each language
                for lang in available_languages:
                    if i < len(article['sentences'][lang]):
                        sentence_entry[lang] = article['sentences'][lang][i]
                
                # Only add if we have sentences in multiple languages
                if len(sentence_entry) > 1:
                    aligned_sentences.append(sentence_entry)
        
        # Sample from aligned sentences using instance sample_size
        sample_count = min(self.sample_size, len(aligned_sentences))
        if aligned_sentences:
            sampled_sentences = random.sample(aligned_sentences, sample_count)
        else:
            sampled_sentences = []
        
        print(f"Extracted {len(sampled_sentences)} aligned sentence entries")
        return sampled_sentences
    
    def process_complete_pipeline(self, 
                                file_paths: Union[Dict[str, str], List[str]],
                                sample_size: Optional[int] = None,
                                special_chars: Optional[List[str]] = None,
                                by_language: bool = False) -> Union[List[str], Dict[str, List[str]]]:
        """
        Complete processing pipeline from GDPR files to evaluation set of sentences.
        """
        # Use provided sample_size or default from instance
        size = sample_size if sample_size is not None else self.sample_size
        
        # Step 1: Process GDPR files and extract sentences
        print("Step 1: Processing GDPR files and extracting sentences...")
        processed_data = self.process_articles_with_sentences(file_paths)
        
        # Step 2: Create evaluation set
        print("Step 2: Creating evaluation set...")
        eval_set = self.create_sentence_evaluation_set(
            processed_data, 
            size,
            by_language
        )
        
        # Step 3: Apply special character substitution if needed
        if special_chars is not None:
            pattern = r'\n\n'
            
            # Determine replacement string
            if len(special_chars) == 1:
                replacement = special_chars[0]
            elif len(special_chars) == 2:
                replacement = special_chars[0] + special_chars[1]
            else:
                replacement = ''.join(special_chars)
            
            # Apply substitution based on return type
            if by_language:
                for lang_code in eval_set:
                    eval_set[lang_code] = [re.sub(pattern, replacement, sentence) for sentence in eval_set[lang_code]]
            else:
                eval_set = [re.sub(pattern, replacement, sentence) for sentence in eval_set]
        
        print("Pipeline completed successfully!")
        return eval_set