# GDPR-Eval

A comprehensive evaluation framework for assessing machine translation quality on GDPR legal documents. GDPR-Eval processes multilingual GDPR texts from [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng), generates translations using local large language models, and evaluates translation quality using multiple automatic metrics.

<img src="Testa_di_Marianna/app_images/Schermata_marianna.png" alt="GDPR-Eval"/>

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Format](#output-format)
- [Dependencies](#dependencies)
- [Customization](#customization)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Features

- **Document Processing**: Automatic extraction and alignment of multilingual GDPR sentences
- **Translation Generation**: Integration with various LLM models (Gemma, GPT, etc.) via Ollama
- **Quality Assessment**: Multiple evaluation metrics including COMET, punctuation consistency, and length ratios
- **Batch Processing**: Efficient processing of large document collections
- **CSV Export**: Results exported in structured format for further analysis

## Project Structure

```
GDPREval/
├── EURlex/
│   └── Eurlex_gdpr.py          # GDPR document processing
├── prompt_eval/
│   ├── automatic_metrics.py    # Evaluation metrics calculation
│   ├── evaluation.py          # Main evaluation pipeline
│   └── translator_processor.py # Translation processing
├── GDPR/
│   └── gdpr_*.txt             # GDPR documents in multiple languages
├── main.py                    # Main execution script
├── full_prompt.json           # Translation prompts configuration
└── requirements.txt           # Project dependencies
```

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd GDPREval
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup environment variables**
Create a `.env` file in the project root:
```bash
HF_TOKEN=your_huggingface_token_here
```

5. **Install Ollama** (for LLM models)
Follow the installation guide at [ollama.ai](https://ollama.ai/)
```bash
# Pull required models:
ollama pull gemma3:27B
```

## Configuration

### Model Configuration
Edit the configuration in `main.py`:

```python
config = {
    "model": "gemma3:27B",           # Ollama model name
    "temperature": 0.2,              # Generation temperature
    "prompt_path": "full_prompt.json", # Prompt configuration file
    "sample_size": 10                # Number of samples to process
}
```

### Translation Prompts
Customize translation prompts in `full_prompt.json`:

```json
{
    "system_prompt": "You are a professional legal translator...",
    "user_prompt": "Translate the following text from {source_language} to {target_language}...",
    "examples": [...]
}
```

## Usage

### Basic Usage

```bash
python main.py
```

This will:
1. Process GDPR documents from the `GDPR/` directory
2. Extract aligned multilingual sentences
3. Generate translations using the configured LLM
4. Evaluate translation quality
5. Save results to `evaluation_results_it.csv`

### Custom Evaluation

```python
from prompt_eval.evaluation import EvaluationPipeline

# Initialize GDPREval pipeline
pipeline = EvaluationPipeline(
    model="your_model",
    temperature=0.2,
    prompt_path="your_prompts.json",
    sample_size=50
)

# Process documents
eval_set = pipeline.extract_eval_set(file_paths)
results = pipeline.run_evaluation(
    eval_set=eval_set,
    src_lan="it",  # Source language
    target_lans=["en", "fr"]  # Target languages
)
```

## Evaluation Metrics

The pipeline calculates several automatic metrics:

### COMET Score
- **Range**: 0-1 (higher is better)
- **Description**: Neural metric trained on human judgments ([Unbabel/COMET](https://github.com/Unbabel/COMET))
- **Usage**: Primary quality indicator

### Punctuation Consistency
- **Range**: 0-1 (1 = perfect consistency)
- **Description**: Checks if punctuation marks are preserved
- **Configurable**: Customize punctuation marks in `AutomaticMetrics`

### Length Ratio
- **Range**: 0-∞ (1.0 = same length)
- **Description**: Ratio of source to target text length
- **Usage**: Detects over/under-translation

### Symbol Consistency (Optional)
- **Range**: 0-1 (1 = perfect consistency)
- **Description**: Custom delimiter-based consistency check
- **Usage**: Domain-specific validation

## Output Format

Results are saved as CSV with columns:

| Column | Description |
|--------|-------------|
| `src` | Source text |
| `mt` | Machine translation |
| `ref` | Reference translation |
| `comet_score` | COMET quality score |
| `punct_marks` | Punctuation consistency |
| `len_ratio` | Length ratio |
| `model` | Model used |
| `temperature` | Generation temperature |

## Dependencies

Core dependencies:
- [`pandas`](https://pandas.pydata.org/): Data manipulation
- [`transformers`](https://huggingface.co/docs/transformers): HuggingFace models
- [`comet-ml`](https://github.com/Unbabel/COMET): COMET evaluation metric
- [`torch`](https://pytorch.org/): PyTorch for model inference
- [`evaluate`](https://huggingface.co/docs/evaluate): HuggingFace evaluation metrics
- [`python-dotenv`](https://github.com/theskumar/python-dotenv): Environment variable management
- [`huggingface-hub`](https://huggingface.co/docs/huggingface_hub): Model downloading

## Customization

### Adding New Metrics

1. Extend `AutomaticMetrics` class:
```python
def _calculate_custom_metric(self, source: str, translation: str) -> float:
    # Your metric implementation
    return score
```

2. Add to `process_data` method:
```python
custom_score = self._calculate_custom_metric(source, translation)
processed_item['custom_metric'] = custom_score
```

### Supporting New Models

1. Update `TranslationProcessor` for new API endpoints
2. Modify prompt templates in `full_prompt.json`
3. Adjust model parameters in configuration

### Document Format Support

Extend `GDPRProcessor` to support additional document formats:
- PDF processing
- XML parsing
- Custom alignment strategies

## Performance Considerations

- **Batch Size**: Adjust COMET batch size for memory optimization
- **Sample Size**: Use smaller samples for quick testing
- **GPU Usage**: COMET evaluation benefits from GPU acceleration
- **Caching**: Models are cached automatically by HuggingFace

## Troubleshooting

### Common Issues

1. **COMET Model Loading**
   - Ensure HuggingFace token is set
   - Check internet connection
   - Verify sufficient disk space

2. **Ollama Connection**
   - Ensure [Ollama service](https://ollama.ai/) is running
   - Verify model is pulled: `ollama list`
   - Check model name spelling

3. **Memory Issues**
   - Reduce batch size in COMET evaluation
   - Decrease sample size
   - Use smaller models

### Error Messages

- `Model not found`: Pull model with `ollama pull model_name`
- `Token authentication failed`: Check `HF_TOKEN` in `.env`
- `CUDA out of memory`: Reduce batch size or use CPU

For more troubleshooting, see the [Issues](../../issues) section.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Add tests for new functionality
4. Submit a [pull request](../../pulls)

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@software{gdpreval,
  title={GDPREval: A Framework for Evaluating Machine Translation Quality on Legal Documents},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/gdpreval}
}
```

## Acknowledgments

- [COMET metric](https://github.com/Unbabel/COMET) by Unbabel
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) library
- [Ollama project](https://ollama.ai/) for local LLM inference
- [EUR-Lex](https://eur-lex.europa.eu/) for GDPR document provision
