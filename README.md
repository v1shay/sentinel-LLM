Post-Hoc Reliability Auditing for Large Language Models
Overview
Sentinel-LLM is a lightweight, model-agnostic NLP system for detecting hallucination risk and unsupported claims in large language model (LLM) outputs. The project focuses on post-generation auditing: evaluating LLM responses without access to internal model weights, making it suitable as a safety layer across diverse LLM deployments.
The system identifies linguistic and structural signals correlated with overconfidence, factual inconsistency, and missing evidence, enabling automated risk flagging in high-stakes or trust-sensitive applications.
Motivation
As LLMs are increasingly deployed in decision-making, education, and research settings, hallucinated or unsupported outputs pose a significant reliability risk. Sentinel-LLM explores whether surface-level language signals alone can be used to predict hallucination likelihood, offering a practical alternative to retrieval-heavy or model-specific approaches.
Approach
Sentinel-LLM uses a supervised NLP pipeline with the following components:
Text preprocessing: normalization, tokenization, and n-gram construction
Feature extraction: TF-IDF vectors capturing lexical and stylistic patterns
Classification: logistic regression trained to estimate hallucination risk
Evaluation: accuracy, precision/recall, and confusion matrix analysis
The design prioritizes interpretability, reproducibility, and ease of integration into existing LLM workflows.
Key Properties
Model-agnostic: works with any LLM output (closed or open-weight)
Post-hoc: no changes to generation process required
Lightweight: fast inference with minimal compute overhead
Interpretable: coefficients reveal linguistic risk indicators
Project Structure
sentinel-ai/
├── data/               # Labeled training and evaluation samples
├── src/
│   ├── train.py        # Model training pipeline
│   ├── evaluate.py     # Evaluation and metrics
│   └── utils.py        # Preprocessing helpers
├── models/             # Saved trained models
├── requirements.txt
└── README.md
Usage
Install dependencies
pip install -r requirements.txt
Train the model
python src/train.py
Evaluate performance
python src/evaluate.py
Integrate as a post-generation filter in LLM pipelines to flag high-risk outputs.
Results
Initial experiments demonstrate that simple lexical and stylistic features can meaningfully separate reliable from unreliable generations, supporting the hypothesis that hallucination risk can be partially inferred from language patterns alone. The project is intended as a foundation for more advanced reliability and alignment research.
Future Work
Multi-class risk scoring (low / medium / high confidence)
Incorporation of semantic consistency checks
Cross-model generalization studies
Extension to retrieval-augmented and agentic systems
Intended Use
Sentinel-LLM is designed for research, experimentation, and educational purposes. It is not a replacement for human verification but serves as an automated early-warning layer for LLM reliability.
