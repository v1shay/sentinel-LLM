Sentinel-LLM
Post-Hoc Reliability Auditing for Large Language Models
Overview
Sentinel-LLM is an end-to-end NLP system designed to identify hallucination risk and unsupported claims in large language model (LLM) outputs. The system operates post-generation, requiring no access to model internals, making it suitable for auditing both closed and open-weight models.
The project investigates whether surface-level linguistic patterns—such as overconfidence markers and structural inconsistencies—can be used to estimate output reliability.
Project Overview
Sentinel-LLM processes generated text, extracts interpretable language features, and applies supervised classifiers to flag high-risk responses. The system is lightweight, fast, and designed to integrate as a safety layer in downstream LLM workflows.
The pipeline is structured to support reproducibility, benchmarking, and future extensions toward semantic and retrieval-based validation.
Core Capabilities
Text preprocessing and normalization
TF-IDF–based feature extraction
Supervised classification (logistic regression)
Hallucination risk scoring
Quantitative evaluation and benchmarking
Modular post-hoc auditing architecture
Repository Structure
sentinel-llm/
├── data/               # Labeled training and evaluation samples
├── src/
│   ├── train.py        # Model training pipeline
│   ├── evaluate.py     # Evaluation and metrics
│   └── utils.py        # Preprocessing utilities
├── models/             # Saved trained models
├── requirements.txt
└── README.md
Intended Use
Sentinel-LLM is intended for research and experimentation in LLM safety, reliability, and alignment. It is not a replacement for human verification but serves as an automated early-warning mechanism.
Future Work
Multi-class risk stratification
Semantic consistency modeling
Cross-model generalization analysis
Integration with agentic and retrieval systems
