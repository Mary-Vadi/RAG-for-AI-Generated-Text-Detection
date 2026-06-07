# RAG for AI-Generated Text Detection

This project explores the use of Retrieval-Augmented Generation (RAG) for detecting AI-generated text. It focuses on building a retrieval-enhanced NLP pipeline using the HC3 dataset, sentence embeddings, and vector search.

## Project Goal

The goal of this project is to understand how retrieval-based methods can support AI-generated text detection by providing additional contextual information before classification.

## Planned Workflow

1. Explore the HC3 dataset
2. Preprocess human-written and AI-generated text
3. Generate sentence embeddings
4. Store embeddings in a FAISS vector database
5. Retrieve semantically similar examples
6. Use retrieved context to support AI-generated text detection

## Repository Structure

```text
RAG-for-AI-Generated-Text-Detection/

├── data/
│   ├── documents/
│   ├── hc3/
│   └── vector_store/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_embeddings.ipynb
│   ├── 03_faiss_retrieval.ipynb
│   ├── 04_rag_pipeline.ipynb
│   └── 05_hc3_detection.ipynb
│
├── .gitignore
├── README.md
├── main.py
├── requirements.txt
└── project_architecture.md
```

## Technologies

* Python
* HC3 Dataset
* Sentence Transformers
* FAISS
* Hugging Face Transformers
* PyTorch
* Scikit-learn
* Jupyter Notebook

## Skills Demonstrated

* Natural Language Processing
* Retrieval-Augmented Generation
* Vector Search
* Sentence Embeddings
* AI-Generated Text Detection
* Machine Learning Evaluation

## Future Work

* Add RoBERTa-based classification
* Compare RAG and non-RAG baselines
* Evaluate accuracy, precision, recall, and F1-score
* Extend the project to multimodal deepfake detection
