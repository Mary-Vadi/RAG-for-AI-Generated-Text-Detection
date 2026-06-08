"""
RAG for AI-Generated Text Detection
------------------------------------
Entry point for running the full pipeline from the command line.

Usage:
    python main.py --text "Your text here"
    python main.py --file path/to/texts.csv
    python main.py --build-index
"""

import argparse
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "data/vector_store/hc3_embeddings.npy"
METADATA_PATH = "data/vector_store/hc3_metadata.csv"
INDEX_PATH = "data/vector_store/hc3_faiss.index"


def load_resources():
    """Load the FAISS index, metadata, and embedding model."""
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}. "
            "Run with --build-index first, or run notebook 03."
        )

    index = faiss.read_index(INDEX_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    model = SentenceTransformer(MODEL_NAME)

    print(f"Loaded index with {index.ntotal} vectors.")
    return index, metadata, model


def build_index():
    """Build and save the FAISS index from saved embeddings."""
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"Embeddings not found at {EMBEDDINGS_PATH}. "
            "Run notebooks 01 and 02 first to generate embeddings."
        )

    embeddings = np.load(EMBEDDINGS_PATH)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, INDEX_PATH)
    print(f"Index built with {index.ntotal} vectors. Saved to {INDEX_PATH}.")


def classify(text, index, metadata, model, top_k=10):
    """
    Classify text as human (0) or AI-generated (1).

    Returns the predicted label and AI confidence score.
    """
    query_vector = model.encode([text], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    weights = [1.0 / (d + 1e-6) for d in distances[0]]
    total_weight = sum(weights)

    ai_score = sum(
        w for w, idx in zip(weights, indices[0])
        if int(metadata.iloc[idx]["label"]) == 1
    )
    ai_confidence = ai_score / total_weight
    predicted_label = 1 if ai_confidence >= 0.5 else 0
    predicted_source = "AI-generated" if predicted_label == 1 else "Human"

    return predicted_label, predicted_source, round(ai_confidence, 4)


def run_on_file(filepath, index, metadata, model):
    """Run classification on a CSV file with an 'answer' column."""
    df = pd.read_csv(filepath)

    if "answer" not in df.columns:
        raise ValueError("CSV must have an 'answer' column.")

    labels, sources, confidences = [], [], []

    for text in df["answer"]:
        label, source, conf = classify(str(text), index, metadata, model)
        labels.append(label)
        sources.append(source)
        confidences.append(conf)

    df["predicted_label"] = labels
    df["predicted_source"] = sources
    df["ai_confidence"] = confidences

    output_path = filepath.replace(".csv", "_predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RAG-based AI-generated text detection using HC3 and FAISS."
    )

    parser.add_argument("--text", type=str, help="Single text string to classify.")
    parser.add_argument("--file", type=str, help="Path to a CSV file with an 'answer' column.")
    parser.add_argument(
        "--build-index", action="store_true",
        help="Build the FAISS index from saved embeddings."
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of nearest neighbours to retrieve (default: 10)."
    )

    args = parser.parse_args()

    if args.build_index:
        build_index()
        return

    if not args.text and not args.file:
        parser.print_help()
        return

    index, metadata, model = load_resources()

    if args.text:
        label, source, confidence = classify(args.text, index, metadata, model, top_k=args.top_k)
        print()
        print(f"  Input:      {args.text[:100]}...")
        print(f"  Prediction: {source}")
        print(f"  Confidence: {confidence}")
        print()

    elif args.file:
        run_on_file(args.file, index, metadata, model)


if __name__ == "__main__":
    main()
