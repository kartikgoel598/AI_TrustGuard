import pandas as pd
import numpy as np

from detection.rag_poison_detector.embedder import DocumentEmbedder

CORPUS_PATH = "dataset/rag_dataset/rag_corpus.csv"


def main():
    corpus = pd.read_csv(CORPUS_PATH)

    basic_docs = corpus[corpus["attack_type"] == "basic"]
    print(f"basic tier: {len(basic_docs)} documents")

    embedder = DocumentEmbedder(device="cuda")
    embeddings = embedder.embed(basic_docs["content"].tolist())

    similarity_matrix = embeddings @ embeddings.T
    np.fill_diagonal(similarity_matrix, -np.inf)

    print(f"max similarity between any two basic-poison docs: {similarity_matrix.max():.4f}")
    print(f"mean of all pairwise similarities: {similarity_matrix[similarity_matrix > -np.inf].mean():.4f}")

    flat = similarity_matrix[similarity_matrix > -np.inf]
    print(f"95th percentile: {np.percentile(flat, 95):.4f}")
    print(f"99th percentile: {np.percentile(flat, 99):.4f}")
    print(f"99.9th percentile: {np.percentile(flat, 99.9):.4f}")

    same_query_groups = basic_docs.groupby("target_query").size()
    print(f"\nnumber of distinct target_query values in basic tier: {len(same_query_groups)}")
    print(f"how many target_queries have more than 1 poisoned doc: {(same_query_groups > 1).sum()}")
    print(f"max docs sharing the same target_query: {same_query_groups.max()}")

    row0 = corpus.iloc[0]
    pair = corpus[corpus["base_document_id"] == row0["base_document_id"]]
    print(f"\nvariants for base_document_id={row0['base_document_id']}:")
    print(pair[["document_id", "attack_type", "content"]].to_string())


if __name__ == "__main__":
    main()