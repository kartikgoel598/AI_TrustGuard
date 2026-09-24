import numpy as np
import networkx as nx
 
from detection.base import Detector, LayerResult, ValidationError
from detection.rag_poison_detector.embedder import DocumentEmbedder
 
K_NEIGHBORS = 10
PRUNE_ALPHA = 2.5
PRUNE_SAMPLE_RATIO = 0.5
MIN_CLIQUE_SIZE = 3
MAX_CLIQUE_SIZE = 11
PREFILTER_DENSITY_THRESHOLD = 1.0

class CleanBaseDetector(Detector):
    def __init__(self, device="cpu"):
        self.embedder = DocumentEmbedder(device=device)
 
    def validate(self, input_data):
        if "documents" not in input_data:
            return ValidationError(layer="layer2_cleanbase", message="missing documents", field="documents")
 
        documents = input_data["documents"]
        if not isinstance(documents, list) or len(documents) < K_NEIGHBORS + 1:
            return ValidationError(
                layer="layer2_cleanbase",
                message=f"need at least {K_NEIGHBORS + 1} documents",
                field="documents",
            )
 
        return None
    
    def build_knn_graph(self,embeddings):
        n = embeddings.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(n))
        similarity_matrix = embeddings @ embeddings.T
        for i in range(n):
            sims = similarity_matrix[i].copy()
            sims[i] = -np.inf
            neighbor_indices = np.argsort(sims)[-K_NEIGHBORS:]
            for j in neighbor_indices:
                weight = similarity_matrix[i, j]
                if G.has_edge(i, j):
                    G[i][j]["weight"] = max(G[i][j]["weight"], weight)
                else:
                    G.add_edge(i, j, weight=weight)
 
        return G
    def prune_graph(self, G):
        edge_weights = np.array([data["weight"] for _, _, data in G.edges(data=True)])
 
        if len(edge_weights) == 0:
            return G
        sample_size = max(1, int(len(edge_weights) * PRUNE_SAMPLE_RATIO))
        sample = np.random.choice(edge_weights, size=sample_size, replace=False)
 
        mean_weight = sample.mean()
        std_weight = sample.std()
        threshold = mean_weight + PRUNE_ALPHA * std_weight
 
        G_pruned = nx.Graph()
        G_pruned.add_nodes_from(G.nodes())
 
        for u, v, data in G.edges(data=True):
            if data["weight"] > threshold:
                G_pruned.add_edge(u, v, weight=data["weight"])
 
        return G_pruned
    def find_flagged_cliques(self, G):
        flagged_indices = set()
        clique_details = []
 
        for clique in nx.find_cliques(G):
            size = len(clique)
            if not (MIN_CLIQUE_SIZE <= size <= MAX_CLIQUE_SIZE):
                continue
 
            subgraph = G.subgraph(clique)
            if subgraph.number_of_edges() == 0:
                continue
 
            edge_weights = [data["weight"] for _, _, data in subgraph.edges(data=True)]
            avg_weight = np.mean(edge_weights)
 
            if avg_weight >= PREFILTER_DENSITY_THRESHOLD:
                continue
 
            flagged_indices.update(clique)
            clique_details.append({
                "size": size,
                "avg_edge_weight": float(avg_weight),
                "member_indices": list(clique),
            })
 
        return flagged_indices, clique_details

    def run(self, input_data):
        error = self.validate(input_data)
        if error is not None:
            return LayerResult(risk_score=0.0, status="error", detail={"error": error.message})
 
        documents = input_data["documents"]
        n = len(documents)
 
        embeddings = self.embedder.embed(documents)
 
        G = self.build_knn_graph(embeddings)
        G = self.prune_graph(G)
        flagged_indices, clique_details = self.find_flagged_cliques(G)
 
        per_document = [{"index": i, "flagged": i in flagged_indices} for i in range(n)]
 
        status = "flagged" if len(flagged_indices) > 0 else "clean"
        risk_score = len(flagged_indices) / n if n > 0 else 0.0
 
        return LayerResult(
            risk_score=risk_score,
            status=status,
            detail={
                "num_documents": n,
                "num_flagged": len(flagged_indices),
                "cliques": clique_details,
                "per_document": per_document,
            },
        )  
