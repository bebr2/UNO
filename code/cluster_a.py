import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
import joblib
import json
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import random

LLM_PATH = os.getenv("LLM_PATH", "../LLM")

class AgglomerativeWrapper:
    def __init__(self, n_clusters=None, distance_threshold=None, linkage='ward', metric='euclidean'):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.metric = metric
        
        if distance_threshold is not None:
            n_clusters = None
            compute_full_tree = True
        else:
            if n_clusters is None:
                n_clusters = 5
            compute_full_tree = False 

        self.model = AgglomerativeClustering(
            n_clusters=n_clusters, 
            distance_threshold=distance_threshold,
            linkage=linkage, 
            metric=metric,
            compute_full_tree=compute_full_tree
        )
        self.centroids = None
        self.labels_ = None
        self.cluster_counts = {}
        
    def fit_calculate_centroids(self, X_cluster, X_centroid):
        self.labels_ = self.model.fit_predict(X_cluster)
        
        if self.distance_threshold is not None:
            self.n_clusters = self.model.n_clusters_
            print(f"[Auto-Cluster] Based on threshold {self.distance_threshold:.4f}, auto-clustered into {self.n_clusters} clusters.")
        else:
            self.n_clusters = self.model.n_clusters
        
        self._compute_centroids(X_centroid, self.labels_)
        return self.labels_

    def _compute_centroids(self, X, labels):
        unique_labels = sorted(list(set(labels)))
        temp_centroids = {}
        self.cluster_counts = {}
        
        for label in unique_labels:
            if label == -1: continue
            cluster_points = X[labels == label]
            count = cluster_points.shape[0]
            centroid = cluster_points.mean(axis=0)
            
            temp_centroids[label] = centroid
            self.cluster_counts[label] = count
            
        final_centroids = []
        max_label = max(temp_centroids.keys()) if temp_centroids else 0
        
        self.centroids = np.zeros((max_label + 1, X.shape[1]))
        for i in range(max_label + 1):
            if i in temp_centroids:
                self.centroids[i] = temp_centroids[i]
                
    def predict(self, X, distance_threshold=None):
        if self.centroids is None:
            raise Exception("Model not fitted yet.")
        
        dists = euclidean_distances(X, self.centroids)
        min_dists = np.min(dists, axis=1)
        labels = np.argmin(dists, axis=1)

        if distance_threshold is not None:
            labels[min_dists > distance_threshold] = -1
            
        return labels

    def update_centroid_online(self, label, new_vector):
        if label >= len(self.centroids):
            raise ValueError(f"Label {label} out of bounds.")
            
        old_centroid = self.centroids[label]
        n = self.cluster_counts.get(label, 0)
        
        new_centroid = (old_centroid * n + new_vector) / (n + 1)
        
        self.centroids[label] = new_centroid
        self.cluster_counts[label] = n + 1

class QuestionClusterer:
    def __init__(self, 
                 n_clusters_k: Optional[int] = 5, 
                 distance_threshold: Optional[float] = None,
                 model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 linkage: str = 'ward'):
        
        self.model_name = model_name
        
        self.distance_threshold = distance_threshold
        self.k = n_clusters_k if distance_threshold is None else None
        
        self.embedder = None
        self.linkage = linkage
        
        self.model = AgglomerativeWrapper(
            n_clusters=self.k, 
            distance_threshold=self.distance_threshold,
            linkage=self.linkage
        )
        
        self._model_file = 'cluster_model.joblib'
        self._data_file = 'source_data.json'
        self._embeddings_file = 'embeddings.npy'
        self._clusters_file = 'clusters_map.json'
        self._results_file = 'cluster_results.json'
        self._config_file = 'cluster_config.json'
        
        self._init_embedder()

    def _init_embedder(self):
        if self.embedder is None:
            try:
                local_path = os.path.join(LLM_PATH, self.model_name.split("/")[-1])
                if os.path.exists(local_path):
                    self.embedder = SentenceTransformer(local_path, device='cuda:0')
                else:
                    self.embedder = SentenceTransformer(self.model_name, device='cuda:0')
            except Exception as e:
                print(f"Error loading embedder: {e}")
                self.embedder = None
        if self.embedder:
            self.embedder.max_seq_length = 8192

    def _get_dual_embeddings(self, texts: List[str]):
        q_parts = []
        r_parts = []
        for t in texts:
            if "[Rules]" in t:
                parts = t.split("[Rules]", 1)
                q_parts.append(parts[0].strip())
                r_parts.append(parts[1].strip())
            else:
                q_parts.append(t.strip())
                r_parts.append("")

        print("Encoding Question parts...")
        emb_q = self.embedder.encode(q_parts, show_progress_bar=True, batch_size=2, device='cuda:0')
        print("Encoding Rule parts...")
        emb_r = self.embedder.encode(r_parts, show_progress_bar=True, batch_size=2, device='cuda:0')
        
        emb_q_norm = normalize(emb_q, norm='l2')
        emb_r_norm = normalize(emb_r, norm='l2')
        
        emb_combined = normalize(np.hstack([emb_q_norm, emb_r_norm]))
        
        return emb_q_norm, emb_combined

    def _get_inference_embedding(self, texts: List[str]):
        clean_texts = [t.split("[Rules]")[0].strip() for t in texts]
        emb = self.embedder.encode(clean_texts, show_progress_bar=False, batch_size=2, device='cuda:0')
        return normalize(emb, norm='l2')

    def fit(self, texts: List[str]):
        if texts and isinstance(texts[0], dict):
            self.texts = [t["question"] for t in texts]
            self.id_to_index = [t["test_idx"] for t in texts]
        else:
            self.texts = texts
            self.id_to_index = [str(i) for i in range(len(self.texts))]
        
        print("Generating features (Mode: Combined Q+Rules)...")
        emb_q, emb_combined = self._get_dual_embeddings(self.texts)
        
        self.embeddings = emb_q 
        
        print(f"Clustering (Mode: {'Auto-Threshold' if self.distance_threshold else 'Fixed-K'})...")
        
        labels = self.model.fit_calculate_centroids(X_cluster=emb_combined, X_centroid=emb_q)
        
        self.k = self.model.n_clusters
        
        self.clusters = {}
        for i in range(self.k):
            self.clusters[str(i)] = np.where(labels == i)[0].tolist()
        
        print(f"Training completed. Total clusters: {self.k}.")
    
    def predict(self, new_texts: List[str], unseen_threshold: Optional[float] = None) -> List[str]:
        new_embeddings = self._get_inference_embedding(new_texts)
        label_ints = self.model.predict(new_embeddings, distance_threshold=unseen_threshold)
        return [str(l) for l in label_ints]

    def add(self, new_texts: List[str]):
        if isinstance(new_texts, str):
            new_texts = [new_texts]
            id_to_index = [0]
        elif isinstance(new_texts[0], dict):
            id_to_index = [t["test_idx"] for t in new_texts]
            new_texts = [t["question"] for t in new_texts]

        else:
            assert False

        print(f"Adding {len(new_texts)} items...")
        
        new_embs = self._get_inference_embedding(new_texts)
        
        label_ints = self.model.predict(new_embs)
        
        start_idx = len(self.texts)
        
        for i, text in enumerate(new_texts):
            label = int(label_ints[i])
            label_str = str(label)
            vec = new_embs[i]
            current_idx = start_idx + i
            
            self.texts.append(text)
            self.embeddings = np.vstack([self.embeddings, vec])
            self.id_to_index.append(id_to_index[i])
            
            if label_str not in self.clusters:
                self.clusters[label_str] = []
            self.clusters[label_str].append(current_idx)
            
            self.model.update_centroid_online(label, vec)

        print(f"Add completed. Current total items: {len(self.texts)}")
        return [str(l) for l in label_ints]

    def save_model(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(output_dir, self._model_file))
        
        np.save(os.path.join(output_dir, self._embeddings_file), self.embeddings)
        
        with open(os.path.join(output_dir, self._data_file), 'w', encoding='utf-8') as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=4)
            
        with open(os.path.join(output_dir, self._clusters_file), 'w', encoding='utf-8') as f:
            json.dump(self.clusters, f, indent=4)
            
        config = {
            "model_name": self.model_name,
            "distance_threshold": self.distance_threshold,
            "n_clusters_k": self.k
        }
        with open(os.path.join(output_dir, self._config_file), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
            
        print(f"Model saved to {output_dir}")

    def load_model(self, output_dir: str):
        print(f"\nLoading model {output_dir}...")
        config_path = os.path.join(output_dir, self._config_file)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f: config = json.load(f)
            self.distance_threshold = config.get("distance_threshold", None)
            self._init_embedder()
        
        self.model = joblib.load(os.path.join(output_dir, self._model_file))
        self.k = self.model.n_clusters
        
        with open(os.path.join(output_dir, self._data_file), 'r') as f: self.texts = json.load(f)
        self.embeddings = np.load(os.path.join(output_dir, self._embeddings_file))
        
        with open(os.path.join(output_dir, self._clusters_file), 'r') as f:
            self.clusters = json.load(f)

        result_filepath = os.path.join(output_dir, self._results_file)
        with open(result_filepath, 'r') as f:
            results = json.load(f)

        self.id_to_index = [None for _ in range(len(self.texts))]
        for cluster_id in self.clusters.keys():
            for j, idx in enumerate(self.clusters[cluster_id]):
                self.id_to_index[int(idx)] = results[cluster_id][j]["id"]
        
        assert None not in self.id_to_index

    def save_cluster_results_json(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, self._results_file)
        results = {}
        sorted_ids = sorted(self.clusters.keys(), key=lambda x: int(x) if x.isdigit() else x)
        
        for cluster_id in sorted_ids:
            indices = self.clusters[cluster_id]
            cluster_content = []
            for idx in indices:
                idx = int(idx)
                cluster_content.append({
                    "id": self.id_to_index[idx], 
                    "text": self.texts[idx]
                })
            results[str(cluster_id)] = cluster_content
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {filepath}")
    
    def print_cluster_status(self):
        pass

def similarity_to_distance(sim_threshold):
    return np.sqrt(2 * (1 - sim_threshold))
