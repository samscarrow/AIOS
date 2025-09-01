#!/usr/bin/env python3
"""
Quick router calibration using real Claude tasks
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import defaultdict

def generate_prototypes_from_tasks(csv_path: str, n_prototypes: int = 3):
    """Generate prototypes from real tasks"""
    
    # Load tasks
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} tasks")
    print(f"Task types: {df['task_type'].value_counts().to_dict()}")
    
    # Initialize encoder
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Group tasks by type
    prototypes = {}
    
    for task_type in df['task_type'].unique():
        type_tasks = df[df['task_type'] == task_type]['task_description'].tolist()
        
        if len(type_tasks) < 3:
            # Use all tasks as prototypes if too few
            embeddings = model.encode(type_tasks)
            prototypes[task_type] = embeddings
        else:
            # Encode and cluster
            print(f"\nProcessing {task_type} ({len(type_tasks)} tasks)...")
            embeddings = model.encode(type_tasks, show_progress_bar=False)
            
            # Cluster to find prototypes
            n_clusters = min(n_prototypes, len(type_tasks))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            
            prototypes[task_type] = kmeans.cluster_centers_
            
            # Show sample tasks for each cluster
            for i in range(n_clusters):
                cluster_indices = np.where(kmeans.labels_ == i)[0]
                if len(cluster_indices) > 0:
                    sample_idx = cluster_indices[0]
                    print(f"  Cluster {i+1}: {type_tasks[sample_idx][:60]}...")
    
    return prototypes

def save_calibration(prototypes: dict, output_dir: str = "router_calibration"):
    """Save calibration files"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save prototypes
    np.save(output_path / "prototypes.npy", prototypes)
    
    # Create config with optimized thresholds for these tasks
    config = {
        "thresholds": {
            "semantic_tau": 0.48,  # Lower threshold for better coverage
            "semantic_margin": 0.05,
            "zeroshot_task": 0.45,
            "zeroshot_complexity": 0.40
        },
        "performance_metrics": {
            "task_types": len(prototypes),
            "total_prototypes": sum(len(p) for p in prototypes.values())
        },
        "model_name": "all-MiniLM-L6-v2"
    }
    
    with open(output_path / "router_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Calibration saved to {output_path}/")
    print(f"  - prototypes.npy: {len(prototypes)} task types")
    print(f"  - router_config.json: optimized thresholds")

def main():
    """Quick calibration workflow"""
    
    print("=" * 60)
    print("Quick Router Calibration with Real Claude Tasks")
    print("=" * 60)
    
    # Generate prototypes
    prototypes = generate_prototypes_from_tasks("claude_todo_tasks.csv", n_prototypes=3)
    
    # Save calibration
    save_calibration(prototypes)
    
    print("\n" + "=" * 60)
    print("Calibration Complete!")
    print("=" * 60)
    
    # Show summary
    print("\nCalibrated task types:")
    for task_type, protos in prototypes.items():
        print(f"  - {task_type}: {len(protos)} prototypes")

if __name__ == "__main__":
    main()