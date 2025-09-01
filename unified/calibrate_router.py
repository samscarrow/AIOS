#!/usr/bin/env python3
"""
Router Calibration Notebook/Script
Tune thresholds, generate prototypes, and evaluate router performance
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# Import our production router
from production_task_router import (
    HardenedSemanticRouter,
    CalibratedZeroShotRouter,
    UnifiedProductionRouter
)


class RouterCalibrator:
    """Calibrate router thresholds and generate prototypes from historical data"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.prototypes = {}
        self.thresholds = {
            "semantic_tau": 0.5,
            "semantic_margin": 0.05,
            "zeroshot_task": 0.45,
            "zeroshot_complexity": 0.40
        }
        self.performance_metrics = {}
    
    def load_historical_data(self, path: str) -> pd.DataFrame:
        """
        Load historical task routing data
        Expected columns: task_description, task_type, complexity, domain (optional)
        """
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        elif path.endswith('.json'):
            df = pd.read_json(path)
        elif path.endswith('.jsonl'):
            df = pd.read_json(path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        # Validate required columns
        required = ['task_description', 'task_type', 'complexity']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        print(f"Loaded {len(df)} historical tasks")
        print(f"Task types: {df['task_type'].value_counts().to_dict()}")
        print(f"Complexity: {df['complexity'].value_counts().to_dict()}")
        
        return df
    
    def generate_prototypes(
        self,
        df: pd.DataFrame,
        n_prototypes_per_class: int = 3,
        min_samples: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Generate multiple prototypes per task type using clustering
        """
        print("\n=== Generating Prototypes ===")
        prototypes = {}
        
        for task_type in df['task_type'].unique():
            # Get all tasks of this type
            type_tasks = df[df['task_type'] == task_type]['task_description'].tolist()
            
            if len(type_tasks) < min_samples:
                print(f"Warning: {task_type} has only {len(type_tasks)} samples (min: {min_samples})")
                # Use all samples as prototypes if too few
                embeddings = self.encoder.encode(type_tasks[:n_prototypes_per_class])
                prototypes[task_type] = embeddings
                continue
            
            print(f"\nProcessing {task_type} ({len(type_tasks)} samples)...")
            
            # Encode all tasks
            embeddings = self.encoder.encode(type_tasks, show_progress_bar=True)
            
            # Cluster to find prototypes
            n_clusters = min(n_prototypes_per_class, len(type_tasks))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            
            # Use cluster centroids as prototypes
            prototypes[task_type] = kmeans.cluster_centers_
            
            # Report cluster quality
            inertia = kmeans.inertia_ / len(embeddings)
            print(f"  - Generated {n_clusters} prototypes")
            print(f"  - Average distance to centroid: {inertia:.4f}")
        
        self.prototypes = prototypes
        return prototypes
    
    def calibrate_semantic_thresholds(
        self,
        df: pd.DataFrame,
        tau_range: Tuple[float, float] = (0.4, 0.7),
        margin_range: Tuple[float, float] = (0.02, 0.15),
        n_steps: int = 10
    ) -> Dict[str, float]:
        """
        Find optimal tau and margin thresholds using grid search
        """
        print("\n=== Calibrating Semantic Router Thresholds ===")
        
        # Split data
        train_df, val_df = train_test_split(df, test_size=0.3, stratify=df['task_type'])
        print(f"Training set: {len(train_df)}, Validation set: {len(val_df)}")
        
        # Generate prototypes from training set only
        train_protos = self.generate_prototypes(train_df, n_prototypes_per_class=3)
        
        # Grid search
        tau_values = np.linspace(tau_range[0], tau_range[1], n_steps)
        margin_values = np.linspace(margin_range[0], margin_range[1], n_steps)
        
        best_f1 = 0
        best_params = {}
        results = []
        
        print("\nGrid search progress:")
        for tau in tqdm(tau_values):
            for margin in margin_values:
                # Create router with current thresholds
                router = HardenedSemanticRouter(
                    proto_map=train_protos,
                    tau=tau,
                    margin_tau=margin
                )
                
                # Evaluate on validation set
                predictions = []
                true_labels = []
                abstentions = 0
                
                for _, row in val_df.iterrows():
                    result = router.route(row['task_description'])
                    
                    if result['abstain']:
                        abstentions += 1
                        predictions.append('abstain')
                    else:
                        predictions.append(result['cognitive_type'])
                    
                    true_labels.append(row['task_type'])
                
                # Calculate metrics
                coverage = 1 - (abstentions / len(val_df))
                
                # F1 score only on non-abstained predictions
                non_abstain_idx = [i for i, p in enumerate(predictions) if p != 'abstain']
                if non_abstain_idx:
                    pred_subset = [predictions[i] for i in non_abstain_idx]
                    true_subset = [true_labels[i] for i in non_abstain_idx]
                    
                    # Calculate accuracy
                    accuracy = sum(p == t for p, t in zip(pred_subset, true_subset)) / len(pred_subset)
                    
                    # Combined metric: balance accuracy and coverage
                    combined_score = accuracy * coverage
                    
                    results.append({
                        'tau': tau,
                        'margin': margin,
                        'accuracy': accuracy,
                        'coverage': coverage,
                        'combined_score': combined_score
                    })
                    
                    if combined_score > best_f1:
                        best_f1 = combined_score
                        best_params = {
                            'tau': tau,
                            'margin': margin,
                            'accuracy': accuracy,
                            'coverage': coverage
                        }
        
        print(f"\nBest parameters found:")
        print(f"  Tau: {best_params['tau']:.3f}")
        print(f"  Margin: {best_params['margin']:.3f}")
        print(f"  Accuracy: {best_params['accuracy']:.3f}")
        print(f"  Coverage: {best_params['coverage']:.3f}")
        print(f"  Combined Score: {best_f1:.3f}")
        
        self.thresholds['semantic_tau'] = best_params['tau']
        self.thresholds['semantic_margin'] = best_params['margin']
        
        # Visualize results
        self._plot_threshold_heatmap(results)
        
        return best_params
    
    def _plot_threshold_heatmap(self, results: List[Dict]):
        """Plot heatmap of threshold performance"""
        df = pd.DataFrame(results)
        pivot = df.pivot_table(values='combined_score', index='margin', columns='tau')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('Semantic Router Threshold Calibration (Combined Score)')
        plt.xlabel('Tau (Confidence Threshold)')
        plt.ylabel('Margin Threshold')
        plt.tight_layout()
        plt.savefig('threshold_calibration.png', dpi=150)
        plt.show()
    
    def evaluate_router_performance(
        self,
        df: pd.DataFrame,
        router: Optional[UnifiedProductionRouter] = None
    ) -> Dict:
        """
        Comprehensive router evaluation
        """
        print("\n=== Evaluating Router Performance ===")
        
        if router is None:
            # Create router with calibrated parameters
            semantic_router = HardenedSemanticRouter(
                proto_map=self.prototypes,
                tau=self.thresholds['semantic_tau'],
                margin_tau=self.thresholds['semantic_margin']
            )
            
            router = UnifiedProductionRouter(
                semantic_router=semantic_router,
                escalation_strategy="waterfall"
            )
        
        # Evaluate
        predictions = []
        true_types = []
        true_complexity = []
        pred_complexity = []
        latencies = []
        routing_paths = []
        confidences = []
        
        print("Running evaluation...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            start = time.perf_counter()
            
            # Route synchronously (convert async to sync for evaluation)
            import asyncio
            decision = asyncio.run(router.route(row['task_description']))
            
            latency = (time.perf_counter() - start) * 1000
            
            predictions.append(decision.task_type or 'abstain')
            true_types.append(row['task_type'])
            true_complexity.append(row['complexity'])
            pred_complexity.append(decision.complexity)
            latencies.append(latency)
            routing_paths.append(decision.routing_path)
            confidences.append(decision.confidence)
        
        # Calculate metrics
        metrics = {
            'task_type_accuracy': sum(p == t for p, t in zip(predictions, true_types)) / len(predictions),
            'complexity_accuracy': sum(p == t for p, t in zip(pred_complexity, true_complexity)) / len(pred_complexity),
            'abstention_rate': sum(p == 'abstain' for p in predictions) / len(predictions),
            'avg_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'avg_confidence': np.mean(confidences),
            'routing_path_distribution': pd.Series(routing_paths).value_counts().to_dict()
        }
        
        print("\n=== Performance Metrics ===")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
        
        # Classification report for task types
        print("\n=== Task Type Classification Report ===")
        non_abstain_idx = [i for i, p in enumerate(predictions) if p != 'abstain']
        if non_abstain_idx:
            pred_subset = [predictions[i] for i in non_abstain_idx]
            true_subset = [true_types[i] for i in non_abstain_idx]
            print(classification_report(true_subset, pred_subset))
        
        # Confusion matrix
        self._plot_confusion_matrix(true_types, predictions)
        
        # Latency distribution
        self._plot_latency_distribution(latencies)
        
        self.performance_metrics = metrics
        return metrics
    
    def _plot_confusion_matrix(self, y_true: List, y_pred: List):
        """Plot confusion matrix"""
        # Get unique labels
        labels = sorted(set(y_true) | set(y_pred))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
        plt.title('Task Type Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150)
        plt.show()
    
    def _plot_latency_distribution(self, latencies: List[float]):
        """Plot latency distribution"""
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(latencies, bins=50, edgecolor='black')
        plt.axvline(np.mean(latencies), color='red', linestyle='--', label=f'Mean: {np.mean(latencies):.1f}ms')
        plt.axvline(np.percentile(latencies, 95), color='orange', linestyle='--', label=f'P95: {np.percentile(latencies, 95):.1f}ms')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Count')
        plt.title('Latency Distribution')
        plt.legend()
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(latencies, vert=True)
        plt.ylabel('Latency (ms)')
        plt.title('Latency Box Plot')
        
        plt.tight_layout()
        plt.savefig('latency_distribution.png', dpi=150)
        plt.show()
    
    def save_calibration(self, output_dir: str = "router_calibration"):
        """Save calibrated parameters and prototypes"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save prototypes
        np.save(output_path / "prototypes.npy", self.prototypes)
        
        # Save thresholds and config
        config = {
            "thresholds": self.thresholds,
            "performance_metrics": self.performance_metrics,
            "model_name": "all-MiniLM-L6-v2"
        }
        
        with open(output_path / "router_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nCalibration saved to {output_path}/")
        print(f"  - prototypes.npy: {len(self.prototypes)} task types")
        print(f"  - router_config.json: thresholds and metrics")
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic task data for testing"""
        print(f"\nGenerating {n_samples} synthetic tasks...")
        
        task_templates = {
            'code_generation': [
                "Write a {lang} function to {action}",
                "Implement a {structure} that {behavior}",
                "Create a {lang} script for {purpose}",
                "Develop a {component} to handle {feature}"
            ],
            'analysis': [
                "Analyze the {metric} of {system}",
                "Investigate why {issue} is happening",
                "Review the {aspect} in {codebase}",
                "Examine {data} for {pattern}"
            ],
            'debugging': [
                "Fix the {error_type} in {component}",
                "Debug why {feature} is {problem}",
                "Resolve the {issue} with {system}",
                "Patch the {vulnerability} in {module}"
            ],
            'system_design': [
                "Design a {architecture} for {scale} users",
                "Architect a {system} with {requirements}",
                "Plan the {infrastructure} for {application}",
                "Create a {design} that handles {load}"
            ],
            'data_analysis': [
                "Analyze the {dataset} for {insights}",
                "Process {data_type} data to find {pattern}",
                "Visualize {metrics} from {source}",
                "Extract {features} from {data}"
            ]
        }
        
        complexity_modifiers = {
            'simple': ['basic', 'simple', 'straightforward'],
            'medium': ['moderate', 'standard', 'typical'],
            'complex': ['complex', 'distributed', 'scalable'],
            'very_complex': ['highly complex', 'enterprise-grade', 'mission-critical']
        }
        
        # Generate tasks
        tasks = []
        for _ in range(n_samples):
            task_type = np.random.choice(list(task_templates.keys()))
            template = np.random.choice(task_templates[task_type])
            
            # Fill template with random values
            task = template.format(
                lang=np.random.choice(['Python', 'JavaScript', 'Go', 'Rust']),
                action=np.random.choice(['sort data', 'process files', 'calculate metrics']),
                structure=np.random.choice(['class', 'module', 'service']),
                behavior=np.random.choice(['handles requests', 'processes data', 'manages state']),
                purpose=np.random.choice(['automation', 'data processing', 'API integration']),
                component=np.random.choice(['API', 'service', 'module']),
                feature=np.random.choice(['authentication', 'caching', 'logging']),
                metric=np.random.choice(['performance', 'memory usage', 'throughput']),
                system=np.random.choice(['database', 'API', 'microservice']),
                issue=np.random.choice(['latency', 'memory leak', 'high CPU']),
                aspect=np.random.choice(['security', 'performance', 'architecture']),
                codebase=np.random.choice(['backend', 'frontend', 'infrastructure']),
                data=np.random.choice(['logs', 'metrics', 'traces']),
                pattern=np.random.choice(['anomalies', 'trends', 'correlations']),
                error_type=np.random.choice(['TypeError', 'NullPointer', 'IndexError']),
                problem=np.random.choice(['failing', 'slow', 'broken']),
                vulnerability=np.random.choice(['SQL injection', 'XSS', 'buffer overflow']),
                module=np.random.choice(['auth module', 'payment system', 'user service']),
                architecture=np.random.choice(['microservices', 'serverless', 'monolithic']),
                scale=np.random.choice(['1M', '10M', '100M']),
                requirements=np.random.choice(['high availability', 'low latency', 'fault tolerance']),
                infrastructure=np.random.choice(['cloud', 'hybrid', 'on-premise']),
                application=np.random.choice(['e-commerce', 'social media', 'fintech']),
                design=np.random.choice(['cache layer', 'message queue', 'load balancer']),
                load=np.random.choice(['1k QPS', '100k QPS', '1M QPS']),
                dataset=np.random.choice(['customer data', 'transaction logs', 'sensor readings']),
                insights=np.random.choice(['patterns', 'anomalies', 'predictions']),
                data_type=np.random.choice(['JSON', 'CSV', 'time-series']),
                metrics=np.random.choice(['KPIs', 'performance metrics', 'user behavior']),
                source=np.random.choice(['database', 'API', 'data lake']),
                features=np.random.choice(['key features', 'embeddings', 'statistics'])
            )
            
            # Assign complexity
            complexity = np.random.choice(list(complexity_modifiers.keys()), p=[0.1, 0.4, 0.35, 0.15])
            
            # Add complexity modifier to task sometimes
            if np.random.random() < 0.3:
                modifier = np.random.choice(complexity_modifiers[complexity])
                task = f"{task} (this is a {modifier} task)"
            
            tasks.append({
                'task_description': task,
                'task_type': task_type,
                'complexity': complexity
            })
        
        return pd.DataFrame(tasks)


def main():
    """Main calibration workflow"""
    print("=" * 60)
    print("Router Calibration Tool")
    print("=" * 60)
    
    calibrator = RouterCalibrator()
    
    # Option 1: Use synthetic data for demonstration
    print("\nGenerating synthetic training data...")
    df = calibrator.generate_synthetic_data(n_samples=2000)
    
    # Option 2: Load your own data
    # df = calibrator.load_historical_data("your_tasks.csv")
    
    # Generate prototypes
    prototypes = calibrator.generate_prototypes(df, n_prototypes_per_class=3)
    
    # Calibrate thresholds
    best_params = calibrator.calibrate_semantic_thresholds(
        df,
        tau_range=(0.45, 0.65),
        margin_range=(0.04, 0.12),
        n_steps=8
    )
    
    # Evaluate performance
    metrics = calibrator.evaluate_router_performance(df)
    
    # Save calibration
    calibrator.save_calibration()
    
    print("\n" + "=" * 60)
    print("Calibration Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the generated plots (threshold_calibration.png, confusion_matrix.png, latency_distribution.png)")
    print("2. Load calibration in production:")
    print("   router = create_production_router(")
    print("       proto_path='router_calibration/prototypes.npy',")
    print("       config_path='router_calibration/router_config.json'")
    print("   )")
    print("3. Monitor drift_score in production and recalibrate when > 0.15")


if __name__ == "__main__":
    main()