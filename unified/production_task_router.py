#!/usr/bin/env python3
"""
Production-Ready Task Router
Hardened implementation with multiple routing strategies and proper abstention/calibration
"""

import torch
import torch.nn as nn
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, pipeline
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import json
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# TIER 1: Ultra-Fast Semantic Router (Primary)
# ============================================================================

class HardenedSemanticRouter:
    """Production semantic router with multi-prototype support and calibration"""
    
    def __init__(
        self, 
        proto_map: Dict[str, np.ndarray],
        model_name: str = "all-MiniLM-L6-v2",
        tau: float = 0.54,
        margin_tau: float = 0.06,
        complexity_rules: Optional[Dict] = None
    ):
        """
        Args:
            proto_map: dict[str, np.ndarray] where each value is shape (k, d) = multiple prototypes per class
            tau: min cosine required to accept a class
            margin_tau: min (top1 - top2) required to avoid abstain
            complexity_rules: optional rules for complexity assessment
        """
        self.model = SentenceTransformer(model_name)
        
        # Stack prototypes into a single matrix while remembering class indices
        self.labels, mats = [], []
        for label, protos in proto_map.items():
            # L2-normalize prototypes
            P = protos / np.clip(norm(protos, axis=1, keepdims=True), 1e-12, None)
            mats.append(P)
            self.labels += [label] * len(P)
        
        self.P = np.vstack(mats)  # (M, d)
        self.labels = np.array(self.labels)
        
        self.tau = tau
        self.margin_tau = margin_tau
        self.complexity_rules = complexity_rules or self._default_complexity_rules()
        
        # Tracking for drift detection
        self.routing_history = []
        self.drift_window = 1000
        
    def _default_complexity_rules(self) -> Dict:
        """Default complexity assessment rules"""
        return {
            'trivial': {
                'max_tokens': 10,
                'keywords': ['what', 'list', 'show', 'display'],
                'patterns': []
            },
            'simple': {
                'max_tokens': 30,
                'keywords': ['basic', 'simple', 'straightforward', 'small'],
                'patterns': []
            },
            'medium': {
                'max_tokens': 60,
                'keywords': ['moderate', 'standard', 'typical'],
                'patterns': []
            },
            'complex': {
                'max_tokens': 120,
                'keywords': ['complex', 'distributed', 'concurrent', 'scalable'],
                'patterns': ['multi-threaded', 'high-performance']
            },
            'very_complex': {
                'max_tokens': None,
                'keywords': ['architecture', 'mission-critical', 'real-time', 'fault-tolerant'],
                'patterns': ['end-to-end', 'production-grade', 'enterprise']
            }
        }
    
    def route(self, text: str) -> Dict[str, Any]:
        """Route task with calibrated abstention"""
        start_time = time.perf_counter()
        
        # Encode and normalize
        x = self.model.encode(text, normalize_embeddings=True)  # (d,)
        sims = self.P @ x  # (M,)
        
        # Max per class by grouping
        best_by_class = {}
        for s, lbl in zip(sims, self.labels):
            if lbl not in best_by_class or s > best_by_class[lbl]:
                best_by_class[lbl] = s
        
        classes, scores = zip(*best_by_class.items())
        order = np.argsort(scores)[::-1]
        classes = np.array(classes)[order]
        scores = np.array(scores)[order]
        
        top1, top2 = scores[0], (scores[1] if len(scores) > 1 else -1.0)
        margin = top1 - top2
        abstain = (top1 < self.tau) or (margin < self.margin_tau)
        
        # Assess complexity
        complexity = self._assess_complexity(text)
        
        # Track for drift detection
        self._track_routing(top1, margin, abstain)
        
        routing_time_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "cognitive_type": (None if abstain else classes[0]),
            "confidence": float(top1),
            "margin": float(margin),
            "abstain": bool(abstain),
            "complexity": complexity,
            "alternatives": [{"label": c, "score": float(s)} for c, s in zip(classes[:3], scores[:3])],
            "routing_time_ms": routing_time_ms,
            "drift_score": self._calculate_drift()
        }
    
    def _assess_complexity(self, text: str) -> str:
        """Fast complexity assessment using rules"""
        token_count = len(text.split())
        text_lower = text.lower()
        
        # Check for code blocks
        has_code = '```' in text or 'def ' in text or 'class ' in text
        
        # Apply rules in order
        for complexity, rules in self.complexity_rules.items():
            max_tokens = rules.get('max_tokens')
            if max_tokens and token_count > max_tokens:
                continue
                
            if any(kw in text_lower for kw in rules['keywords']):
                return complexity
                
            if any(pattern in text_lower for pattern in rules['patterns']):
                return complexity
        
        # Default based on length
        if token_count < 15:
            return 'simple'
        elif token_count < 50:
            return 'medium'
        else:
            return 'complex'
    
    def _track_routing(self, confidence: float, margin: float, abstain: bool):
        """Track routing decisions for drift detection"""
        self.routing_history.append({
            'confidence': confidence,
            'margin': margin,
            'abstain': abstain,
            'timestamp': time.time()
        })
        
        # Keep window size limited
        if len(self.routing_history) > self.drift_window:
            self.routing_history.pop(0)
    
    def _calculate_drift(self) -> float:
        """Calculate drift score based on recent routing patterns"""
        if len(self.routing_history) < 100:
            return 0.0
        
        recent = self.routing_history[-100:]
        older = self.routing_history[-200:-100] if len(self.routing_history) >= 200 else self.routing_history[:100]
        
        recent_conf = np.mean([r['confidence'] for r in recent])
        older_conf = np.mean([r['confidence'] for r in older])
        
        recent_abstain = np.mean([r['abstain'] for r in recent])
        older_abstain = np.mean([r['abstain'] for r in older])
        
        drift = abs(recent_conf - older_conf) + abs(recent_abstain - older_abstain)
        return float(drift)


# ============================================================================
# TIER 2: Zero-Shot NLI Router (For Abstentions)
# ============================================================================

class CalibratedZeroShotRouter:
    """Zero-shot router with better label descriptions and calibration"""
    
    def __init__(
        self,
        model: str = "facebook/bart-large-mnli",
        device: int = -1,
        task_threshold: float = 0.45,
        complexity_threshold: float = 0.40
    ):
        self.clf = pipeline(
            "zero-shot-classification",
            model=model,
            device=device,
            hypothesis_template="This task is about {}."
        )
        
        self.task_labels = {
            "code generation": "writing or modifying source code, functions, classes, or scripts",
            "bug fixing": "reproducing, diagnosing, and patching failures or exceptions",
            "performance analysis": "profiling, throughput/latency tuning, memory/CPU bottlenecks",
            "system design": "architecting components, APIs, storage, scaling, trade-off analysis",
            "data analysis": "statistics, EDA, visualization, datasets, feature engineering",
            "logical reasoning": "step-by-step deduction, proofs, puzzles, math/logic problems",
            "documentation": "writing or improving README, comments, guides, or specs"
        }
        
        self.complexity_labels = {
            "trivial task": "one-liner or obvious answer",
            "simple task": "few steps, low ambiguity",
            "moderate complexity": "multiple steps or tradeoffs",
            "complex task": "many moving parts, integration concerns",
            "very complex task": "architecture-level, long investigation"
        }
        
        self.task_threshold = task_threshold
        self.complexity_threshold = complexity_threshold
    
    def _classify(self, text: str, labels_dict: Dict, multi: bool = False) -> Tuple:
        """Classify text against label descriptions"""
        labels = list(labels_dict.keys())
        descs = list(labels_dict.values())
        
        out = self.clf(text, candidate_labels=descs, multi_label=multi)
        
        # Map back to short labels
        scores = {labels[i]: float(out["scores"][i]) for i in range(len(labels))}
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary, primary_score = ordered[0]
        
        return primary, primary_score, ordered
    
    def route(self, task: str) -> Dict[str, Any]:
        """Route with calibrated thresholds"""
        start_time = time.perf_counter()
        
        t_label, t_score, t_all = self._classify(task, self.task_labels, multi=True)
        c_label, c_score, c_all = self._classify(task, self.complexity_labels, multi=False)
        
        task_abstain = (t_score < self.task_threshold)
        complexity_abstain = (c_score < self.complexity_threshold)
        
        routing_time_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "primary_type": t_label if not task_abstain else None,
            "all_types": t_all[:5],
            "complexity": c_label if not complexity_abstain else None,
            "task_confidence": t_score,
            "complexity_confidence": c_score,
            "abstain": task_abstain,
            "routing_time_ms": routing_time_ms
        }


# ============================================================================
# TIER 3: Fine-Tuned Multi-Head Router (Maximum Accuracy)
# ============================================================================

class TriHeadRouter(nn.Module):
    """Production multi-head router with separate classifiers"""
    
    def __init__(
        self,
        backbone: str = "microsoft/deberta-v3-small",
        n_task: int = 5,
        n_complex: int = 5,
        n_domain: int = 5,
        p_drop: float = 0.1
    ):
        super().__init__()
        self.enc = AutoModel.from_pretrained(backbone)
        hid = self.enc.config.hidden_size
        self.dropout = nn.Dropout(p_drop)
        self.task_head = nn.Linear(hid, n_task)
        self.complex_head = nn.Linear(hid, n_complex)
        self.domain_head = nn.Linear(hid, n_domain)
    
    def forward(self, **batch):
        x = self.enc(**batch).last_hidden_state[:, 0]  # CLS token
        x = self.dropout(x)
        return {
            "task_logits": self.task_head(x),
            "complex_logits": self.complex_head(x),
            "domain_logits": self.domain_head(x),
        }


class CalibratedFineTunedRouter:
    """Fine-tuned router with calibration and thresholds"""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = "microsoft/deberta-v3-small",
        thresholds: Optional[Dict] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = TriHeadRouter()
        
        # Load trained weights
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
        # Calibrated thresholds (should be tuned on validation set)
        self.thresholds = thresholds or {
            "task": 0.5,
            "complex": 0.4,
            "domain": 0.45
        }
        
        self.label_maps = {
            "task": ["code_generation", "analysis", "reasoning", "design", "debugging"],
            "complex": ["trivial", "simple", "medium", "complex", "very_complex"],
            "domain": ["web_development", "data_science", "systems", "algorithms", "infrastructure"]
        }
    
    @torch.no_grad()
    def route(self, text: str) -> Dict[str, Any]:
        """Route with calibrated predictions"""
        start_time = time.perf_counter()
        
        self.model.eval()
        toks = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)
        
        outs = self.model(**toks)
        probs = {
            k.replace("_logits", ""): torch.sigmoid(v).squeeze(0).cpu().numpy()
            for k, v in outs.items()
        }
        
        def pick(labels, ps, tau):
            idx = int(np.argmax(ps))
            return (labels[idx] if ps[idx] >= tau else None), float(ps[idx])
        
        task_type, task_conf = pick(self.label_maps["task"], probs["task"], self.thresholds["task"])
        complexity, complex_conf = pick(self.label_maps["complex"], probs["complex"], self.thresholds["complex"])
        domain, domain_conf = pick(self.label_maps["domain"], probs["domain"], self.thresholds["domain"])
        
        routing_time_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "task_type": task_type,
            "complexity": complexity,
            "domain": domain,
            "task_confidence": task_conf,
            "complexity_confidence": complex_conf,
            "domain_confidence": domain_conf,
            "abstain": task_type is None,
            "raw_probs": {k: v.tolist() for k, v in probs.items()},
            "routing_time_ms": routing_time_ms
        }


# ============================================================================
# UNIFIED PRODUCTION ROUTER
# ============================================================================

@dataclass
class RoutingDecision:
    """Structured routing decision"""
    task_type: Optional[str]
    complexity: str
    domain: Optional[str]
    confidence: float
    abstain: bool
    routing_time_ms: float
    routing_path: str  # which router was used
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for standardized data flow"""
        return {
            'task_type': self.task_type,
            'complexity': self.complexity,
            'domain': self.domain,
            'confidence': self.confidence,
            'abstain': self.abstain,
            'routing_time_ms': self.routing_time_ms,
            'routing_path': self.routing_path,
            'recommended_models': self.metadata.get('recommended_models', []),
            'metadata': self.metadata
        }


class UnifiedProductionRouter:
    """
    Production router with tiered strategies:
    1. Fast semantic router (primary)
    2. Zero-shot NLI (on abstentions)
    3. Fine-tuned model (when available)
    """
    
    def __init__(
        self,
        semantic_router: Optional[HardenedSemanticRouter] = None,
        zeroshot_router: Optional[CalibratedZeroShotRouter] = None,
        finetuned_router: Optional[CalibratedFineTunedRouter] = None,
        escalation_strategy: str = "waterfall"  # waterfall, parallel, or finetuned_only
    ):
        self.semantic_router = semantic_router
        self.zeroshot_router = zeroshot_router
        self.finetuned_router = finetuned_router
        self.escalation_strategy = escalation_strategy
        
        # Metrics tracking
        self.metrics = {
            "total_requests": 0,
            "semantic_routes": 0,
            "zeroshot_routes": 0,
            "finetuned_routes": 0,
            "abstentions": 0,
            "total_latency_ms": 0.0
        }
        
        # Normalization mappings for consistent output
        self.complexity_normalization = {
            # Zero-shot labels to standard labels
            "trivial task": "trivial",
            "simple task": "simple",
            "moderate complexity": "medium",
            "complex task": "complex",
            "very complex task": "very_complex",
            # Already normalized
            "trivial": "trivial",
            "simple": "simple",
            "medium": "medium",
            "complex": "complex",
            "very_complex": "very_complex"
        }
    
    def _normalize_complexity(self, complexity: str) -> str:
        """Normalize complexity labels across different routers"""
        if not complexity:
            return "medium"  # Default
        return self.complexity_normalization.get(complexity.lower(), complexity)
    
    async def route(self, task: str) -> RoutingDecision:
        """Unified routing with escalation"""
        start_time = time.perf_counter()
        self.metrics["total_requests"] += 1
        
        # Strategy: Fine-tuned only (if available)
        if self.escalation_strategy == "finetuned_only" and self.finetuned_router:
            result = self.finetuned_router.route(task)
            self.metrics["finetuned_routes"] += 1
            routing_path = "finetuned"
            
            decision = RoutingDecision(
                task_type=result.get("task_type"),
                complexity=result.get("complexity", "medium"),
                domain=result.get("domain"),
                confidence=result.get("task_confidence", 0.0),
                abstain=result.get("abstain", False),
                routing_time_ms=result.get("routing_time_ms", 0.0),
                routing_path=routing_path,
                metadata=result
            )
        
        # Strategy: Waterfall (semantic -> zeroshot -> finetuned)
        elif self.escalation_strategy == "waterfall":
            # Try semantic router first
            if self.semantic_router:
                result = self.semantic_router.route(task)
                self.metrics["semantic_routes"] += 1
                
                if not result["abstain"]:
                    decision = RoutingDecision(
                        task_type=result["cognitive_type"],
                        complexity=self._normalize_complexity(result["complexity"]),
                        domain=None,
                        confidence=result["confidence"],
                        abstain=False,
                        routing_time_ms=result["routing_time_ms"],
                        routing_path="semantic",
                        metadata=result
                    )
                else:
                    # Escalate to zero-shot
                    if self.zeroshot_router:
                        zs_result = self.zeroshot_router.route(task)
                        self.metrics["zeroshot_routes"] += 1
                        
                        # Normalize complexity from zero-shot router
                        complexity = self._normalize_complexity(
                            zs_result.get("complexity", result["complexity"])
                        )
                        
                        decision = RoutingDecision(
                            task_type=zs_result["primary_type"],
                            complexity=complexity,
                            domain=None,
                            confidence=zs_result["task_confidence"],
                            abstain=zs_result["abstain"],
                            routing_time_ms=result["routing_time_ms"] + zs_result["routing_time_ms"],
                            routing_path="semantic->zeroshot",
                            metadata={**result, **zs_result}
                        )
                    else:
                        # No escalation available
                        self.metrics["abstentions"] += 1
                        decision = RoutingDecision(
                            task_type=None,
                            complexity="medium",
                            domain=None,
                            confidence=0.0,
                            abstain=True,
                            routing_time_ms=result["routing_time_ms"],
                            routing_path="semantic->abstain",
                            metadata=result
                        )
            else:
                # No semantic router, default response
                decision = RoutingDecision(
                    task_type=None,
                    complexity="medium",
                    domain=None,
                    confidence=0.0,
                    abstain=True,
                    routing_time_ms=0.0,
                    routing_path="no_router",
                    metadata={}
                )
        
        # Track total latency
        total_time_ms = (time.perf_counter() - start_time) * 1000
        self.metrics["total_latency_ms"] += total_time_ms
        
        return decision
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics"""
        total = max(self.metrics["total_requests"], 1)
        return {
            **self.metrics,
            "avg_latency_ms": self.metrics["total_latency_ms"] / total,
            "semantic_rate": self.metrics["semantic_routes"] / total,
            "zeroshot_rate": self.metrics["zeroshot_routes"] / total,
            "finetuned_rate": self.metrics["finetuned_routes"] / total,
            "abstention_rate": self.metrics["abstentions"] / total
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_production_router(
    config_path: Optional[str] = None,
    proto_path: Optional[str] = None,
    model_path: Optional[str] = None
) -> UnifiedProductionRouter:
    """
    Create a production router with appropriate configuration
    
    Args:
        config_path: Path to router configuration JSON
        proto_path: Path to prototype embeddings
        model_path: Path to fine-tuned model weights
    """
    
    # Load configuration
    config = {
        "semantic_tau": 0.54,
        "semantic_margin": 0.06,
        "zeroshot_threshold": 0.45,
        "escalation_strategy": "waterfall"
    }
    
    if config_path:
        with open(config_path) as f:
            config.update(json.load(f))
    
    # Initialize routers
    semantic_router = None
    if proto_path:
        with open(proto_path, 'rb') as f:
            proto_map = np.load(f, allow_pickle=True).item()
        semantic_router = HardenedSemanticRouter(
            proto_map=proto_map,
            tau=config["semantic_tau"],
            margin_tau=config["semantic_margin"]
        )
    
    zeroshot_router = CalibratedZeroShotRouter(
        task_threshold=config["zeroshot_threshold"]
    )
    
    finetuned_router = None
    if model_path:
        finetuned_router = CalibratedFineTunedRouter(model_path=model_path)
    
    return UnifiedProductionRouter(
        semantic_router=semantic_router,
        zeroshot_router=zeroshot_router,
        finetuned_router=finetuned_router,
        escalation_strategy=config["escalation_strategy"]
    )


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    # Create default prototypes for semantic router
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    proto_map = {
        "code_generation": model.encode([
            "write a function to",
            "implement a class that",
            "create code for",
            "develop a program"
        ]),
        "analysis": model.encode([
            "analyze the performance",
            "investigate why",
            "examine the data",
            "review this code"
        ]),
        "debugging": model.encode([
            "fix the bug",
            "debug this error",
            "resolve the issue",
            "patch the problem"
        ])
    }
    
    # Create routers
    semantic = HardenedSemanticRouter(proto_map)
    zeroshot = CalibratedZeroShotRouter()
    
    # Create unified router
    router = UnifiedProductionRouter(
        semantic_router=semantic,
        zeroshot_router=zeroshot,
        escalation_strategy="waterfall"
    )
    
    # Test routing
    async def test():
        tasks = [
            "Write a Python function to calculate fibonacci numbers",
            "Debug why my API endpoint returns 500 errors",
            "Design a distributed cache system for 1M QPS",
            "What is 2 + 2?"
        ]
        
        for task in tasks:
            decision = await router.route(task)
            print(f"\nTask: {task[:50]}...")
            print(f"Decision: {decision.task_type} ({decision.complexity})")
            print(f"Confidence: {decision.confidence:.3f}")
            print(f"Path: {decision.routing_path}")
            print(f"Time: {decision.routing_time_ms:.1f}ms")
        
        print(f"\nMetrics: {router.get_metrics()}")
    
    asyncio.run(test())