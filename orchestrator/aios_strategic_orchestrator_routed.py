#!/usr/bin/env python3
"""
AIOS Strategic Orchestrator with Integrated Production Router
Provides intelligent task routing and collaborative reasoning for AIOS agents
"""

import asyncio
import json
import sqlite3
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import os
import numpy as np

logger = logging.getLogger(__name__)

# Import production router
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'unified'))
try:
    from production_task_router import (
        create_production_router,
        UnifiedProductionRouter,
        HardenedSemanticRouter
    )
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False
    logger.warning("Production router not available - falling back to manual routing")

# Import AIOS infrastructure
unified_path = os.path.join(os.path.dirname(__file__), '..', 'unified')
sys.path.insert(0, unified_path)

try:
    # Import AIOS unified modules
    import model_memory_manager
    ModelMemoryManager = model_memory_manager.ModelMemoryManager
    
    # Try to import provider kernel
    try:
        from providers.universal_provider_kernel import UniversalProviderKernel
        from config import get_config
    except ImportError:
        UniversalProviderKernel = None
        get_config = None
    
    AIOS_AVAILABLE = True
    print("🧠 AIOS unified modules detected")
    
except ImportError as e:
    # Fallback if AIOS modules not available
    print(f"Warning: AIOS modules not found ({e}), using standalone mode")
    ModelMemoryManager = None
    UniversalProviderKernel = None
    get_config = None
    AIOS_AVAILABLE = False

@dataclass
class CognitiveTask:
    """AIOS-aware cognitive task with router-enhanced intelligence"""
    task_id: str
    description: str
    cognitive_type: str  # From router decision
    complexity: str      # From router decision
    attention_weight: float  # From router decision
    semantic_context: Dict[str, Any]
    expected_duration: Optional[int] = None
    recommended_models: List[str] = None  # From router
    routing_confidence: float = 0.0  # Router confidence
    routing_path: str = "manual"  # How task was routed

@dataclass
class ThoughtStream:
    """Represents an AIOS thought stream for collaborative reasoning"""
    stream_id: str
    initiating_agent: str
    cognitive_task: CognitiveTask
    thought_chain: List[Dict[str, Any]]
    active_models: List[str]
    cognitive_state: str  # "thinking", "synthesizing", "concluded"
    attention_level: float
    created_at: str
    routing_metadata: Dict[str, Any] = None  # Router decision details

class AIOSStrategicOrchestratorRouted:
    """Strategic orchestrator with integrated production task routing"""
    
    def __init__(self, aios_kernel=None, router_config: Optional[Dict] = None):
        self.aios_kernel = aios_kernel
        self.memory_manager = None
        self.provider_kernel = None
        self.db_path = os.path.join(os.path.dirname(__file__), '..', 'memory', 'strategic_reasoning.db')
        
        # Initialize production router
        self.router = None
        if ROUTER_AVAILABLE:
            self._initialize_router(router_config)
        
        # Initialize AIOS components if available
        if AIOS_AVAILABLE:
            try:
                self.memory_manager = ModelMemoryManager()
                if get_config:
                    config = get_config()
                    self.provider_kernel = UniversalProviderKernel(config)
                print("✅ AIOS infrastructure loaded successfully")
            except Exception as e:
                print(f"Warning: Could not initialize AIOS components: {e}")
        
        # Model selection strategy based on task routing
        self.model_selection_strategy = {
            'code_generation': ['qwen/qwen2.5-coder-14b', 'deepseek-coder-v2'],
            'analysis': ['qwen/qwen2.5-14b', 'llama-3.2-3b'],
            'debugging': ['qwen/qwen2.5-coder-14b', 'codestral-22b'],
            'system_design': ['qwen/qwen2.5-14b', 'llama-3.2-8b'],
            'data_analysis': ['qwen/qwen2.5-14b', 'llama-3.2-3b'],
            'logical_reasoning': ['qwen/qwen3-4b-thinking', 'llama-3.2-8b'],
            'documentation': ['llama-3.2-3b', 'qwen/qwen2.5-7b']
        }
        
        self.init_cognitive_database()
    
    def _initialize_router(self, router_config: Optional[Dict] = None):
        """Initialize the production task router"""
        try:
            # Check for calibrated router files
            proto_path = os.path.join(os.path.dirname(__file__), '..', 'router_calibration', 'prototypes.npy')
            config_path = os.path.join(os.path.dirname(__file__), '..', 'router_calibration', 'router_config.json')
            
            # Use calibrated router if available
            if os.path.exists(proto_path) and os.path.exists(config_path):
                print("📊 Loading calibrated router...")
                self.router = create_production_router(
                    proto_path=proto_path,
                    config_path=config_path
                )
            else:
                # Create default router with standard prototypes
                print("📊 Initializing default router...")
                self._create_default_router()
            
            print("✅ Production router initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize router: {e}")
            self.router = None
    
    def _create_default_router(self):
        """Create a default router with standard prototypes"""
        from sentence_transformers import SentenceTransformer
        
        # Create default prototypes
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
            ]),
            "system_design": model.encode([
                "design a system",
                "architect a solution",
                "plan the infrastructure",
                "create a scalable"
            ]),
            "reasoning": model.encode([
                "explain why",
                "understand the logic",
                "solve this problem",
                "figure out how"
            ])
        }
        
        # Create router with default thresholds
        semantic_router = HardenedSemanticRouter(
            proto_map=proto_map,
            tau=0.5,
            margin_tau=0.06
        )
        
        self.router = UnifiedProductionRouter(
            semantic_router=semantic_router,
            escalation_strategy="waterfall"
        )
        
    def init_cognitive_database(self):
        """Initialize database with router-aware schema"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        
        # Drop old tables to ensure clean schema
        conn.execute("DROP TABLE IF EXISTS thought_streams")
        conn.execute("DROP TABLE IF EXISTS cognitive_interactions") 
        conn.execute("DROP TABLE IF EXISTS router_performance")
        
        # Enhanced thought streams table with routing metadata
        conn.execute("""
            CREATE TABLE thought_streams (
                stream_id TEXT PRIMARY KEY,
                initiating_agent TEXT,
                cognitive_task TEXT,
                thought_chain TEXT,
                active_models TEXT,
                cognitive_state TEXT,
                attention_level REAL,
                created_at TEXT,
                semantic_tags TEXT,
                routing_metadata TEXT,
                routing_confidence REAL,
                routing_path TEXT
            )
        """)
        
        # Cognitive interactions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_interactions (
                interaction_id TEXT PRIMARY KEY,
                stream_id TEXT,
                agent_name TEXT,
                timestamp TEXT,
                cognitive_action TEXT,
                thought_contribution TEXT,
                attention_consumed REAL,
                model_used TEXT,
                routing_assisted BOOLEAN,
                FOREIGN KEY (stream_id) REFERENCES thought_streams (stream_id)
            )
        """)
        
        # Router performance tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS router_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                task_description TEXT,
                routed_type TEXT,
                routed_complexity TEXT,
                confidence REAL,
                routing_time_ms REAL,
                routing_path TEXT,
                abstained BOOLEAN
            )
        """)
        
        conn.commit()
        conn.close()

    async def initiate_thought_stream(
        self, 
        task_description: str, 
        agent_name: str,
        cognitive_type: Optional[str] = None,  # Optional - router will determine
        complexity: Optional[str] = None,       # Optional - router will determine
        semantic_tags: Optional[List[str]] = None,
        use_router: bool = True  # Allow override for compatibility
    ) -> Dict[str, Any]:
        """
        AIOS Tool: Initiate a new collaborative thought stream with intelligent routing
        """
        stream_id = f"ts_{str(uuid.uuid4())[:8]}"
        routing_metadata = {}
        
        # Use production router for intelligent task analysis
        if use_router and self.router:
            try:
                start_time = time.perf_counter()
                routing_decision = await self.router.route(task_description)
                routing_time = (time.perf_counter() - start_time) * 1000
                
                # Use router decisions if not manually specified
                if not cognitive_type:
                    cognitive_type = routing_decision.task_type or "reasoning"
                if not complexity:
                    complexity = routing_decision.complexity
                
                # Get recommended models
                recommended_models = self.model_selection_strategy.get(
                    routing_decision.task_type,
                    ['qwen/qwen2.5-7b']  # Default
                )
                
                routing_metadata = {
                    "router_type": routing_decision.task_type,
                    "router_complexity": routing_decision.complexity,
                    "confidence": routing_decision.confidence,
                    "routing_time_ms": routing_time,
                    "routing_path": routing_decision.routing_path,
                    "abstained": routing_decision.abstain,
                    "alternatives": routing_decision.metadata.get("alternatives", [])
                }
                
                # Log router performance
                self._log_router_performance(
                    task_description,
                    routing_decision,
                    routing_time
                )
                
                print(f"🎯 Router decision: {cognitive_type} ({complexity}) - confidence: {routing_decision.confidence:.3f}")
                
            except Exception as e:
                logger.warning(f"Router failed, using defaults: {e}")
                cognitive_type = cognitive_type or "reasoning"
                complexity = complexity or "medium"
                recommended_models = None
        else:
            # Fallback to manual specification
            cognitive_type = cognitive_type or "reasoning"
            complexity = complexity or "medium"
            recommended_models = None
        
        # Create cognitive task with router enhancements
        cognitive_task = CognitiveTask(
            task_id=str(uuid.uuid4())[:8],
            description=task_description,
            cognitive_type=cognitive_type,
            complexity=complexity,
            attention_weight=self._calculate_attention_weight(complexity),
            semantic_context={"tags": semantic_tags or [], "domain": cognitive_type},
            recommended_models=recommended_models,
            routing_confidence=routing_metadata.get("confidence", 0.0),
            routing_path=routing_metadata.get("routing_path", "manual")
        )
        
        # Create thought stream
        thought_stream = ThoughtStream(
            stream_id=stream_id,
            initiating_agent=agent_name,
            cognitive_task=cognitive_task,
            thought_chain=[],
            active_models=recommended_models[:2] if recommended_models else [],
            cognitive_state="thinking",
            attention_level=cognitive_task.attention_weight,
            created_at=datetime.now().isoformat(),
            routing_metadata=routing_metadata
        )
        
        # Store in database with routing metadata
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO thought_streams 
            (stream_id, initiating_agent, cognitive_task, thought_chain, 
             active_models, cognitive_state, attention_level, created_at, 
             semantic_tags, routing_metadata, routing_confidence, routing_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stream_id, agent_name, json.dumps(asdict(cognitive_task)),
            json.dumps([]), json.dumps(thought_stream.active_models), "thinking", 
            cognitive_task.attention_weight, thought_stream.created_at,
            json.dumps(semantic_tags or []),
            json.dumps(routing_metadata),
            routing_metadata.get("confidence", 0.0),
            routing_metadata.get("routing_path", "manual")
        ))
        conn.commit()
        conn.close()
        
        # Prepare models if recommended
        if self.memory_manager and recommended_models:
            asyncio.create_task(
                self.memory_manager.ensure_models_available(
                    recommended_models[:2],
                    priority=int(cognitive_task.attention_weight * 10)
                )
            )
        
        return {
            "stream_id": stream_id,
            "status": "success",
            "message": f"Thought stream {stream_id} initiated with intelligent routing",
            "cognitive_task": asdict(cognitive_task),
            "attention_allocated": cognitive_task.attention_weight,
            "recommended_models": recommended_models,
            "routing_metadata": routing_metadata
        }
    
    def _calculate_attention_weight(self, complexity: str) -> float:
        """Calculate attention weight based on complexity"""
        weights = {
            "trivial": 0.2,
            "simple": 0.3,
            "medium": 0.5,
            "complex": 0.7,
            "very_complex": 0.9
        }
        return weights.get(complexity, 0.5)
    
    def _log_router_performance(self, task: str, decision: Any, routing_time: float):
        """Log router performance for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO router_performance 
                (timestamp, task_description, routed_type, routed_complexity, 
                 confidence, routing_time_ms, routing_path, abstained)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                task[:500],  # Truncate long tasks
                decision.task_type,
                decision.complexity,
                decision.confidence,
                routing_time,
                decision.routing_path,
                decision.abstain
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log router performance: {e}")
    
    async def get_router_analytics(self) -> Dict[str, Any]:
        """Get analytics on router performance"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent router performance
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_routes,
                AVG(confidence) as avg_confidence,
                AVG(routing_time_ms) as avg_routing_time,
                SUM(CASE WHEN abstained = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as abstention_rate,
                COUNT(DISTINCT routed_type) as unique_types,
                COUNT(DISTINCT routed_complexity) as unique_complexities
            FROM router_performance
            WHERE datetime(timestamp) > datetime('now', '-1 day')
        """)
        
        stats = cursor.fetchone()
        
        # Get routing path distribution
        cursor = conn.execute("""
            SELECT routing_path, COUNT(*) as count
            FROM router_performance
            WHERE datetime(timestamp) > datetime('now', '-1 day')
            GROUP BY routing_path
        """)
        
        path_dist = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get type distribution
        cursor = conn.execute("""
            SELECT routed_type, COUNT(*) as count
            FROM router_performance
            WHERE datetime(timestamp) > datetime('now', '-1 day')
            AND routed_type IS NOT NULL
            GROUP BY routed_type
        """)
        
        type_dist = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        # Get current router metrics if available
        router_metrics = {}
        if self.router:
            router_metrics = self.router.get_metrics()
        
        return {
            "last_24h_stats": {
                "total_routes": stats[0] if stats else 0,
                "avg_confidence": stats[1] if stats else 0,
                "avg_routing_time_ms": stats[2] if stats else 0,
                "abstention_rate": stats[3] if stats else 0,
                "unique_types": stats[4] if stats else 0,
                "unique_complexities": stats[5] if stats else 0
            },
            "routing_path_distribution": path_dist,
            "task_type_distribution": type_dist,
            "current_session_metrics": router_metrics
        }
    
    async def contribute_thought(
        self,
        stream_id: str,
        agent_name: str,
        thought_content: str,
        cognitive_action: str = "analysis",
        attention_request: float = 0.5
    ) -> Dict[str, Any]:
        """
        AIOS Tool: Contribute a thought to an existing stream
        """
        # Get stream info to check recommended models
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT cognitive_task, active_models FROM thought_streams WHERE stream_id = ?",
            (stream_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return {"status": "error", "message": f"Thought stream {stream_id} not found"}
        
        cognitive_task = json.loads(row[0])
        active_models = json.loads(row[1])
        
        # Check AIOS attention availability
        if self.aios_kernel:
            attention_granted = await self._request_attention(attention_request)
            if not attention_granted:
                return {
                    "status": "attention_denied",
                    "message": "Insufficient cognitive resources available"
                }
        
        thought_contribution = {
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "cognitive_action": cognitive_action,
            "thought": thought_content,
            "attention_consumed": attention_request,
            "contribution_id": str(uuid.uuid4())[:6],
            "model_recommendation": active_models[0] if active_models else None
        }
        
        # Update thought stream
        cursor = conn.execute(
            "SELECT thought_chain FROM thought_streams WHERE stream_id = ?",
            (stream_id,)
        )
        row = cursor.fetchone()
        
        thought_chain = json.loads(row[0])
        thought_chain.append(thought_contribution)
        
        conn.execute(
            "UPDATE thought_streams SET thought_chain = ? WHERE stream_id = ?",
            (json.dumps(thought_chain), stream_id)
        )
        
        # Log cognitive interaction with routing assistance flag
        conn.execute("""
            INSERT INTO cognitive_interactions 
            (interaction_id, stream_id, agent_name, timestamp, cognitive_action, 
             thought_contribution, attention_consumed, model_used, routing_assisted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()), stream_id, agent_name, 
            thought_contribution["timestamp"], cognitive_action,
            json.dumps(thought_contribution), attention_request,
            active_models[0] if active_models else None,
            True  # This orchestrator always uses routing assistance
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "message": f"Thought contributed to stream {stream_id}",
            "contribution_id": thought_contribution["contribution_id"],
            "total_thoughts": len(thought_chain),
            "attention_consumed": attention_request,
            "recommended_model": active_models[0] if active_models else None
        }
    
    async def _request_attention(self, amount: float) -> bool:
        """Request attention from AIOS kernel"""
        # Stub for AIOS kernel integration
        return True
    
    async def _register_with_aios_kernel(self, thought_stream: ThoughtStream):
        """Register thought stream with AIOS kernel"""
        # Stub for AIOS kernel integration
        pass


# Example usage and testing
async def demo_routed_orchestrator():
    """Demonstrate the router-enhanced orchestrator"""
    
    orchestrator = AIOSStrategicOrchestratorRouted()
    
    # Test various task types
    test_tasks = [
        ("Write a Python function to calculate fibonacci numbers", "CodeAgent"),
        ("Debug why my API returns 500 errors intermittently", "DebugAgent"),
        ("Design a distributed cache system for 1M QPS", "ArchitectAgent"),
        ("Analyze the performance bottlenecks in our database", "AnalystAgent"),
        ("Explain why gradient descent works for optimization", "TeacherAgent")
    ]
    
    print("\n" + "="*60)
    print("AIOS Strategic Orchestrator with Production Routing")
    print("="*60)
    
    for task, agent in test_tasks:
        print(f"\n📝 Task: {task[:60]}...")
        print(f"👤 Agent: {agent}")
        
        # Initiate thought stream with routing
        result = await orchestrator.initiate_thought_stream(
            task_description=task,
            agent_name=agent,
            # Note: NOT specifying cognitive_type or complexity - let router decide
            semantic_tags=["demo", "test"]
        )
        
        print(f"✅ Stream ID: {result['stream_id']}")
        print(f"🎯 Routed Type: {result['cognitive_task']['cognitive_type']}")
        print(f"📊 Complexity: {result['cognitive_task']['complexity']}")
        print(f"⚡ Attention: {result['cognitive_task']['attention_weight']:.2f}")
        print(f"🤖 Models: {', '.join(result['recommended_models'][:2]) if result['recommended_models'] else 'None'}")
        print(f"📈 Confidence: {result['routing_metadata'].get('confidence', 0):.3f}")
        print(f"🛤️ Path: {result['routing_metadata'].get('routing_path', 'unknown')}")
    
    # Get router analytics
    print("\n" + "="*60)
    print("Router Analytics")
    print("="*60)
    
    analytics = await orchestrator.get_router_analytics()
    print(f"Session Metrics: {json.dumps(analytics['current_session_metrics'], indent=2)}")


if __name__ == "__main__":
    asyncio.run(demo_routed_orchestrator())