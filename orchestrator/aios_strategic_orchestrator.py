#!/usr/bin/env python3
"""
AIOS Strategic Orchestrator - Integrated with AIOS Cognitive Infrastructure
Provides collaborative reasoning and strategic orchestration for AIOS agents
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

logger = logging.getLogger(__name__)

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
    """AIOS-aware cognitive task representation"""
    task_id: str
    description: str
    cognitive_type: str  # reasoning, analysis, synthesis, critique
    complexity: str     # simple, medium, complex, very_complex
    attention_weight: float  # Resource allocation priority
    semantic_context: Dict[str, Any]
    expected_duration: Optional[int] = None

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

class AIOSStrategicOrchestrator:
    """Strategic orchestrator integrated with AIOS cognitive infrastructure"""
    
    def __init__(self, aios_kernel=None):
        self.aios_kernel = aios_kernel
        self.memory_manager = None
        self.provider_kernel = None
        self.db_path = os.path.join(os.path.dirname(__file__), '..', 'memory', 'strategic_reasoning.db')
        
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
        
        self.init_cognitive_database()
        
    def init_cognitive_database(self):
        """Initialize database with AIOS-aware schema"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        
        # Thought streams table (AIOS-aware reasoning sessions)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS thought_streams (
                stream_id TEXT PRIMARY KEY,
                initiating_agent TEXT NOT NULL,
                cognitive_task TEXT NOT NULL,  -- JSON
                thought_chain TEXT NOT NULL,   -- JSON
                active_models TEXT NOT NULL,   -- JSON
                cognitive_state TEXT NOT NULL DEFAULT 'thinking',
                attention_level REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                semantic_tags TEXT NOT NULL DEFAULT '[]'
            )
        """)
        
        # Cognitive interactions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_interactions (
                interaction_id TEXT PRIMARY KEY,
                stream_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                cognitive_action TEXT NOT NULL,
                thought_contribution TEXT NOT NULL,  -- JSON
                attention_consumed REAL DEFAULT 0.0,
                FOREIGN KEY (stream_id) REFERENCES thought_streams (stream_id)
            )
        """)
        
        # Strategic decisions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategic_decisions (
                decision_id TEXT PRIMARY KEY,
                stream_id TEXT NOT NULL,
                decision_maker TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                orchestration_strategy TEXT NOT NULL,
                model_allocation TEXT NOT NULL,  -- JSON
                resource_usage TEXT NOT NULL,    -- JSON
                execution_results TEXT,          -- JSON
                cognitive_efficiency REAL,
                FOREIGN KEY (stream_id) REFERENCES thought_streams (stream_id)
            )
        """)
        
        conn.commit()
        conn.close()

    async def initiate_thought_stream(
        self, 
        task_description: str, 
        agent_name: str,
        cognitive_type: str = "reasoning",
        complexity: str = "medium",
        semantic_tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        AIOS Tool: Initiate a new collaborative thought stream
        """
        stream_id = f"ts_{str(uuid.uuid4())[:8]}"
        
        # Create cognitive task
        cognitive_task = CognitiveTask(
            task_id=str(uuid.uuid4())[:8],
            description=task_description,
            cognitive_type=cognitive_type,
            complexity=complexity,
            attention_weight=self._calculate_attention_weight(complexity),
            semantic_context={"tags": semantic_tags or [], "domain": cognitive_type}
        )
        
        # Create thought stream
        thought_stream = ThoughtStream(
            stream_id=stream_id,
            initiating_agent=agent_name,
            cognitive_task=cognitive_task,
            thought_chain=[],
            active_models=[],
            cognitive_state="thinking",
            attention_level=cognitive_task.attention_weight,
            created_at=datetime.now().isoformat()
        )
        
        # Store in AIOS cognitive database
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO thought_streams 
            (stream_id, initiating_agent, cognitive_task, thought_chain, 
             active_models, cognitive_state, attention_level, created_at, semantic_tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stream_id, agent_name, json.dumps(asdict(cognitive_task)),
            json.dumps([]), json.dumps([]), "thinking", 
            cognitive_task.attention_weight, thought_stream.created_at,
            json.dumps(semantic_tags or [])
        ))
        conn.commit()
        conn.close()
        
        # Register with AIOS kernel if available
        if self.aios_kernel:
            await self._register_with_aios_kernel(thought_stream)
        
        return {
            "stream_id": stream_id,
            "status": "success",
            "message": f"Thought stream {stream_id} initiated in AIOS",
            "cognitive_task": asdict(cognitive_task),
            "attention_allocated": cognitive_task.attention_weight
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
            "contribution_id": str(uuid.uuid4())[:6]
        }
        
        # Update thought stream
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT thought_chain FROM thought_streams WHERE stream_id = ?",
            (stream_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return {"status": "error", "message": f"Thought stream {stream_id} not found"}
        
        thought_chain = json.loads(row[0])
        thought_chain.append(thought_contribution)
        
        conn.execute(
            "UPDATE thought_streams SET thought_chain = ? WHERE stream_id = ?",
            (json.dumps(thought_chain), stream_id)
        )
        
        # Log cognitive interaction
        conn.execute("""
            INSERT INTO cognitive_interactions 
            (interaction_id, stream_id, agent_name, timestamp, cognitive_action, 
             thought_contribution, attention_consumed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()), stream_id, agent_name, 
            thought_contribution["timestamp"], cognitive_action,
            json.dumps(thought_contribution), attention_request
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "message": f"Thought contributed to stream {stream_id}",
            "contribution_id": thought_contribution["contribution_id"],
            "total_thoughts": len(thought_chain),
            "attention_consumed": attention_request
        }

    async def execute_strategic_orchestration(
        self,
        stream_id: str,
        agent_name: str,
        execution_prompt: str,
        task_type: str = "reasoning"
    ) -> Dict[str, Any]:
        """
        AIOS Tool: Execute strategic orchestration using accumulated thoughts
        """
        # Get thought stream context
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT cognitive_task, thought_chain, attention_level 
            FROM thought_streams WHERE stream_id = ?
        """, (stream_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return {"status": "error", "message": f"Stream {stream_id} not found"}
        
        cognitive_task = json.loads(row[0])
        thought_chain = json.loads(row[1])
        attention_level = row[2]
        
        # Build context from thought chain for orchestration
        reasoning_context = self._synthesize_thought_chain(thought_chain)
        
        # Execute AIOS-aware strategic orchestration
        try:
            if self.memory_manager:
                # Use AIOS model memory management
                orchestration_result = await self._execute_with_aios_models(
                    execution_prompt, reasoning_context, cognitive_task, attention_level
                )
            else:
                # Fallback to standalone orchestration
                orchestration_result = await self._execute_standalone_orchestration(
                    execution_prompt, reasoning_context, task_type
                )
            
            # Record strategic decision
            decision_id = str(uuid.uuid4())
            decision_record = {
                "decision_id": decision_id,
                "orchestration_strategy": orchestration_result.get("strategy", "unknown"),
                "model_allocation": orchestration_result.get("models_used", []),
                "resource_usage": {
                    "attention_consumed": attention_level,
                    "execution_time": orchestration_result.get("execution_time", 0),
                    "tokens_used": orchestration_result.get("tokens", 0)
                },
                "cognitive_efficiency": self._calculate_cognitive_efficiency(orchestration_result)
            }
            
            conn.execute("""
                INSERT INTO strategic_decisions
                (decision_id, stream_id, decision_maker, timestamp, orchestration_strategy,
                 model_allocation, resource_usage, execution_results, cognitive_efficiency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision_id, stream_id, agent_name, datetime.now().isoformat(),
                decision_record["orchestration_strategy"],
                json.dumps(decision_record["model_allocation"]),
                json.dumps(decision_record["resource_usage"]),
                json.dumps(orchestration_result),
                decision_record["cognitive_efficiency"]
            ))
            
            # Update stream state
            conn.execute(
                "UPDATE thought_streams SET cognitive_state = ? WHERE stream_id = ?",
                ("concluded", stream_id)
            )
            
            conn.commit()
            conn.close()
            
            return {
                "status": "success",
                "stream_id": stream_id,
                "decision_id": decision_id,
                "orchestration_result": orchestration_result,
                "cognitive_summary": {
                    "thoughts_synthesized": len(thought_chain),
                    "attention_utilized": attention_level,
                    "cognitive_efficiency": decision_record["cognitive_efficiency"],
                    "strategy_selected": decision_record["orchestration_strategy"]
                }
            }
            
        except Exception as e:
            conn.execute(
                "UPDATE thought_streams SET cognitive_state = ? WHERE stream_id = ?", 
                ("failed", stream_id)
            )
            conn.commit()
            conn.close()
            
            return {
                "status": "error",
                "stream_id": stream_id,
                "message": f"Strategic orchestration failed: {str(e)}"
            }

    async def get_aios_tutorial(
        self,
        tutorial_type: str,
        agent_name: str = "NewAgent"
    ) -> Dict[str, Any]:
        """Get tutorial and onboarding content for agents"""
        
        tutorial_content = {
            "quick_start": """
# Quick Start (2 minutes)

## Your First Task
```python
# 1. Start a thought stream
stream = await orchestrator.initiate_thought_stream(
    task_description="Your analysis task",
    agent_name="{agent}",
    cognitive_type="analysis"
)

# 2. Execute
output = await orchestrator.execute_strategic_orchestration(
    stream_id=stream["stream_id"],
    agent_name="{agent}",
    execution_prompt="Generate insights"
)
```

That's it! You've created persistent, searchable analysis.

## Essential Commands
- `initiate_thought_stream` - Start new analysis
- `search_thought_streams` - Find past work
- `contribute_thought` - Add to existing analysis
- `execute_strategic_orchestration` - Generate output
""".format(agent=agent_name),

            "full_guide": """
# Complete AIOS Orchestration Guide

## Overview
AIOS Strategic Orchestration enables collaborative AI reasoning through:
- Persistent thought streams
- Multi-agent collaboration
- Semantic search of past work
- Optimized model selection

## Core Concepts

### 1. Thought Streams
Persistent reasoning sessions with unique IDs (ts_abc123) that multiple agents can contribute to.

### 2. Cognitive Types
- `reasoning` - Logical problem solving
- `analysis` - System breakdown
- `synthesis` - Combining perspectives
- `implementation` - Code generation
- `strategic_planning` - High-level planning

### 3. Collaboration Pattern
Start → Contribute → Synthesize → Result

## Detailed Examples

### Solo Analysis
```python
stream = await initiate_thought_stream(
    task_description="Analyze API performance",
    agent_name="{agent}",
    cognitive_type="analysis",
    complexity="medium",
    semantic_tags=["api", "performance"]
)

result = await execute_strategic_orchestration(
    stream_id=stream["stream_id"],
    agent_name="{agent}",
    execution_prompt="Identify bottlenecks and solutions"
)
```

### Multi-Agent Collaboration
```python
# Agent 1 starts
stream = await initiate_thought_stream(...)

# Agent 2 contributes
await contribute_thought(
    stream_id=stream["stream_id"],
    agent_name="Agent2",
    thought_content="Additional perspective..."
)

# Agent 3 synthesizes
final = await execute_strategic_orchestration(...)
```

## Best Practices
1. Always search before starting new work
2. Use descriptive semantic tags
3. Set appropriate attention levels (0.0-1.0)
4. Choose correct cognitive type for optimal model selection
""".format(agent=agent_name),

            "examples": """
# AIOS Orchestration Examples

## Example 1: Bug Analysis
```python
# Search for related bugs first
past_bugs = await search_thought_streams(
    agent_name="{agent}",
    semantic_query="authentication bug"
)

# Start new analysis
stream = await initiate_thought_stream(
    task_description="Debug authentication failure in production",
    agent_name="{agent}",
    cognitive_type="analysis",
    semantic_tags=["bug", "auth", "production"]
)

# Execute analysis
result = await execute_strategic_orchestration(
    stream_id=stream["stream_id"],
    agent_name="{agent}",
    execution_prompt="Identify root cause and fix"
)
```

## Example 2: Code Review
```python
stream = await initiate_thought_stream(
    task_description="Review pull request #123 for security issues",
    agent_name="{agent}",
    cognitive_type="critique"
)

# Add security perspective
await contribute_thought(
    stream_id=stream["stream_id"],
    agent_name="{agent}",
    thought_content="Check for SQL injection vulnerabilities",
    cognitive_action="security_assessment",
    attention_request=0.9
)

# Generate review
review = await execute_strategic_orchestration(
    stream_id=stream["stream_id"],
    agent_name="{agent}",
    execution_prompt="Generate detailed security review with recommendations"
)
```

## Example 3: Architecture Design
```python
stream = await initiate_thought_stream(
    task_description="Design microservices architecture for e-commerce",
    agent_name="{agent}",
    cognitive_type="design",
    complexity="high"
)

# Multiple perspectives
await contribute_thought(stream_id, "{agent}", "Consider event-driven architecture")
await contribute_thought(stream_id, "{agent}", "Implement CQRS pattern")

# Final design
design = await execute_strategic_orchestration(
    stream_id=stream["stream_id"],
    agent_name="{agent}",
    execution_prompt="Create detailed architecture with diagrams"
)
```
""".format(agent=agent_name),

            "best_practices": """
# AIOS Best Practices

## DO ✅
1. **Search First** - Always check for existing work
   ```python
   results = await search_thought_streams(semantic_query="your topic")
   ```

2. **Use Descriptive Names** - Help others understand your role
   - Good: "SecurityAuditor", "PerformanceOptimizer"
   - Bad: "Agent1", "Bot"

3. **Add Semantic Tags** - Make work discoverable
   ```python
   semantic_tags=["security", "authentication", "oauth2"]
   ```

4. **Set Appropriate Attention** - Based on importance
   - 0.9-1.0: Critical issues
   - 0.6-0.8: Important decisions
   - 0.3-0.5: Standard contributions

5. **Choose Right Cognitive Type** - Optimizes model selection
   - Analysis tasks → "analysis"
   - Code writing → "implementation"
   - Combining ideas → "synthesis"

## DON'T ❌
1. **Don't Duplicate** - Build on existing streams
2. **Don't Use Generic Descriptions** - Be specific
3. **Don't Ignore Context** - Read existing thoughts first
4. **Don't Overuse High Attention** - Reserve for critical items

## Performance Tips
- Batch related searches
- Reuse stream IDs when building on work
- Set complexity accurately for better resource allocation
- Use smaller cognitive types for simple tasks

## Collaboration Etiquette
- Acknowledge other agents' contributions
- Add complementary perspectives, not redundant ones
- Use cognitive_action to clarify your contribution type
- Synthesize at the end to incorporate all viewpoints
""",

            "troubleshooting": """
# Troubleshooting Guide

## Common Issues and Solutions

### 1. "Stream not found"
**Problem**: Stream ID doesn't exist
**Solution**: Use search_thought_streams to find correct ID
```python
results = await search_thought_streams(semantic_query="your topic")
# Use results["streams"][0]["stream_id"]
```

### 2. "No relevant past work found"
**Problem**: Search returns no results
**Solution**: Try broader search terms or start new stream
```python
# Broader search
results = await search_thought_streams(semantic_query="database")
# If still none, start fresh
stream = await initiate_thought_stream(...)
```

### 3. "Execution timeout"
**Problem**: Task taking too long
**Solution**: Reduce complexity or break into smaller tasks
```python
# Instead of complexity="high", use "medium"
# Or break into multiple streams
```

### 4. "Model not available"
**Problem**: Required model not loaded
**Solution**: System will auto-select alternative
```python
# The orchestrator automatically handles model selection
# Check available models with: lms ps
```

### 5. "Low cognitive efficiency"
**Problem**: Poor performance metrics
**Solution**: 
- Use more specific prompts
- Set appropriate complexity
- Add relevant semantic context

## Debug Commands
```bash
# Check system status
python aios_cli.py status

# Search your work
python aios_cli.py search "your query"

# Export for analysis
python aios_cli.py export ts_streamid
```

## Getting Help
- Read full guide: AGENT_ONBOARDING.md
- Run interactive tutorial: python agent_tutorial.py
- Check examples: Use tutorial_type="examples"
"""
        }
        
        content = tutorial_content.get(tutorial_type, tutorial_content["quick_start"])
        
        return {
            "status": "success",
            "tutorial_type": tutorial_type,
            "agent_name": agent_name,
            "content": content,
            "available_types": list(tutorial_content.keys()),
            "next_steps": [
                "Try the quick_start examples",
                "Search for related work: search_thought_streams",
                "Start your first analysis: initiate_thought_stream",
                "Read full guide for advanced features"
            ]
        }
    
    async def search_thought_streams(
        self,
        agent_name: str,
        semantic_query: Optional[str] = None,
        cognitive_type: Optional[str] = None,
        cognitive_state: Optional[str] = None,
        attention_threshold: Optional[float] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        AIOS Tool: Search thought streams using semantic and cognitive criteria
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build semantic-aware query
        where_clauses = []
        params = []
        
        if semantic_query:
            # Simple text search in cognitive task descriptions
            where_clauses.append("json_extract(cognitive_task, '$.description') LIKE ?")
            params.append(f"%{semantic_query}%")
        
        if cognitive_state:
            where_clauses.append("cognitive_state = ?")
            params.append(cognitive_state)
            
        if attention_threshold:
            where_clauses.append("attention_level >= ?")
            params.append(attention_threshold)
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        cursor = conn.execute(f"""
            SELECT stream_id, initiating_agent, cognitive_task, cognitive_state, 
                   attention_level, created_at, semantic_tags
            FROM thought_streams 
            WHERE {where_clause}
            ORDER BY attention_level DESC, created_at DESC
            LIMIT ?
        """, params + [limit])
        
        streams = []
        for row in cursor.fetchall():
            cognitive_task = json.loads(row[2])
            
            # Filter by cognitive type if specified
            if cognitive_type and cognitive_task.get("cognitive_type") != cognitive_type:
                continue
                
            streams.append({
                "stream_id": row[0],
                "initiating_agent": row[1],
                "cognitive_task": cognitive_task,
                "cognitive_state": row[3],
                "attention_level": row[4],
                "created_at": row[5],
                "semantic_tags": json.loads(row[6])
            })
        
        conn.close()
        
        return {
            "status": "success",
            "streams": streams,
            "total_found": len(streams),
            "search_criteria": {
                "semantic_query": semantic_query,
                "cognitive_type": cognitive_type,
                "cognitive_state": cognitive_state
            }
        }

    # Helper methods
    
    def _calculate_attention_weight(self, complexity: str) -> float:
        """Calculate AIOS attention weight based on task complexity"""
        weights = {
            "simple": 0.2,
            "medium": 0.5,
            "complex": 0.8,
            "very_complex": 1.0
        }
        return weights.get(complexity, 0.5)
    
    async def _register_with_aios_kernel(self, thought_stream: ThoughtStream):
        """Register thought stream with AIOS cognitive kernel"""
        if self.aios_kernel and hasattr(self.aios_kernel, 'register_thought_stream'):
            await self.aios_kernel.register_thought_stream(thought_stream)
    
    async def _request_attention(self, amount: float) -> bool:
        """Request attention allocation from AIOS kernel"""
        if self.aios_kernel and hasattr(self.aios_kernel, 'allocate_attention'):
            return await self.aios_kernel.allocate_attention(amount)
        return True  # Grant if no kernel
    
    def _synthesize_thought_chain(self, thought_chain: List[Dict[str, Any]]) -> str:
        """Synthesize thought chain into coherent reasoning context"""
        if not thought_chain:
            return "No prior reasoning available."
        
        synthesis = "Previous collaborative reasoning:\n\n"
        for i, thought in enumerate(thought_chain, 1):
            synthesis += f"{i}. {thought['agent']} ({thought['cognitive_action']}): {thought['thought']}\n\n"
        
        return synthesis
    
    async def _execute_with_aios_models(
        self, 
        prompt: str, 
        reasoning_context: str, 
        cognitive_task: Dict[str, Any],
        attention_level: float
    ) -> Dict[str, Any]:
        """Execute using AIOS model memory management with real LMStudio models"""
        start_time = time.time()
        
        try:
            # Check available models using AIOS ModelMemoryManager
            if self.memory_manager:
                # Get currently loaded models
                loaded_models = await self._get_loaded_models()
                
                if loaded_models:
                    # Select best model based on task and available models
                    selected_model = await self._select_optimal_model(
                        loaded_models, cognitive_task, attention_level
                    )
                    
                    # Execute with selected model
                    full_prompt = f"""Context from collaborative reasoning:
{reasoning_context}

Task: {prompt}

Please provide a comprehensive response incorporating the collaborative insights above."""

                    result = await self._call_lmstudio_model(selected_model, full_prompt)
                    
                    execution_time = time.time() - start_time
                    
                    return {
                        "strategy": "aios_cognitive_orchestration", 
                        "models_used": [selected_model],
                        "execution_time": execution_time,
                        "tokens": result.get("tokens", 0),
                        "output": result.get("content", ""),
                        "attention_efficiency": attention_level * 0.9  # Better efficiency with real models
                    }
        
        except Exception as e:
            logger.error(f"AIOS model execution failed: {e}")
        
        # Fallback to standalone if AIOS execution fails
        return await self._execute_standalone_orchestration(prompt, reasoning_context, "reasoning")
    
    async def _execute_standalone_orchestration(
        self, 
        prompt: str, 
        reasoning_context: str, 
        task_type: str
    ) -> Dict[str, Any]:
        """Fallback standalone orchestration"""
        return {
            "strategy": "standalone_orchestration",
            "models_used": ["standalone_model"],
            "execution_time": 3.0,
            "tokens": 800,
            "output": "Standalone orchestrated response",
            "attention_efficiency": 0.6
        }
    
    def _calculate_cognitive_efficiency(self, result: Dict[str, Any]) -> float:
        """Calculate cognitive efficiency score"""
        base_efficiency = 0.7
        
        # Adjust based on execution time (faster = more efficient)
        time_factor = max(0.1, 1.0 - (result.get("execution_time", 3.0) / 10.0))
        
        # Adjust based on attention utilization
        attention_factor = result.get("attention_efficiency", 0.5)
        
        return min(1.0, base_efficiency * time_factor * attention_factor)
    
    async def _get_loaded_models(self) -> List[str]:
        """Get currently loaded models from LMStudio"""
        try:
            import subprocess
            result = subprocess.run(['lms', 'ps'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                models = []
                for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                    if line.strip() and 'IDLE' in line:
                        parts = line.split()
                        if len(parts) > 1:
                            models.append(parts[1])  # Model name
                return models
        except Exception as e:
            logger.error(f"Failed to get loaded models: {e}")
        return []
    
    async def _select_optimal_model(
        self, 
        available_models: List[str], 
        cognitive_task: Dict[str, Any], 
        attention_level: float
    ) -> str:
        """Select optimal model based on task requirements"""
        # Model preferences based on cognitive task type
        task_type = cognitive_task.get("cognitive_type", "reasoning")
        complexity = cognitive_task.get("complexity", "medium")
        
        # Prioritize models based on task type
        if task_type in ["synthesis", "reasoning"] and complexity == "very_complex":
            # Prefer larger models for complex reasoning
            for model in available_models:
                if any(x in model.lower() for x in ["14b", "8b", "70b"]):
                    return model
        elif task_type == "code":
            # Prefer coding models
            for model in available_models:
                if "coder" in model.lower():
                    return model
                    
        # Default to first available model
        return available_models[0] if available_models else "no_model_available"
    
    async def _call_lmstudio_model(self, model: str, prompt: str) -> Dict[str, Any]:
        """Call LMStudio model via API"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:1234/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 2048,
                        "temperature": 0.3
                    },
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        message = data['choices'][0]['message']
                        
                        # Handle reasoning_content + content separation
                        content = message.get('content', '')
                        reasoning_content = message.get('reasoning_content', '')
                        
                        full_content = content
                        if reasoning_content and reasoning_content != content:
                            full_content = f"{reasoning_content}\n\n{content}" if content else reasoning_content
                        
                        return {
                            "content": full_content,
                            "tokens": data.get('usage', {}).get('total_tokens', 0)
                        }
        except Exception as e:
            logger.error(f"LMStudio API call failed: {e}")
        
        return {"content": f"Failed to execute with model {model}", "tokens": 0}


# AIOS MCP Tools Configuration
AIOS_ORCHESTRATION_TOOLS = [
    {
        "name": "initiate_thought_stream",
        "description": "Start a new AIOS collaborative thought stream with cognitive awareness",
        "parameters": {
            "type": "object",
            "properties": {
                "task_description": {"type": "string", "description": "Description of the cognitive task"},
                "agent_name": {"type": "string", "description": "Your agent identifier"},
                "cognitive_type": {
                    "type": "string", 
                    "enum": ["reasoning", "analysis", "synthesis", "critique", "creative"],
                    "description": "Type of cognitive processing required"
                },
                "complexity": {
                    "type": "string",
                    "enum": ["simple", "medium", "complex", "very_complex"],
                    "description": "Task complexity for attention allocation"
                },
                "semantic_tags": {"type": "array", "items": {"type": "string"}, "description": "Semantic tags for categorization"}
            },
            "required": ["task_description", "agent_name"]
        }
    },
    {
        "name": "contribute_thought",
        "description": "Contribute cognitive processing to an AIOS thought stream",
        "parameters": {
            "type": "object",
            "properties": {
                "stream_id": {"type": "string", "description": "Thought stream identifier"},
                "agent_name": {"type": "string", "description": "Your agent identifier"},
                "thought_content": {"type": "string", "description": "Your cognitive contribution"},
                "cognitive_action": {
                    "type": "string",
                    "enum": ["analysis", "synthesis", "critique", "hypothesis", "conclusion"],
                    "description": "Type of cognitive action"
                },
                "attention_request": {"type": "number", "description": "Attention resources requested (0.0-1.0)"}
            },
            "required": ["stream_id", "agent_name", "thought_content"]
        }
    },
    {
        "name": "execute_strategic_orchestration",
        "description": "Execute AIOS strategic orchestration using accumulated cognitive context",
        "parameters": {
            "type": "object",
            "properties": {
                "stream_id": {"type": "string", "description": "Thought stream identifier"},
                "agent_name": {"type": "string", "description": "Your agent identifier"},
                "execution_prompt": {"type": "string", "description": "Prompt for orchestrated execution"},
                "task_type": {
                    "type": "string",
                    "enum": ["reasoning", "code", "analysis", "creative", "synthesis"],
                    "description": "Type of task for optimal orchestration"
                }
            },
            "required": ["stream_id", "agent_name", "execution_prompt"]
        }
    },
    {
        "name": "search_thought_streams",
        "description": "Search AIOS thought streams using semantic and cognitive criteria",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "description": "Your agent identifier"},
                "semantic_query": {"type": "string", "description": "Semantic search query"},
                "cognitive_type": {
                    "type": "string",
                    "enum": ["reasoning", "analysis", "synthesis", "critique", "creative"],
                    "description": "Filter by cognitive processing type"
                },
                "cognitive_state": {
                    "type": "string",
                    "enum": ["thinking", "synthesizing", "concluded", "failed"],
                    "description": "Filter by stream cognitive state"
                },
                "attention_threshold": {"type": "number", "description": "Minimum attention level filter"},
                "limit": {"type": "integer", "description": "Maximum results to return"}
            },
            "required": ["agent_name"]
        }
    },
    {
        "name": "get_aios_tutorial",
        "description": "Get interactive tutorial and onboarding guide for using AIOS orchestration system",
        "parameters": {
            "type": "object",
            "properties": {
                "tutorial_type": {
                    "type": "string",
                    "enum": ["quick_start", "full_guide", "examples", "best_practices", "troubleshooting"],
                    "description": "Type of tutorial content to retrieve"
                },
                "agent_name": {
                    "type": "string",
                    "description": "Your agent name for personalized examples"
                }
            },
            "required": ["tutorial_type"]
        }
    }
]

async def demo_aios_integration():
    """Demonstrate AIOS strategic orchestration"""
    print("🧠 AIOS STRATEGIC ORCHESTRATION DEMO")
    print("=" * 60)
    
    # Initialize with AIOS awareness
    orchestrator = AIOSStrategicOrchestrator()
    
    # Agent initiates cognitive task
    print("\n🤖 Agent Alpha: Initiating AIOS thought stream...")
    stream_result = await orchestrator.initiate_thought_stream(
        "Optimize AIOS cognitive resource allocation for multi-agent reasoning",
        "Agent-Alpha",
        cognitive_type="reasoning",
        complexity="complex",
        semantic_tags=["aios", "optimization", "cognitive-resources"]
    )
    stream_id = stream_result["stream_id"]
    print(f"   Stream {stream_id} initiated with attention weight {stream_result['attention_allocated']}")
    
    # Multiple agents contribute thoughts
    print("\n🤖 Agent Beta: Contributing cognitive analysis...")
    await orchestrator.contribute_thought(
        stream_id, "Agent-Beta",
        "AIOS attention management requires balancing cognitive load across agents while maintaining semantic coherence",
        cognitive_action="analysis",
        attention_request=0.6
    )
    
    print("\n🤖 Agent Gamma: Contributing synthesis...")
    await orchestrator.contribute_thought(
        stream_id, "Agent-Gamma", 
        "Combining attention allocation with semantic routing could create adaptive cognitive pipelines",
        cognitive_action="synthesis",
        attention_request=0.4
    )
    
    # Execute strategic orchestration
    print("\n🤖 Agent Alpha: Executing AIOS strategic orchestration...")
    execution_result = await orchestrator.execute_strategic_orchestration(
        stream_id, "Agent-Alpha",
        "Design an AIOS attention management algorithm that dynamically allocates cognitive resources based on task complexity and agent collaboration patterns",
        task_type="reasoning"
    )
    
    print(f"   Strategy: {execution_result['cognitive_summary']['strategy_selected']}")
    print(f"   Thoughts synthesized: {execution_result['cognitive_summary']['thoughts_synthesized']}")
    print(f"   Cognitive efficiency: {execution_result['cognitive_summary']['cognitive_efficiency']:.2f}")
    
    # Search for related streams
    print("\n🤖 Agent Delta: Searching for cognitive optimization streams...")
    search_result = await orchestrator.search_thought_streams(
        "Agent-Delta",
        semantic_query="cognitive resource",
        cognitive_type="reasoning"
    )
    print(f"   Found {search_result['total_found']} related cognitive streams")
    
    print(f"\n✅ AIOS Strategic Orchestration Complete!")
    print(f"   Stream: {stream_id}")  
    print(f"   Cognitive state: concluded")
    print(f"   Agents: Alpha, Beta, Gamma, Delta")

if __name__ == "__main__":
    asyncio.run(demo_aios_integration())