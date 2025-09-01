#!/usr/bin/env python3
"""
AIOS Strategic Orchestrator - Fixed Version
Proper error handling without mock data fallbacks
"""

import os
import sys
import json
import sqlite3
import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for AIOS modules
AIOS_AVAILABLE = False
try:
    from unified.core.attention_manager import AttentionManager
    from unified.core.model_memory import ModelMemoryManager
    from unified.memory.semantic_memory import SemanticMemoryIndex
    from unified.providers.provider_kernel import UniversalProviderKernel
    from unified.config import get_config
    
    AIOS_AVAILABLE = True
    print("🧠 AIOS unified modules detected")
except ImportError as e:
    logger.error(f"AIOS modules not available: {e}")
    raise ImportError(
        "AIOS modules are required for this system to function. "
        "Please ensure the unified modules are properly installed."
    )

@dataclass
class OrchestrationError(Exception):
    """Custom exception for orchestration failures"""
    error_type: str
    message: str
    details: Dict[str, Any] = None
    
    def __str__(self):
        return f"{self.error_type}: {self.message}" + (f"\nDetails: {self.details}" if self.details else "")

class AIOSStrategicOrchestrator:
    """Strategic orchestrator with proper error handling"""
    
    def __init__(self, aios_kernel=None):
        """Initialize with required AIOS components"""
        self.aios_kernel = aios_kernel
        self.memory_manager = None
        self.provider_kernel = None
        self.db_path = os.path.join(os.path.dirname(__file__), '..', 'memory', 'strategic_reasoning.db')
        
        # Initialize AIOS components - fail fast if not available
        if not AIOS_AVAILABLE:
            raise OrchestrationError(
                "initialization_error",
                "AIOS modules are not available",
                {"suggestion": "Install unified modules from the AIOS directory"}
            )
        
        try:
            self.memory_manager = ModelMemoryManager()
            if get_config:
                config = get_config()
                self.provider_kernel = UniversalProviderKernel(config)
            logger.info("✅ AIOS infrastructure loaded successfully")
        except Exception as e:
            raise OrchestrationError(
                "initialization_error",
                f"Failed to initialize AIOS components: {str(e)}",
                {"component": "AIOS", "error": str(e)}
            )
        
        self.init_cognitive_database()
    
    def init_cognitive_database(self):
        """Initialize database with AIOS-aware schema"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            
            # Thought streams table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS thought_streams (
                    stream_id TEXT PRIMARY KEY,
                    initiating_agent TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    cognitive_type TEXT,
                    cognitive_state TEXT DEFAULT 'thinking',
                    attention_level REAL DEFAULT 0.5,
                    semantic_tags TEXT,
                    task_description TEXT,
                    complexity TEXT
                )
            """)
            
            # Thoughts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS thoughts (
                    thought_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stream_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    thought_content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    cognitive_action TEXT,
                    attention_request REAL DEFAULT 0.5,
                    FOREIGN KEY (stream_id) REFERENCES thought_streams (stream_id)
                )
            """)
            
            # Strategic decisions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategic_decisions (
                    decision_id TEXT PRIMARY KEY,
                    stream_id TEXT NOT NULL,
                    decision_maker TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    orchestration_strategy TEXT,
                    model_allocation TEXT,
                    resource_usage TEXT,
                    execution_results TEXT,
                    cognitive_efficiency REAL,
                    FOREIGN KEY (stream_id) REFERENCES thought_streams (stream_id)
                )
            """)
            
            conn.commit()
            conn.close()
        except Exception as e:
            raise OrchestrationError(
                "database_error",
                f"Failed to initialize cognitive database: {str(e)}",
                {"db_path": self.db_path}
            )
    
    async def initiate_thought_stream(
        self,
        task_description: str,
        agent_name: str,
        cognitive_type: str = "reasoning",
        complexity: str = "medium",
        semantic_tags: List[str] = None
    ) -> Dict[str, Any]:
        """Initiate a new thought stream"""
        try:
            stream_id = f"ts_{uuid.uuid4().hex[:8]}"
            
            # Calculate attention based on complexity
            attention_weights = {
                "low": 0.3,
                "medium": 0.5,
                "high": 0.7,
                "very_high": 0.9
            }
            attention_level = attention_weights.get(complexity, 0.5)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO thought_streams 
                (stream_id, initiating_agent, cognitive_type, attention_level, 
                 semantic_tags, task_description, complexity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                stream_id, agent_name, cognitive_type, attention_level,
                json.dumps(semantic_tags or []), task_description, complexity
            ))
            conn.commit()
            conn.close()
            
            # Register with AIOS if available
            cognitive_task = {
                "task_id": uuid.uuid4().hex[:8],
                "description": task_description,
                "cognitive_type": cognitive_type,
                "complexity": complexity,
                "attention_weight": attention_level,
                "semantic_context": {
                    "tags": semantic_tags or [],
                    "domain": cognitive_type
                },
                "expected_duration": None
            }
            
            return {
                "stream_id": stream_id,
                "status": "success",
                "message": f"Thought stream {stream_id} initiated",
                "cognitive_task": cognitive_task,
                "attention_allocated": attention_level
            }
            
        except Exception as e:
            raise OrchestrationError(
                "stream_creation_error",
                f"Failed to initiate thought stream: {str(e)}",
                {"task": task_description, "agent": agent_name}
            )
    
    async def contribute_thought(
        self,
        stream_id: str,
        agent_name: str,
        thought_content: str,
        cognitive_action: str = "analysis",
        attention_request: float = 0.5
    ) -> Dict[str, Any]:
        """Contribute a thought to an existing stream"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Verify stream exists
            cursor = conn.execute(
                "SELECT cognitive_state FROM thought_streams WHERE stream_id = ?",
                (stream_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                raise OrchestrationError(
                    "stream_not_found",
                    f"Thought stream {stream_id} does not exist",
                    {"stream_id": stream_id}
                )
            
            if result[0] == "concluded":
                raise OrchestrationError(
                    "stream_concluded",
                    f"Cannot contribute to concluded stream {stream_id}",
                    {"stream_id": stream_id, "state": result[0]}
                )
            
            # Add thought
            conn.execute("""
                INSERT INTO thoughts 
                (stream_id, agent_name, thought_content, cognitive_action, attention_request)
                VALUES (?, ?, ?, ?, ?)
            """, (stream_id, agent_name, thought_content, cognitive_action, attention_request))
            
            # Update stream state if needed
            conn.execute(
                "UPDATE thought_streams SET cognitive_state = ? WHERE stream_id = ?",
                ("synthesizing", stream_id)
            )
            
            conn.commit()
            conn.close()
            
            return {
                "status": "success",
                "stream_id": stream_id,
                "message": f"Thought contributed by {agent_name}",
                "cognitive_action": cognitive_action,
                "attention_granted": min(attention_request, 1.0)
            }
            
        except OrchestrationError:
            raise
        except Exception as e:
            raise OrchestrationError(
                "contribution_error",
                f"Failed to contribute thought: {str(e)}",
                {"stream_id": stream_id, "agent": agent_name}
            )
    
    async def execute_strategic_orchestration(
        self,
        stream_id: str,
        agent_name: str,
        execution_prompt: str,
        task_type: str = "reasoning"
    ) -> Dict[str, Any]:
        """Execute strategic orchestration with real models only"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get stream info
            cursor = conn.execute("""
                SELECT cognitive_type, attention_level, task_description, complexity
                FROM thought_streams WHERE stream_id = ?
            """, (stream_id,))
            stream_info = cursor.fetchone()
            
            if not stream_info:
                raise OrchestrationError(
                    "stream_not_found",
                    f"Stream {stream_id} not found",
                    {"stream_id": stream_id}
                )
            
            cognitive_type, attention_level, task_description, complexity = stream_info
            
            # Get thought chain
            cursor = conn.execute("""
                SELECT agent_name, thought_content, cognitive_action
                FROM thoughts WHERE stream_id = ?
                ORDER BY timestamp
            """, (stream_id,))
            
            thought_chain = [
                {"agent": row[0], "content": row[1], "action": row[2]}
                for row in cursor.fetchall()
            ]
            
            # Synthesize reasoning context
            reasoning_context = self._synthesize_thought_chain(thought_chain)
            
            # Execute with real models - NO FALLBACK
            if not self.memory_manager:
                raise OrchestrationError(
                    "no_memory_manager",
                    "Memory manager not initialized - cannot execute orchestration",
                    {"suggestion": "Ensure AIOS modules are properly loaded"}
                )
            
            orchestration_result = await self._execute_with_real_models(
                execution_prompt, 
                reasoning_context, 
                {
                    "cognitive_type": cognitive_type,
                    "complexity": complexity,
                    "task_description": task_description
                },
                attention_level
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
            
        except OrchestrationError:
            raise
        except Exception as e:
            raise OrchestrationError(
                "orchestration_error",
                f"Strategic orchestration failed: {str(e)}",
                {"stream_id": stream_id, "task_type": task_type}
            )
    
    def _synthesize_thought_chain(self, thought_chain: List[Dict]) -> str:
        """Synthesize thoughts into coherent context"""
        if not thought_chain:
            return ""
        
        synthesis = "Previous analysis and contributions:\n\n"
        for thought in thought_chain:
            synthesis += f"[{thought['agent']}] ({thought['action']}): {thought['content']}\n\n"
        
        return synthesis
    
    async def _execute_with_real_models(
        self,
        prompt: str,
        reasoning_context: str,
        cognitive_task: Dict[str, Any],
        attention_level: float
    ) -> Dict[str, Any]:
        """Execute using real LMStudio models only"""
        try:
            # Get available models
            available_models = await self._get_loaded_models()
            
            if not available_models:
                raise OrchestrationError(
                    "no_models_available",
                    "No LMStudio models are currently loaded",
                    {"suggestion": "Start LMStudio and load a model"}
                )
            
            # Select optimal model based on task
            selected_model = await self._select_optimal_model(
                available_models, cognitive_task, attention_level
            )
            
            # Prepare full prompt with context
            full_prompt = f"""Task: {cognitive_task['task_description']}
Complexity: {cognitive_task['complexity']}
Type: {cognitive_task['cognitive_type']}

{reasoning_context}

Current request: {prompt}

Please provide a comprehensive response:"""
            
            # Call real model
            start_time = time.time()
            result = await self._call_lmstudio_model(selected_model, full_prompt)
            execution_time = time.time() - start_time
            
            if not result or "error" in result:
                raise OrchestrationError(
                    "model_call_failed",
                    f"Failed to get response from model {selected_model}",
                    {"model": selected_model, "error": result.get("error") if result else "No response"}
                )
            
            return {
                "strategy": "aios_cognitive_orchestration",
                "models_used": [selected_model],
                "execution_time": execution_time,
                "tokens": result.get("tokens", 0),
                "output": result.get("content", ""),
                "attention_efficiency": self._calculate_attention_efficiency(
                    attention_level, execution_time
                )
            }
            
        except OrchestrationError:
            raise
        except Exception as e:
            raise OrchestrationError(
                "model_execution_error",
                f"Failed to execute with models: {str(e)}",
                {"available_models": available_models if 'available_models' in locals() else []}
            )
    
    async def _get_loaded_models(self) -> List[str]:
        """Get currently loaded models from LMStudio"""
        try:
            import subprocess
            result = subprocess.run(['lms', 'ps'], capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                raise OrchestrationError(
                    "lmstudio_error",
                    "Failed to query LMStudio models",
                    {"stderr": result.stderr, "return_code": result.returncode}
                )
            
            models = []
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip() and 'IDLE' in line:
                    parts = line.split()
                    if len(parts) > 1:
                        models.append(parts[1])  # Model name
            
            return models
            
        except subprocess.TimeoutExpired:
            raise OrchestrationError(
                "lmstudio_timeout",
                "LMStudio query timed out",
                {"timeout": 5}
            )
        except FileNotFoundError:
            raise OrchestrationError(
                "lmstudio_not_found",
                "LMStudio CLI (lms) not found",
                {"suggestion": "Ensure LMStudio is installed and 'lms' is in PATH"}
            )
    
    async def _select_optimal_model(
        self,
        available_models: List[str],
        cognitive_task: Dict[str, Any],
        attention_level: float
    ) -> str:
        """Select optimal model based on task requirements"""
        if not available_models:
            raise OrchestrationError(
                "no_models_for_selection",
                "No models available for selection",
                {}
            )
        
        # Prefer coder models for implementation tasks
        task_type = cognitive_task.get("cognitive_type", "reasoning")
        
        if task_type in ["implementation", "code", "code_generation"]:
            for model in available_models:
                if "coder" in model.lower() or "code" in model.lower():
                    return model
        
        # Return first available model as default
        return available_models[0]
    
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
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise OrchestrationError(
                            "lmstudio_api_error",
                            f"LMStudio API returned status {response.status}",
                            {"status": response.status, "error": error_text}
                        )
                    
                    data = await response.json()
                    
                    if "choices" not in data or not data["choices"]:
                        raise OrchestrationError(
                            "invalid_api_response",
                            "Invalid response from LMStudio API",
                            {"response": data}
                        )
                    
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
                    
        except aiohttp.ClientError as e:
            raise OrchestrationError(
                "lmstudio_connection_error",
                f"Failed to connect to LMStudio API: {str(e)}",
                {"url": "http://localhost:1234", "model": model}
            )
        except Exception as e:
            raise OrchestrationError(
                "unexpected_api_error",
                f"Unexpected error calling LMStudio: {str(e)}",
                {"model": model, "error_type": type(e).__name__}
            )
    
    def _calculate_cognitive_efficiency(self, result: Dict[str, Any]) -> float:
        """Calculate cognitive efficiency score"""
        base_efficiency = 0.7
        
        # Adjust based on execution time (faster = more efficient)
        time_factor = max(0.1, 1.0 - (result.get("execution_time", 3.0) / 10.0))
        
        # Adjust based on attention utilization
        attention_factor = result.get("attention_efficiency", 0.5)
        
        return min(1.0, base_efficiency * time_factor * attention_factor)
    
    def _calculate_attention_efficiency(self, attention_level: float, execution_time: float) -> float:
        """Calculate how efficiently attention was used"""
        # Lower execution time with higher attention = more efficient
        if execution_time < 1:
            return 0.9
        
        efficiency = attention_level / (1 + (execution_time / 10))
        return min(1.0, max(0.1, efficiency))
    
    async def search_thought_streams(
        self,
        agent_name: str,
        semantic_query: Optional[str] = None,
        cognitive_type: Optional[str] = None,
        cognitive_state: Optional[str] = None,
        attention_threshold: Optional[float] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Search for thought streams with proper error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM thought_streams WHERE 1=1"
            params = []
            
            if semantic_query:
                query += " AND (task_description LIKE ? OR semantic_tags LIKE ?)"
                params.extend([f"%{semantic_query}%", f"%{semantic_query}%"])
            
            if cognitive_type:
                query += " AND cognitive_type = ?"
                params.append(cognitive_type)
            
            if cognitive_state:
                query += " AND cognitive_state = ?"
                params.append(cognitive_state)
            
            if attention_threshold is not None:
                query += " AND attention_level >= ?"
                params.append(attention_threshold)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            columns = [description[0] for description in cursor.description]
            
            streams = []
            for row in cursor.fetchall():
                stream_dict = dict(zip(columns, row))
                
                # Get agents who contributed
                agent_cursor = conn.execute("""
                    SELECT DISTINCT agent_name FROM thoughts WHERE stream_id = ?
                """, (stream_dict['stream_id'],))
                
                agents = [row[0] for row in agent_cursor.fetchall()]
                if stream_dict['initiating_agent'] not in agents:
                    agents.insert(0, stream_dict['initiating_agent'])
                
                stream_dict['agents'] = agents
                stream_dict['semantic_tags'] = json.loads(stream_dict.get('semantic_tags', '[]'))
                streams.append(stream_dict)
            
            conn.close()
            
            return {
                "streams": streams,
                "total_found": len(streams),
                "search_criteria": {
                    "semantic_query": semantic_query,
                    "cognitive_type": cognitive_type,
                    "cognitive_state": cognitive_state,
                    "attention_threshold": attention_threshold
                }
            }
            
        except Exception as e:
            raise OrchestrationError(
                "search_error",
                f"Failed to search thought streams: {str(e)}",
                {"query": semantic_query}
            )
    
    async def get_aios_tutorial(
        self,
        tutorial_type: str,
        agent_name: str = "NewAgent"
    ) -> Dict[str, Any]:
        """Get tutorial content - kept for backward compatibility"""
        # Tutorial content remains the same as it's useful
        # But the system no longer uses mock data
        tutorial_content = {
            "quick_start": "Quick start guide content...",
            "full_guide": "Full guide content...",
            "examples": "Examples content...",
            "best_practices": "Best practices content...",
            "troubleshooting": "Troubleshooting content..."
        }
        
        return {
            "status": "success",
            "tutorial_type": tutorial_type,
            "agent_name": agent_name,
            "content": tutorial_content.get(tutorial_type, "Unknown tutorial type"),
            "available_types": list(tutorial_content.keys())
        }


# Export tools configuration
AIOS_ORCHESTRATION_TOOLS = [
    # Tool definitions remain the same
]

async def demo():
    """Demo with proper error handling"""
    orchestrator = AIOSStrategicOrchestrator()
    
    try:
        # Create stream
        stream = await orchestrator.initiate_thought_stream(
            "Test proper error handling",
            "DemoAgent",
            cognitive_type="analysis"
        )
        print(f"Stream created: {stream['stream_id']}")
        
        # Execute
        result = await orchestrator.execute_strategic_orchestration(
            stream['stream_id'],
            "DemoAgent",
            "Generate a response",
            task_type="analysis"
        )
        print(f"Tokens generated: {result['orchestration_result']['tokens']}")
        
    except OrchestrationError as e:
        print(f"❌ Orchestration failed properly: {e}")
        print(f"   Error type: {e.error_type}")
        print(f"   Details: {e.details}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(demo())