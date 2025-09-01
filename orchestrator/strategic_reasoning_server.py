#!/usr/bin/env python3
"""
MCP Server for LMStudio agents to access orchestration capabilities
Provides tools for strategic reasoning and shared memory access
"""

import asyncio
import json
import sqlite3
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Import our orchestration components
from engine_reasoning_separation import (
    IntegratedOrchestrator, 
    MockReasoningAPI,
    TaskMetadata, 
    TaskComplexity,
    ReasoningDecision
)

@dataclass
class ReasoningSession:
    """Represents a reasoning session that agents can create and access"""
    session_id: str
    created_by: str
    created_at: str
    task_description: str
    reasoning_chain: List[Dict[str, Any]]
    final_decision: Optional[ReasoningDecision]
    execution_results: Optional[Dict[str, Any]]
    status: str  # "reasoning", "decided", "executed", "failed"
    tags: List[str]

class OrchestrationMCPServer:
    """MCP Server exposing orchestration tools for LMStudio agents"""
    
    def __init__(self, db_path: str = "orchestration_memory.db"):
        self.db_path = db_path
        self.orchestrator = None
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for persistent reasoning storage"""
        conn = sqlite3.connect(self.db_path)
        
        # Reasoning sessions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_sessions (
                session_id TEXT PRIMARY KEY,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                task_description TEXT NOT NULL,
                reasoning_chain TEXT NOT NULL,  -- JSON
                final_decision TEXT,           -- JSON
                execution_results TEXT,        -- JSON
                status TEXT NOT NULL DEFAULT 'reasoning',
                tags TEXT NOT NULL DEFAULT '[]'  -- JSON
            )
        """)
        
        # Agent interactions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_interactions (
                interaction_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                action_type TEXT NOT NULL,
                action_data TEXT NOT NULL,  -- JSON
                FOREIGN KEY (session_id) REFERENCES reasoning_sessions (session_id)
            )
        """)
        
        conn.commit()
        conn.close()
        
    async def get_orchestrator(self):
        """Lazy initialize orchestrator"""
        if self.orchestrator is None:
            mock_api = MockReasoningAPI()
            self.orchestrator = IntegratedOrchestrator(mock_api)
        return self.orchestrator

    # MCP Tool Methods - These are callable by LMStudio agents
    
    async def initiate_reasoning_session(
        self, 
        task_description: str, 
        agent_name: str, 
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Tool: Initiate a new reasoning session
        
        Args:
            task_description: Description of the task to reason about
            agent_name: Name of the agent initiating the session
            tags: Optional tags for categorization
            
        Returns:
            Dict with session_id and initial session info
        """
        session_id = str(uuid.uuid4())[:8]  # Short ID for readability
        created_at = datetime.now().isoformat()
        
        session = ReasoningSession(
            session_id=session_id,
            created_by=agent_name,
            created_at=created_at,
            task_description=task_description,
            reasoning_chain=[],
            final_decision=None,
            execution_results=None,
            status="reasoning",
            tags=tags or []
        )
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO reasoning_sessions 
            (session_id, created_by, created_at, task_description, reasoning_chain, status, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, agent_name, created_at, task_description, 
            json.dumps([]), "reasoning", json.dumps(tags or [])
        ))
        conn.commit()
        conn.close()
        
        # Log agent interaction
        await self._log_interaction(session_id, agent_name, "initiate", {
            "task_description": task_description,
            "tags": tags
        })
        
        return {
            "session_id": session_id,
            "status": "success",
            "message": f"Reasoning session {session_id} initiated",
            "session": asdict(session)
        }
    
    async def add_reasoning_step(
        self, 
        session_id: str, 
        agent_name: str, 
        reasoning_text: str, 
        step_type: str = "analysis"
    ) -> Dict[str, Any]:
        """
        Tool: Add a reasoning step to an existing session
        
        Args:
            session_id: ID of the reasoning session
            agent_name: Name of the agent adding reasoning
            reasoning_text: The reasoning content
            step_type: Type of reasoning step (analysis, hypothesis, conclusion, etc.)
            
        Returns:
            Dict with success status and updated chain length
        """
        step = {
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "step_type": step_type,
            "reasoning": reasoning_text,
            "step_id": str(uuid.uuid4())[:6]
        }
        
        # Get current reasoning chain
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT reasoning_chain FROM reasoning_sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return {"status": "error", "message": f"Session {session_id} not found"}
        
        reasoning_chain = json.loads(row[0])
        reasoning_chain.append(step)
        
        # Update database
        conn.execute(
            "UPDATE reasoning_sessions SET reasoning_chain = ? WHERE session_id = ?",
            (json.dumps(reasoning_chain), session_id)
        )
        conn.commit()
        conn.close()
        
        # Log interaction
        await self._log_interaction(session_id, agent_name, "add_reasoning", {
            "step_type": step_type,
            "reasoning_length": len(reasoning_text)
        })
        
        return {
            "status": "success",
            "message": f"Reasoning step added to session {session_id}",
            "chain_length": len(reasoning_chain),
            "step_id": step["step_id"]
        }
    
    async def execute_orchestrated_decision(
        self, 
        session_id: str, 
        agent_name: str,
        prompt: str,
        task_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Tool: Execute orchestrated reasoning on a prompt and store results
        
        Args:
            session_id: ID of the reasoning session
            agent_name: Name of the agent requesting execution
            prompt: The prompt to orchestrate
            task_type: Type of task (code, reasoning, analysis, etc.)
            
        Returns:
            Dict with execution results and reasoning decision
        """
        orchestrator = await self.get_orchestrator()
        
        try:
            # Execute orchestration with reasoning
            result = await orchestrator.orchestrate_with_reasoning(prompt, task_type)
            
            # Store decision and results in session
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                UPDATE reasoning_sessions 
                SET final_decision = ?, execution_results = ?, status = ?
                WHERE session_id = ?
            """, (
                json.dumps(result.get('decision_metadata', {})),
                json.dumps(result),
                "executed",
                session_id
            ))
            conn.commit()
            conn.close()
            
            # Log interaction
            await self._log_interaction(session_id, agent_name, "execute", {
                "task_type": task_type,
                "strategy": result.get('decision_metadata', {}).get('recommended_strategy', 'unknown'),
                "tokens_allocated": result.get('decision_metadata', {}).get('tokens_allocated', 0)
            })
            
            return {
                "status": "success",
                "session_id": session_id,
                "execution_results": result,
                "reasoning_summary": {
                    "strategy": result.get('decision_metadata', {}).get('recommended_strategy'),
                    "models_used": result.get('decision_metadata', {}).get('models_selected'),
                    "confidence": result.get('strategy_confidence'),
                    "reasoning": result.get('reasoning_used')
                }
            }
            
        except Exception as e:
            # Update session with error
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "UPDATE reasoning_sessions SET status = ? WHERE session_id = ?",
                ("failed", session_id)
            )
            conn.commit()
            conn.close()
            
            return {
                "status": "error",
                "session_id": session_id,
                "message": f"Execution failed: {str(e)}"
            }
    
    async def get_reasoning_session(
        self, 
        session_id: str, 
        agent_name: str
    ) -> Dict[str, Any]:
        """
        Tool: Retrieve a reasoning session and its full history
        
        Args:
            session_id: ID of the reasoning session to retrieve
            agent_name: Name of the agent requesting access
            
        Returns:
            Dict with full session data including reasoning chain
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT session_id, created_by, created_at, task_description, 
                   reasoning_chain, final_decision, execution_results, status, tags
            FROM reasoning_sessions 
            WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {"status": "error", "message": f"Session {session_id} not found"}
        
        # Log interaction
        await self._log_interaction(session_id, agent_name, "retrieve", {})
        
        return {
            "status": "success",
            "session": {
                "session_id": row[0],
                "created_by": row[1],
                "created_at": row[2],
                "task_description": row[3],
                "reasoning_chain": json.loads(row[4]),
                "final_decision": json.loads(row[5]) if row[5] else None,
                "execution_results": json.loads(row[6]) if row[6] else None,
                "status": row[7],
                "tags": json.loads(row[8])
            }
        }
    
    async def search_reasoning_sessions(
        self, 
        agent_name: str,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Tool: Search for reasoning sessions by various criteria
        
        Args:
            agent_name: Name of the agent searching
            query: Text to search in task descriptions
            tags: Filter by tags
            status: Filter by session status
            limit: Maximum number of results
            
        Returns:
            Dict with matching sessions
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build dynamic query
        where_clauses = []
        params = []
        
        if query:
            where_clauses.append("task_description LIKE ?")
            params.append(f"%{query}%")
            
        if status:
            where_clauses.append("status = ?")
            params.append(status)
            
        # Basic query without tag filtering for now (SQLite JSON handling is complex)
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        cursor = conn.execute(f"""
            SELECT session_id, created_by, created_at, task_description, status, tags
            FROM reasoning_sessions 
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """, params + [limit])
        
        rows = cursor.fetchall()
        conn.close()
        
        sessions = []
        for row in rows:
            # Filter by tags if specified
            session_tags = json.loads(row[5])
            if tags and not any(tag in session_tags for tag in tags):
                continue
                
            sessions.append({
                "session_id": row[0],
                "created_by": row[1], 
                "created_at": row[2],
                "task_description": row[3],
                "status": row[4],
                "tags": session_tags
            })
        
        # Log interaction
        await self._log_interaction("SEARCH", agent_name, "search", {
            "query": query,
            "results_count": len(sessions)
        })
        
        return {
            "status": "success",
            "sessions": sessions,
            "total_found": len(sessions)
        }
    
    async def get_orchestration_capabilities(self, agent_name: str) -> Dict[str, Any]:
        """
        Tool: Get information about available orchestration capabilities
        
        Args:
            agent_name: Name of the agent requesting info
            
        Returns:
            Dict with capability information
        """
        orchestrator = await self.get_orchestrator()
        system_state = orchestrator.engine.get_system_state()
        
        return {
            "status": "success",
            "capabilities": {
                "available_models": system_state["available_models"],
                "memory_limit": system_state["memory_limit"],
                "current_memory_usage": system_state["current_memory_usage"],
                "supported_strategies": [
                    "concurrent", "chain", "specialized", "hybrid"
                ],
                "supported_task_types": [
                    "code", "reasoning", "analysis", "general", "creative"
                ]
            },
            "usage_instructions": {
                "workflow": [
                    "1. Call initiate_reasoning_session() with your task",
                    "2. Add reasoning steps with add_reasoning_step()",
                    "3. Execute orchestrated decision with execute_orchestrated_decision()",
                    "4. Retrieve results anytime with get_reasoning_session()",
                    "5. Search past sessions with search_reasoning_sessions()"
                ],
                "collaboration": "Multiple agents can contribute to the same session by adding reasoning steps"
            }
        }
    
    async def _log_interaction(
        self, 
        session_id: str, 
        agent_name: str, 
        action_type: str, 
        action_data: Dict[str, Any]
    ):
        """Log agent interaction for analytics"""
        interaction_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO agent_interactions 
            (interaction_id, session_id, agent_name, timestamp, action_type, action_data)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            interaction_id, session_id, agent_name, 
            datetime.now().isoformat(), action_type, json.dumps(action_data)
        ))
        conn.commit()
        conn.close()

# MCP Server Tool Definitions
MCP_TOOLS = [
    {
        "name": "initiate_reasoning_session",
        "description": "Start a new collaborative reasoning session that multiple agents can contribute to",
        "parameters": {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Description of the task or problem to reason about"
                },
                "agent_name": {
                    "type": "string", 
                    "description": "Your agent name for identification"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorizing this session"
                }
            },
            "required": ["task_description", "agent_name"]
        }
    },
    {
        "name": "add_reasoning_step", 
        "description": "Add your reasoning contribution to an existing session",
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "ID of the reasoning session"
                },
                "agent_name": {
                    "type": "string",
                    "description": "Your agent name"
                },
                "reasoning_text": {
                    "type": "string", 
                    "description": "Your reasoning content or analysis"
                },
                "step_type": {
                    "type": "string",
                    "enum": ["analysis", "hypothesis", "conclusion", "critique", "synthesis"],
                    "description": "Type of reasoning step"
                }
            },
            "required": ["session_id", "agent_name", "reasoning_text"]
        }
    },
    {
        "name": "execute_orchestrated_decision",
        "description": "Execute strategic orchestration using the accumulated reasoning",
        "parameters": {
            "type": "object", 
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "ID of the reasoning session"
                },
                "agent_name": {
                    "type": "string",
                    "description": "Your agent name"
                },
                "prompt": {
                    "type": "string",
                    "description": "The prompt to orchestrate execution on"
                },
                "task_type": {
                    "type": "string",
                    "enum": ["code", "reasoning", "analysis", "general", "creative"],
                    "description": "Type of task for optimal orchestration"
                }
            },
            "required": ["session_id", "agent_name", "prompt"]
        }
    },
    {
        "name": "get_reasoning_session",
        "description": "Retrieve a reasoning session with full history and results",
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string", 
                    "description": "ID of the reasoning session to retrieve"
                },
                "agent_name": {
                    "type": "string",
                    "description": "Your agent name"
                }
            },
            "required": ["session_id", "agent_name"]
        }
    },
    {
        "name": "search_reasoning_sessions",
        "description": "Search for previous reasoning sessions by various criteria",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Your agent name"
                },
                "query": {
                    "type": "string",
                    "description": "Text to search for in task descriptions"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags"
                },
                "status": {
                    "type": "string",
                    "enum": ["reasoning", "decided", "executed", "failed"],
                    "description": "Filter by session status"
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of results"
                }
            },
            "required": ["agent_name"]
        }
    },
    {
        "name": "get_orchestration_capabilities",
        "description": "Get information about available orchestration models and strategies",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Your agent name"
                }
            },
            "required": ["agent_name"]
        }
    }
]

async def demo_agent_workflow():
    """Demonstrate how LMStudio agents would use these tools"""
    print("🤖 LMSTUDIO AGENT ORCHESTRATION DEMO")
    print("=" * 60)
    
    server = OrchestrationMCPServer()
    
    # Agent 1: Initiates a reasoning session
    print("\n👤 Agent Alice: Initiating reasoning session...")
    session_result = await server.initiate_reasoning_session(
        "How should we implement load balancing in our AI orchestration system?",
        "Alice",
        ["architecture", "performance"]
    )
    session_id = session_result["session_id"]
    print(f"   Session {session_id} created")
    
    # Agent 1: Adds initial reasoning
    print("\n👤 Agent Alice: Adding initial analysis...")
    await server.add_reasoning_step(
        session_id, 
        "Alice",
        "Load balancing is critical for preventing model overload. We should consider round-robin, least-connections, and weighted algorithms.",
        "analysis"
    )
    
    # Agent 2: Joins the session and adds reasoning
    print("\n👤 Agent Bob: Joining session and adding critique...")
    await server.add_reasoning_step(
        session_id,
        "Bob", 
        "Alice's analysis is good, but we also need to consider memory constraints. A model might be idle but using lots of memory.",
        "critique"
    )
    
    # Agent 3: Adds synthesis
    print("\n👤 Agent Charlie: Adding synthesis...")
    await server.add_reasoning_step(
        session_id,
        "Charlie",
        "Combining both perspectives: implement weighted round-robin that factors both current load AND memory usage.",
        "synthesis" 
    )
    
    # Agent 1: Executes the orchestrated decision
    print("\n👤 Agent Alice: Executing orchestrated decision...")
    exec_result = await server.execute_orchestrated_decision(
        session_id,
        "Alice", 
        "Implement a load balancing algorithm that considers both CPU load and memory usage for model selection in Python",
        "code"
    )
    
    print(f"   Strategy chosen: {exec_result['reasoning_summary']['strategy']}")
    print(f"   Models used: {exec_result['reasoning_summary']['models_used']}")
    print(f"   Confidence: {exec_result['reasoning_summary']['confidence']:.2f}")
    
    # Agent 4: Later retrieves the session
    print("\n👤 Agent David: Retrieving session for reference...")
    session_data = await server.get_reasoning_session(session_id, "David")
    reasoning_chain = session_data["session"]["reasoning_chain"]
    print(f"   Found {len(reasoning_chain)} reasoning steps from {len(set(step['agent'] for step in reasoning_chain))} agents")
    
    # Agent 4: Searches for similar sessions
    print("\n👤 Agent David: Searching for load balancing sessions...")
    search_result = await server.search_reasoning_sessions(
        "David",
        query="load balancing",
        tags=["architecture"]
    )
    print(f"   Found {search_result['total_found']} related sessions")
    
    print(f"\n✅ Multi-agent collaborative reasoning complete!")
    print(f"   Session ID: {session_id}")
    print(f"   Agents involved: Alice, Bob, Charlie, David")
    print(f"   Reasoning chain: {len(reasoning_chain)} steps")
    print(f"   Status: {session_data['session']['status']}")

if __name__ == "__main__":
    asyncio.run(demo_agent_workflow())