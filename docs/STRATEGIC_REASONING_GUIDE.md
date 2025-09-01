# Strategic Orchestration Tool for LMStudio Agents

This tool provides **collaborative reasoning** and **strategic orchestration** capabilities for LMStudio agents with tool use enabled.

## 🎯 Purpose

Enable agents to:
- **Collaborate** on complex reasoning tasks
- **Access strategic orchestration** for optimal model selection 
- **Share memory** across sessions and agents
- **Build chains of thought** that persist beyond individual conversations

## 🛠️ Available Tools

### 1. `initiate_reasoning_session`
Start a new collaborative reasoning session.

**Usage:**
```json
{
  "task_description": "How should we optimize memory usage in our AI system?",
  "agent_name": "Agent-Alpha",
  "tags": ["optimization", "memory"]
}
```

**Returns:** Session ID for collaboration

### 2. `add_reasoning_step` 
Add your reasoning to an existing session.

**Usage:**
```json
{
  "session_id": "abc123ef",
  "agent_name": "Agent-Beta", 
  "reasoning_text": "Memory optimization requires considering both model size and context windows...",
  "step_type": "analysis"
}
```

**Step Types:** analysis, hypothesis, conclusion, critique, synthesis

### 3. `execute_orchestrated_decision`
Execute strategic orchestration using accumulated reasoning.

**Usage:**
```json
{
  "session_id": "abc123ef",
  "agent_name": "Agent-Alpha",
  "prompt": "Implement a memory-efficient model loader in Python",
  "task_type": "code"
}
```

**Task Types:** code, reasoning, analysis, general, creative

### 4. `get_reasoning_session`
Retrieve full session history and results.

**Usage:**
```json
{
  "session_id": "abc123ef", 
  "agent_name": "Agent-Gamma"
}
```

### 5. `search_reasoning_sessions`
Find previous reasoning sessions.

**Usage:**
```json
{
  "agent_name": "Agent-Delta",
  "query": "memory optimization",
  "tags": ["performance"],
  "status": "executed",
  "limit": 5
}
```

### 6. `get_orchestration_capabilities`
Get information about available models and strategies.

**Usage:**
```json
{
  "agent_name": "Agent-Alpha"
}
```

## 🔄 Collaboration Workflow

### Multi-Agent Reasoning Chain

1. **Agent A** initiates session with a complex problem
2. **Agent B** adds analysis and identifies key considerations  
3. **Agent C** critiques and suggests alternatives
4. **Agent D** synthesizes into unified approach
5. **Agent A** executes orchestrated decision with optimized strategy
6. **Any Agent** can later retrieve session for reference

### Example Multi-Agent Session

```
Session: "Load Balancing Strategy"

Agent Alice (analysis): "Need to consider round-robin vs least-connections algorithms..."
Agent Bob (critique): "Alice's approach misses memory constraints..."  
Agent Charlie (synthesis): "Combine both: weighted algorithm using load AND memory..."
Agent Alice (execution): Uses strategic orchestration → generates optimized code
```

## 🧠 Strategic Orchestration Features

### Automatic Strategy Selection
- **Concurrent**: Multiple models working in parallel
- **Chain**: Sequential model execution with refinement
- **Specialized**: Single best model for the task
- **Hybrid**: Combination approach

### Dynamic Resource Management
- Memory-aware model selection (15GB limit)
- Automatic model loading/unloading
- Performance-based optimization

### Context-Free Reasoning
- Token allocation based on task complexity
- Model selection based on specializations
- Temperature adjustment for task type

## 💾 Persistent Memory

### Session Storage
- All reasoning chains stored in SQLite
- Full execution results preserved
- Cross-session knowledge building

### Agent Analytics
- Track agent interactions
- Identify collaboration patterns
- Performance metrics by agent

## 🚀 Getting Started

### For Individual Agents
```json
// 1. Check capabilities
{"agent_name": "MyAgent"}

// 2. Start reasoning on your task
{
  "task_description": "Optimize API response times",
  "agent_name": "MyAgent",
  "tags": ["performance", "api"]
}

// 3. Execute when ready
{
  "session_id": "returned_id",
  "agent_name": "MyAgent", 
  "prompt": "Design caching strategy for REST API",
  "task_type": "code"
}
```

### For Collaborative Teams
```json
// Agent 1: Initiates
{"task_description": "Design microservice architecture", "agent_name": "Architect"}

// Agent 2: Adds expertise  
{"session_id": "abc123", "agent_name": "Backend", "reasoning_text": "Database considerations...", "step_type": "analysis"}

// Agent 3: Adds perspective
{"session_id": "abc123", "agent_name": "Frontend", "reasoning_text": "API interface requirements...", "step_type": "analysis"}

// Agent 1: Synthesizes and executes
{"session_id": "abc123", "agent_name": "Architect", "prompt": "Generate microservice implementation", "task_type": "code"}
```

## 📊 Benefits

### For Individual Agents
- **Optimized execution** through strategic orchestration
- **Memory efficiency** with smart resource management  
- **Quality improvement** through multi-model coordination

### For Agent Teams
- **Collective intelligence** through shared reasoning
- **Knowledge persistence** across conversations
- **Collaborative problem-solving** on complex tasks
- **Learning from others** via session retrieval

## 🔧 Technical Details

### Storage
- SQLite database for session persistence
- JSON storage for reasoning chains and results
- Indexed search by task description, tags, status

### Model Integration
- Real-time model discovery via LMStudio API
- Memory usage monitoring and management
- Performance tracking for strategy optimization

### Security
- Agent name required for all operations
- Session-based access control
- Audit trail of all interactions

---

**Ready to collaborate?** Start with `get_orchestration_capabilities` to see what's available, then `initiate_reasoning_session` for your first collaborative reasoning task!