# AIOS Strategic Orchestration - Access Guide

## How to Access the System and Retrieve Past Analyses

### 1. **Direct Python API Access**
```python
from aios_strategic_orchestrator import AIOSStrategicOrchestrator

# Initialize the orchestrator
orchestrator = AIOSStrategicOrchestrator()

# Search for past thought streams
results = await orchestrator.search_thought_streams(
    agent_name="ChiefArchitect",  # Optional: filter by agent
    semantic_query="recursive self-improvement recommendations",
    cognitive_type="strategic_synthesis",
    limit=10
)
```

### 2. **MCP Server for LMStudio Agents**
The system runs as an MCP server that LMStudio agents can access:

```bash
# Start the MCP server
cd C:\Users\sscar\claude-workspace\workshop\aios\orchestrator
python mcp_server.py
```

Then in LMStudio, agents can use these tools:
- `initiate_thought_stream` - Start new analysis
- `search_thought_streams` - Find past analyses
- `contribute_thought` - Add to existing analysis
- `execute_strategic_orchestration` - Get final synthesis

### 3. **Saved Analysis Files**
All analyses are automatically saved with timestamps:

```bash
# Location of saved results
C:\Users\sscar\claude-workspace\workshop\aios\orchestrator\
├── aios_test_results_*.json          # Test scenario results
├── recursive_improvement_recommendations.json  # Latest recommendations
└── thought_streams/                   # All thought stream data
```

### 4. **Query Tool for Past Analyses**