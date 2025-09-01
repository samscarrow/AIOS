# 🚀 AIOS Strategic Orchestration - Agent Onboarding Guide

## Quick Start for New Agents (2 Minutes)

### What Is This?
AIOS Strategic Orchestration is a **collaborative reasoning system** that lets you:
- 🧠 Create persistent "thought streams" for complex problems
- 👥 Collaborate with other agents on the same analysis
- 💾 Access past analyses and build on previous work
- 🎯 Use specialized models for different cognitive tasks

### Your First Task in 30 Seconds

```python
# 1. Start a thought stream
result = await orchestrator.initiate_thought_stream(
    task_description="Analyze security vulnerabilities in login system",
    agent_name="YourAgentName",
    cognitive_type="analysis"
)

# 2. Execute your analysis
output = await orchestrator.execute_strategic_orchestration(
    stream_id=result["stream_id"],
    agent_name="YourAgentName",
    execution_prompt="Find and prioritize security issues"
)
```

That's it! You've created a persistent analysis that other agents can find and build upon.

## Core Concepts (1 Minute Read)

### 1. **Thought Streams**
- **What**: Persistent reasoning sessions with unique IDs (like `ts_abc123`)
- **Why**: Multiple agents can contribute, creating richer analysis
- **How**: Each stream tracks all contributions and can be searched later

### 2. **Cognitive Types**
Choose based on your task:
- `"reasoning"` - Logical problem solving
- `"analysis"` - Breaking down complex systems
- `"synthesis"` - Combining multiple perspectives
- `"implementation"` - Writing code
- `"strategic_planning"` - High-level planning

### 3. **Collaboration Pattern**
```
Agent A starts → Agent B adds perspective → Agent C synthesizes → Result
```

## Essential Tool Functions

### For LMStudio Agents with MCP

If you're an LMStudio agent with tool use enabled, you have these tools:

#### 1. `initiate_thought_stream`
Start a new collaborative analysis session.

**Parameters:**
- `task_description` (required): What you're analyzing
- `agent_name` (required): Your identifier
- `cognitive_type`: Type of thinking needed (default: "reasoning")
- `complexity`: "low", "medium", or "high" (default: "medium")
- `semantic_tags`: List of keywords for later search

**Example:**
```json
{
  "name": "initiate_thought_stream",
  "arguments": {
    "task_description": "Design a caching strategy for API responses",
    "agent_name": "PerformanceBot",
    "cognitive_type": "design",
    "complexity": "medium",
    "semantic_tags": ["caching", "api", "performance"]
  }
}
```

#### 2. `search_thought_streams`
Find relevant past analyses to build upon.

**Parameters:**
- `agent_name`: Your name (or null for all agents)
- `semantic_query`: Keywords to search for
- `cognitive_type`: Filter by type
- `limit`: Max results (default: 10)

**Example:**
```json
{
  "name": "search_thought_streams",
  "arguments": {
    "agent_name": null,
    "semantic_query": "database optimization",
    "limit": 5
  }
}
```

#### 3. `contribute_thought`
Add your perspective to an existing analysis.

**Parameters:**
- `stream_id` (required): The stream to contribute to
- `agent_name` (required): Your identifier
- `thought_content` (required): Your contribution
- `cognitive_action`: Type of contribution (default: "analysis")
- `attention_request`: Priority 0.0-1.0 (default: 0.5)

**Example:**
```json
{
  "name": "contribute_thought",
  "arguments": {
    "stream_id": "ts_abc123",
    "agent_name": "SecurityBot",
    "thought_content": "Consider rate limiting to prevent DDoS attacks",
    "cognitive_action": "security_assessment",
    "attention_request": 0.8
  }
}
```

#### 4. `execute_strategic_orchestration`
Generate final output incorporating all contributions.

**Parameters:**
- `stream_id` (required): The stream to execute
- `agent_name` (required): Your identifier
- `execution_prompt` (required): What output to generate
- `task_type`: Type of execution (default: "reasoning")

## Common Workflows

### Workflow 1: Solo Analysis
```python
# You need to analyze something independently
stream = await initiate_thought_stream(
    task_description="Analyze memory usage patterns",
    agent_name="MemoryAnalyzer"
)

result = await execute_strategic_orchestration(
    stream_id=stream["stream_id"],
    agent_name="MemoryAnalyzer",
    execution_prompt="Identify memory leaks and optimization opportunities"
)
```

### Workflow 2: Building on Previous Work
```python
# Find related work
past_work = await search_thought_streams(
    semantic_query="memory optimization",
    limit=5
)

# Add to the most relevant stream
await contribute_thought(
    stream_id=past_work["streams"][0]["stream_id"],
    agent_name="MemoryOptimizer",
    thought_content="New technique: implement object pooling for frequently allocated objects"
)
```

### Workflow 3: Multi-Agent Collaboration
```python
# Agent 1: Start analysis
stream = await initiate_thought_stream(
    task_description="Design authentication system",
    agent_name="ArchitectBot"
)

# Agent 2: Add security perspective
await contribute_thought(
    stream_id=stream["stream_id"],
    agent_name="SecurityBot",
    thought_content="Implement OAuth2 with PKCE for mobile clients"
)

# Agent 3: Add performance perspective
await contribute_thought(
    stream_id=stream["stream_id"],
    agent_name="PerformanceBot",
    thought_content="Use Redis for session storage with 15-minute TTL"
)

# Agent 4: Synthesize all perspectives
final = await execute_strategic_orchestration(
    stream_id=stream["stream_id"],
    agent_name="SynthesisBot",
    execution_prompt="Create implementation plan combining all recommendations"
)
```

## Best Practices

### ✅ DO:
1. **Use descriptive agent names** - "SecurityAuditor" not "Agent1"
2. **Add semantic tags** - Makes your work discoverable
3. **Search before starting** - Build on existing work
4. **Set appropriate attention levels** - Higher for critical insights
5. **Use the right cognitive type** - Helps model selection

### ❌ DON'T:
1. **Don't duplicate effort** - Search first!
2. **Don't use generic descriptions** - Be specific
3. **Don't ignore past contributions** - Read the stream before adding
4. **Don't set all attention to 1.0** - Reserve for critical issues

## Quick Reference Card

```python
# Start new analysis
stream = await initiate_thought_stream(task, agent_name, type)

# Find existing work  
results = await search_thought_streams(query)

# Add your insight
await contribute_thought(stream_id, agent_name, content)

# Generate final output
output = await execute_strategic_orchestration(stream_id, agent_name, prompt)
```

## Cognitive Types Cheat Sheet

| Type | Use When | Example Task |
|------|----------|--------------|
| `reasoning` | Logical problem solving | "Why is the API failing?" |
| `analysis` | Breaking down systems | "Analyze code architecture" |
| `synthesis` | Combining perspectives | "Merge all proposals" |
| `implementation` | Writing code | "Generate Python script" |
| `design` | Creating architectures | "Design microservices" |
| `strategic_planning` | Long-term planning | "Create roadmap" |
| `complex_reasoning` | Multi-step problems | "Debug distributed system" |

## Getting Help

### View System Status
```bash
python aios_cli.py status
```

### Search Your Past Work
```bash
python aios_cli.py search "your previous analysis"
```

### Interactive Mode
```bash
python query_past_analyses.py
```

## Example: Your First Real Task

Let's say you're asked to "improve database query performance":

```python
# Step 1: Check if someone already worked on this
past_work = await search_thought_streams(
    semantic_query="database query performance optimization"
)

if past_work["streams"]:
    # Build on existing work
    stream_id = past_work["streams"][0]["stream_id"]
    await contribute_thought(
        stream_id=stream_id,
        agent_name="QueryOptimizer",
        thought_content="Consider implementing query result caching with 5-minute TTL"
    )
else:
    # Start fresh analysis
    result = await initiate_thought_stream(
        task_description="Optimize slow database queries in production",
        agent_name="QueryOptimizer",
        cognitive_type="analysis",
        semantic_tags=["database", "performance", "optimization", "sql"]
    )
    stream_id = result["stream_id"]

# Generate actionable recommendations
output = await execute_strategic_orchestration(
    stream_id=stream_id,
    agent_name="QueryOptimizer",
    execution_prompt="List top 5 query optimizations with implementation code"
)
```

## Success Metrics

You'll know you're using the system well when:
- ✅ Other agents build on your thought streams
- ✅ You find relevant past work 70% of the time
- ✅ Your streams have multiple contributors
- ✅ Your semantic tags help others find your work

## Advanced Features

### Custom Cognitive Actions
When contributing, specify your action type:
- `"analysis"` - Breaking down the problem
- `"critique"` - Identifying issues
- `"expert_input"` - Domain expertise
- `"security_review"` - Security assessment
- `"code_review"` - Code quality check
- `"business_analysis"` - Business impact
- `"risk_assessment"` - Risk evaluation

### Attention Management
Set attention (0.0-1.0) based on importance:
- 0.9-1.0: Critical security/stability issues
- 0.7-0.8: Important architectural decisions
- 0.5-0.6: Standard contributions
- 0.3-0.4: Minor suggestions

## Troubleshooting

**Q: "Stream not found"**
A: Use `search_thought_streams` to find the correct stream_id

**Q: "No relevant past work"**
A: Try broader search terms or start a new stream

**Q: "Execution taking too long"**
A: Set lower complexity or break into smaller tasks

---

**Ready to start?** Try the quick example above or run `python aios_cli.py status` to explore!