# 🎯 How New Agents Become Familiar with AIOS Orchestration

## For LMStudio Agents with MCP Tool Access

### Step 1: Use the Tutorial Tool (30 seconds)
The fastest way to learn is through the built-in tutorial tool:

```json
{
  "name": "get_aios_tutorial",
  "arguments": {
    "tutorial_type": "quick_start",
    "agent_name": "YourAgentName"
  }
}
```

This returns personalized examples and instructions in seconds.

### Step 2: Progressive Learning Path

1. **Quick Start** (2 minutes)
   ```json
   {"tutorial_type": "quick_start"}
   ```
   - Learn basic commands
   - Run your first analysis
   - Understand core concepts

2. **Examples** (5 minutes)
   ```json
   {"tutorial_type": "examples"}
   ```
   - Bug analysis workflow
   - Code review process
   - Architecture design patterns

3. **Best Practices** (3 minutes)
   ```json
   {"tutorial_type": "best_practices"}
   ```
   - Do's and don'ts
   - Performance tips
   - Collaboration etiquette

4. **Full Guide** (10 minutes)
   ```json
   {"tutorial_type": "full_guide"}
   ```
   - Complete documentation
   - All features explained
   - Advanced usage patterns

5. **Troubleshooting** (as needed)
   ```json
   {"tutorial_type": "troubleshooting"}
   ```
   - Common issues
   - Debug commands
   - Getting help

## For Human Users or Direct Python Access

### Option 1: Interactive Tutorial (15 minutes)
```bash
cd C:\Users\sscar\claude-workspace\workshop\aios\orchestrator
python agent_tutorial.py
```
- Hands-on learning experience
- Step-by-step guidance
- Real tasks with immediate feedback

### Option 2: Quick Demo (2 minutes)
```bash
python agent_tutorial.py --demo
```
- Non-interactive demonstration
- See the system in action
- Understand capabilities quickly

### Option 3: CLI Exploration (5 minutes)
```bash
# Check system status
python aios_cli.py status

# Search existing work
python aios_cli.py search "topic"

# Start new analysis
python aios_cli.py analyze "your task"
```

### Option 4: Read Documentation
- **AGENT_ONBOARDING.md** - Comprehensive guide
- **access_guide.md** - How to access past work
- **agent_instructions.md** - Usage examples

## Learning by Doing - First Task Template

For any new agent, here's a template for your first real task:

```python
# 1. Get tutorial first
tutorial = await get_aios_tutorial("quick_start", "YourName")

# 2. Search for related work
existing = await search_thought_streams(
    agent_name="YourName",
    semantic_query="your task keywords"
)

# 3. Start your analysis
if not existing["streams"]:
    stream = await initiate_thought_stream(
        task_description="Your specific task",
        agent_name="YourName",
        cognitive_type="analysis",  # or appropriate type
        semantic_tags=["tag1", "tag2"]
    )
else:
    # Build on existing work
    stream_id = existing["streams"][0]["stream_id"]
    await contribute_thought(
        stream_id=stream_id,
        agent_name="YourName",
        thought_content="Your insights"
    )

# 4. Generate results
result = await execute_strategic_orchestration(
    stream_id=stream["stream_id"],
    agent_name="YourName",
    execution_prompt="What you want as output"
)
```

## Quick Reference Card for New Agents

### Essential Tools (in order of importance)
1. **get_aios_tutorial** - Learn the system
2. **search_thought_streams** - Find existing work
3. **initiate_thought_stream** - Start new analysis
4. **execute_strategic_orchestration** - Generate output
5. **contribute_thought** - Add to existing work

### Cognitive Types to Choose
- `"reasoning"` - Problem solving
- `"analysis"` - Breaking down systems
- `"synthesis"` - Combining ideas
- `"implementation"` - Writing code
- `"critique"` - Review and feedback

### Complexity Levels
- `"low"` - Simple, quick tasks
- `"medium"` - Standard analysis
- `"high"` - Complex, multi-faceted
- `"very_high"` - Enterprise-level

## Success Metrics for Familiarization

An agent is considered familiar when they can:
✅ Use the tutorial tool to get help
✅ Search for existing work before starting new
✅ Create a thought stream with proper parameters
✅ Execute orchestration to generate output
✅ Contribute to another agent's stream

## Time to Productivity

- **Minimal (2 min)**: Use tutorial tool → Run first task
- **Basic (10 min)**: Interactive tutorial → Understand concepts
- **Proficient (30 min)**: Full guide → Best practices → Real tasks
- **Expert (2 hours)**: All tutorials → Multiple collaborations → Optimization

## Getting Stuck?

If an agent gets stuck, they should:
1. Use `get_aios_tutorial("troubleshooting")`
2. Search for similar successful analyses
3. Start with simpler complexity levels
4. Check system status with CLI

## The Power of the Tutorial Tool

The `get_aios_tutorial` tool is the key innovation that makes onboarding instant:
- **No external documentation needed** - Everything is in the tool
- **Personalized examples** - Uses the agent's name
- **Progressive disclosure** - Start simple, go deeper as needed
- **Always available** - Can be called anytime during work
- **Context-aware help** - Different types for different needs

This means any LMStudio agent can become productive with AIOS orchestration in literally seconds by calling the tutorial tool!