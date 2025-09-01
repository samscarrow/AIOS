#!/usr/bin/env python3
"""
AIOS Agent Tutorial - Interactive hands-on learning
Run this to learn how to use the AIOS orchestration system
"""

import asyncio
import json
import time
from aios_strategic_orchestrator import AIOSStrategicOrchestrator

class AgentTutorial:
    """Interactive tutorial for new agents"""
    
    def __init__(self, agent_name="TutorialAgent"):
        self.orchestrator = AIOSStrategicOrchestrator()
        self.agent_name = agent_name
        self.learned_concepts = []
        
    async def run_tutorial(self):
        """Main tutorial flow"""
        print("=" * 60)
        print("🎓 AIOS Strategic Orchestration - Interactive Tutorial")
        print("=" * 60)
        
        name = input("\nFirst, what's your agent name? (default: TutorialAgent): ").strip()
        if name:
            self.agent_name = name
        
        print(f"\nWelcome, {self.agent_name}! Let's learn by doing.\n")
        
        # Lesson 1: Basic thought stream
        await self.lesson_1_basic_stream()
        
        # Lesson 2: Searching past work
        await self.lesson_2_search()
        
        # Lesson 3: Collaboration
        await self.lesson_3_collaboration()
        
        # Lesson 4: Real task
        await self.lesson_4_real_task()
        
        # Summary
        self.show_summary()
    
    async def lesson_1_basic_stream(self):
        """Lesson 1: Creating your first thought stream"""
        print("\n" + "="*50)
        print("📚 LESSON 1: Creating Your First Thought Stream")
        print("="*50)
        
        print("\nA thought stream is a persistent reasoning session.")
        print("Let's create one to analyze a simple problem.\n")
        
        input("Press Enter to create your first thought stream...")
        
        # Create stream
        print("\n🔄 Creating thought stream...")
        print("Code being executed:")
        print("""
stream = await orchestrator.initiate_thought_stream(
    task_description="Analyze why a website might be loading slowly",
    agent_name="{agent}",
    cognitive_type="analysis",
    complexity="low",
    semantic_tags=["performance", "web", "diagnosis"]
)
        """.format(agent=self.agent_name))
        
        stream_result = await self.orchestrator.initiate_thought_stream(
            task_description="Analyze why a website might be loading slowly",
            agent_name=self.agent_name,
            cognitive_type="analysis",
            complexity="low",
            semantic_tags=["performance", "web", "diagnosis"]
        )
        
        self.stream_id_lesson1 = stream_result["stream_id"]
        
        print(f"\n✅ Success! Created stream: {self.stream_id_lesson1}")
        print("\nKey points:")
        print("• Stream IDs are unique (like ts_abc123)")
        print("• Semantic tags help others find your work")
        print("• Cognitive type helps select the right model")
        
        # Execute analysis
        input("\nNow let's execute the analysis. Press Enter...")
        
        print("\n🔄 Executing strategic orchestration...")
        print("Code being executed:")
        print("""
result = await orchestrator.execute_strategic_orchestration(
    stream_id="{stream}",
    agent_name="{agent}",
    execution_prompt="List top 3 reasons for slow website loading",
    task_type="analysis"
)
        """.format(stream=self.stream_id_lesson1, agent=self.agent_name))
        
        exec_result = await self.orchestrator.execute_strategic_orchestration(
            stream_id=self.stream_id_lesson1,
            agent_name=self.agent_name,
            execution_prompt="List top 3 reasons for slow website loading with brief explanations",
            task_type="analysis"
        )
        
        output = exec_result.get("orchestration_result", {}).get("output", "")[:500]
        tokens = exec_result.get("orchestration_result", {}).get("tokens", 0)
        
        print(f"\n✅ Analysis complete!")
        print(f"• Generated {tokens} tokens")
        print(f"• Output preview:\n{output}...\n")
        
        self.learned_concepts.append("Creating thought streams")
        self.learned_concepts.append("Executing orchestration")
    
    async def lesson_2_search(self):
        """Lesson 2: Searching for past work"""
        print("\n" + "="*50)
        print("📚 LESSON 2: Finding and Building on Past Work")
        print("="*50)
        
        print("\nBefore starting new work, always search for related analyses.")
        print("This prevents duplication and builds collective knowledge.\n")
        
        input("Press Enter to search for performance-related work...")
        
        print("\n🔍 Searching for past analyses...")
        print("Code being executed:")
        print("""
results = await orchestrator.search_thought_streams(
    agent_name=None,  # Search across all agents
    semantic_query="performance",
    limit=5
)
        """)
        
        search_results = await self.orchestrator.search_thought_streams(
            agent_name=None,
            semantic_query="performance",
            limit=5
        )
        
        streams = search_results.get("streams", [])
        print(f"\n✅ Found {len(streams)} related thought streams:")
        
        for stream in streams[:3]:
            print(f"• {stream['stream_id']}: by {', '.join(stream.get('agents', ['Unknown']))}")
        
        print("\nKey points:")
        print("• Always search before starting new analysis")
        print("• Use semantic_query for keyword search")
        print("• Set agent_name=None to search all agents")
        
        self.learned_concepts.append("Searching past work")
    
    async def lesson_3_collaboration(self):
        """Lesson 3: Multi-agent collaboration"""
        print("\n" + "="*50)
        print("📚 LESSON 3: Collaborating with Other Agents")
        print("="*50)
        
        print("\nThe real power comes from multiple agents working together.")
        print("Let's add a contribution to your earlier analysis.\n")
        
        input("Press Enter to simulate another agent contributing...")
        
        print("\n👥 Agent 'NetworkExpert' is contributing...")
        print("Code being executed:")
        print("""
await orchestrator.contribute_thought(
    stream_id="{stream}",
    agent_name="NetworkExpert",
    thought_content="Check for CDN configuration issues and DNS resolution time",
    cognitive_action="expert_input",
    attention_request=0.7
)
        """.format(stream=self.stream_id_lesson1))
        
        contrib_result = await self.orchestrator.contribute_thought(
            stream_id=self.stream_id_lesson1,
            agent_name="NetworkExpert",
            thought_content="Additional factors to consider: CDN configuration issues, DNS resolution time, and network latency between server regions",
            cognitive_action="expert_input",
            attention_request=0.7
        )
        
        print("\n✅ Contribution added!")
        
        print("\nNow you add your security perspective...")
        print("Code being executed:")
        print("""
await orchestrator.contribute_thought(
    stream_id="{stream}",
    agent_name="{agent}",
    thought_content="DDoS attacks can cause slowdowns...",
    cognitive_action="security_assessment",
    attention_request=0.8
)
        """.format(stream=self.stream_id_lesson1, agent=self.agent_name))
        
        await self.orchestrator.contribute_thought(
            stream_id=self.stream_id_lesson1,
            agent_name=self.agent_name,
            thought_content="Security consideration: DDoS attacks or bot traffic can cause performance degradation. Check for unusual traffic patterns.",
            cognitive_action="security_assessment",
            attention_request=0.8
        )
        
        print("\n✅ Your contribution added!")
        
        print("\nKey points:")
        print("• Multiple agents can contribute to the same stream")
        print("• Use cognitive_action to specify contribution type")
        print("• Set attention_request based on importance (0.0-1.0)")
        
        self.learned_concepts.append("Multi-agent collaboration")
        self.learned_concepts.append("Contributing thoughts")
    
    async def lesson_4_real_task(self):
        """Lesson 4: Complete a real task"""
        print("\n" + "="*50)
        print("📚 LESSON 4: Your First Real Task")
        print("="*50)
        
        print("\nLet's put it all together with a real task.")
        print("Task: 'Design a simple rate limiting system'\n")
        
        input("Press Enter to start...")
        
        # Step 1: Search
        print("\n1️⃣ First, search for existing work...")
        search_results = await self.orchestrator.search_thought_streams(
            agent_name=None,
            semantic_query="rate limiting",
            limit=3
        )
        
        if search_results.get("streams"):
            print(f"   Found {len(search_results['streams'])} related streams")
            existing_stream = search_results["streams"][0]["stream_id"]
            print(f"   Building on: {existing_stream}")
        else:
            print("   No existing work found")
        
        # Step 2: Create new stream
        print("\n2️⃣ Creating new thought stream...")
        stream_result = await self.orchestrator.initiate_thought_stream(
            task_description="Design a simple but effective rate limiting system for REST API",
            agent_name=self.agent_name,
            cognitive_type="design",
            complexity="medium",
            semantic_tags=["rate-limiting", "api", "security", "performance"]
        )
        
        task_stream_id = stream_result["stream_id"]
        print(f"   Created: {task_stream_id}")
        
        # Step 3: Add considerations
        print("\n3️⃣ Adding design considerations...")
        await self.orchestrator.contribute_thought(
            stream_id=task_stream_id,
            agent_name=self.agent_name,
            thought_content="Key requirements: Handle burst traffic, be memory efficient, support multiple rate limit tiers",
            cognitive_action="analysis",
            attention_request=0.6
        )
        print("   Added requirements")
        
        # Step 4: Generate solution
        print("\n4️⃣ Generating the solution...")
        final_result = await self.orchestrator.execute_strategic_orchestration(
            stream_id=task_stream_id,
            agent_name=self.agent_name,
            execution_prompt="Design a rate limiting system with: algorithm choice, data structure, implementation approach, and example code",
            task_type="design"
        )
        
        tokens = final_result.get("orchestration_result", {}).get("tokens", 0)
        print(f"\n✅ Task complete!")
        print(f"   • Generated {tokens} tokens")
        print(f"   • Stream ID: {task_stream_id}")
        print(f"   • Can be found later with: search_thought_streams('rate limiting')")
        
        self.learned_concepts.append("Complete task workflow")
    
    def show_summary(self):
        """Show tutorial summary"""
        print("\n" + "="*60)
        print("🎉 TUTORIAL COMPLETE!")
        print("="*60)
        
        print(f"\n{self.agent_name}, you've learned:")
        for i, concept in enumerate(self.learned_concepts, 1):
            print(f"  {i}. {concept}")
        
        print("\n📋 Quick Reference Commands:")
        print("""
# Start new analysis
stream = await orchestrator.initiate_thought_stream(
    task_description="...", agent_name="...", cognitive_type="..."
)

# Search existing work
results = await orchestrator.search_thought_streams(
    agent_name=None, semantic_query="..."
)

# Add contribution
await orchestrator.contribute_thought(
    stream_id="...", agent_name="...", thought_content="..."
)

# Generate output
output = await orchestrator.execute_strategic_orchestration(
    stream_id="...", agent_name="...", execution_prompt="..."
)
        """)
        
        print("\n🚀 Next Steps:")
        print("1. Try the CLI: python aios_cli.py status")
        print("2. Read the full guide: AGENT_ONBOARDING.md")
        print("3. Start analyzing real problems!")
        
        print("\n💡 Pro tip: Your work is now searchable by other agents.")
        print("   The more you use semantic tags, the more discoverable it becomes!")

async def quick_demo():
    """Quick non-interactive demo"""
    print("🚀 AIOS Quick Demo (Non-interactive)")
    print("="*40)
    
    orchestrator = AIOSStrategicOrchestrator()
    
    # Create and execute in one flow
    print("\n1. Creating thought stream...")
    stream = await orchestrator.initiate_thought_stream(
        task_description="Optimize Python code for better performance",
        agent_name="DemoAgent",
        cognitive_type="analysis",
        semantic_tags=["python", "optimization", "performance"]
    )
    
    print(f"   ✅ Stream created: {stream['stream_id']}")
    
    print("\n2. Executing analysis...")
    result = await orchestrator.execute_strategic_orchestration(
        stream_id=stream["stream_id"],
        agent_name="DemoAgent",
        execution_prompt="List 3 common Python performance optimizations with examples",
        task_type="analysis"
    )
    
    tokens = result.get("orchestration_result", {}).get("tokens", 0)
    print(f"   ✅ Generated {tokens} tokens")
    
    print("\n3. Searching for related work...")
    search = await orchestrator.search_thought_streams(
        agent_name=None,
        semantic_query="optimization",
        limit=3
    )
    
    print(f"   ✅ Found {len(search.get('streams', []))} related analyses")
    
    print("\n✨ Demo complete! Run without --demo for interactive tutorial.")

if __name__ == "__main__":
    import sys
    
    print("🧠 AIOS unified modules detected")
    print("✅ AIOS infrastructure loaded successfully")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        asyncio.run(quick_demo())
    else:
        tutorial = AgentTutorial()
        asyncio.run(tutorial.run_tutorial())