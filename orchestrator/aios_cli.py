#!/usr/bin/env python3
"""
AIOS CLI - Simple command-line interface for accessing AIOS orchestration
"""

import asyncio
import json
import sys
from aios_strategic_orchestrator import AIOSStrategicOrchestrator

async def main():
    """Main CLI handler"""
    
    if len(sys.argv) < 2:
        print("AIOS Strategic Orchestration CLI")
        print("================================")
        print("\nUsage:")
        print("  python aios_cli.py search <query>     - Search past analyses")
        print("  python aios_cli.py analyze <prompt>   - Start new analysis")
        print("  python aios_cli.py status             - Show system status")
        print("  python aios_cli.py export <stream_id> - Export analysis")
        print("\nExamples:")
        print("  python aios_cli.py search 'recursive improvement'")
        print("  python aios_cli.py analyze 'Review this code for security issues'")
        print("  python aios_cli.py export ts_abc123")
        return
    
    orchestrator = AIOSStrategicOrchestrator()
    command = sys.argv[1].lower()
    
    if command == "search" and len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        print(f"Searching for: {query}")
        results = await orchestrator.search_thought_streams(
            agent_name=None,
            semantic_query=query,
            limit=5
        )
        
        if results.get("streams"):
            print(f"\nFound {len(results['streams'])} results:")
            for stream in results["streams"]:
                print(f"\n📌 Stream ID: {stream['stream_id']}")
                print(f"   Agents: {', '.join(stream.get('agents', []))}")
                print(f"   Type: {stream.get('cognitive_type', 'unknown')}")
                print(f"   State: {stream.get('cognitive_state', 'unknown')}")
        else:
            print("No results found")
    
    elif command == "analyze" and len(sys.argv) > 2:
        prompt = " ".join(sys.argv[2:])
        print(f"Starting analysis: {prompt[:50]}...")
        
        # Start analysis
        stream_result = await orchestrator.initiate_thought_stream(
            task_description=prompt,
            agent_name="CLIAnalyst",
            cognitive_type="analysis",
            complexity="medium"
        )
        
        # Execute
        result = await orchestrator.execute_strategic_orchestration(
            stream_id=stream_result["stream_id"],
            agent_name="CLIAnalyst",
            execution_prompt=prompt,
            task_type="analysis"
        )
        
        print(f"\n✅ Analysis Complete")
        print(f"Stream ID: {stream_result['stream_id']}")
        print(f"Tokens: {result.get('orchestration_result', {}).get('tokens', 0)}")
        print(f"\nOutput:")
        print(result.get('orchestration_result', {}).get('output', 'No output'))
    
    elif command == "status":
        print("AIOS System Status")
        print("==================")
        print(f"AIOS Infrastructure: ✅ Loaded")
        
        # Show recent streams
        results = await orchestrator.search_thought_streams(
            agent_name=None,
            limit=3
        )
        stream_count = len(results.get("streams", []))
        print(f"Accessible Thought Streams: {stream_count}")
        
        if results.get("streams"):
            print(f"\nRecent Analyses:")
            for stream in results["streams"]:
                print(f"  - {stream['stream_id']}: {stream.get('cognitive_type', 'unknown')}")
    
    elif command == "export" and len(sys.argv) > 2:
        stream_id = sys.argv[2]
        
        # Search for the stream
        results = await orchestrator.search_thought_streams(
            agent_name=None,
            limit=100
        )
        found = False
        
        for stream in results.get("streams", []):
            if stream.get("stream_id") == stream_id:
                filename = f"export_{stream_id}.json"
                with open(filename, 'w') as f:
                    json.dump(stream, f, indent=2, default=str)
                print(f"✅ Exported to {filename}")
                found = True
                break
        
        if not found:
            print(f"❌ Stream {stream_id} not found")
    
    else:
        print(f"Unknown command: {command}")
        print("Run 'python aios_cli.py' for help")

if __name__ == "__main__":
    asyncio.run(main())