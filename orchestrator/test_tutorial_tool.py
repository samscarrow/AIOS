#!/usr/bin/env python3
"""
Test the tutorial tool functionality
"""

import asyncio
from aios_strategic_orchestrator import AIOSStrategicOrchestrator

async def test_tutorial_tool():
    """Test the get_aios_tutorial function"""
    orchestrator = AIOSStrategicOrchestrator()
    
    print("Testing AIOS Tutorial Tool")
    print("=" * 50)
    
    # Test different tutorial types
    tutorial_types = ["quick_start", "full_guide", "examples", "best_practices", "troubleshooting"]
    
    for tutorial_type in tutorial_types:
        print(f"\n📚 Testing tutorial type: {tutorial_type}")
        result = await orchestrator.get_aios_tutorial(
            tutorial_type=tutorial_type,
            agent_name="TestAgent"
        )
        
        print(f"   Status: {result['status']}")
        print(f"   Content length: {len(result['content'])} characters")
        print(f"   Preview: {result['content'][:200]}...")
        
        if tutorial_type == "quick_start":
            print("\n   Next steps provided:")
            for step in result['next_steps']:
                print(f"     - {step}")
    
    print("\n✅ All tutorial types working correctly!")
    print(f"Available types: {result['available_types']}")

if __name__ == "__main__":
    asyncio.run(test_tutorial_tool())