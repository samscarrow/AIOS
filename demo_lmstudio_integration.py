#!/usr/bin/env python3
"""
Demo: AIOS Cognitive OS with Real LMStudio Integration
Routes tasks through production router and executes on LMStudio models
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from unified.production_task_router import UnifiedProductionRouter
from orchestrator.aios_strategic_orchestrator_routed import AIOSStrategicOrchestratorRouted

class LMStudioCognitiveDemo:
    """Demo integrating AIOS with real LMStudio models"""
    
    def __init__(self):
        self.orchestrator = AIOSStrategicOrchestratorRouted()
        self.router = self.orchestrator.router  # Use the orchestrator's router
        self.results = []
        
    async def query_lmstudio(self, prompt: str, model: str = None) -> Dict[str, Any]:
        """Query LMStudio directly via HTTP API"""
        import aiohttp
        
        try:
            # LMStudio local API endpoint
            url = "http://localhost:1234/v1/chat/completions"
            
            # Prepare request
            payload = {
                "model": model or "qwen/qwen3-8b",
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant specialized in software engineering."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "response": data['choices'][0]['message']['content'],
                            "model_used": model or "qwen/qwen3-8b",
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {await response.text()}",
                            "model_attempted": model,
                            "timestamp": datetime.now().isoformat()
                        }
                        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_attempted": model,
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_task(self, task_description: str, agent_name: str = "CognitiveAgent"):
        """Process a task through the full pipeline"""
        print(f"\n{'='*60}")
        print(f"🎯 Task: {task_description[:100]}...")
        print(f"{'='*60}")
        
        # Step 1: Route the task
        print("\n📊 Routing task...")
        routing_result = await self.router.route(task_description)
        
        print(f"  📍 Type: {routing_result.task_type}")
        print(f"  📊 Complexity: {routing_result.complexity}")
        print(f"  🎯 Confidence: {routing_result.confidence:.3f}")
        print(f"  🛤️ Path: {routing_result.routing_path}")
        
        # Get model recommendations from metadata
        recommended_models = routing_result.metadata.get('recommended_models', ['qwen/qwen3-8b'])
        print(f"  🤖 Recommended Models: {', '.join(recommended_models[:2])}")
        
        # Step 2: Initialize thought stream in orchestrator
        print("\n🧠 Initializing cognitive stream...")
        stream_result = await self.orchestrator.initiate_thought_stream(
            task_description=task_description,
            agent_name=agent_name,
            use_router=True
        )
        
        print(f"  ✅ Stream ID: {stream_result['stream_id']}")
        print(f"  ⚡ Attention Weight: {stream_result['attention_allocated']:.2f}")
        
        # Step 3: Query LMStudio with the task
        print("\n🚀 Querying LMStudio...")
        
        # Select model based on routing recommendation
        recommended_models = routing_result.metadata.get('recommended_models', ['qwen/qwen3-8b'])
        model_to_use = None
        
        # Map common model patterns to available LMStudio models
        model_mapping = {
            "qwen": "qwen/qwen3-8b",
            "codestral": "mistralai/codestral-22b-v0.1",
            "deepseek": "deepseek/deepseek-r1-0528-qwen3-8b",
            "hermes": "nousresearch/hermes-4-70b"
        }
        
        for recommended in recommended_models:
            for pattern, lmstudio_model in model_mapping.items():
                if pattern in recommended.lower():
                    model_to_use = lmstudio_model
                    break
            if model_to_use:
                break
        
        if not model_to_use:
            model_to_use = "qwen/qwen3-8b"  # Default fallback
        
        print(f"  🤖 Using model: {model_to_use}")
        
        # Create a focused prompt based on task type
        prompt_templates = {
            "code_generation": f"Write clean, efficient code for: {task_description}",
            "debugging": f"Debug and fix the following issue: {task_description}",
            "analysis": f"Analyze and provide insights on: {task_description}",
            "system_design": f"Design a system architecture for: {task_description}",
            "documentation": f"Create clear documentation for: {task_description}",
            "testing": f"Create comprehensive tests for: {task_description}",
            "refactoring": f"Refactor and improve: {task_description}"
        }
        
        task_type = routing_result.task_type
        prompt = prompt_templates.get(task_type, task_description)
        
        # Query LMStudio
        lmstudio_result = await self.query_lmstudio(prompt, model_to_use)
        
        if lmstudio_result['success']:
            print(f"  ✅ Response received")
            print(f"\n📝 Response Preview:")
            response_text = str(lmstudio_result['response'])[:500]
            print(f"  {response_text}...")
        else:
            print(f"  ❌ Error: {lmstudio_result['error']}")
        
        # Step 4: Record processing completion
        print("\n📊 Processing completed successfully")
        
        # Store results
        self.results.append({
            "task": task_description,
            "routing": routing_result.to_dict(),  # Use standardized conversion
            "stream": stream_result,
            "lmstudio": lmstudio_result
        })
        
        return lmstudio_result
    
    async def run_demo(self):
        """Run comprehensive demo with various tasks"""
        
        print("\n" + "="*60)
        print("🚀 AIOS Cognitive OS - LMStudio Integration Demo")
        print("="*60)
        
        # Demo tasks covering different types and complexities
        demo_tasks = [
            # Code generation
            ("Write a Python function that implements binary search on a sorted array", "CodeAgent"),
            
            # Debugging
            ("Debug why my REST API returns 500 errors when handling concurrent requests", "DebugAgent"),
            
            # System design
            ("Design a microservices architecture for an e-commerce platform with 10M daily users", "ArchitectAgent"),
            
            # Analysis
            ("Analyze the time complexity of quicksort vs mergesort for different data distributions", "AnalystAgent"),
            
            # Documentation
            ("Document the authentication flow for a JWT-based API system", "DocAgent"),
        ]
        
        for task, agent in demo_tasks:
            try:
                await self.process_task(task, agent)
                await asyncio.sleep(1)  # Brief pause between tasks
            except Exception as e:
                print(f"❌ Error processing task: {e}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Demo Summary")
        print("="*60)
        
        successful = sum(1 for r in self.results if r['lmstudio']['success'])
        total = len(self.results)
        
        print(f"\n✅ Successful queries: {successful}/{total}")
        print(f"🎯 Success rate: {(successful/total)*100:.1f}%")
        
        # Task type distribution
        task_types = {}
        for result in self.results:
            task_type = result['routing']['task_type']
            task_types[task_type] = task_types.get(task_type, 0) + 1
        
        print(f"\n📊 Task Type Distribution:")
        for task_type, count in task_types.items():
            print(f"  - {task_type}: {count}")
        
        # Complexity distribution
        complexities = {}
        for result in self.results:
            complexity = result['routing']['complexity']
            complexities[complexity] = complexities.get(complexity, 0) + 1
        
        print(f"\n📈 Complexity Distribution:")
        for complexity, count in complexities.items():
            print(f"  - {complexity}: {count}")
        
        # Router performance
        avg_confidence = sum(r['routing']['confidence'] for r in self.results) / len(self.results)
        print(f"\n🎯 Average Routing Confidence: {avg_confidence:.3f}")
        
        # Save results
        output_file = "lmstudio_demo_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n💾 Full results saved to {output_file}")

async def main():
    """Main entry point"""
    demo = LMStudioCognitiveDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())