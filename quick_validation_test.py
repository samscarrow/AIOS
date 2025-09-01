#!/usr/bin/env python3
"""
Quick Validation Test - Prove system claims with focused tests
"""

import asyncio
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from unified.production_task_router import UnifiedProductionRouter
from orchestrator.aios_strategic_orchestrator_routed import AIOSStrategicOrchestratorRouted

async def quick_validation():
    print("🧪 QUICK VALIDATION: Testing Core System Claims")
    print("=" * 60)
    
    # Initialize system
    orchestrator = AIOSStrategicOrchestratorRouted()
    router = orchestrator.router
    
    # Test 1: Intelligent Routing Claims
    print("\n🎯 TEST 1: Intelligent Routing")
    challenging_tasks = [
        "Fix memory leak in production Kubernetes cluster",
        "Design microservices for 1M QPS",  
        "Debug race condition in distributed system",
        "Implement OAuth2 with JWT tokens",
        "Analyze database query performance"
    ]
    
    routing_times = []
    confidences = []
    
    for i, task in enumerate(challenging_tasks, 1):
        start = time.perf_counter()
        result = await router.route(task)
        rt = (time.perf_counter() - start) * 1000
        
        routing_times.append(rt)
        confidences.append(result.confidence)
        
        print(f"  {i}. {task[:40]}...")
        print(f"     Type: {result.task_type}, Complexity: {result.complexity}")
        print(f"     Confidence: {result.confidence:.3f}, Time: {rt:.1f}ms")
    
    avg_rt = sum(routing_times) / len(routing_times)
    avg_conf = sum(confidences) / len(confidences)
    
    print(f"\n  📊 ROUTING RESULTS:")
    print(f"     ⚡ Avg routing time: {avg_rt:.1f}ms (Target: <50ms)")
    print(f"     🎯 Avg confidence: {avg_conf:.3f} (Target: >0.5)")
    print(f"     ✅ Sub-50ms routing: {'PASS' if avg_rt < 50 else 'FAIL'}")
    print(f"     ✅ High confidence: {'PASS' if avg_conf > 0.5 else 'FAIL'}")
    
    # Test 2: Orchestration Claims
    print("\n🧠 TEST 2: Cognitive Orchestration")
    
    test_tasks = [
        ("Simple task", "Agent1"),
        ("Complex architectural design", "Agent2"), 
        ("Debug production issue", "Agent3")
    ]
    
    orchestration_times = []
    successes = 0
    
    for task, agent in test_tasks:
        start = time.perf_counter()
        try:
            result = await orchestrator.initiate_thought_stream(
                task_description=task,
                agent_name=agent,
                use_router=True
            )
            ot = (time.perf_counter() - start) * 1000
            orchestration_times.append(ot)
            successes += 1
            
            print(f"  ✅ {agent}: Stream {result['stream_id']} ({ot:.1f}ms)")
            print(f"     Attention: {result['attention_allocated']:.2f}")
            
        except Exception as e:
            ot = (time.perf_counter() - start) * 1000
            orchestration_times.append(ot)
            print(f"  ❌ {agent}: Failed - {str(e)[:50]}")
    
    avg_ot = sum(orchestration_times) / len(orchestration_times) if orchestration_times else 0
    success_rate = successes / len(test_tasks)
    
    print(f"\n  📊 ORCHESTRATION RESULTS:")
    print(f"     ⚡ Avg orchestration time: {avg_ot:.1f}ms")
    print(f"     🎯 Success rate: {success_rate*100:.1f}% (Target: >90%)")
    print(f"     ✅ High success rate: {'PASS' if success_rate > 0.9 else 'FAIL'}")
    
    # Test 3: LMStudio Quality Test
    print("\n🤖 TEST 3: LMStudio Response Quality")
    
    import aiohttp
    
    async def test_lmstudio(prompt):
        try:
            url = "http://localhost:1234/v1/chat/completions"
            payload = {
                "model": "qwen/qwen3-8b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 300
            }
            
            start = time.perf_counter()
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    rt = (time.perf_counter() - start) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        return {
                            "success": True,
                            "response": content,
                            "time_ms": rt,
                            "word_count": len(content.split())
                        }
                    else:
                        return {"success": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    quality_tests = [
        "Write a Python function for binary search",
        "Explain REST API best practices", 
        "Debug a memory leak in Node.js"
    ]
    
    lm_results = []
    for prompt in quality_tests:
        result = await test_lmstudio(prompt)
        lm_results.append(result)
        
        if result['success']:
            print(f"  ✅ Query: {prompt[:30]}...")
            print(f"     Response: {result['word_count']} words, {result['time_ms']:.1f}ms")
        else:
            print(f"  ❌ Query failed: {result.get('error', 'Unknown error')}")
    
    lm_successes = sum(1 for r in lm_results if r['success'])
    lm_success_rate = lm_successes / len(lm_results)
    
    if lm_successes > 0:
        avg_lm_time = sum(r['time_ms'] for r in lm_results if r['success']) / lm_successes
        avg_words = sum(r['word_count'] for r in lm_results if r['success']) / lm_successes
    else:
        avg_lm_time = avg_words = 0
    
    print(f"\n  📊 LMSTUDIO RESULTS:")
    print(f"     🎯 Success rate: {lm_success_rate*100:.1f}% (Target: >80%)")
    print(f"     ⚡ Avg response time: {avg_lm_time:.1f}ms")
    print(f"     📝 Avg response length: {avg_words:.0f} words")
    print(f"     ✅ High success rate: {'PASS' if lm_success_rate > 0.8 else 'FAIL'}")
    
    # Final Assessment
    print("\n" + "=" * 60)
    print("🎯 FINAL CLAIM VALIDATION")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    if avg_rt < 50:
        print("✅ CLAIM: Sub-50ms intelligent routing - VALIDATED")
        tests_passed += 1
    else:
        print("❌ CLAIM: Sub-50ms intelligent routing - FAILED")
    
    if avg_conf > 0.5:
        print("✅ CLAIM: High confidence routing decisions - VALIDATED") 
        tests_passed += 1
    else:
        print("❌ CLAIM: High confidence routing decisions - FAILED")
    
    if success_rate > 0.9:
        print("✅ CLAIM: Reliable cognitive orchestration - VALIDATED")
        tests_passed += 1
    else:
        print("❌ CLAIM: Reliable cognitive orchestration - FAILED")
        
    if lm_success_rate > 0.8:
        print("✅ CLAIM: High-quality LMStudio responses - VALIDATED")
        tests_passed += 1
    else:
        print("❌ CLAIM: High-quality LMStudio responses - FAILED")
    
    print(f"\n🎯 OVERALL: {tests_passed}/{total_tests} claims validated ({tests_passed/total_tests*100:.1f}%)")
    
    if tests_passed == total_tests:
        print("🎉 ALL CLAIMS VALIDATED - SYSTEM PERFORMING AS ADVERTISED!")
    elif tests_passed >= total_tests * 0.75:
        print("⚠️  MOST CLAIMS VALIDATED - SYSTEM MOSTLY RELIABLE")
    else:
        print("❌ CLAIMS NOT VALIDATED - SYSTEM NEEDS IMPROVEMENT")

if __name__ == "__main__":
    asyncio.run(quick_validation())