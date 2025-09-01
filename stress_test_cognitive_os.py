#!/usr/bin/env python3
"""
Comprehensive Stress Test Suite for Unified Cognitive OS
Tests all system claims with challenging real-world scenarios
"""

import asyncio
import json
import time
import statistics
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from unified.production_task_router import UnifiedProductionRouter
from orchestrator.aios_strategic_orchestrator_routed import AIOSStrategicOrchestratorRouted

class CognitiveOSStressTester:
    """Comprehensive stress testing for all system components"""
    
    def __init__(self):
        self.orchestrator = AIOSStrategicOrchestratorRouted()
        self.router = self.orchestrator.router
        self.results = []
        self.performance_metrics = {
            "routing_times": [],
            "orchestration_times": [],
            "lmstudio_times": [],
            "end_to_end_times": [],
            "success_rates": [],
            "confidence_scores": []
        }
        
    async def query_lmstudio(self, prompt: str, model: str = "qwen/qwen3-8b") -> Dict[str, Any]:
        """Query LMStudio with timing"""
        import aiohttp
        
        start_time = time.perf_counter()
        try:
            url = "http://localhost:1234/v1/chat/completions"
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert AI assistant. Provide detailed, accurate, and comprehensive responses."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1500
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    response_time = (time.perf_counter() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "response": data['choices'][0]['message']['content'],
                            "model_used": model,
                            "response_time_ms": response_time,
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}",
                            "response_time_ms": response_time
                        }
                        
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "response_time_ms": response_time
            }
    
    async def stress_test_routing(self) -> Dict[str, Any]:
        """Test intelligent routing with challenging edge cases"""
        print("\n🎯 STRESS TEST 1: Intelligent Task Routing")
        print("=" * 60)
        
        # Edge cases and challenging tasks
        challenging_tasks = [
            # Ambiguous tasks
            "Fix the thing that's broken",
            "Make it faster",
            "Optimize performance",
            
            # Complex multi-domain tasks
            "Build a distributed real-time analytics system with machine learning pipelines, microservices architecture, and edge computing capabilities for processing 1TB/day of IoT sensor data",
            
            # Technical debt scenarios
            "Refactor this legacy monolith into microservices while maintaining backwards compatibility and zero downtime deployment",
            
            # Domain-specific challenges
            "Implement GDPR-compliant data processing with differential privacy, homomorphic encryption, and federated learning",
            
            # Novel/emerging tech
            "Create a quantum-classical hybrid algorithm for portfolio optimization using QAOA with classical preprocessing",
            
            # Integration nightmares
            "Integrate Salesforce, SAP, Oracle, and custom APIs while handling rate limits, authentication, and data transformation",
            
            # Performance critical
            "Optimize database queries that are causing 30-second page load times under 10K concurrent users",
            
            # Debugging nightmares
            "Debug intermittent memory leak in C++ multithreaded application that only occurs in production under specific load patterns"
        ]
        
        routing_results = []
        routing_times = []
        
        for i, task in enumerate(challenging_tasks, 1):
            print(f"\n⚡ Test {i}/10: {task[:80]}...")
            
            start_time = time.perf_counter()
            routing_result = await self.router.route(task)
            routing_time = (time.perf_counter() - start_time) * 1000
            
            routing_times.append(routing_time)
            routing_results.append({
                "task": task,
                "result": routing_result.to_dict(),
                "routing_time_ms": routing_time
            })
            
            print(f"  📍 Type: {routing_result.task_type}")
            print(f"  📊 Complexity: {routing_result.complexity}")
            print(f"  🎯 Confidence: {routing_result.confidence:.3f}")
            print(f"  ⚡ Time: {routing_time:.1f}ms")
            print(f"  🛤️ Path: {routing_result.routing_path}")
        
        # Analysis
        avg_routing_time = statistics.mean(routing_times)
        max_routing_time = max(routing_times)
        min_routing_time = min(routing_times)
        
        confidences = [r['result']['confidence'] for r in routing_results]
        avg_confidence = statistics.mean(confidences)
        
        abstentions = sum(1 for r in routing_results if r['result']['abstain'])
        
        print(f"\n📊 ROUTING PERFORMANCE:")
        print(f"  ⚡ Avg Response Time: {avg_routing_time:.1f}ms")
        print(f"  🚀 Min Response Time: {min_routing_time:.1f}ms") 
        print(f"  🐌 Max Response Time: {max_routing_time:.1f}ms")
        print(f"  🎯 Avg Confidence: {avg_confidence:.3f}")
        print(f"  ❓ Abstention Rate: {abstentions/len(challenging_tasks)*100:.1f}%")
        
        return {
            "test": "intelligent_routing",
            "tasks_tested": len(challenging_tasks),
            "avg_routing_time_ms": avg_routing_time,
            "max_routing_time_ms": max_routing_time,
            "avg_confidence": avg_confidence,
            "abstention_rate": abstentions/len(challenging_tasks),
            "results": routing_results
        }
    
    async def stress_test_orchestration(self) -> Dict[str, Any]:
        """Test cognitive orchestration under concurrent load"""
        print("\n🧠 STRESS TEST 2: Cognitive Orchestration Under Load")
        print("=" * 60)
        
        # Concurrent tasks of varying complexity
        concurrent_tasks = [
            ("Simple validation task", "ValidationAgent"),
            ("Complex system design challenge", "ArchitectAgent"),
            ("Debug production issue", "DebugAgent"),
            ("Refactor legacy codebase", "RefactorAgent"),
            ("Analyze performance bottlenecks", "AnalystAgent"),
            ("Design API documentation", "DocAgent"),
            ("Implement security audit", "SecurityAgent"),
            ("Create testing strategy", "TestAgent")
        ]
        
        print(f"🚀 Launching {len(concurrent_tasks)} concurrent thought streams...")
        
        start_time = time.perf_counter()
        
        # Launch all tasks concurrently
        tasks = []
        for task_desc, agent in concurrent_tasks:
            task = asyncio.create_task(
                self.orchestrator.initiate_thought_stream(
                    task_description=task_desc,
                    agent_name=agent,
                    use_router=True
                )
            )
            tasks.append((task_desc, agent, task))
        
        # Wait for all to complete
        results = []
        for task_desc, agent, task in tasks:
            try:
                result = await task
                results.append({
                    "task": task_desc,
                    "agent": agent,
                    "success": True,
                    "result": result
                })
                print(f"  ✅ {agent}: Stream {result['stream_id']} initialized")
            except Exception as e:
                results.append({
                    "task": task_desc,
                    "agent": agent,
                    "success": False,
                    "error": str(e)
                })
                print(f"  ❌ {agent}: Failed - {e}")
        
        total_time = (time.perf_counter() - start_time) * 1000
        successful = sum(1 for r in results if r['success'])
        
        print(f"\n📊 ORCHESTRATION PERFORMANCE:")
        print(f"  ⚡ Total Time: {total_time:.1f}ms")
        print(f"  🎯 Success Rate: {successful}/{len(concurrent_tasks)} ({successful/len(concurrent_tasks)*100:.1f}%)")
        print(f"  🚀 Avg Time per Stream: {total_time/len(concurrent_tasks):.1f}ms")
        
        return {
            "test": "cognitive_orchestration",
            "concurrent_streams": len(concurrent_tasks),
            "total_time_ms": total_time,
            "success_rate": successful/len(concurrent_tasks),
            "avg_time_per_stream_ms": total_time/len(concurrent_tasks),
            "results": results
        }
    
    async def stress_test_lmstudio_quality(self) -> Dict[str, Any]:
        """Test LMStudio model quality with challenging prompts"""
        print("\n🤖 STRESS TEST 3: LMStudio Model Quality")
        print("=" * 60)
        
        # Challenging prompts across different domains
        quality_tests = [
            {
                "category": "Code Quality",
                "prompt": "Write a production-ready Python class for a thread-safe LRU cache with TTL support, comprehensive error handling, and monitoring hooks. Include proper documentation and type hints.",
                "expected_elements": ["thread-safe", "LRU", "TTL", "error handling", "type hints", "documentation"]
            },
            {
                "category": "System Architecture",
                "prompt": "Design a fault-tolerant microservices architecture for a financial trading platform that needs to handle 100K transactions/second with sub-millisecond latency requirements. Include disaster recovery and compliance considerations.",
                "expected_elements": ["microservices", "fault-tolerant", "latency", "disaster recovery", "compliance"]
            },
            {
                "category": "Problem Solving",
                "prompt": "Debug this scenario: A distributed system works perfectly in staging but fails randomly in production with race conditions. Load is identical, but production has 3x more instances. What are the likely causes and how would you debug systematically?",
                "expected_elements": ["race conditions", "distributed", "debugging", "systematic approach"]
            },
            {
                "category": "Technical Analysis",
                "prompt": "Analyze the trade-offs between GraphQL and REST APIs for a mobile-first application with offline capabilities, considering performance, caching, bandwidth, and developer experience.",
                "expected_elements": ["GraphQL", "REST", "mobile", "offline", "performance", "trade-offs"]
            },
            {
                "category": "Security",
                "prompt": "Design a zero-trust security model for a multi-cloud Kubernetes deployment with service mesh, considering identity management, network segmentation, and threat detection.",
                "expected_elements": ["zero-trust", "kubernetes", "service mesh", "identity", "network segmentation"]
            }
        ]
        
        quality_results = []
        
        for i, test in enumerate(quality_tests, 1):
            print(f"\n🧪 Quality Test {i}/5: {test['category']}")
            print(f"   Prompt: {test['prompt'][:100]}...")
            
            result = await self.query_lmstudio(test['prompt'])
            
            if result['success']:
                response = result['response']
                
                # Quality assessment
                elements_found = sum(1 for element in test['expected_elements'] 
                                   if element.lower() in response.lower())
                completeness_score = elements_found / len(test['expected_elements'])
                
                # Length and detail assessment
                word_count = len(response.split())
                detail_score = min(1.0, word_count / 300)  # Expect at least 300 words for detailed response
                
                quality_score = (completeness_score + detail_score) / 2
                
                print(f"   ✅ Response received ({word_count} words)")
                print(f"   📊 Completeness: {completeness_score:.2f} ({elements_found}/{len(test['expected_elements'])} elements)")
                print(f"   📝 Detail Score: {detail_score:.2f}")
                print(f"   🎯 Quality Score: {quality_score:.2f}")
                
                quality_results.append({
                    "category": test['category'],
                    "success": True,
                    "quality_score": quality_score,
                    "completeness_score": completeness_score,
                    "detail_score": detail_score,
                    "word_count": word_count,
                    "response_time_ms": result['response_time_ms']
                })
            else:
                print(f"   ❌ Failed: {result['error']}")
                quality_results.append({
                    "category": test['category'],
                    "success": False,
                    "error": result['error']
                })
        
        # Overall quality analysis
        successful_tests = [r for r in quality_results if r['success']]
        if successful_tests:
            avg_quality = statistics.mean([r['quality_score'] for r in successful_tests])
            avg_response_time = statistics.mean([r['response_time_ms'] for r in successful_tests])
            avg_word_count = statistics.mean([r['word_count'] for r in successful_tests])
        else:
            avg_quality = avg_response_time = avg_word_count = 0
        
        print(f"\n📊 QUALITY ASSESSMENT:")
        print(f"  🎯 Success Rate: {len(successful_tests)}/{len(quality_tests)} ({len(successful_tests)/len(quality_tests)*100:.1f}%)")
        print(f"  📈 Avg Quality Score: {avg_quality:.2f}/1.00")
        print(f"  ⚡ Avg Response Time: {avg_response_time:.1f}ms")
        print(f"  📝 Avg Word Count: {avg_word_count:.0f} words")
        
        return {
            "test": "lmstudio_quality",
            "tests_conducted": len(quality_tests),
            "success_rate": len(successful_tests)/len(quality_tests),
            "avg_quality_score": avg_quality,
            "avg_response_time_ms": avg_response_time,
            "avg_word_count": avg_word_count,
            "results": quality_results
        }
    
    async def stress_test_end_to_end(self) -> Dict[str, Any]:
        """Test complete end-to-end pipeline performance"""
        print("\n🌐 STRESS TEST 4: End-to-End Pipeline Performance")
        print("=" * 60)
        
        # Real-world scenarios that test the full pipeline
        e2e_scenarios = [
            "Analyze why our Kubernetes pods are being OOMKilled and propose solutions with implementation steps",
            "Design a CI/CD pipeline for a microservices architecture with automated testing, security scanning, and blue-green deployment",
            "Debug and fix a memory leak in a Node.js application that's causing production outages every 3 days",
            "Implement a real-time recommendation engine using collaborative filtering with Redis caching and Apache Kafka",
            "Create a disaster recovery plan for a multi-region AWS deployment with RTO < 1 hour and RPO < 5 minutes"
        ]
        
        e2e_results = []
        
        for i, scenario in enumerate(e2e_scenarios, 1):
            print(f"\n🎬 End-to-End Test {i}/5:")
            print(f"   Scenario: {scenario[:80]}...")
            
            total_start = time.perf_counter()
            
            # Step 1: Route
            routing_start = time.perf_counter()
            routing_result = await self.router.route(scenario)
            routing_time = (time.perf_counter() - routing_start) * 1000
            
            # Step 2: Orchestrate
            orchestration_start = time.perf_counter()
            try:
                stream_result = await self.orchestrator.initiate_thought_stream(
                    task_description=scenario,
                    agent_name="E2EAgent",
                    use_router=True
                )
                orchestration_time = (time.perf_counter() - orchestration_start) * 1000
                orchestration_success = True
            except Exception as e:
                orchestration_time = (time.perf_counter() - orchestration_start) * 1000
                orchestration_success = False
                stream_result = {"error": str(e)}
            
            # Step 3: Query LMStudio
            if orchestration_success:
                lmstudio_result = await self.query_lmstudio(scenario)
                lmstudio_success = lmstudio_result['success']
                lmstudio_time = lmstudio_result['response_time_ms']
            else:
                lmstudio_success = False
                lmstudio_time = 0
            
            total_time = (time.perf_counter() - total_start) * 1000
            
            # Results
            result = {
                "scenario": scenario,
                "total_time_ms": total_time,
                "routing_time_ms": routing_time,
                "orchestration_time_ms": orchestration_time,
                "lmstudio_time_ms": lmstudio_time,
                "routing_success": not routing_result.abstain,
                "orchestration_success": orchestration_success,
                "lmstudio_success": lmstudio_success,
                "end_to_end_success": not routing_result.abstain and orchestration_success and lmstudio_success,
                "confidence": routing_result.confidence,
                "complexity": routing_result.complexity
            }
            
            e2e_results.append(result)
            
            print(f"   ⚡ Total Time: {total_time:.1f}ms")
            print(f"   🎯 Routing: {'✅' if result['routing_success'] else '❌'} ({routing_time:.1f}ms)")
            print(f"   🧠 Orchestration: {'✅' if result['orchestration_success'] else '❌'} ({orchestration_time:.1f}ms)")
            print(f"   🤖 LMStudio: {'✅' if result['lmstudio_success'] else '❌'} ({lmstudio_time:.1f}ms)")
            print(f"   🌐 End-to-End: {'✅' if result['end_to_end_success'] else '❌'}")
        
        # Analysis
        successful_e2e = sum(1 for r in e2e_results if r['end_to_end_success'])
        avg_total_time = statistics.mean([r['total_time_ms'] for r in e2e_results])
        avg_confidence = statistics.mean([r['confidence'] for r in e2e_results])
        
        print(f"\n📊 END-TO-END PERFORMANCE:")
        print(f"  🎯 Success Rate: {successful_e2e}/{len(e2e_scenarios)} ({successful_e2e/len(e2e_scenarios)*100:.1f}%)")
        print(f"  ⚡ Avg Total Time: {avg_total_time:.1f}ms")
        print(f"  🎯 Avg Confidence: {avg_confidence:.3f}")
        
        return {
            "test": "end_to_end_performance",
            "scenarios_tested": len(e2e_scenarios),
            "success_rate": successful_e2e/len(e2e_scenarios),
            "avg_total_time_ms": avg_total_time,
            "avg_confidence": avg_confidence,
            "results": e2e_results
        }
    
    async def run_comprehensive_stress_test(self):
        """Run all stress tests and generate comprehensive report"""
        print("🚀 COMPREHENSIVE COGNITIVE OS STRESS TEST SUITE")
        print("=" * 80)
        print(f"Started at: {datetime.now().isoformat()}")
        print("=" * 80)
        
        # Run all stress tests
        test_results = {}
        
        try:
            test_results["routing"] = await self.stress_test_routing()
        except Exception as e:
            print(f"❌ Routing stress test failed: {e}")
            test_results["routing"] = {"error": str(e)}
        
        try:
            test_results["orchestration"] = await self.stress_test_orchestration()
        except Exception as e:
            print(f"❌ Orchestration stress test failed: {e}")
            test_results["orchestration"] = {"error": str(e)}
        
        try:
            test_results["quality"] = await self.stress_test_lmstudio_quality()
        except Exception as e:
            print(f"❌ Quality stress test failed: {e}")
            test_results["quality"] = {"error": str(e)}
        
        try:
            test_results["end_to_end"] = await self.stress_test_end_to_end()
        except Exception as e:
            print(f"❌ End-to-end stress test failed: {e}")
            test_results["end_to_end"] = {"error": str(e)}
        
        # Generate comprehensive report
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE STRESS TEST REPORT")
        print("=" * 80)
        
        # Summary
        total_tests = 0
        total_successes = 0
        
        for test_name, result in test_results.items():
            if "error" not in result:
                if "success_rate" in result:
                    print(f"\n✅ {test_name.upper()}: {result['success_rate']*100:.1f}% success rate")
                else:
                    print(f"\n✅ {test_name.upper()}: Completed")
                
                # Extract metrics based on test type
                if test_name == "routing":
                    print(f"   ⚡ Avg routing time: {result.get('avg_routing_time_ms', 0):.1f}ms")
                    print(f"   🎯 Avg confidence: {result.get('avg_confidence', 0):.3f}")
                elif test_name == "orchestration":
                    print(f"   ⚡ Avg orchestration time: {result.get('avg_time_per_stream_ms', 0):.1f}ms")
                elif test_name == "quality":
                    print(f"   🎯 Avg quality score: {result.get('avg_quality_score', 0):.2f}/1.00")
                    print(f"   ⚡ Avg response time: {result.get('avg_response_time_ms', 0):.1f}ms")
                elif test_name == "end_to_end":
                    print(f"   ⚡ Avg total time: {result.get('avg_total_time_ms', 0):.1f}ms")
                
                total_tests += 1
                if result.get('success_rate', 0) > 0.8:  # 80% threshold
                    total_successes += 1
            else:
                print(f"\n❌ {test_name.upper()}: FAILED - {result['error']}")
                total_tests += 1
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"   Tests passed: {total_successes}/{total_tests}")
        print(f"   Overall success rate: {total_successes/total_tests*100:.1f}%")
        
        # Save detailed results
        output_file = f"cognitive_os_stress_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        print(f"   📄 Detailed results saved to: {output_file}")
        
        print("\n" + "=" * 80)
        if total_successes == total_tests:
            print("🎉 ALL STRESS TESTS PASSED - CLAIMS VALIDATED! 🎉")
        elif total_successes >= total_tests * 0.75:
            print("⚠️  MOST TESTS PASSED - SYSTEM MOSTLY RELIABLE")
        else:
            print("❌ MULTIPLE FAILURES - CLAIMS NOT VALIDATED")
        print("=" * 80)

async def main():
    """Main test runner"""
    tester = CognitiveOSStressTester()
    await tester.run_comprehensive_stress_test()

if __name__ == "__main__":
    asyncio.run(main())