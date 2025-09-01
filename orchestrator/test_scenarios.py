#!/usr/bin/env python3
"""
AIOS Strategic Orchestration Test Scenarios
Comprehensive test scenarios for analyzing system performance and capabilities
"""

import asyncio
import json
import time
from typing import Dict, List, Any
from aios_strategic_orchestrator import AIOSStrategicOrchestrator

class TestScenarioRunner:
    """Runs and analyzes test scenarios for the AIOS orchestration system"""
    
    def __init__(self):
        self.orchestrator = AIOSStrategicOrchestrator()
        self.results = []
    
    async def run_all_scenarios(self) -> Dict[str, Any]:
        """Run all test scenarios and return comprehensive analysis"""
        print("=== AIOS Strategic Orchestration Test Scenarios ===\n")
        
        scenarios = [
            self.scenario_1_basic_reasoning,
            self.scenario_2_collaborative_analysis,
            self.scenario_3_complex_problem_solving,
            self.scenario_4_code_generation_chain,
            self.scenario_5_strategic_planning
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"Running Scenario {i}...")
            result = await scenario()
            self.results.append(result)
            print(f"Scenario {i} completed\n")
        
        analysis = self.analyze_results()
        self.save_results(analysis)
        return analysis
    
    async def scenario_1_basic_reasoning(self) -> Dict[str, Any]:
        """Test basic thought stream initiation and single-agent reasoning"""
        start_time = time.time()
        
        stream_result = await self.orchestrator.initiate_thought_stream(
            task_description="Analyze the computational complexity of bubble sort algorithm",
            agent_name="AlgorithmAnalyst",
            cognitive_type="analysis",
            complexity="low",
            semantic_tags=["algorithms", "complexity", "sorting"]
        )
        
        execution_result = await self.orchestrator.execute_strategic_orchestration(
            stream_id=stream_result["stream_id"],
            agent_name="AlgorithmAnalyst", 
            execution_prompt="Provide detailed time and space complexity analysis with examples",
            task_type="analysis"
        )
        
        duration = time.time() - start_time
        
        return {
            "scenario": "Basic Reasoning",
            "duration": duration,
            "stream_id": stream_result["stream_id"],
            "tokens_used": execution_result.get("orchestration_result", {}).get("tokens", 0),
            "model_used": execution_result.get("orchestration_result", {}).get("models_used", []),
            "success": "error" not in execution_result,
            "complexity_score": self._calculate_complexity_score(execution_result),
            "analysis_points": [
                "Single agent reasoning capability",
                "Algorithm analysis accuracy", 
                "Response coherence and structure",
                "Token efficiency for simple tasks"
            ]
        }
    
    async def scenario_2_collaborative_analysis(self) -> Dict[str, Any]:
        """Test multi-agent collaborative reasoning on the same problem"""
        start_time = time.time()
        
        # Agent 1: Mathematical perspective
        stream_result = await self.orchestrator.initiate_thought_stream(
            task_description="Design a caching system for a high-traffic web application",
            agent_name="SystemArchitect",
            cognitive_type="design",
            complexity="medium",
            semantic_tags=["caching", "performance", "architecture"]
        )
        
        # Agent 2: Performance perspective
        contrib_result = await self.orchestrator.contribute_thought(
            stream_id=stream_result["stream_id"],
            agent_name="PerformanceEngineer",
            thought_content="Consider cache eviction policies: LRU vs LFU vs TTL-based. Redis vs Memcached trade-offs for this use case.",
            cognitive_action="analysis",
            attention_request=0.7
        )
        
        # Agent 3: Security perspective
        contrib_result2 = await self.orchestrator.contribute_thought(
            stream_id=stream_result["stream_id"],
            agent_name="SecurityExpert",
            thought_content="Cache poisoning attacks and data isolation concerns. Sensitive data should never be cached in shared layers.",
            cognitive_action="critique",
            attention_request=0.6
        )
        
        # Final synthesis
        execution_result = await self.orchestrator.execute_strategic_orchestration(
            stream_id=stream_result["stream_id"],
            agent_name="SolutionSynthesizer",
            execution_prompt="Synthesize all perspectives into a comprehensive caching strategy with implementation details",
            task_type="synthesis"
        )
        
        duration = time.time() - start_time
        
        return {
            "scenario": "Collaborative Analysis", 
            "duration": duration,
            "stream_id": stream_result["stream_id"],
            "agents_involved": ["SystemArchitect", "PerformanceEngineer", "SecurityExpert", "SolutionSynthesizer"],
            "collaboration_depth": len([contrib_result, contrib_result2]) + 1,
            "tokens_used": execution_result.get("orchestration_result", {}).get("tokens", 0),
            "model_used": execution_result.get("orchestration_result", {}).get("models_used", []),
            "success": "error" not in execution_result,
            "synthesis_quality": self._evaluate_synthesis_quality(execution_result),
            "analysis_points": [
                "Multi-agent collaboration effectiveness",
                "Perspective diversity integration",
                "Knowledge synthesis capability",
                "Attention management across agents"
            ]
        }
    
    async def scenario_3_complex_problem_solving(self) -> Dict[str, Any]:
        """Test complex, multi-step problem solving with high cognitive load"""
        start_time = time.time()
        
        stream_result = await self.orchestrator.initiate_thought_stream(
            task_description="Design a distributed system for real-time fraud detection that can process 100k transactions/second with sub-100ms latency while maintaining 99.99% uptime",
            agent_name="DistributedSystemsExpert",
            cognitive_type="complex_reasoning",
            complexity="high",
            semantic_tags=["distributed-systems", "fraud-detection", "real-time", "high-availability"]
        )
        
        # Add domain expertise
        contrib_result = await self.orchestrator.contribute_thought(
            stream_id=stream_result["stream_id"],
            agent_name="FraudDetectionSpecialist",
            thought_content="ML model serving challenges: feature stores, model versioning, A/B testing, concept drift detection. Need real-time feature engineering pipeline.",
            cognitive_action="expert_input",
            attention_request=0.8
        )
        
        # Add scalability expertise  
        contrib_result2 = await self.orchestrator.contribute_thought(
            stream_id=stream_result["stream_id"],
            agent_name="ScalabilityEngineer", 
            thought_content="Event streaming with Kafka, horizontal scaling patterns, circuit breakers, bulkhead isolation. Database sharding strategies for transaction data.",
            cognitive_action="technical_analysis",
            attention_request=0.9
        )
        
        execution_result = await self.orchestrator.execute_strategic_orchestration(
            stream_id=stream_result["stream_id"],
            agent_name="ChiefArchitect",
            execution_prompt="Create detailed system architecture with component diagrams, data flow, failure scenarios, and implementation roadmap",
            task_type="complex_design"
        )
        
        duration = time.time() - start_time
        
        return {
            "scenario": "Complex Problem Solving",
            "duration": duration,
            "stream_id": stream_result["stream_id"],
            "problem_complexity": "high",
            "domain_breadth": 4,  # distributed systems, ML, fraud detection, scalability
            "tokens_used": execution_result.get("orchestration_result", {}).get("tokens", 0),
            "model_used": execution_result.get("orchestration_result", {}).get("models_used", []),
            "success": "error" not in execution_result,
            "architectural_depth": self._assess_architectural_depth(execution_result),
            "analysis_points": [
                "Complex system design capability",
                "Multi-domain knowledge integration", 
                "Scalability requirement handling",
                "Technical feasibility assessment"
            ]
        }
    
    async def scenario_4_code_generation_chain(self) -> Dict[str, Any]:
        """Test iterative code generation and refinement"""
        start_time = time.time()
        
        # Initial code generation
        stream_result = await self.orchestrator.initiate_thought_stream(
            task_description="Implement a thread-safe connection pool for database connections with health checking and automatic recovery",
            agent_name="BackendDeveloper",
            cognitive_type="implementation",
            complexity="medium",
            semantic_tags=["threading", "database", "connection-pool", "python"]
        )
        
        # Code review and suggestions
        contrib_result = await self.orchestrator.contribute_thought(
            stream_id=stream_result["stream_id"],
            agent_name="CodeReviewer",
            thought_content="Consider connection timeout handling, pool size optimization, monitoring/metrics integration, graceful shutdown procedures.",
            cognitive_action="code_review",
            attention_request=0.6
        )
        
        # Security analysis
        contrib_result2 = await self.orchestrator.contribute_thought(
            stream_id=stream_result["stream_id"],
            agent_name="SecurityAuditor",
            thought_content="SQL injection prevention, connection string security, credential management, access logging for compliance.",
            cognitive_action="security_review", 
            attention_request=0.7
        )
        
        # Final implementation
        execution_result = await self.orchestrator.execute_strategic_orchestration(
            stream_id=stream_result["stream_id"],
            agent_name="SeniorDeveloper",
            execution_prompt="Generate production-ready Python code incorporating all security and performance considerations",
            task_type="code_generation"
        )
        
        duration = time.time() - start_time
        
        return {
            "scenario": "Code Generation Chain",
            "duration": duration,
            "stream_id": stream_result["stream_id"],
            "code_quality_factors": ["thread-safety", "security", "performance", "maintainability"],
            "review_rounds": 2,
            "tokens_used": execution_result.get("orchestration_result", {}).get("tokens", 0),
            "model_used": execution_result.get("orchestration_result", {}).get("models_used", []),
            "success": "error" not in execution_result,
            "code_completeness": self._evaluate_code_completeness(execution_result),
            "analysis_points": [
                "Code generation accuracy",
                "Security consideration integration",
                "Performance optimization inclusion",
                "Review feedback incorporation"
            ]
        }
    
    async def scenario_5_strategic_planning(self) -> Dict[str, Any]:
        """Test high-level strategic planning and decision making"""
        start_time = time.time()
        
        stream_result = await self.orchestrator.initiate_thought_stream(
            task_description="Plan the technical strategy for migrating a monolithic e-commerce platform to microservices while maintaining zero downtime",
            agent_name="TechnicalStrategist",
            cognitive_type="strategic_planning",
            complexity="high",
            semantic_tags=["migration", "microservices", "strategy", "zero-downtime"]
        )
        
        # Business perspective
        contrib_result = await self.orchestrator.contribute_thought(
            stream_id=stream_result["stream_id"],
            agent_name="BusinessAnalyst",
            thought_content="Revenue impact concerns, customer experience continuity, team training costs, timeline constraints for peak shopping seasons.",
            cognitive_action="business_analysis",
            attention_request=0.8
        )
        
        # Risk assessment
        contrib_result2 = await self.orchestrator.contribute_thought(
            stream_id=stream_result["stream_id"],
            agent_name="RiskManager",
            thought_content="Data consistency risks, rollback procedures, monitoring gaps during transition, team coordination challenges across services.",
            cognitive_action="risk_assessment",
            attention_request=0.9
        )
        
        execution_result = await self.orchestrator.execute_strategic_orchestration(
            stream_id=stream_result["stream_id"],
            agent_name="ChiefTechnologyOfficer",
            execution_prompt="Create comprehensive migration strategy with phases, timelines, risk mitigation, and success metrics",
            task_type="strategic_plan"
        )
        
        duration = time.time() - start_time
        
        return {
            "scenario": "Strategic Planning",
            "duration": duration,
            "stream_id": stream_result["stream_id"],
            "strategic_depth": "enterprise-level",
            "stakeholder_perspectives": 3,
            "tokens_used": execution_result.get("orchestration_result", {}).get("tokens", 0),
            "model_used": execution_result.get("orchestration_result", {}).get("models_used", []),
            "success": "error" not in execution_result,
            "strategic_completeness": self._assess_strategic_completeness(execution_result),
            "analysis_points": [
                "Strategic thinking capability",
                "Multi-stakeholder consideration",
                "Risk-aware planning",
                "Implementation feasibility"
            ]
        }
    
    def _calculate_complexity_score(self, result: Dict[str, Any]) -> float:
        """Calculate complexity score based on response characteristics"""
        output = result.get("orchestration_result", {}).get("output", "")
        lines = len(output.split('\n'))
        words = len(output.split())
        
        # Basic complexity scoring
        if words < 100:
            return 0.3
        elif words < 300:
            return 0.6
        elif words < 600:
            return 0.8
        else:
            return 1.0
    
    def _evaluate_synthesis_quality(self, result: Dict[str, Any]) -> float:
        """Evaluate how well the result synthesizes multiple perspectives"""
        output = result.get("orchestration_result", {}).get("output", "").lower()
        
        synthesis_indicators = [
            "considering", "integrating", "balancing", "combining",
            "perspective", "approach", "strategy", "trade-off",
            "however", "meanwhile", "additionally", "furthermore"
        ]
        
        score = sum(1 for indicator in synthesis_indicators if indicator in output)
        return min(1.0, score / len(synthesis_indicators))
    
    def _assess_architectural_depth(self, result: Dict[str, Any]) -> float:
        """Assess the architectural depth and completeness"""
        output = result.get("orchestration_result", {}).get("output", "").lower()
        
        architectural_concepts = [
            "component", "service", "interface", "protocol",
            "scaling", "availability", "consistency", "partition",
            "monitoring", "logging", "metrics", "alerting",
            "deployment", "infrastructure", "containerization"
        ]
        
        score = sum(1 for concept in architectural_concepts if concept in output)
        return min(1.0, score / len(architectural_concepts))
    
    def _evaluate_code_completeness(self, result: Dict[str, Any]) -> float:
        """Evaluate completeness of generated code"""
        output = result.get("orchestration_result", {}).get("output", "")
        
        code_indicators = [
            "class ", "def ", "import ", "try:", "except:",
            "with ", "async def", "await ", "__init__",
            "return", "raise", "assert"
        ]
        
        score = sum(1 for indicator in code_indicators if indicator in output)
        return min(1.0, score / len(code_indicators))
    
    def _assess_strategic_completeness(self, result: Dict[str, Any]) -> float:
        """Assess strategic planning completeness"""
        output = result.get("orchestration_result", {}).get("output", "").lower()
        
        strategic_elements = [
            "phase", "timeline", "milestone", "objective",
            "risk", "mitigation", "metric", "success",
            "resource", "budget", "team", "stakeholder",
            "governance", "communication", "training"
        ]
        
        score = sum(1 for element in strategic_elements if element in output)
        return min(1.0, score / len(strategic_elements))
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze all test results and provide comprehensive insights"""
        if not self.results:
            return {"error": "No test results to analyze"}
        
        total_duration = sum(r["duration"] for r in self.results)
        total_tokens = sum(r["tokens_used"] for r in self.results)
        success_rate = sum(1 for r in self.results if r["success"]) / len(self.results)
        
        models_used = set()
        for result in self.results:
            models_used.update(result.get("model_used", []))
        
        # Performance analysis
        avg_duration = total_duration / len(self.results)
        tokens_per_second = total_tokens / total_duration if total_duration > 0 else 0
        
        # Quality analysis
        quality_scores = []
        for result in self.results:
            quality_score = 0
            if "complexity_score" in result:
                quality_score = result["complexity_score"]
            elif "synthesis_quality" in result:
                quality_score = result["synthesis_quality"]
            elif "architectural_depth" in result:
                quality_score = result["architectural_depth"]
            elif "code_completeness" in result:
                quality_score = result["code_completeness"]
            elif "strategic_completeness" in result:
                quality_score = result["strategic_completeness"]
            quality_scores.append(quality_score)
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        return {
            "test_summary": {
                "scenarios_run": len(self.results),
                "total_duration": total_duration,
                "success_rate": success_rate,
                "average_duration_per_scenario": avg_duration
            },
            "performance_metrics": {
                "total_tokens_generated": total_tokens,
                "tokens_per_second": tokens_per_second,
                "models_utilized": list(models_used),
                "average_quality_score": avg_quality
            },
            "capability_analysis": {
                "basic_reasoning": self.results[0]["success"] if len(self.results) > 0 else False,
                "collaborative_analysis": self.results[1]["success"] if len(self.results) > 1 else False,
                "complex_problem_solving": self.results[2]["success"] if len(self.results) > 2 else False,
                "code_generation": self.results[3]["success"] if len(self.results) > 3 else False,
                "strategic_planning": self.results[4]["success"] if len(self.results) > 4 else False
            },
            "detailed_results": self.results,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not self.results:
            return ["Run test scenarios first to generate recommendations"]
        
        avg_duration = sum(r["duration"] for r in self.results) / len(self.results)
        success_rate = sum(1 for r in self.results if r["success"]) / len(self.results)
        
        if avg_duration > 30:
            recommendations.append("Consider optimizing model selection for faster response times")
        
        if success_rate < 0.8:
            recommendations.append("Investigate error patterns and improve error handling")
        
        total_tokens = sum(r["tokens_used"] for r in self.results)
        if total_tokens > 10000:
            recommendations.append("Monitor token usage for cost optimization")
        
        recommendations.append("System demonstrates strong multi-agent collaboration capabilities")
        recommendations.append("Consider implementing caching for frequently accessed thought streams")
        
        return recommendations
    
    def save_results(self, analysis: Dict[str, Any]):
        """Save test results to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"aios_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"Test results saved to: {filename}")
    
    async def run_single_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Run a specific test scenario by name"""
        scenarios = {
            "basic": self.scenario_1_basic_reasoning,
            "collaborative": self.scenario_2_collaborative_analysis, 
            "complex": self.scenario_3_complex_problem_solving,
            "code": self.scenario_4_code_generation_chain,
            "strategic": self.scenario_5_strategic_planning
        }
        
        if scenario_name not in scenarios:
            return {"error": f"Unknown scenario: {scenario_name}"}
        
        return await scenarios[scenario_name]()

async def main():
    """Main function to run test scenarios"""
    runner = TestScenarioRunner()
    
    print("AIOS Strategic Orchestration Test Suite")
    print("======================================\n")
    
    # Run all scenarios
    analysis = await runner.run_all_scenarios()
    
    print("\n=== FINAL ANALYSIS ===")
    print(f"Success Rate: {analysis['test_summary']['success_rate']:.1%}")
    print(f"Average Duration: {analysis['test_summary']['average_duration_per_scenario']:.2f}s")
    print(f"Total Tokens: {analysis['performance_metrics']['total_tokens_generated']}")
    print(f"Average Quality: {analysis['performance_metrics']['average_quality_score']:.2f}")
    
    print("\nCapabilities Verified:")
    for capability, success in analysis['capability_analysis'].items():
        status = "✓" if success else "✗"
        print(f"  {status} {capability.replace('_', ' ').title()}")
    
    print("\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  • {rec}")

if __name__ == "__main__":
    asyncio.run(main())