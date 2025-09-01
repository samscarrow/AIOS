#!/usr/bin/env python3
"""
Clean separation between Engine (programmatic) and Reasoning Model (strategic decisions)
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium" 
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

@dataclass
class ModelCapabilities:
    """Structured model metadata for engine"""
    name: str
    memory_gb: float
    context_limit: int
    specializations: List[str]
    avg_throughput: float
    success_rate: float
    cost_per_token: float = 0.0

@dataclass
class TaskMetadata:
    """Structured task information for reasoning"""
    prompt_length: int
    task_type: str
    complexity: TaskComplexity
    expected_output_length: int
    quality_requirements: List[str]
    time_constraints: Optional[int] = None

@dataclass 
class ReasoningDecision:
    """Structured response from reasoning model"""
    recommended_strategy: str
    model_selection: List[str]
    token_allocation: int
    temperature: float
    confidence: float
    reasoning: str
    fallback_strategies: List[str]

class OrchestrationEngine:
    """Programmatic engine that manages resources and executes decisions"""
    
    def __init__(self):
        self.available_models: Dict[str, ModelCapabilities] = {}
        self.current_memory_usage = 0.0
        self.memory_limit = 15.0
        self.performance_history = []
        
    def register_model(self, model: ModelCapabilities):
        """Register model with its capabilities"""
        self.available_models[model.name] = model
        
    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state for reasoning model"""
        return {
            'available_models': [
                {
                    'name': model.name,
                    'memory_gb': model.memory_gb,
                    'context_limit': model.context_limit,
                    'specializations': model.specializations,
                    'avg_throughput': model.avg_throughput,
                    'success_rate': model.success_rate,
                    'currently_loaded': self._is_model_loaded(model.name)
                }
                for model in self.available_models.values()
            ],
            'current_memory_usage': self.current_memory_usage,
            'memory_limit': self.memory_limit,
            'recent_performance': self.performance_history[-5:] if self.performance_history else []
        }
        
    def validate_decision(self, decision: ReasoningDecision, task: TaskMetadata) -> bool:
        """Validate that reasoning decision is executable"""
        # Check model availability
        for model_name in decision.model_selection:
            if model_name not in self.available_models:
                return False
                
        # Check memory constraints
        required_memory = sum(
            self.available_models[name].memory_gb 
            for name in decision.model_selection
        )
        if required_memory > self.memory_limit:
            return False
            
        # Check token allocation is reasonable
        if decision.token_allocation > 4096:  # Conservative max
            return False
            
        return True
    
    async def execute_decision(self, decision: ReasoningDecision, task: TaskMetadata) -> Dict[str, Any]:
        """Execute the reasoning model's decision"""
        if not self.validate_decision(decision, task):
            raise ValueError("Invalid decision cannot be executed")
            
        # Load required models
        for model_name in decision.model_selection:
            await self._ensure_model_loaded(model_name)
            
        # Execute with specified parameters
        result = await self._execute_with_params(
            models=decision.model_selection,
            strategy=decision.recommended_strategy,
            max_tokens=decision.token_allocation,
            temperature=decision.temperature,
            task=task
        )
        
        # Record performance
        self._record_performance(decision, result)
        
        return result
    
    def _is_model_loaded(self, model_name: str) -> bool:
        """Check if model is currently loaded (stub)"""
        # Would integrate with actual model management
        return False
        
    async def _ensure_model_loaded(self, model_name: str):
        """Load model if needed (stub)"""
        # Would integrate with actual model loading
        pass
        
    async def _execute_with_params(self, models, strategy, max_tokens, temperature, task):
        """Execute orchestration with specified parameters (stub)"""
        # Would integrate with actual orchestration execution
        return {
            'output': 'Executed successfully',
            'tokens': max_tokens // 2,
            'throughput': 50.0,
            'strategy_used': strategy
        }
        
    def _record_performance(self, decision: ReasoningDecision, result: Dict[str, Any]):
        """Record performance for future decisions"""
        self.performance_history.append({
            'strategy': decision.recommended_strategy,
            'models': decision.model_selection,
            'tokens_allocated': decision.token_allocation,
            'tokens_used': result.get('tokens', 0),
            'throughput': result.get('throughput', 0),
            'success': result.get('success', True)
        })

class ContextFreeReasoner:
    """Context-free reasoning model for strategic decisions"""
    
    def __init__(self, reasoning_model_api):
        self.reasoning_model = reasoning_model_api
        
    async def make_orchestration_decision(
        self, 
        task: TaskMetadata, 
        system_state: Dict[str, Any]
    ) -> ReasoningDecision:
        """Make strategic decision about how to orchestrate the task"""
        
        decision_prompt = self._build_decision_prompt(task, system_state)
        
        response = await self.reasoning_model.query(decision_prompt)
        
        # Parse structured response
        decision = self._parse_decision_response(response)
        
        return decision
        
    def _build_decision_prompt(self, task: TaskMetadata, system_state: Dict[str, Any]) -> str:
        """Build structured prompt for reasoning model"""
        return f"""You are making orchestration decisions. Analyze the task and system state, then provide a structured decision.

TASK ANALYSIS:
- Type: {task.task_type}
- Complexity: {task.complexity.value}
- Prompt length: {task.prompt_length} chars
- Expected output: {task.expected_output_length} tokens
- Quality requirements: {task.quality_requirements}

SYSTEM STATE:
- Available models: {len(system_state['available_models'])}
- Memory usage: {system_state['current_memory_usage']:.1f}GB / {system_state['memory_limit']:.1f}GB
- Recent performance: {system_state['recent_performance']}

MODEL OPTIONS:
{json.dumps(system_state['available_models'], indent=2)}

DECISION REQUIRED:
Provide your recommendation in this exact JSON format:
{{
    "recommended_strategy": "concurrent|chain|specialized|hybrid",
    "model_selection": ["model1", "model2"],
    "token_allocation": 1024,
    "temperature": 0.7,
    "confidence": 0.85,
    "reasoning": "Brief explanation of why this approach",
    "fallback_strategies": ["alternative1", "alternative2"]
}}

Base your decision on:
1. Task complexity and requirements
2. Model capabilities and specializations
3. Memory constraints
4. Performance history patterns
5. Quality vs speed tradeoffs

Provide JSON decision now:"""

    def _parse_decision_response(self, response: str) -> ReasoningDecision:
        """Parse structured response into decision object"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
                
            json_str = response[json_start:json_end]
            decision_dict = json.loads(json_str)
            
            return ReasoningDecision(
                recommended_strategy=decision_dict['recommended_strategy'],
                model_selection=decision_dict['model_selection'],
                token_allocation=decision_dict['token_allocation'],
                temperature=decision_dict['temperature'],
                confidence=decision_dict['confidence'],
                reasoning=decision_dict['reasoning'],
                fallback_strategies=decision_dict.get('fallback_strategies', [])
            )
            
        except Exception as e:
            # Fallback to safe defaults
            return ReasoningDecision(
                recommended_strategy="specialized",
                model_selection=["qwen/qwen2.5-coder-14b"],
                token_allocation=1024,
                temperature=0.7,
                confidence=0.5,
                reasoning=f"Failed to parse response: {e}",
                fallback_strategies=["concurrent", "chain"]
            )

class IntegratedOrchestrator:
    """Integrated system combining engine and reasoning"""
    
    def __init__(self, reasoning_model_api):
        self.engine = OrchestrationEngine()
        self.reasoner = ContextFreeReasoner(reasoning_model_api)
        
        # Initialize with model capabilities
        self._setup_model_registry()
        
    def _setup_model_registry(self):
        """Register available models with their capabilities"""
        models = [
            ModelCapabilities(
                name="qwen/qwen2.5-coder-14b",
                memory_gb=9.0,
                context_limit=4096,
                specializations=["code", "programming", "analysis"],
                avg_throughput=45.0,
                success_rate=0.92
            ),
            ModelCapabilities(
                name="qwen3-8b@q4_k_m", 
                memory_gb=8.0,
                context_limit=4096,
                specializations=["reasoning", "analysis", "general"],
                avg_throughput=38.0,
                success_rate=0.88
            ),
            ModelCapabilities(
                name="qwen/qwen3-4b-thinking-2507",
                memory_gb=3.0,
                context_limit=2048,
                specializations=["reasoning", "thinking", "logic"],
                avg_throughput=65.0,
                success_rate=0.85
            )
        ]
        
        for model in models:
            self.engine.register_model(model)
            
    async def orchestrate_with_reasoning(self, prompt: str, task_type: str) -> Dict[str, Any]:
        """Main orchestration method using engine + reasoning separation"""
        
        # Analyze task programmatically
        task = TaskMetadata(
            prompt_length=len(prompt),
            task_type=task_type,
            complexity=self._assess_complexity(prompt, task_type),
            expected_output_length=self._estimate_output_length(prompt, task_type),
            quality_requirements=self._extract_quality_requirements(prompt)
        )
        
        # Get current system state
        system_state = self.engine.get_system_state()
        
        # Let reasoning model make strategic decision
        decision = await self.reasoner.make_orchestration_decision(task, system_state)
        
        # Validate and execute decision
        result = await self.engine.execute_decision(decision, task)
        
        # Return enriched result
        return {
            **result,
            'reasoning_used': decision.reasoning,
            'strategy_confidence': decision.confidence,
            'task_complexity': task.complexity.value,
            'decision_metadata': {
                'recommended_strategy': decision.recommended_strategy,
                'models_selected': decision.model_selection,
                'tokens_allocated': decision.token_allocation
            }
        }
    
    def _assess_complexity(self, prompt: str, task_type: str) -> TaskComplexity:
        """Programmatically assess task complexity"""
        if len(prompt) < 100:
            return TaskComplexity.SIMPLE
        elif len(prompt) < 500:
            return TaskComplexity.MEDIUM
        elif len(prompt) < 1500:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.VERY_COMPLEX
            
    def _estimate_output_length(self, prompt: str, task_type: str) -> int:
        """Estimate expected output token length"""
        if task_type == "code":
            return min(len(prompt) * 2, 2048)
        elif task_type == "reasoning":
            return min(len(prompt) * 3, 1500)
        else:
            return min(len(prompt), 1024)
            
    def _extract_quality_requirements(self, prompt: str) -> List[str]:
        """Extract quality requirements from prompt"""
        requirements = []
        if "detailed" in prompt.lower():
            requirements.append("detailed")
        if "accurate" in prompt.lower():
            requirements.append("accurate")  
        if "fast" in prompt.lower():
            requirements.append("fast")
        return requirements

# Mock reasoning model API for testing
class MockReasoningAPI:
    async def query(self, prompt: str) -> str:
        # Simulate reasoning model response with structured JSON
        return '''Based on the task analysis, I recommend:

{
    "recommended_strategy": "specialized",
    "model_selection": ["qwen/qwen2.5-coder-14b"],
    "token_allocation": 1024,
    "temperature": 0.3,
    "confidence": 0.87,
    "reasoning": "Code generation task with medium complexity requires specialized model with focused temperature",
    "fallback_strategies": ["chain", "concurrent"]
}

This approach balances quality and efficiency for the given task.'''

async def demonstrate_separation():
    """Demonstrate the engine + reasoning model separation"""
    print("🧠 ENGINE + REASONING MODEL SEPARATION DEMO")
    print("=" * 60)
    
    # Initialize integrated orchestrator
    mock_api = MockReasoningAPI()
    orchestrator = IntegratedOrchestrator(mock_api)
    
    # Test with a code generation task
    test_prompt = "Implement a load balancing algorithm for model selection in Python"
    
    result = await orchestrator.orchestrate_with_reasoning(test_prompt, "code")
    
    print("📋 ORCHESTRATION RESULT:")
    print(f"Strategy: {result['decision_metadata']['recommended_strategy']}")
    print(f"Models: {result['decision_metadata']['models_selected']}")  
    print(f"Tokens allocated: {result['decision_metadata']['tokens_allocated']}")
    print(f"Confidence: {result['strategy_confidence']:.2f}")
    print(f"Complexity: {result['task_complexity']}")
    print(f"Reasoning: {result['reasoning_used']}")
    
    print(f"\n✅ Clean separation achieved:")
    print("- Engine handled metadata, validation, execution")  
    print("- Reasoning model made strategic decisions")
    print("- Structured interface enabled seamless integration")

if __name__ == "__main__":
    asyncio.run(demonstrate_separation())