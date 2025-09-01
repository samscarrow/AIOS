#!/usr/bin/env python3
"""
Cognitive Feedback Loop Experiment
Tests: LLM Response → Nomic Embedder → Cognitive Preprocessor → Routing

This explores what happens when we feed LLM outputs back into the cognitive system
to create recursive reasoning loops and emergent behaviors.
"""

import asyncio
import json
import time
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from unified.production_task_router import UnifiedProductionRouter
from orchestrator.aios_strategic_orchestrator_routed import AIOSStrategicOrchestratorRouted
from sentence_transformers import SentenceTransformer

class CognitiveFeedbackExplorer:
    """Explores cognitive feedback loops and emergent behaviors"""
    
    def __init__(self):
        self.orchestrator = AIOSStrategicOrchestratorRouted()
        self.router = self.orchestrator.router
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Nomic-style embedder
        self.feedback_history = []
        
    async def query_lmstudio(self, prompt: str) -> Dict[str, Any]:
        """Query LMStudio for initial response"""
        import aiohttp
        
        try:
            url = "http://localhost:1234/v1/chat/completions"
            payload = {
                "model": "qwen/qwen3-8b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "response": data['choices'][0]['message']['content']
                        }
                    else:
                        return {"success": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_embeddings(self, text: str) -> np.ndarray:
        """Create embeddings using nomic-style encoder"""
        return self.embedder.encode(text)
    
    def analyze_embedding_space(self, embedding: np.ndarray) -> Dict[str, Any]:
        """Analyze embedding characteristics and patterns"""
        
        # Basic statistics
        mean_val = float(np.mean(embedding))
        std_val = float(np.std(embedding))
        norm_val = float(np.linalg.norm(embedding))
        
        # Dimensionality analysis
        high_dims = np.sum(np.abs(embedding) > 0.1)  # High-activity dimensions
        dominant_dims = np.argsort(np.abs(embedding))[-10:]  # Top 10 dimensions
        
        # Semantic density (rough approximation)
        semantic_density = float(high_dims / len(embedding))
        
        # Create analysis prompt based on embedding characteristics
        if semantic_density > 0.3:
            complexity_hint = "complex, multi-dimensional"
        elif semantic_density > 0.15:
            complexity_hint = "moderate complexity"
        else:
            complexity_hint = "simple, focused"
            
        if norm_val > 10:
            intensity_hint = "high semantic intensity"
        elif norm_val > 5:
            intensity_hint = "moderate semantic load"
        else:
            intensity_hint = "low semantic activation"
        
        return {
            "mean": mean_val,
            "std": std_val,
            "norm": norm_val,
            "semantic_density": semantic_density,
            "high_activity_dims": int(high_dims),
            "dominant_dimensions": dominant_dims.tolist(),
            "complexity_hint": complexity_hint,
            "intensity_hint": intensity_hint,
            "embedding_signature": f"semantic_{complexity_hint.replace(' ', '_')}_{intensity_hint.replace(' ', '_')}"
        }
    
    def generate_cognitive_meta_prompt(self, embedding_analysis: Dict[str, Any], original_response: str) -> str:
        """Generate a meta-cognitive prompt based on embedding analysis"""
        
        signature = embedding_analysis['embedding_signature']
        complexity = embedding_analysis['complexity_hint']
        intensity = embedding_analysis['intensity_hint']
        density = embedding_analysis['semantic_density']
        
        meta_prompt = f"""
Meta-Cognitive Analysis Request:

Original Response Embedding Analysis:
- Semantic Signature: {signature}
- Complexity Profile: {complexity} 
- Semantic Intensity: {intensity}
- Activation Density: {density:.3f} ({embedding_analysis['high_activity_dims']}/{len(embedding_analysis.get('dominant_dimensions', []))})

Original Response Preview: "{original_response[:200]}..."

Based on this embedding analysis, what deeper cognitive patterns emerge? 
What are the implicit assumptions, missing perspectives, or unexplored dimensions?
How could this response be refined through recursive cognitive processing?

Provide a meta-analysis that explores:
1. Hidden cognitive biases in the original response
2. Unexplored conceptual dimensions 
3. Potential recursive improvements
4. Alternative framing perspectives
"""
        
        return meta_prompt.strip()
    
    async def run_feedback_loop_experiment(self, initial_prompt: str, max_iterations: int = 3):
        """Run cognitive feedback loop experiment"""
        
        print(f"🔄 COGNITIVE FEEDBACK LOOP EXPERIMENT")
        print("=" * 60)
        print(f"Initial Prompt: {initial_prompt}")
        print(f"Max Iterations: {max_iterations}")
        print("=" * 60)
        
        current_prompt = initial_prompt
        
        for iteration in range(max_iterations):
            print(f"\n🔄 ITERATION {iteration + 1}")
            print("-" * 40)
            
            # Step 1: Route the current prompt
            print("📊 Step 1: Routing current prompt...")
            routing_start = time.perf_counter()
            routing_result = await self.router.route(current_prompt)
            routing_time = (time.perf_counter() - routing_start) * 1000
            
            print(f"   Type: {routing_result.task_type}")
            print(f"   Complexity: {routing_result.complexity}")
            print(f"   Confidence: {routing_result.confidence:.3f}")
            print(f"   Routing time: {routing_time:.1f}ms")
            
            # Step 2: Get LLM response  
            print("\n🤖 Step 2: Querying LMStudio...")
            lm_result = await self.query_lmstudio(current_prompt)
            
            if not lm_result['success']:
                print(f"   ❌ LMStudio query failed: {lm_result['error']}")
                break
                
            response = lm_result['response']
            print(f"   ✅ Response received ({len(response)} chars)")
            print(f"   Preview: {response[:150]}...")
            
            # Step 3: Create embeddings of the response
            print("\n🧠 Step 3: Creating embeddings...")
            embedding_start = time.perf_counter()
            response_embedding = self.create_embeddings(response)
            embedding_time = (time.perf_counter() - embedding_start) * 1000
            
            print(f"   ✅ Embedding created ({len(response_embedding)} dims)")
            print(f"   Embedding time: {embedding_time:.1f}ms")
            
            # Step 4: Analyze embedding space
            print("\n🔍 Step 4: Analyzing embedding space...")
            embedding_analysis = self.analyze_embedding_space(response_embedding)
            
            print(f"   Semantic Density: {embedding_analysis['semantic_density']:.3f}")
            print(f"   Complexity Profile: {embedding_analysis['complexity_hint']}")
            print(f"   Intensity: {embedding_analysis['intensity_hint']}")
            print(f"   High-Activity Dims: {embedding_analysis['high_activity_dims']}")
            
            # Step 5: Generate meta-cognitive prompt
            print("\n🎭 Step 5: Generating meta-cognitive prompt...")
            if iteration < max_iterations - 1:  # Don't generate for last iteration
                meta_prompt = self.generate_cognitive_meta_prompt(embedding_analysis, response)
                print(f"   ✅ Meta-prompt generated ({len(meta_prompt)} chars)")
                print(f"   Meta-prompt preview: {meta_prompt[:100]}...")
                
                # This becomes the next iteration's prompt
                current_prompt = meta_prompt
            
            # Step 6: Route the meta-cognitive prompt (if not last iteration)
            if iteration < max_iterations - 1:
                print("\n🔄 Step 6: Routing meta-cognitive prompt...")
                meta_routing_result = await self.router.route(meta_prompt)
                
                print(f"   Meta-Type: {meta_routing_result.task_type}")
                print(f"   Meta-Complexity: {meta_routing_result.complexity}") 
                print(f"   Meta-Confidence: {meta_routing_result.confidence:.3f}")
                
                # Store feedback loop data
                feedback_data = {
                    "iteration": iteration + 1,
                    "original_prompt": initial_prompt if iteration == 0 else "meta_prompt",
                    "routing": routing_result.to_dict(),
                    "lm_response": response,
                    "embedding_analysis": embedding_analysis,
                    "meta_routing": meta_routing_result.to_dict() if iteration < max_iterations - 1 else None,
                    "meta_prompt": meta_prompt if iteration < max_iterations - 1 else None
                }
                
                self.feedback_history.append(feedback_data)
        
        # Analysis of feedback loop
        print(f"\n📈 FEEDBACK LOOP ANALYSIS")
        print("=" * 60)
        
        if len(self.feedback_history) > 1:
            # Track how routing decisions evolved
            complexities = [item['routing']['complexity'] for item in self.feedback_history]
            confidences = [item['routing']['confidence'] for item in self.feedback_history]
            types = [item['routing']['task_type'] for item in self.feedback_history]
            
            print("🔄 Cognitive Evolution:")
            for i, (complexity, confidence, task_type) in enumerate(zip(complexities, confidences, types)):
                print(f"   Iteration {i+1}: {task_type} ({complexity}) - conf: {confidence:.3f}")
            
            # Semantic density evolution
            densities = [item['embedding_analysis']['semantic_density'] for item in self.feedback_history]
            print(f"\n🧠 Semantic Density Evolution:")
            for i, density in enumerate(densities):
                print(f"   Iteration {i+1}: {density:.3f}")
            
            # Check for convergence or divergence patterns
            if len(confidences) >= 2:
                confidence_trend = confidences[-1] - confidences[0]
                density_trend = densities[-1] - densities[0]
                
                print(f"\n📊 Convergence Analysis:")
                print(f"   Confidence trend: {confidence_trend:+.3f}")
                print(f"   Semantic density trend: {density_trend:+.3f}")
                
                if abs(confidence_trend) < 0.1 and abs(density_trend) < 0.1:
                    print("   🎯 System appears to be converging")
                elif confidence_trend > 0.2 or density_trend > 0.2:
                    print("   📈 System showing increasing complexity")
                else:
                    print("   🌊 System showing dynamic evolution")
        
        # Save detailed results
        output_file = f"cognitive_feedback_experiment_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(self.feedback_history, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        return self.feedback_history
    
    async def run_comparative_experiments(self):
        """Run multiple experiments with different initial prompts"""
        
        print("🧪 COMPARATIVE COGNITIVE FEEDBACK EXPERIMENTS")
        print("=" * 80)
        
        test_scenarios = [
            "Design a machine learning system for fraud detection",
            "Debug a performance issue in a distributed system", 
            "Explain the philosophical implications of artificial consciousness"
        ]
        
        all_experiments = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n🔬 EXPERIMENT {i}: {scenario}")
            print("=" * 60)
            
            # Reset history for each experiment
            self.feedback_history = []
            
            # Run feedback loop
            experiment_result = await self.run_feedback_loop_experiment(scenario, max_iterations=3)
            all_experiments.append({
                "scenario": scenario,
                "results": experiment_result
            })
        
        # Cross-experiment analysis
        print(f"\n🔍 CROSS-EXPERIMENT ANALYSIS")
        print("=" * 80)
        
        for i, exp in enumerate(all_experiments, 1):
            print(f"\n📊 Experiment {i}: {exp['scenario'][:50]}...")
            if exp['results']:
                final_confidence = exp['results'][-1]['routing']['confidence']
                final_density = exp['results'][-1]['embedding_analysis']['semantic_density']
                complexity_changes = len(set(r['routing']['complexity'] for r in exp['results']))
                
                print(f"   Final confidence: {final_confidence:.3f}")
                print(f"   Final semantic density: {final_density:.3f}")
                print(f"   Complexity transitions: {complexity_changes}")
        
        return all_experiments

async def main():
    """Main experiment runner"""
    explorer = CognitiveFeedbackExplorer()
    
    # Single experiment
    print("Running single feedback loop experiment...")
    await explorer.run_feedback_loop_experiment(
        "How can artificial intelligence systems develop genuine understanding rather than just pattern matching?",
        max_iterations=3
    )
    
    print("\n" + "="*80)
    print("🎉 COGNITIVE FEEDBACK LOOP EXPERIMENT COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())