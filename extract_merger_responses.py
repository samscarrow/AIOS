#!/usr/bin/env python3
"""
Extract merger implementation responses from cognitive platform logs
"""
import json
import re
from pathlib import Path

def extract_merger_responses():
    """Extract the generated merger implementation code from JSONL logs"""
    
    cognitive_platform_path = Path("C:/Users/sscar/claude-workspace/workshop/cognitive-platform")
    jsonl_file = cognitive_platform_path / "lmstudio_golden_bus.jsonl"
    
    merger_responses = {}
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                # Look for outcome spans with merger task responses
                if (data.get('span') == 'outcome' and 
                    'result' in data.get('attrs', {}) and 
                    data['attrs']['result'].get('ok')):
                    
                    response = data['attrs']['result'].get('response', '')
                    
                    # Look for merger implementation tasks
                    if 'MERGER IMPLEMENTATION TASK' in response:
                        # Extract task type from response
                        if 'Unified Configuration' in response:
                            merger_responses['unified_config'] = response
                        elif 'Real Provider Kernel' in response:
                            merger_responses['provider_kernel'] = response  
                        elif 'Production MCP Bridge' in response:
                            merger_responses['mcp_bridge'] = response
                        elif 'Learning System Integ' in response:
                            merger_responses['learning_integration'] = response
                        elif 'Cost-Aware Decision' in response:
                            merger_responses['cost_optimization'] = response
                    elif 'Complex task requiring careful analysis' in response and 'MERGER' in response:
                        # This is likely the Real Provider Kernel task
                        merger_responses['provider_kernel'] = response
                        
            except json.JSONDecodeError:
                continue
    
    return merger_responses

def save_responses_to_files(responses):
    """Save extracted responses to individual files for review"""
    output_dir = Path("C:/Users/sscar/claude-workspace/workshop/aios/extracted_responses")
    output_dir.mkdir(exist_ok=True)
    
    for task_id, response in responses.items():
        output_file = output_dir / f"{task_id}_response.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response)
        print(f"Saved {task_id} response to {output_file}")

if __name__ == "__main__":
    print("Extracting merger implementation responses...")
    responses = extract_merger_responses()
    
    print(f"Found {len(responses)} merger task responses:")
    for task_id in responses.keys():
        print(f"  - {task_id}")
    
    save_responses_to_files(responses)
    print("Extraction complete!")