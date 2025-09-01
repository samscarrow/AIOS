#!/usr/bin/env python3
"""
Extract task data from Claude conversation history for router calibration
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from collections import Counter

def extract_tasks_from_jsonl(file_path: str) -> List[Dict]:
    """Extract user messages that look like tasks from a JSONL file"""
    tasks = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    
                    # Look for user messages
                    if entry.get('type') == 'message' and entry.get('role') == 'user':
                        content = entry.get('content', '')
                        
                        # Skip very short messages
                        if len(content) < 10:
                            continue
                        
                        # Extract text content (handle different formats)
                        if isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    text_parts.append(item.get('text', ''))
                            content = ' '.join(text_parts)
                        
                        # Filter for task-like messages
                        if is_task_like(content):
                            tasks.append({
                                'task_description': clean_text(content),
                                'source_file': os.path.basename(file_path)
                            })
                
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return tasks

def is_task_like(text: str) -> bool:
    """Determine if text looks like a task/request"""
    # Task indicators
    task_patterns = [
        r'\b(write|create|implement|build|develop|make|generate|design|fix|debug|solve|analyze|explain|help|show|find|search|update|modify|refactor|optimize|test|review|check|integrate|setup|configure)\b',
        r'\b(can you|could you|please|would you|I need|I want|let\'s|we need|how to|how do|what is|why does)\b',
        r'[\?\!]',  # Questions or commands
    ]
    
    # Skip patterns (not tasks)
    skip_patterns = [
        r'^(yes|no|okay|sure|thanks|thank you|got it|understood|continue|proceed)[\s\.\!]*$',
        r'^[\d\s\.\,\-\+\*\/\=]+$',  # Just numbers/math
        r'^[^\w\s]{1,10}$',  # Just symbols
    ]
    
    text_lower = text.lower().strip()
    
    # Check skip patterns
    for pattern in skip_patterns:
        if re.match(pattern, text_lower):
            return False
    
    # Check task patterns
    for pattern in task_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove markdown code blocks for cleaner text
    text = re.sub(r'```[\s\S]*?```', '[code block]', text)
    
    # Truncate very long texts
    if len(text) > 500:
        text = text[:497] + '...'
    
    return text.strip()

def classify_task(task_text: str) -> Tuple[str, str]:
    """Classify task type and complexity based on content"""
    text_lower = task_text.lower()
    
    # Task type classification
    task_type = "general"
    
    type_keywords = {
        'code_generation': ['write', 'implement', 'create function', 'create class', 'code', 'program'],
        'debugging': ['fix', 'debug', 'error', 'bug', 'issue', 'problem', 'broken', 'failing'],
        'analysis': ['analyze', 'investigate', 'examine', 'review', 'assess', 'evaluate', 'performance'],
        'system_design': ['design', 'architect', 'plan', 'structure', 'system', 'infrastructure'],
        'data_analysis': ['data', 'dataset', 'statistics', 'csv', 'json', 'database', 'query'],
        'testing': ['test', 'unit test', 'integration', 'pytest', 'assert', 'validate'],
        'documentation': ['document', 'readme', 'comment', 'explain', 'description', 'guide'],
        'refactoring': ['refactor', 'clean', 'optimize', 'improve', 'restructure', 'organize'],
        'configuration': ['config', 'setup', 'install', 'deploy', 'environment', 'settings']
    }
    
    # Score each type
    type_scores = {}
    for type_name, keywords in type_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            type_scores[type_name] = score
    
    if type_scores:
        task_type = max(type_scores, key=type_scores.get)
    
    # Complexity classification
    complexity = "medium"
    
    # Simple indicators
    simple_indicators = ['simple', 'basic', 'easy', 'quick', 'small', 'trivial']
    complex_indicators = ['complex', 'advanced', 'distributed', 'scalable', 'production', 
                         'enterprise', 'architecture', 'multi-threaded', 'concurrent', 'async']
    
    word_count = len(task_text.split())
    
    if any(ind in text_lower for ind in complex_indicators) or word_count > 100:
        complexity = "complex"
    elif any(ind in text_lower for ind in simple_indicators) or word_count < 20:
        complexity = "simple"
    elif word_count > 150:
        complexity = "very_complex"
    
    return task_type, complexity

def extract_all_tasks(claude_dir: str = "~/.claude") -> pd.DataFrame:
    """Extract tasks from all Claude conversation files"""
    claude_path = Path(os.path.expanduser(claude_dir))
    
    # Focus on AIOS and workshop projects
    target_patterns = [
        "*aios*.jsonl",
        "*workshop*.jsonl",
        "*cognitive*.jsonl",
        "*orchestrat*.jsonl"
    ]
    
    all_tasks = []
    
    # Find relevant files
    project_dir = claude_path / "projects"
    if project_dir.exists():
        for pattern in target_patterns:
            for jsonl_file in project_dir.rglob(pattern):
                print(f"Processing {jsonl_file.name}...")
                tasks = extract_tasks_from_jsonl(str(jsonl_file))
                all_tasks.extend(tasks)
    
    # Also check recent large files
    large_files = [
        "C--Users-sscar-claude-workspace-workshop-aios/3f6acb2b-cb7a-447b-afc6-e9fd53be3121.jsonl",
        "C--Users-sscar-claude-workspace-workshop-aios/301df31a-8fbb-45b6-b7dc-a3515e855bb1.jsonl",
        "C--Users-sscar-claude-workspace-workshop-aios/bcc37563-a3e1-414d-bd9d-8bc17b61fe5b.jsonl"
    ]
    
    for file_path in large_files:
        full_path = project_dir / file_path
        if full_path.exists():
            print(f"Processing large file {full_path.name}...")
            tasks = extract_tasks_from_jsonl(str(full_path))
            all_tasks.extend(tasks)
    
    print(f"\nExtracted {len(all_tasks)} total tasks")
    
    # Classify tasks
    classified_tasks = []
    for task in all_tasks:
        task_type, complexity = classify_task(task['task_description'])
        classified_tasks.append({
            'task_description': task['task_description'],
            'task_type': task_type,
            'complexity': complexity,
            'source': task['source_file']
        })
    
    df = pd.DataFrame(classified_tasks)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['task_description'])
    
    print(f"After deduplication: {len(df)} unique tasks")
    
    return df

def analyze_tasks(df: pd.DataFrame):
    """Analyze extracted tasks"""
    print("\n" + "="*60)
    print("Task Analysis")
    print("="*60)
    
    print(f"\nTotal unique tasks: {len(df)}")
    
    print("\nTask Type Distribution:")
    print(df['task_type'].value_counts())
    
    print("\nComplexity Distribution:")
    print(df['complexity'].value_counts())
    
    print("\nSample tasks by type:")
    for task_type in df['task_type'].unique()[:5]:
        print(f"\n{task_type.upper()}:")
        samples = df[df['task_type'] == task_type]['task_description'].head(2)
        for i, sample in enumerate(samples, 1):
            print(f"  {i}. {sample[:100]}...")

def main():
    """Main extraction and analysis"""
    print("Extracting Claude conversation tasks for router calibration...")
    
    # Extract tasks
    df = extract_all_tasks()
    
    if len(df) == 0:
        print("No tasks found. Generating synthetic data instead...")
        # Fall back to synthetic data
        from calibrate_router import RouterCalibrator
        calibrator = RouterCalibrator()
        df = calibrator.generate_synthetic_data(n_samples=1000)
    
    # Analyze
    analyze_tasks(df)
    
    # Save to CSV for calibration
    output_path = "claude_tasks_for_calibration.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} tasks to {output_path}")
    
    # Also save a sample for review
    sample_path = "claude_tasks_sample.json"
    df.head(20).to_json(sample_path, orient='records', indent=2)
    print(f"Saved sample to {sample_path}")
    
    return df

if __name__ == "__main__":
    df = main()