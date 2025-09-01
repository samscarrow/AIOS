#!/usr/bin/env python3
"""
Extract tasks from Claude todo files for router calibration
"""

import json
import os
import glob
from pathlib import Path
import pandas as pd
import re

def extract_tasks_from_todos(todos_dir: str = "~/.claude/todos") -> pd.DataFrame:
    """Extract all tasks from Claude todo files"""
    todos_path = Path(os.path.expanduser(todos_dir))
    
    all_tasks = []
    
    # Process all JSON files in todos directory
    for json_file in todos_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content and content != "[]":
                    todos = json.loads(content)
                    
                    for todo in todos:
                        if isinstance(todo, dict) and 'content' in todo:
                            task_text = todo.get('content', '')
                            if len(task_text) > 10:  # Skip very short entries
                                all_tasks.append({
                                    'task_description': task_text,
                                    'status': todo.get('status', 'unknown'),
                                    'source': json_file.name[:20]
                                })
        except (json.JSONDecodeError, Exception) as e:
            continue
    
    print(f"Extracted {len(all_tasks)} tasks from todo files")
    return pd.DataFrame(all_tasks)

def classify_task(task_text: str) -> tuple:
    """Classify task type and complexity"""
    text_lower = task_text.lower()
    
    # Task type patterns
    type_patterns = {
        'code_generation': r'\b(write|create|implement|build|develop|generate)\b.*\b(function|class|code|script|module|component)\b',
        'debugging': r'\b(fix|debug|resolve|troubleshoot|error|issue|problem|failing|broken)\b',
        'testing': r'\b(test|verify|validate|check|ensure|confirm)\b',
        'configuration': r'\b(config|setup|install|deploy|configure|update.*config|restart)\b',
        'analysis': r'\b(analyze|investigate|examine|review|assess|evaluate|check)\b',
        'documentation': r'\b(document|readme|update.*doc|write.*guide|explain)\b',
        'refactoring': r'\b(refactor|clean|optimize|improve|restructure)\b',
        'system_design': r'\b(design|architect|plan|structure)\b.*\b(system|infrastructure|architecture)\b',
        'integration': r'\b(integrate|connect|link|bridge|combine)\b',
        'data_processing': r'\b(process|parse|extract|transform|convert)\b.*\b(data|json|csv|file)\b'
    }
    
    # Find matching type
    task_type = 'general'
    for type_name, pattern in type_patterns.items():
        if re.search(pattern, text_lower):
            task_type = type_name
            break
    
    # Complexity assessment
    complexity = 'medium'
    
    # Word count factor
    word_count = len(task_text.split())
    
    # Complexity indicators
    simple_words = ['simple', 'basic', 'small', 'minimal', 'test', 'check', 'verify']
    complex_words = ['complex', 'multiple', 'distributed', 'architecture', 'infrastructure', 
                     'production', 'enterprise', 'comprehensive', 'advanced']
    
    if any(w in text_lower for w in complex_words) or word_count > 50:
        complexity = 'complex'
    elif any(w in text_lower for w in simple_words) or word_count < 15:
        complexity = 'simple'
    elif 'implement' in text_lower and 'system' in text_lower:
        complexity = 'complex'
    
    return task_type, complexity

def prepare_calibration_data():
    """Prepare calibration data from Claude todos"""
    
    # Extract tasks
    df = extract_tasks_from_todos()
    
    if len(df) == 0:
        print("No tasks found in todo files")
        return None
    
    # Classify tasks
    df['task_type'] = df['task_description'].apply(lambda x: classify_task(x)[0])
    df['complexity'] = df['task_description'].apply(lambda x: classify_task(x)[1])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['task_description'])
    
    print(f"\nTask Statistics:")
    print(f"Total unique tasks: {len(df)}")
    print(f"\nTask Type Distribution:")
    print(df['task_type'].value_counts())
    print(f"\nComplexity Distribution:")
    print(df['complexity'].value_counts())
    
    # Save for calibration
    output_file = 'claude_todo_tasks.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved {len(df)} tasks to {output_file}")
    
    # Show samples
    print("\nSample Tasks:")
    for task_type in df['task_type'].unique()[:5]:
        print(f"\n{task_type.upper()}:")
        samples = df[df['task_type'] == task_type].head(2)
        for _, row in samples.iterrows():
            print(f"  - {row['task_description'][:80]}...")
            print(f"    Complexity: {row['complexity']}")
    
    return df

if __name__ == "__main__":
    df = prepare_calibration_data()