#!/usr/bin/env python3
"""
Router CLI Tool
Command-line interface for task routing and calibration
"""

import click
import asyncio
import json
import sys
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import print as rprint
import time

# Import router components
from production_task_router import create_production_router
from calibrate_router import RouterCalibrator

console = Console()

@click.group()
def cli():
    """Task Router CLI - Route tasks and manage calibration"""
    pass

@cli.command()
@click.argument('task', required=False)
@click.option('--file', '-f', help='File containing tasks (one per line or JSON/CSV)')
@click.option('--proto-path', default='router_calibration/prototypes.npy', help='Path to prototypes')
@click.option('--config-path', default='router_calibration/router_config.json', help='Path to config')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
def route(task, file, proto_path, config_path, output_json):
    """Route a task or batch of tasks"""
    
    # Load router
    router = create_production_router(
        proto_path=proto_path if Path(proto_path).exists() else None,
        config_path=config_path if Path(config_path).exists() else None
    )
    
    # Get tasks to route
    tasks = []
    if task:
        tasks = [task]
    elif file:
        if file.endswith('.json'):
            with open(file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    tasks = data if isinstance(data[0], str) else [d['task'] for d in data]
                else:
                    tasks = [data['task']]
        elif file.endswith('.csv'):
            df = pd.read_csv(file)
            tasks = df['task_description'].tolist() if 'task_description' in df else df.iloc[:, 0].tolist()
        else:
            with open(file) as f:
                tasks = [line.strip() for line in f if line.strip()]
    else:
        # Read from stdin
        console.print("[yellow]Enter task (Ctrl+D to finish):[/yellow]")
        task = sys.stdin.read().strip()
        if task:
            tasks = [task]
    
    if not tasks:
        console.print("[red]No tasks provided[/red]")
        return
    
    # Route tasks
    results = []
    
    async def route_all():
        for task_text in track(tasks, description="Routing tasks..."):
            decision = await router.route(task_text)
            results.append({
                'task': task_text[:100] + '...' if len(task_text) > 100 else task_text,
                'type': decision.task_type or 'unknown',
                'complexity': decision.complexity,
                'confidence': decision.confidence,
                'abstain': decision.abstain,
                'path': decision.routing_path,
                'time_ms': decision.routing_time_ms
            })
    
    asyncio.run(route_all())
    
    # Output results
    if output_json:
        print(json.dumps(results, indent=2))
    else:
        # Create table
        table = Table(title=f"Routing Results ({len(results)} tasks)")
        table.add_column("Task", style="cyan", no_wrap=False, max_width=50)
        table.add_column("Type", style="green")
        table.add_column("Complexity", style="yellow")
        table.add_column("Confidence", justify="right")
        table.add_column("Path", style="blue")
        table.add_column("Time (ms)", justify="right")
        
        for r in results:
            confidence_color = "green" if r['confidence'] > 0.7 else "yellow" if r['confidence'] > 0.5 else "red"
            table.add_row(
                r['task'],
                r['type'] if not r['abstain'] else '[red]abstain[/red]',
                r['complexity'],
                f"[{confidence_color}]{r['confidence']:.3f}[/{confidence_color}]",
                r['path'],
                f"{r['time_ms']:.1f}"
            )
        
        console.print(table)
        
        # Summary stats
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        abstention_rate = sum(r['abstain'] for r in results) / len(results)
        avg_time = sum(r['time_ms'] for r in results) / len(results)
        
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Average confidence: {avg_confidence:.3f}")
        console.print(f"  Abstention rate: {abstention_rate:.1%}")
        console.print(f"  Average time: {avg_time:.1f}ms")

@cli.command()
@click.option('--data', '-d', required=True, help='Training data (CSV/JSON)')
@click.option('--output', '-o', default='router_calibration', help='Output directory')
@click.option('--n-prototypes', default=3, type=int, help='Prototypes per class')
@click.option('--tau-min', default=0.45, type=float, help='Min tau threshold')
@click.option('--tau-max', default=0.65, type=float, help='Max tau threshold')
@click.option('--steps', default=8, type=int, help='Grid search steps')
def calibrate(data, output, n_prototypes, tau_min, tau_max, steps):
    """Calibrate router thresholds from historical data"""
    
    console.print(f"[bold blue]Router Calibration[/bold blue]")
    console.print(f"Loading data from: {data}")
    
    calibrator = RouterCalibrator()
    
    # Load data
    df = calibrator.load_historical_data(data)
    
    # Generate prototypes
    console.print(f"\n[yellow]Generating {n_prototypes} prototypes per class...[/yellow]")
    prototypes = calibrator.generate_prototypes(df, n_prototypes_per_class=n_prototypes)
    
    # Calibrate thresholds
    console.print(f"\n[yellow]Calibrating thresholds (tau: {tau_min}-{tau_max})...[/yellow]")
    best_params = calibrator.calibrate_semantic_thresholds(
        df,
        tau_range=(tau_min, tau_max),
        margin_range=(0.04, 0.12),
        n_steps=steps
    )
    
    # Evaluate
    console.print(f"\n[yellow]Evaluating performance...[/yellow]")
    metrics = calibrator.evaluate_router_performance(df)
    
    # Save
    calibrator.save_calibration(output)
    
    console.print(f"\n[bold green]Calibration complete![/bold green]")
    console.print(f"Results saved to: {output}/")

@cli.command()
@click.option('--n-samples', '-n', default=1000, type=int, help='Number of samples')
@click.option('--output', '-o', default='synthetic_tasks.csv', help='Output file')
def generate(n_samples, output):
    """Generate synthetic task data for testing"""
    
    console.print(f"[yellow]Generating {n_samples} synthetic tasks...[/yellow]")
    
    calibrator = RouterCalibrator()
    df = calibrator.generate_synthetic_data(n_samples=n_samples)
    
    # Save to file
    if output.endswith('.csv'):
        df.to_csv(output, index=False)
    elif output.endswith('.json'):
        df.to_json(output, orient='records', indent=2)
    else:
        df.to_csv(output, index=False)
    
    # Show sample
    table = Table(title="Sample Generated Tasks")
    table.add_column("Task", no_wrap=False, max_width=60)
    table.add_column("Type", style="green")
    table.add_column("Complexity", style="yellow")
    
    for _, row in df.head(5).iterrows():
        table.add_row(
            row['task_description'][:60] + '...' if len(row['task_description']) > 60 else row['task_description'],
            row['task_type'],
            row['complexity']
        )
    
    console.print(table)
    console.print(f"\n[green]Generated {n_samples} tasks saved to: {output}[/green]")

@cli.command()
@click.option('--host', default='localhost', help='API server host')
@click.option('--port', default=8000, type=int, help='API server port')
def test_api(host, port):
    """Test the router API server"""
    
    import requests
    
    base_url = f"http://{host}:{port}"
    
    console.print(f"[yellow]Testing API at {base_url}...[/yellow]")
    
    # Test health endpoint
    try:
        resp = requests.get(f"{base_url}/health")
        resp.raise_for_status()
        health = resp.json()
        
        table = Table(title="API Health Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Status", health['status'])
        table.add_row("Uptime", f"{health['uptime_seconds']:.1f}s")
        table.add_row("Total Requests", str(health['total_requests']))
        table.add_row("Avg Latency", f"{health['avg_latency_ms']:.1f}ms")
        table.add_row("Cache Hit Rate", f"{health['cache_hit_rate']:.1%}")
        table.add_row("Abstention Rate", f"{health['abstention_rate']:.1%}")
        
        console.print(table)
        
    except requests.exceptions.ConnectionError:
        console.print(f"[red]Cannot connect to API at {base_url}[/red]")
        console.print("[yellow]Start the API server with: python router_api.py[/yellow]")
        return
    
    # Test routing
    test_tasks = [
        "Write a Python function to calculate fibonacci numbers",
        "Debug why my API returns 500 errors",
        "Design a distributed cache for 1M QPS"
    ]
    
    console.print("\n[yellow]Testing task routing...[/yellow]")
    
    for task in test_tasks:
        resp = requests.post(f"{base_url}/route", json={"task": task})
        result = resp.json()
        
        console.print(f"\nTask: [cyan]{task[:60]}...[/cyan]")
        console.print(f"  Type: [green]{result['task_type']}[/green]")
        console.print(f"  Complexity: [yellow]{result['complexity']}[/yellow]")
        console.print(f"  Confidence: {result['confidence']:.3f}")
        console.print(f"  Models: {', '.join(result['recommended_models'][:2])}")
        console.print(f"  Time: {result['routing_time_ms']:.1f}ms")

@cli.command()
def benchmark():
    """Benchmark router performance"""
    
    console.print("[bold blue]Router Performance Benchmark[/bold blue]\n")
    
    # Create router
    router = create_production_router()
    
    # Test tasks of varying complexity
    test_tasks = [
        ("Simple", "What is 2 + 2?"),
        ("Medium", "Write a function to sort an array using quicksort"),
        ("Complex", "Design a distributed system for real-time fraud detection handling 100k TPS"),
        ("Very Long", " ".join(["Analyze this complex system"] * 50))
    ]
    
    results = []
    
    async def benchmark_tasks():
        # Warmup
        console.print("[yellow]Warming up...[/yellow]")
        for _ in range(10):
            await router.route("warmup task")
        
        console.print("\n[yellow]Running benchmark...[/yellow]")
        
        for label, task in test_tasks:
            times = []
            
            # Run multiple iterations
            for _ in track(range(100), description=f"Testing {label}..."):
                start = time.perf_counter()
                decision = await router.route(task)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            results.append({
                'label': label,
                'task_len': len(task),
                'p50': np.percentile(times, 50),
                'p95': np.percentile(times, 95),
                'p99': np.percentile(times, 99),
                'min': min(times),
                'max': max(times)
            })
    
    import numpy as np
    asyncio.run(benchmark_tasks())
    
    # Display results
    table = Table(title="Benchmark Results (100 iterations each)")
    table.add_column("Task Type", style="cyan")
    table.add_column("Length", justify="right")
    table.add_column("P50 (ms)", justify="right", style="green")
    table.add_column("P95 (ms)", justify="right", style="yellow")
    table.add_column("P99 (ms)", justify="right", style="red")
    table.add_column("Min (ms)", justify="right")
    table.add_column("Max (ms)", justify="right")
    
    for r in results:
        table.add_row(
            r['label'],
            str(r['task_len']),
            f"{r['p50']:.1f}",
            f"{r['p95']:.1f}",
            f"{r['p99']:.1f}",
            f"{r['min']:.1f}",
            f"{r['max']:.1f}"
        )
    
    console.print(table)
    
    # Overall stats
    all_p50 = np.mean([r['p50'] for r in results])
    all_p95 = np.mean([r['p95'] for r in results])
    
    console.print(f"\n[bold]Overall Performance:[/bold]")
    console.print(f"  Average P50: [green]{all_p50:.1f}ms[/green]")
    console.print(f"  Average P95: [yellow]{all_p95:.1f}ms[/yellow]")
    
    if all_p50 < 10:
        console.print("  [bold green]✓ Excellent: Sub-10ms median latency![/bold green]")
    elif all_p50 < 20:
        console.print("  [green]✓ Good: Fast routing performance[/green]")
    else:
        console.print("  [yellow]⚠ Consider optimization for better performance[/yellow]")

if __name__ == "__main__":
    cli()