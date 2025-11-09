import json
import matplotlib.pyplot as plt
import numpy as np

def parse_results_file(file_path):
    """Parse the results file and extract performance metrics"""
    results = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                throughput = data.get('Throughput (requests/second)', 0)
                avg_latency = data.get('Latency Distribution', {}).get('Average Latency (microseconds)', 0)
                knobs = data.get('knobs')
                results.append({
                    'throughput': throughput,
                    'avg_latency': avg_latency,
                    'is_baseline': knobs is None
                })
    
    return results

def plot_improvement_from_baseline(file_path, metric='tps'):
    """Create a graph showing improvement from baseline using running best
    
    Args:
        file_path: Path to results file
        metric: 'tps' for throughput or 'latency' for average latency
    """
    results = parse_results_file(file_path)
    
    if not results:
        print("No results found in file")
        return
    
    # Get baseline (first run with knobs=null)
    baseline_value = None
    metric_key = 'throughput' if metric == 'tps' else 'avg_latency'
    
    for result in results:
        if result['is_baseline']:
            baseline_value = result[metric_key]
            break
    
    if baseline_value is None:
        print("No baseline found in results")
        return
    
    # Extract metric values (excluding baseline)
    metric_values = [r[metric_key] for r in results if not r['is_baseline']]
    
    if not metric_values:
        print("No tuning results found")
        return
    
    # Calculate running best (depends on metric)
    running_best = []
    current_best = metric_values[0]
    
    for value in metric_values:
        if metric == 'tps':
            # Higher TPS is better
            if value > current_best:
                current_best = value
        else:
            # Lower latency is better
            if value < current_best:
                current_best = value
        running_best.append(current_best)
    
    # Calculate improvement percentages
    if metric == 'tps':
        baseline_improvement = [(val - baseline_value) / baseline_value * 100 for val in running_best]
        ylabel = 'TPS Improvement from Baseline (%)'
        title_metric = 'Throughput (TPS)'
    else:
        baseline_improvement = [(baseline_value - val) / baseline_value * 100 for val in running_best]
        ylabel = 'Latency Improvement from Baseline (%)'
        title_metric = 'Average Latency'
    trial_numbers = list(range(1, len(running_best) + 1))
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot running best improvement
    plt.plot(trial_numbers, baseline_improvement, 'b-', linewidth=2, marker='o', 
             markersize=4, label='Best Performance So Far')
    
    # Add horizontal line at 0% (baseline)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Baseline Performance')
    
    # Formatting
    plt.xlabel('Trial Number', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Determine workload type from file path
    workload_type = 'TPC-C' if 'tpcc' in file_path.lower() else 'TPC-H' if 'tpch' in file_path.lower() else 'Unknown'
    
    plt.title(f'Database Tuning Performance Improvement Over Time\n({workload_type} {title_metric})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotations for key points
    max_improvement = max(baseline_improvement)
    max_trial = baseline_improvement.index(max_improvement) + 1
    
    plt.annotate(f'Best: +{max_improvement:.1f}%', 
                xy=(max_trial, max_improvement), 
                xytext=(max_trial + 2, max_improvement + 2),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')
    
    # Show final improvement
    final_improvement = baseline_improvement[-1]
    plt.annotate(f'Final: +{final_improvement:.1f}%', 
                xy=(len(trial_numbers), final_improvement), 
                xytext=(len(trial_numbers) - 3, final_improvement + 3),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')
    
    plt.tight_layout()
    
    # Print summary statistics
    if metric == 'tps':
        print(f"Baseline TPS: {baseline_value:.2f}")
        print(f"Best TPS: {max(metric_values):.2f}")
        print(f"Maximum improvement: +{max_improvement:.1f}%")
        print(f"Final improvement: +{final_improvement:.1f}%")
        print(f"Total trials: {len(metric_values)}")
        
        # Save plot
        plt.savefig('tuning_improvement_tps.png', dpi=300, bbox_inches='tight')
        print("Graph saved as 'tuning_improvement_tps.png'")
    else:
        print(f"Baseline Latency: {baseline_value:.0f} μs ({baseline_value/1000:.1f} ms)")
        print(f"Best Latency: {min(metric_values):.0f} μs ({min(metric_values)/1000:.1f} ms)")
        print(f"Maximum improvement: +{max_improvement:.1f}% (lower latency)")
        print(f"Final improvement: +{final_improvement:.1f}% (lower latency)")
        print(f"Total trials: {len(metric_values)}")
        
        # Save plot
        plt.savefig('tuning_improvement_latency.png', dpi=300, bbox_inches='tight')
        print("Graph saved as 'tuning_improvement_latency.png'")
    
    plt.show()

if __name__ == "__main__":
    import os
    
    # Define file paths for both workloads
    tpcc_file_path = "/home/karimnazarovj/LATuner/Tuning/benchbase_tpcc_50_1762472889.6701279/results_tps.res"
    tpch_file_path = "/home/karimnazarovj/LATuner/Tuning/benchbase_tpch_5_run1/results_lat.res"
    
    plots_generated = 0
    
    # Check and plot TPC-C TPS results if file exists
    if os.path.exists(tpcc_file_path):
        print("=== TPC-C Throughput (TPS) Results ===")
        plot_improvement_from_baseline(tpcc_file_path, metric='tps')
        plots_generated += 1
    else:
        print(f"TPC-C file not found: {tpcc_file_path}")
    
    # Add separator if both files exist
    if os.path.exists(tpcc_file_path) and os.path.exists(tpch_file_path):
        print("\n" + "="*50 + "\n")
    
    # Check and plot TPC-H Latency results if file exists
    if os.path.exists(tpch_file_path):
        print("=== TPC-H Average Latency Results ===")
        plot_improvement_from_baseline(tpch_file_path, metric='latency')
        plots_generated += 1
    else:
        print(f"TPC-H file not found: {tpch_file_path}")
    
    if plots_generated == 0:
        print("No results files found. Please check the file paths.")
    else:
        print(f"\nGenerated {plots_generated} plot(s).")