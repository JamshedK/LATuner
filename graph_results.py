import json
import matplotlib.pyplot as plt
import numpy as np

def parse_results_file(file_path):
    """Parse the results file and extract TPS values"""
    results = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                goodput = data.get('Goodput (requests/second)', 0)
                knobs = data.get('knobs')
                results.append({
                    'goodput': goodput,
                    'is_baseline': knobs is None
                })
    
    return results

def plot_improvement_from_baseline(file_path):
    """Create a graph showing improvement from baseline using running minimum"""
    results = parse_results_file(file_path)
    
    if not results:
        print("No results found in file")
        return
    
    # Get baseline (first run with knobs=null)
    baseline_tps = None
    for result in results:
        if result['is_baseline']:
            baseline_tps = result['goodput']
            break
    
    if baseline_tps is None:
        print("No baseline found in results")
        return
    
    # Extract TPS values (excluding baseline)
    tps_values = [r['goodput'] for r in results if not r['is_baseline']]
    
    if not tps_values:
        print("No tuning results found")
        return
    
    # Calculate running minimum (best performance so far)
    running_best = []
    current_best = tps_values[0]
    
    for tps in tps_values:
        if tps > current_best:  # Higher TPS is better
            current_best = tps
        running_best.append(current_best)
    
    # Calculate improvement percentages
    baseline_improvement = [(tps - baseline_tps) / baseline_tps * 100 for tps in running_best]
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
    plt.ylabel('Improvement from Baseline (%)', fontsize=12)
    plt.title('Database Tuning Performance Improvement Over Time\n(TPC-C Goodput)', fontsize=14)
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
    print(f"Baseline TPS: {baseline_tps:.2f}")
    print(f"Best TPS: {max(tps_values):.2f}")
    print(f"Maximum improvement: +{max_improvement:.1f}%")
    print(f"Final improvement: +{final_improvement:.1f}%")
    print(f"Total trials: {len(tps_values)}")
    
    plt.show()
    
    # Save the plot
    plt.savefig('tuning_improvement.png', dpi=300, bbox_inches='tight')
    print("Graph saved as 'tuning_improvement.png'")

if __name__ == "__main__":
    # Example usage - adjust the path to your results file
    file_path = "/home/karimnazarovj/LATuner/Tuning/benchbase_tpcc_50_16_1761509323.299166/results_tps.res"
    plot_improvement_from_baseline(file_path)