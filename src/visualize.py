"""
Visualization utilities for plasticity loss experiments.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path


def plot_plasticity_metrics(results, save_path=None):
    """
    Plot plasticity metrics across tasks.

    Args:
        results: Results dictionary from experiment
        save_path: Path to save figure (optional)
    """
    methods = [k for k in results.keys() if k != 'config']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Plasticity Loss Across Tasks', fontsize=16)

    for method in methods:
        task_results = results[method]

        # Extract metrics
        task_nums = [r['task_idx'] for r in task_results]
        test_accs = [r['test_acc'] for r in task_results]

        # Dead units (average across layers)
        dead_units = []
        for r in task_results:
            du = r['dead_units']
            avg_du = np.mean(list(du.values())) if du else 0
            dead_units.append(avg_du)

        # Weight magnitude
        weight_mags = [r['weight_magnitude']['overall'] for r in task_results]

        # Effective rank (average across layers)
        eff_ranks = []
        for r in task_results:
            er = r['effective_rank']
            avg_er = np.mean(list(er.values())) if er else 0
            eff_ranks.append(avg_er)

        # Plot
        axes[0, 0].plot(task_nums, test_accs, label=method, marker='o', markersize=3)
        axes[0, 1].plot(task_nums, dead_units, label=method, marker='o', markersize=3)
        axes[1, 0].plot(task_nums, weight_mags, label=method, marker='o', markersize=3)
        axes[1, 1].plot(task_nums, eff_ranks, label=method, marker='o', markersize=3)

    # Configure axes
    axes[0, 0].set_xlabel('Task Number')
    axes[0, 0].set_ylabel('Test Accuracy (%)')
    axes[0, 0].set_title('Test Accuracy Over Tasks')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Task Number')
    axes[0, 1].set_ylabel('Dead Units (%)')
    axes[0, 1].set_title('Dead Units Over Tasks')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Task Number')
    axes[1, 0].set_ylabel('Average Weight Magnitude')
    axes[1, 0].set_title('Weight Magnitude Over Tasks')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Task Number')
    axes[1, 1].set_ylabel('Effective Rank')
    axes[1, 1].set_title('Effective Rank Over Tasks')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_prediction_analysis(results, method='preventive', save_path=None):
    """
    Plot prediction statistics for preventive reinitialization.

    Args:
        results: Results dictionary
        method: Method name (should have predictor stats)
        save_path: Path to save figure
    """
    if method not in results:
        print(f"Method {method} not found in results")
        return

    task_results = results[method]

    # Extract prediction statistics
    task_nums = []
    avg_death_scores = []
    at_risk_counts = []
    intervention_counts = []

    for r in task_results:
        task_nums.append(r['task_idx'])

        if 'predictor_stats' in r:
            stats = r['predictor_stats']
            # Get average death score across all layers
            death_scores = [v for k, v in stats.items() if 'avg_death_score' in k]
            avg_death_scores.append(np.mean(death_scores) if death_scores else 0)

            # Get at-risk count
            at_risk = [v for k, v in stats.items() if 'at_risk_count' in k]
            at_risk_counts.append(np.sum(at_risk) if at_risk else 0)

        if 'intervention_stats' in r:
            intervention_counts.append(r['intervention_stats']['total_interventions'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Neuron Death Prediction Analysis', fontsize=16)

    # Plot death scores
    axes[0].plot(task_nums, avg_death_scores, marker='o', markersize=4)
    axes[0].set_xlabel('Task Number')
    axes[0].set_ylabel('Average Death Score')
    axes[0].set_title('Average Death Prediction Score')
    axes[0].grid(True, alpha=0.3)

    # Plot at-risk neurons
    axes[1].plot(task_nums, at_risk_counts, marker='o', markersize=4, color='orange')
    axes[1].set_xlabel('Task Number')
    axes[1].set_ylabel('Number of At-Risk Neurons')
    axes[1].set_title('Neurons Predicted to Die')
    axes[1].grid(True, alpha=0.3)

    # Plot interventions
    if intervention_counts:
        axes[2].plot(task_nums, intervention_counts, marker='o', markersize=4, color='red')
        axes[2].set_xlabel('Task Number')
        axes[2].set_ylabel('Cumulative Interventions')
        axes[2].set_title('Preventive Reinitializations')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def create_summary_report(results, save_path=None):
    """
    Create a summary report of the experiment.

    Args:
        results: Results dictionary
        save_path: Path to save report
    """
    methods = [k for k in results.keys() if k != 'config']

    report = []
    report.append("=" * 60)
    report.append("PLASTICITY LOSS EXPERIMENT SUMMARY")
    report.append("=" * 60)
    report.append("")

    # Configuration
    if 'config' in results:
        report.append("Configuration:")
        for key, value in results['config'].items():
            report.append(f"  {key}: {value}")
        report.append("")

    # Method comparison
    report.append("Method Comparison:")
    report.append("")

    for method in methods:
        task_results = results[method]

        # Calculate statistics
        first_acc = task_results[0]['test_acc']
        last_acc = task_results[-1]['test_acc']
        acc_drop = first_acc - last_acc

        first_dead = np.mean(list(task_results[0]['dead_units'].values()))
        last_dead = np.mean(list(task_results[-1]['dead_units'].values()))

        first_weight = task_results[0]['weight_magnitude']['overall']
        last_weight = task_results[-1]['weight_magnitude']['overall']

        report.append(f"{method.upper()}:")
        report.append(f"  Accuracy: {first_acc:.2f}% → {last_acc:.2f}% (Δ {-acc_drop:.2f}%)")
        report.append(f"  Dead Units: {first_dead:.2f}% → {last_dead:.2f}%")
        report.append(f"  Weight Mag: {first_weight:.4f} → {last_weight:.4f}")

        # Prediction stats if available
        if method == 'preventive' and 'intervention_stats' in task_results[-1]:
            interventions = task_results[-1]['intervention_stats']['total_interventions']
            report.append(f"  Total Interventions: {interventions}")

        report.append("")

    report_text = "\n".join(report)
    print(report_text)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {save_path}")

    return report_text


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualize.py <results_file.json>")
        sys.exit(1)

    results_file = sys.argv[1]

    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Create output directory
    output_dir = Path(results_file).parent
    base_name = Path(results_file).stem

    # Generate plots
    print("Generating plots...")

    fig1 = plot_plasticity_metrics(results, save_path=output_dir / f"{base_name}_metrics.png")
    plt.close(fig1)

    if 'preventive' in results:
        fig2 = plot_prediction_analysis(results, save_path=output_dir / f"{base_name}_predictions.png")
        plt.close(fig2)

    # Generate report
    create_summary_report(results, save_path=output_dir / f"{base_name}_report.txt")

    print("\nVisualization complete!")
