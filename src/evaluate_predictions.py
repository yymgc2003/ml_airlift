#!/usr/bin/env python3
"""
Evaluate model predictions and create scatter plots for each target variable.
Creates prediction vs ground truth plots for all 6 targets.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def load_predictions_and_truths(pred_path: str, truth_path: str):
    """Load prediction and ground truth arrays."""
    y_pred = np.load(pred_path)
    y_true = np.load(truth_path)
    
    print(f"Loaded predictions: {y_pred.shape}")
    print(f"Loaded ground truth: {y_true.shape}")
    
    return y_pred, y_true


def create_prediction_plots(y_pred: np.ndarray, y_true: np.ndarray, 
                          output_dir: str = "evaluation_plots",
                          target_names: list = None):
    """
    Create prediction vs ground truth scatter plots for each target.
    
    Args:
        y_pred: Predicted values (N, 6)
        y_true: Ground truth values (N, 6)
        output_dir: Directory to save plots
        target_names: Names for each target variable
    """
    print("📊 Creating Prediction Evaluation Plots")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Default target names
    if target_names is None:
        target_names = [f"Target {i+1}" for i in range(y_pred.shape[1])]
    
    # Calculate metrics for each target
    metrics = []
    for i in range(y_pred.shape[1]):
        pred_i = y_pred[:, i]
        true_i = y_true[:, i]
        
        r2 = r2_score(true_i, pred_i)
        mse = mean_squared_error(true_i, pred_i)
        mae = mean_absolute_error(true_i, pred_i)
        rmse = np.sqrt(mse)
        
        metrics.append({
            'target': i+1,
            'name': target_names[i],
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        })
        
        print(f"Target {i+1} ({target_names[i]}): R²={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    # Create individual scatter plots for each target
    print("\nCreating individual scatter plots...")
    create_individual_plots(y_pred, y_true, target_names, metrics, output_dir)
    
    # Create combined overview plot
    print("Creating combined overview plot...")
    create_overview_plot(y_pred, y_true, target_names, metrics, output_dir)
    
    # Create metrics summary table
    print("Creating metrics summary...")
    create_metrics_summary(metrics, output_dir)
    
    print(f"\n✅ All plots saved to: {output_dir}/")


def create_individual_plots(y_pred: np.ndarray, y_true: np.ndarray, 
                          target_names: list, metrics: list, output_dir: str):
    """Create individual scatter plots for each target."""
    
    n_targets = y_pred.shape[1]
    
    for i in range(n_targets):
        pred_i = y_pred[:, i]
        true_i = y_true[:, i]
        metric = metrics[i]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Create scatter plot
        ax.scatter(true_i, pred_i, alpha=0.6, s=30, color='blue', edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line (y=x)
        min_val = min(np.min(true_i), np.min(pred_i))
        max_val = max(np.max(true_i), np.max(pred_i))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Set labels and title
        ax.set_xlabel('Ground Truth', fontsize=12)
        ax.set_ylabel('Predicted Value', fontsize=12)
        ax.set_title(f'{target_names[i]} - Prediction vs Ground Truth\n'
                    f'R² = {metric["r2"]:.4f}, RMSE = {metric["rmse"]:.4f}, MAE = {metric["mae"]:.4f}',
                    fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Add statistics text box
        stats_text = f'R² = {metric["r2"]:.4f}\nRMSE = {metric["rmse"]:.4f}\nMAE = {metric["mae"]:.4f}\nMSE = {metric["mse"]:.4f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'target_{i+1:02d}_prediction_plot.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def create_overview_plot(y_pred: np.ndarray, y_true: np.ndarray, 
                        target_names: list, metrics: list, output_dir: str):
    """Create overview plot with all targets in subplots."""
    
    n_targets = y_pred.shape[1]
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_targets):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        pred_i = y_pred[:, i]
        true_i = y_true[:, i]
        metric = metrics[i]
        
        # Create scatter plot
        ax.scatter(true_i, pred_i, alpha=0.6, s=20, color='blue', edgecolors='black', linewidth=0.3)
        
        # Perfect prediction line
        min_val = min(np.min(true_i), np.min(pred_i))
        max_val = max(np.max(true_i), np.max(pred_i))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='y=x')
        
        # Set labels and title
        ax.set_xlabel('Ground Truth', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.set_title(f'{target_names[i]}\nR²={metric["r2"]:.3f}, RMSE={metric["rmse"]:.3f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots
    for i in range(n_targets, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overview_prediction_plots.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()


def create_metrics_summary(metrics: list, output_dir: str):
    """Create metrics summary table and save as text file."""
    
    # Create summary table
    summary_lines = [
        "Model Evaluation Summary",
        "=" * 50,
        f"{'Target':<12} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'MSE':<8}",
        "-" * 50
    ]
    
    for metric in metrics:
        summary_lines.append(
            f"{metric['name']:<12} {metric['r2']:<8.4f} {metric['rmse']:<8.4f} "
            f"{metric['mae']:<8.4f} {metric['mse']:<8.4f}"
        )
    
    # Calculate overall metrics
    overall_r2 = np.mean([m['r2'] for m in metrics])
    overall_rmse = np.mean([m['rmse'] for m in metrics])
    overall_mae = np.mean([m['mae'] for m in metrics])
    overall_mse = np.mean([m['mse'] for m in metrics])
    
    summary_lines.extend([
        "-" * 50,
        f"{'Average':<12} {overall_r2:<8.4f} {overall_rmse:<8.4f} "
        f"{overall_mae:<8.4f} {overall_mse:<8.4f}",
        "=" * 50
    ])
    
    # Save summary
    summary_text = "\n".join(summary_lines)
    with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write(summary_text)
    
    print("\nEvaluation Summary:")
    print(summary_text)


def create_residual_plots(y_pred: np.ndarray, y_true: np.ndarray, 
                         target_names: list, output_dir: str):
    """Create residual plots for each target."""
    
    print("Creating residual plots...")
    
    n_targets = y_pred.shape[1]
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_targets):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        pred_i = y_pred[:, i]
        true_i = y_true[:, i]
        residuals = pred_i - true_i
        
        # Create residual plot
        ax.scatter(true_i, residuals, alpha=0.6, s=20, color='red', edgecolors='black', linewidth=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        ax.set_xlabel('Ground Truth', fontsize=10)
        ax.set_ylabel('Residuals (Pred - True)', fontsize=10)
        ax.set_title(f'{target_names[i]} - Residuals', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_targets, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_plots.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate model predictions')
    parser.add_argument('--pred_path', 
                       default='/home/smatsubara/documents/airlift/data/outputs_real/simple/y_pred.npy',
                       help='Path to prediction file')
    parser.add_argument('--truth_path', 
                       default='/home/smatsubara/documents/airlift/data/outputs_real/simple/y_true.npy',
                       help='Path to ground truth file')
    parser.add_argument('--output_dir', default='evaluation_plots',
                       help='Output directory for plots')
    parser.add_argument('--target_names', nargs=6, 
                       default=['Solid Velocity', 'Gas Velocity', 'Liquid Velocity', 
                               'Solid Volume Fraction', 'Gas Volume Fraction', 'Liquid Volume Fraction'],
                       help='Names for target variables')
    parser.add_argument('--create_residuals', action='store_true',
                       help='Create residual plots as well')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading prediction and ground truth data...")
    y_pred, y_true = load_predictions_and_truths(args.pred_path, args.truth_path)
    
    # Create plots
    create_prediction_plots(y_pred, y_true, args.output_dir, args.target_names)
    
    # Create residual plots if requested
    if args.create_residuals:
        create_residual_plots(y_pred, y_true, args.target_names, args.output_dir)
    
    print("\n🎉 Evaluation complete!")


if __name__ == "__main__":
    main()
