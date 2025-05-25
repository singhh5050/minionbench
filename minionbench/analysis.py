import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

class BenchmarkAnalyzer:
    """Analyze and visualize MinionBench results"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.df = None
        self.detailed_results = None
    
    def load_latest_results(self) -> bool:
        """Load the most recent benchmark results"""
        if not self.results_dir.exists():
            print(f"❌ Results directory {self.results_dir} not found")
            return False
        
        # Find latest CSV file
        csv_files = list(self.results_dir.glob("summary_results_*.csv"))
        if not csv_files:
            print("❌ No CSV results found")
            return False
        
        latest_csv = max(csv_files, key=os.path.getctime)
        print(f"📊 Loading results from {latest_csv}")
        
        # Load CSV data
        self.df = pd.read_csv(latest_csv)
        
        # Try to load corresponding detailed JSON results
        timestamp = latest_csv.stem.split('_')[-1]
        json_file = self.results_dir / f"detailed_results_{timestamp}.json"
        
        if json_file.exists():
            with open(json_file, 'r') as f:
                self.detailed_results = json.load(f)
            print(f"📄 Loaded detailed results from {json_file}")
        
        return True
    
    def summary_statistics(self):
        """Print comprehensive summary statistics"""
        if self.df is None:
            print("❌ No data loaded")
            return
        
        print("\n📊 BENCHMARK SUMMARY STATISTICS")
        print("=" * 60)
        
        # Basic counts
        total_experiments = len(self.df)
        successful_experiments = len(self.df[~self.df['failed']])
        failed_experiments = len(self.df[self.df['failed']])
        
        print(f"Total Experiments: {total_experiments}")
        print(f"Successful: {successful_experiments} ({successful_experiments/total_experiments*100:.1f}%)")
        print(f"Failed: {failed_experiments} ({failed_experiments/total_experiments*100:.1f}%)")
        
        if successful_experiments == 0:
            print("❌ No successful experiments to analyze")
            return
        
        # Filter successful experiments
        success_df = self.df[~self.df['failed']].copy()
        
        # Performance metrics
        print(f"\n🚀 PERFORMANCE METRICS (Successful Experiments Only)")
        print("-" * 40)
        
        for metric, unit in [
            ('avg_ttft_seconds', 's'),
            ('avg_latency_seconds', 's'), 
            ('avg_throughput_tokens_per_sec', 'tokens/s'),
            ('total_tokens', 'tokens')
        ]:
            if metric in success_df.columns:
                values = success_df[metric].dropna()
                if len(values) > 0:
                    print(f"{metric.replace('_', ' ').title()}:")
                    print(f"  Mean: {values.mean():.3f} {unit}")
                    print(f"  Std:  {values.std():.3f} {unit}")
                    print(f"  Min:  {values.min():.3f} {unit}")
                    print(f"  Max:  {values.max():.3f} {unit}")
                    print()
        
        # Breakdown by categories
        print("📋 EXPERIMENT BREAKDOWN")
        print("-" * 25)
        
        for category in ['workload', 'deployment', 'reasoning', 'quantization', 'remote_model']:
            if category in success_df.columns:
                counts = success_df[category].value_counts()
                print(f"{category.title()}:")
                for value, count in counts.items():
                    if pd.notna(value):
                        print(f"  {value}: {count}")
                print()
    
    def create_performance_comparison(self, save_path: Optional[str] = None):
        """Create performance comparison visualizations"""
        if self.df is None:
            print("❌ No data loaded")
            return
        
        success_df = self.df[~self.df['failed']].copy()
        if len(success_df) == 0:
            print("❌ No successful experiments to plot")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MinionBench Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. TTFT by Deployment Type
        if 'avg_ttft_seconds' in success_df.columns and 'deployment' in success_df.columns:
            ttft_data = success_df.dropna(subset=['avg_ttft_seconds', 'deployment'])
            if len(ttft_data) > 0:
                sns.boxplot(data=ttft_data, x='deployment', y='avg_ttft_seconds', ax=axes[0,0])
                axes[0,0].set_title('Time to First Token by Deployment')
                axes[0,0].set_ylabel('TTFT (seconds)')
                axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Throughput by Workload
        if 'avg_throughput_tokens_per_sec' in success_df.columns and 'workload' in success_df.columns:
            throughput_data = success_df.dropna(subset=['avg_throughput_tokens_per_sec', 'workload'])
            if len(throughput_data) > 0:
                sns.boxplot(data=throughput_data, x='workload', y='avg_throughput_tokens_per_sec', ax=axes[0,1])
                axes[0,1].set_title('Throughput by Workload Type')
                axes[0,1].set_ylabel('Throughput (tokens/sec)')
                axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Latency by Reasoning
        if 'avg_latency_seconds' in success_df.columns and 'reasoning' in success_df.columns:
            latency_data = success_df.dropna(subset=['avg_latency_seconds', 'reasoning'])
            if len(latency_data) > 0:
                sns.boxplot(data=latency_data, x='reasoning', y='avg_latency_seconds', ax=axes[1,0])
                axes[1,0].set_title('Latency by Reasoning Mode')
                axes[1,0].set_ylabel('Latency (seconds)')
        
        # 4. Performance by Quantization (for local/hybrid only)
        quant_data = success_df[success_df['deployment'].isin(['Local', 'Hybrid'])].dropna(subset=['avg_throughput_tokens_per_sec', 'quantization'])
        if len(quant_data) > 0:
            sns.boxplot(data=quant_data, x='quantization', y='avg_throughput_tokens_per_sec', ax=axes[1,1])
            axes[1,1].set_title('Throughput by Quantization Level')
            axes[1,1].set_ylabel('Throughput (tokens/sec)')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Performance plots saved to {save_path}")
        else:
            plt.show()
    
    def create_heatmap_analysis(self, save_path: Optional[str] = None):
        """Create heatmap showing performance across different configurations"""
        if self.df is None:
            print("❌ No data loaded")
            return
        
        success_df = self.df[~self.df['failed']].copy()
        if len(success_df) == 0:
            print("❌ No successful experiments to plot")
            return
        
        # Create pivot tables for different metrics
        metrics = ['avg_ttft_seconds', 'avg_latency_seconds', 'avg_throughput_tokens_per_sec']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in success_df.columns:
                # Create pivot table: workload vs deployment
                pivot_data = success_df.pivot_table(
                    values=metric, 
                    index='workload', 
                    columns='deployment', 
                    aggfunc='mean'
                )
                
                if not pivot_data.empty:
                    sns.heatmap(
                        pivot_data, 
                        annot=True, 
                        fmt='.3f', 
                        cmap='RdYlBu_r' if 'throughput' not in metric else 'RdYlBu',
                        ax=axes[i]
                    )
                    axes[i].set_title(f'{metric.replace("_", " ").title()}')
        
        plt.suptitle('Performance Heatmaps: Workload vs Deployment', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🔥 Heatmap saved to {save_path}")
        else:
            plt.show()
    
    def export_analysis_report(self, output_path: str = "analysis_report.html"):
        """Export a comprehensive HTML analysis report"""
        if self.df is None:
            print("❌ No data loaded")
            return
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MinionBench Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e8f4fd; }}
                .summary {{ background-color: #fff3cd; }}
            </style>
        </head>
        <body>
            <h1>🚀 MinionBench Analysis Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>📊 Summary Statistics</h2>
            <div class="summary">
                <p><strong>Total Experiments:</strong> {len(self.df)}</p>
                <p><strong>Successful:</strong> {len(self.df[~self.df['failed']])}</p>
                <p><strong>Failed:</strong> {len(self.df[self.df['failed']])}</p>
            </div>
            
            <h2>📈 Performance Data</h2>
            {self.df[~self.df['failed']].describe().to_html(classes='metric')}
            
            <h2>📋 Detailed Results</h2>
            {self.df.to_html(classes='metric')}
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"📄 Analysis report exported to {output_path}")
    
    def find_best_configurations(self, metric: str = 'avg_throughput_tokens_per_sec', top_n: int = 5):
        """Find the best performing configurations for a given metric"""
        if self.df is None:
            print("❌ No data loaded")
            return
        
        success_df = self.df[~self.df['failed']].copy()
        if len(success_df) == 0 or metric not in success_df.columns:
            print(f"❌ No data available for metric: {metric}")
            return
        
        # Remove NaN values
        metric_data = success_df.dropna(subset=[metric])
        
        # Sort by metric (higher is better for throughput, lower is better for latency/ttft)
        ascending = 'throughput' not in metric.lower()
        best_configs = metric_data.nsmallest(top_n, metric) if ascending else metric_data.nlargest(top_n, metric)
        
        print(f"\n🏆 TOP {top_n} CONFIGURATIONS FOR {metric.upper()}")
        print("=" * 60)
        
        for i, (_, row) in enumerate(best_configs.iterrows(), 1):
            print(f"{i}. {metric}: {row[metric]:.3f}")
            print(f"   Workload: {row['workload']}")
            print(f"   Deployment: {row['deployment']}")
            print(f"   Reasoning: {row['reasoning']}")
            print(f"   Quantization: {row['quantization']}")
            print(f"   Remote Model: {row['remote_model']}")
            print()

def analyze_latest_results(results_dir: str = "results"):
    """Convenience function to analyze the latest benchmark results"""
    analyzer = BenchmarkAnalyzer(results_dir)
    
    if not analyzer.load_latest_results():
        print("❌ Could not load results")
        return None
    
    # Run comprehensive analysis
    analyzer.summary_statistics()
    
    # Create visualizations
    plots_dir = Path(results_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    analyzer.create_performance_comparison(plots_dir / "performance_comparison.png")
    analyzer.create_heatmap_analysis(plots_dir / "performance_heatmaps.png")
    
    # Export report
    analyzer.export_analysis_report(str(Path(results_dir) / "analysis_report.html"))
    
    # Find best configurations
    analyzer.find_best_configurations('avg_throughput_tokens_per_sec')
    analyzer.find_best_configurations('avg_ttft_seconds')
    
    return analyzer

if __name__ == "__main__":
    analyze_latest_results() 