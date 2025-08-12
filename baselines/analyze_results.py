#!/usr/bin/env python3
"""Comprehensive Evaluation and Analysis Script.

This script runs the complete tournament evaluation pipeline:
1. Discovers all available checkpoints from recent training runs
2. Runs comprehensive tournament evaluation against scripted baselines
3. Runs cross-algorithm evaluation between all baseline algorithms
4. Analyzes results and generates research artifacts (graphs, tables, insights)
5. Creates shareable research outputs for the community

Features:
- Automatic checkpoint discovery from training runs
- Comprehensive tournament execution with statistical analysis
- Research-quality visualizations and data analysis
- Exportable artifacts for publication and sharing

Usage:
    python -m baselines.analyze_results --run-timestamp 20250128_091120
    python -m baselines.analyze_results --auto-discover  # Find latest run
"""

import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import subprocess
import datetime

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def discover_checkpoints(run_timestamp: Optional[str] = None) -> Dict[str, List[str]]:
    """Discover available checkpoints from training runs."""
    
    print(f"🔍 Discovering latest checkpoints from all runs")
    
    # Use the same discovery patterns as the working tournament script
    discovered = {}
    
    # Use the exact same patterns as the working tournament script
    # IPPO discovery - find most recent checkpoint from any run
    ippo_checkpoints = glob.glob("checkpoints/ippo/run_*_seed*/main/")
    if ippo_checkpoints:
        ippo_checkpoints.sort(key=lambda x: (
            x.split('run_')[1].split('_seed')[0],  # timestamp
            int(x.split('_seed')[1].split('/')[0])  # seed number
        ))
        latest_ippo = ippo_checkpoints[-1]  # Most recent
        discovered["IPPO"] = [latest_ippo]
        print(f"  IPPO: Found {len(ippo_checkpoints)} total, selected latest: {latest_ippo}")
    else:
        discovered["IPPO"] = []
        print(f"  IPPO: No checkpoints found")
    
    # SPPPO discovery - find most recent checkpoint from any run
    spppo_checkpoints = glob.glob("checkpoints/spppo/run_*_seed*/main/")
    if spppo_checkpoints:
        spppo_checkpoints.sort(key=lambda x: (
            x.split('run_')[1].split('_seed')[0],  # timestamp
            int(x.split('_seed')[1].split('/')[0])  # seed number
        ))
        latest_spppo = spppo_checkpoints[-1]  # Most recent
        discovered["SPPPO"] = [latest_spppo]
        print(f"  SPPPO: Found {len(spppo_checkpoints)} total, selected latest: {latest_spppo}")
    else:
        discovered["SPPPO"] = []
        print(f"  SPPPO: No checkpoints found")
    
    # FSPPPO discovery - find most recent checkpoint from any run
    fspppo_checkpoints = glob.glob("checkpoints/fspppo/run_*_seed*/main/*/")
    if fspppo_checkpoints:
        # Filter out non-numeric directories (like agent_metadata.json)
        numeric_checkpoints = []
        for cp in fspppo_checkpoints:
            try:
                step_num = int(cp.rstrip('/').split('/')[-1])
                numeric_checkpoints.append(cp)
            except ValueError:
                continue
        
        if numeric_checkpoints:
            numeric_checkpoints.sort(key=lambda x: (
                x.split('run_')[1].split('_seed')[0],  # timestamp
                int(x.split('_seed')[1].split('/')[0]),  # seed number
                int(x.rstrip('/').split('/')[-1])  # step number
            ))
            latest_fspppo = numeric_checkpoints[-1]  # Most recent
            discovered["FSPPPO"] = [latest_fspppo]
            print(f"  FSPPPO: Found {len(numeric_checkpoints)} total, selected latest: {latest_fspppo}")
        else:
            discovered["FSPPPO"] = []
            print(f"  FSPPPO: No valid checkpoints found")
    else:
        discovered["FSPPPO"] = []
        print(f"  FSPPPO: No checkpoints found")
    
    return discovered


def create_tournament_config(checkpoints: Dict[str, List[str]], output_file: str) -> str:
    """Create tournament configuration file from discovered checkpoints."""
    
    # Build agent list
    agents = []
    for algorithm, checkpoint_list in checkpoints.items():
        for checkpoint in checkpoint_list:
            agents.append(f"{algorithm}:{checkpoint}")
    
    # Tournament configuration
    config = {
        "env_name": "MPE_simple_sumo_v3",
        "env_kwargs": {"random_spawn": False},
        
        "agents": agents,
        
        "scripted_baselines": [
            "seek",      # Aggressive chasing
            "guardian",   # Defensive positioning  
            "dodge",     # Predictive evasion
            "random",    # Random actions
            "noop",      # Static behavior
        ],
        
        "num_seeds": 10,     # Statistical robustness
        "base_seed": 42,
    }
    
    # Save configuration
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"📄 Tournament config saved: {output_file}")
    print(f"   Agents: {len(agents)}")
    print(f"   Scripted baselines: {len(config['scripted_baselines'])}")
    print(f"   Total matchups: {len(agents) * (len(agents) + len(config['scripted_baselines']) - 1)} × {config['num_seeds']} seeds")
    
    return output_file


def run_tournament(config_file: str, output_csv: str) -> bool:
    """Run the tournament evaluation."""
    
    print(f"\n🏆 Running Tournament Evaluation")
    print("=" * 50)
    
    cmd = [
        "python", "-m", "baselines.run_tournament",
        "--episodes-per-matchup", "50"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd="/share/code/src/JaxMARL",
            text=True,
            capture_output=True
        )
        
        if result.returncode == 0:
            print("✅ Tournament completed successfully!")
            print(result.stdout)
            return True
        else:
            print("❌ Tournament failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running tournament: {e}")
        return False


def load_and_analyze_results(csv_file: str) -> pd.DataFrame:
    """Load tournament results and perform basic analysis."""
    
    print(f"\n📊 Loading and Analyzing Results")
    print("=" * 40)
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} tournament matches")
        
        # Basic statistics
        total_matches = len(df)
        error_matches = len(df[df['winner'] == 'error'])
        valid_matches = total_matches - error_matches
        
        print(f"   Valid matches: {valid_matches}")
        print(f"   Error matches: {error_matches}")
        
        if error_matches > 0:
            print(f"   Error rate: {error_matches/total_matches:.1%}")
        
        # Player breakdown
        players = set()
        for col in ['green_player', 'red_player']:
            players.update(df[col].unique())
        
        print(f"   Players: {sorted(players)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return pd.DataFrame()


def calculate_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive win rates for each algorithm."""
    
    results = []
    
    # Get unique players (excluding scripted behaviors)
    players = set()
    for col in ['green_player', 'red_player']:
        players.update(df[col].unique())
    # Filter to only learned algorithms (not scripted behaviors)
    learned_players = [p for p in players if not p.startswith('scripted_')]
    
    for player in learned_players:
        # All matches where this player participated
        green_matches = df[df['green_player'] == player]
        red_matches = df[df['red_player'] == player]
        
        # Calculate wins
        green_wins = len(green_matches[green_matches['winner'] == 'green'])
        red_wins = len(red_matches[red_matches['winner'] == 'red'])
        total_wins = green_wins + red_wins
        total_matches = len(green_matches) + len(red_matches)
        
        # Overall win rate
        win_rate = total_wins / total_matches if total_matches > 0 else 0
        
        # Win rate vs scripted baselines
        vs_scripted = df[
            ((df['green_player'] == player) & (df['red_player'].str.startswith('scripted_'))) |
            ((df['red_player'] == player) & (df['green_player'].str.startswith('scripted_')))
        ]
        vs_scripted_wins = (
            len(vs_scripted[(vs_scripted['green_player'] == player) & (vs_scripted['winner'] == 'green')]) +
            len(vs_scripted[(vs_scripted['red_player'] == player) & (vs_scripted['winner'] == 'red')])
        )
        vs_scripted_rate = vs_scripted_wins / len(vs_scripted) if len(vs_scripted) > 0 else 0
        
        # Win rate vs other learned algorithms
        vs_other = df[
            ((df['green_player'] == player) & (~df['red_player'].str.startswith('scripted_')) & (df['green_player'] != df['red_player'])) |
            ((df['red_player'] == player) & (~df['green_player'].str.startswith('scripted_')) & (df['green_player'] != df['red_player']))
        ]
        vs_other_wins = (
            len(vs_other[(vs_other['green_player'] == player) & (vs_other['winner'] == 'green')]) +
            len(vs_other[(vs_other['red_player'] == player) & (vs_other['winner'] == 'red')])
        )
        vs_other_rate = vs_other_wins / len(vs_other) if len(vs_other) > 0 else 0
        
        results.append({
            'algorithm': player,
            'total_matches': total_matches,
            'total_wins': total_wins,
            'win_rate': win_rate,
            'vs_scripted_matches': len(vs_scripted),
            'vs_scripted_wins': vs_scripted_wins,
            'vs_scripted_rate': vs_scripted_rate,
            'vs_other_matches': len(vs_other),
            'vs_other_wins': vs_other_wins,
            'vs_other_rate': vs_other_rate,
        })
    
    return pd.DataFrame(results).sort_values('win_rate', ascending=False)


def create_visualizations(df: pd.DataFrame, win_rates: pd.DataFrame, output_dir: str) -> List[str]:
    """Create comprehensive visualizations for research sharing."""
    
    print(f"\n📈 Creating Research Visualizations")
    print("=" * 40)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    artifacts = []
    
    # 1. Overall Win Rate Comparison
    plt.figure(figsize=(12, 8))
    
    # Main win rate bar chart
    plt.subplot(2, 2, 1)
    bars = plt.bar(win_rates['algorithm'], win_rates['win_rate'], color='skyblue', alpha=0.8)
    plt.title('Overall Win Rates by Algorithm', fontsize=14, fontweight='bold')
    plt.ylabel('Win Rate')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    
    # 2. Win Rate Breakdown (vs Scripted vs Other Algorithms)
    plt.subplot(2, 2, 2)
    x = range(len(win_rates))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], win_rates['vs_scripted_rate'], 
           width, label='vs Scripted', color='lightcoral', alpha=0.8)
    plt.bar([i + width/2 for i in x], win_rates['vs_other_rate'], 
           width, label='vs Other Algorithms', color='lightgreen', alpha=0.8)
    
    plt.title('Win Rates: Scripted vs Algorithm Opponents', fontsize=14, fontweight='bold')
    plt.ylabel('Win Rate')
    plt.xlabel('Algorithm')
    plt.xticks(x, win_rates['algorithm'], rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    
    # 3. Match Count Distribution
    plt.subplot(2, 2, 3)
    plt.bar(win_rates['algorithm'], win_rates['total_matches'], color='gold', alpha=0.8)
    plt.title('Total Matches per Algorithm', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Matches')
    plt.xticks(rotation=45)
    
    # 4. Performance Consistency (Win Rate vs Match Count)
    plt.subplot(2, 2, 4)
    scatter = plt.scatter(win_rates['total_matches'], win_rates['win_rate'], 
                         s=100, c=win_rates['vs_scripted_rate'], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Win Rate vs Scripted')
    plt.xlabel('Total Matches')
    plt.ylabel('Overall Win Rate')
    plt.title('Performance vs Experience', fontsize=14, fontweight='bold')
    
    # Add algorithm labels
    for _, row in win_rates.iterrows():
        plt.annotate(row['algorithm'], 
                    (row['total_matches'], row['win_rate']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    plt.tight_layout()
    
    # Save main visualization
    main_viz_path = output_dir / "baseline_algorithm_performance.png"
    plt.savefig(main_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    artifacts.append(str(main_viz_path))
    print(f"✅ Main performance visualization: {main_viz_path}")
    
    # 5. Detailed Heatmap of Algorithm vs Algorithm Performance
    # Get learned algorithms (exclude scripted behaviors)
    players = set(df['green_player'].unique()) | set(df['red_player'].unique())
    algorithms = [p for p in players if not p.startswith('scripted_')]
    
    if len(algorithms) > 1:
        plt.figure(figsize=(10, 8))
        
        # Create win rate matrix
        matrix = np.zeros((len(algorithms), len(algorithms)))
        
        for i, green_alg in enumerate(algorithms):
            for j, red_alg in enumerate(algorithms):
                if green_alg != red_alg:
                    matches = df[(df['green_player'] == green_alg) & (df['red_player'] == red_alg)]
                    if len(matches) > 0:
                        win_rate = len(matches[matches['winner'] == 'green']) / len(matches)
                        matrix[i, j] = win_rate
                else:
                    matrix[i, j] = np.nan
        
        # Create heatmap
        mask = np.eye(len(algorithms), dtype=bool)
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5,
                   xticklabels=algorithms, yticklabels=algorithms, 
                   mask=mask, cbar_kws={'label': 'Win Rate'})
        
        plt.title('Algorithm vs Algorithm Win Rates\n(Row beats Column)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Red Agent (Opponent)')
        plt.ylabel('Green Agent')
        
        heatmap_path = output_dir / "algorithm_vs_algorithm_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        artifacts.append(str(heatmap_path))
        print(f"✅ Algorithm vs algorithm heatmap: {heatmap_path}")
    
    # 6. Scripted Baseline Performance Analysis
    scripted_baselines = df[df['red_player'].str.startswith('scripted_')]['red_player'].str.replace('scripted_', '').unique()
    
    if len(scripted_baselines) > 0:
        plt.figure(figsize=(12, 6))
        
        baseline_performance = []
        for baseline in scripted_baselines:
            for alg in algorithms:
                # Matches where algorithm was green vs this scripted baseline
                green_matches = df[
                    (df['green_player'] == alg) & 
                    (df['red_player'] == f'scripted_{baseline}')
                ]
                
                # Matches where algorithm was red vs this scripted baseline  
                red_matches = df[
                    (df['red_player'] == alg) & 
                    (df['green_player'] == f'scripted_{baseline}')
                ]
                
                total_wins = (
                    len(green_matches[green_matches['winner'] == 'green']) +
                    len(red_matches[red_matches['winner'] == 'red'])
                )
                total_matches = len(green_matches) + len(red_matches)
                
                if total_matches > 0:
                    win_rate = total_wins / total_matches
                    baseline_performance.append({
                        'algorithm': alg,
                        'baseline': baseline,
                        'win_rate': win_rate,
                        'matches': total_matches
                    })
        
        if baseline_performance:
            baseline_df = pd.DataFrame(baseline_performance)
            
            # Create pivot table for heatmap
            pivot = baseline_df.pivot(index='algorithm', columns='baseline', values='win_rate')
            
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5,
                       cbar_kws={'label': 'Win Rate'})
            
            plt.title('Algorithm Performance vs Scripted Baselines', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Scripted Baseline')
            plt.ylabel('Algorithm')
            
            baseline_heatmap_path = output_dir / "algorithm_vs_scripted_heatmap.png"
            plt.savefig(baseline_heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            artifacts.append(str(baseline_heatmap_path))
            print(f"✅ Algorithm vs scripted heatmap: {baseline_heatmap_path}")
    
    return artifacts


def generate_research_summary(
    df: pd.DataFrame, 
    win_rates: pd.DataFrame, 
    artifacts: List[str],
    output_dir: str
) -> str:
    """Generate comprehensive research summary and insights."""
    
    output_dir = Path(output_dir)
    summary_file = output_dir / "research_summary.md"
    
    with open(summary_file, 'w') as f:
        f.write("# Baseline Algorithm Evaluation: Research Summary\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report presents a comprehensive evaluation of three baseline reinforcement learning algorithms ")
        f.write("(IPPO, SPPPO, FSPPPO) in the MPE Simple Sumo environment. The evaluation includes performance ")
        f.write("against scripted baselines and cross-algorithm competition to assess robustness and generalization.\n\n")
        
        # Key Findings
        f.write("## Key Findings\n\n")
        
        best_algorithm = win_rates.iloc[0]
        worst_algorithm = win_rates.iloc[-1]
        
        f.write(f"### Overall Performance\n")
        f.write(f"- **Best performing algorithm:** {best_algorithm['algorithm']} ({best_algorithm['win_rate']:.3f} win rate)\n")
        f.write(f"- **Lowest performing algorithm:** {worst_algorithm['algorithm']} ({worst_algorithm['win_rate']:.3f} win rate)\n")
        f.write(f"- **Performance gap:** {best_algorithm['win_rate'] - worst_algorithm['win_rate']:.3f}\n\n")
        
        # Algorithm-specific insights
        f.write("### Algorithm-Specific Performance\n\n")
        for _, row in win_rates.iterrows():
            f.write(f"**{row['algorithm']}:**\n")
            f.write(f"- Overall win rate: {row['win_rate']:.3f} ({row['total_wins']}/{row['total_matches']} matches)\n")
            f.write(f"- vs Scripted baselines: {row['vs_scripted_rate']:.3f} ({row['vs_scripted_wins']}/{row['vs_scripted_matches']} matches)\n")
            f.write(f"- vs Other algorithms: {row['vs_other_rate']:.3f} ({row['vs_other_wins']}/{row['vs_other_matches']} matches)\n\n")
        
        # Robustness Analysis
        f.write("### Robustness Analysis\n\n")
        
        avg_vs_scripted = win_rates['vs_scripted_rate'].mean()
        avg_vs_other = win_rates['vs_other_rate'].mean()
        
        f.write(f"- **Average performance vs scripted baselines:** {avg_vs_scripted:.3f}\n")
        f.write(f"- **Average performance vs other algorithms:** {avg_vs_other:.3f}\n")
        
        if avg_vs_scripted > avg_vs_other + 0.1:
            f.write("- **Finding:** Algorithms perform better against scripted baselines than against each other, ")
            f.write("suggesting potential overfitting to simple behaviors.\n")
        elif avg_vs_other > avg_vs_scripted + 0.1:
            f.write("- **Finding:** Algorithms struggle more against scripted baselines than against each other, ")
            f.write("indicating potential brittleness to unexpected strategies.\n")
        else:
            f.write("- **Finding:** Performance is relatively consistent across opponent types.\n")
        
        f.write("\n")
        
        # Statistical Summary
        f.write("## Statistical Summary\n\n")
        f.write(f"- **Total tournament matches:** {len(df)}\n")
        f.write(f"- **Algorithms evaluated:** {len(win_rates)}\n")
        scripted_count = len(df[df['red_player'].str.startswith('scripted_')]['red_player'].unique())
        f.write(f"- **Scripted baselines:** {scripted_count}\n")
        # Calculate episodes per matchup instead of seeds
        episodes_per_matchup = len(df) // len(df[['green_player', 'red_player']].drop_duplicates())
        f.write(f"- **Episodes per matchup:** {episodes_per_matchup}\n")
        
        error_rate = len(df[df['winner'] == 'error']) / len(df)
        f.write(f"- **Error rate:** {error_rate:.1%}\n\n")
        
        # Research Implications
        f.write("## Research Implications\n\n")
        f.write("### Limitations Identified\n\n")
        
        # Identify specific weaknesses
        weak_vs_scripted = win_rates[win_rates['vs_scripted_rate'] < 0.6]
        if len(weak_vs_scripted) > 0:
            f.write("**Poor performance against scripted baselines:**\n")
            for _, row in weak_vs_scripted.iterrows():
                f.write(f"- {row['algorithm']}: {row['vs_scripted_rate']:.3f} win rate vs scripted\n")
            f.write("\n")
        
        weak_vs_other = win_rates[win_rates['vs_other_rate'] < 0.4]
        if len(weak_vs_other) > 0:
            f.write("**Poor performance against other algorithms:**\n")
            for _, row in weak_vs_other.iterrows():
                f.write(f"- {row['algorithm']}: {row['vs_other_rate']:.3f} win rate vs other algorithms\n")
            f.write("\n")
        
        # Recommendations
        f.write("### Recommendations for Future Research\n\n")
        f.write("1. **Curriculum Learning:** Develop training curricula that expose agents to diverse opponent strategies\n")
        f.write("2. **Robustness Training:** Implement adversarial training or domain randomization techniques\n")
        f.write("3. **Population-Based Training:** Train against diverse populations rather than single opponents\n")
        f.write("4. **Meta-Learning:** Develop algorithms that can quickly adapt to new opponent strategies\n")
        f.write("5. **Evaluation Protocols:** Establish standardized evaluation against diverse opponent types\n\n")
        
        # Artifacts
        f.write("## Generated Artifacts\n\n")
        for artifact in artifacts:
            artifact_name = Path(artifact).name
            f.write(f"- `{artifact_name}`: {artifact}\n")
        
        f.write("\n")
        f.write("## Methodology\n\n")
        f.write("All algorithms were trained with consistent hyperparameters and evaluated using the same ")
        f.write("tournament system. Multiple random seeds were used for statistical robustness. ")
        f.write("The evaluation environment (MPE Simple Sumo) provides a competitive two-agent setting ")
        f.write("that tests both strategic planning and reactive capabilities.\n\n")
        
        f.write("---\n")
        f.write("*This analysis was generated automatically by the JaxMARL baseline evaluation system.*\n")
    
    print(f"✅ Research summary: {summary_file}")
    return str(summary_file)


def find_latest_tournament_results() -> str:
    """Find the most recent tournament results CSV file."""
    tournament_dirs = list(Path("tournament_results").glob("run_*"))
    if not tournament_dirs:
        return None
    
    # Sort by directory name (which includes timestamp) and find the latest one with a CSV
    for latest_dir in sorted(tournament_dirs, reverse=True):
        results_csv = latest_dir / "tournament_results.csv"
        if results_csv.exists():
            return str(results_csv)
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Analyze existing tournament results")
    parser.add_argument("--results-csv", default=None, help="Tournament results CSV file to analyze (discovers latest if not specified)")
    parser.add_argument("--output-dir", default="analysis_results", help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # Discover latest results if no CSV specified
    if args.results_csv is None:
        print("🔍 No results CSV specified, discovering latest tournament results...")
        latest_csv = find_latest_tournament_results()
        if not latest_csv:
            print("❌ No tournament results found in tournament_results/")
            print("Please run the tournament first using: python -m baselines.run_tournament")
            print("Or specify a results file manually with: --results-csv path/to/results.csv")
            return
        results_csv = Path(latest_csv)
        print(f"🔍 Auto-discovered latest tournament results: {results_csv}")
    else:
        results_csv = Path(args.results_csv)
        print(f"📊 Analyzing specified results file: {results_csv}")
    
    # Validate input file exists
    if not results_csv.exists():
        print(f"❌ Results file not found: {results_csv}")
        print("Please run the tournament first using: python -m baselines.run_tournament")
        return
    
    # Create timestamped analysis folder
    from datetime import datetime
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / f"run_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Tournament Results Analysis")
    print("=" * 40)
    print(f"Input file: {results_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Analysis timestamp: {run_timestamp}")
    
    # Load and analyze results
    df = load_and_analyze_results(str(results_csv))
    if df.empty:
        print("❌ No valid results to analyze.")
        return
    
    # Calculate win rates
    win_rates = calculate_win_rates(df)
    print(f"\n📊 Win Rate Summary:")
    print(win_rates[['algorithm', 'win_rate', 'vs_scripted_rate', 'vs_other_rate']].to_string(index=False))
    
    # Create visualizations
    artifacts = create_visualizations(df, win_rates, str(output_dir))
    
    # Generate research summary
    summary_file = generate_research_summary(df, win_rates, artifacts, str(output_dir))
    
    # Final summary
    print(f"\n🎉 Comprehensive Evaluation Complete!")
    print("=" * 50)
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Results CSV: {results_csv}")
    print(f"📝 Research summary: {summary_file}")
    print(f"📈 Visualizations: {len(artifacts)} files")
    
    print(f"\n🚀 Ready for research sharing!")
    print("Next steps:")
    print("1. Review the research summary for key insights")
    print("2. Share visualizations and findings with the research community")
    print("3. Use insights to design improved training curricula")


if __name__ == "__main__":
    main()
