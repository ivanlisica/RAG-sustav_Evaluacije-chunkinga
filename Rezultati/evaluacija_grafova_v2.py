import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({'figure.dpi': 150}) 

def format_strategy_name(name):
    """Pretvara tehnička imena u čitljiv format."""
    mapping = {
        "micro_overlap20": "128 Size, 20 Overlap",
        "micro_overlap40": "128 Size, 40 Overlap",
        "standard_overlap50": "512 Size, 50 Overlap",
        "standard_overlap100": "512 Size, 100 Overlap",
        "macro_overlap100": "1024 Size, 100 Overlap",
        "macro_overlap200": "1024 Size, 200 Overlap",
        "semantic": "Semantic Chunking"
    }
    return mapping.get(name, name)

def generate_custom_graphs(csv_path):
    if not os.path.exists(csv_path):
        print(f"Datoteka {csv_path} nije pronađena!")
        return

    df = pd.read_csv(csv_path)
    
    df['Readable Strategy'] = df['strategy'].apply(format_strategy_name)
    
    strategy_order = [
        "128 Size, 20 Overlap", "128 Size, 40 Overlap", 
        "512 Size, 50 Overlap", "512 Size, 100 Overlap", 
        "1024 Size, 100 Overlap", "1024 Size, 200 Overlap", 
        "Semantic Chunking"
    ]
    
    existing_strategies = [s for s in strategy_order if s in df['Readable Strategy'].unique()]

    output_dir = "Grafovi_Evaluacije_Final"
    os.makedirs(output_dir, exist_ok=True)

    print("Generiram Graf 1...")
    metrics = ['faithfulness', 'context_relevancy', 'answer_correctness']
    metric_labels = {'faithfulness': 'Faithfulness', 'context_relevancy': 'Relevancy', 'answer_correctness': 'Correctness'}
    
    df_melted = df.melt(id_vars=['Readable Strategy'], value_vars=metrics, var_name='Metric', value_name='Score')
    df_melted['Metric'] = df_melted['Metric'].map(metric_labels)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df_melted, 
        x='Readable Strategy', 
        y='Score', 
        hue='Metric',
        order=existing_strategies,
        palette="viridis",
        errorbar=None
    )
    
    plt.xlabel("") 
    plt.ylabel("Score (0-1)")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 0.85)
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/1_metrics_comparison.png", bbox_inches='tight')
    plt.show()

    print("Generiram Graf 2...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df, 
        x='Readable Strategy', 
        y='answer_correctness',
        order=existing_strategies,
        palette="Blues"
    )
    plt.xlabel("")
    plt.ylabel("Answer Correctness")
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2_accuracy_distribution.png", bbox_inches='tight')
    plt.show()

    print("Generiram Graf 3...")
    summary = df.groupby('Readable Strategy').agg({
        'answer_correctness': 'mean',
        'latency': 'mean'
    }).reset_index()

    plt.figure(figsize=(10, 7))
    

    sns.scatterplot(
        data=summary,
        x='latency',
        y='answer_correctness',
        hue='Readable Strategy', 
        s=300,
        marker='o',
        legend=False,
        palette="deep"
    )
    
    for i in range(summary.shape[0]):
        plt.text(
            x=summary.latency[i] + 0.05, 
            y=summary.answer_correctness[i] + 0.005, 
            s=summary['Readable Strategy'][i], 
            fontsize=10,
            weight='bold'
        )

    plt.xlabel("Latency (seconds)")
    plt.ylabel("Answer Correctness (0-1)")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.xlim(summary.latency.min() - 0.2, summary.latency.max() + 1.2)
    plt.ylim(summary.answer_correctness.min() - 0.02, summary.answer_correctness.max() + 0.02)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/3_latency_vs_accuracy.png", bbox_inches='tight')
    plt.show()

    print("Generiram Graf 4...")
    df['Overall Score'] = (df['faithfulness'] + df['context_relevancy'] + df['answer_correctness']) / 3
    overall_summary = df.groupby('Readable Strategy')['Overall Score'].mean().sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(overall_summary.index, overall_summary.values, color='teal', alpha=0.7)
    
    plt.xlabel("Average Composite Score (F + R + C) / 3")
    plt.xlim(0, 0.7)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/4_overall_ranking.png", bbox_inches='tight')
    plt.show()


    print("Generiram Graf 5...")
    latency_summary = df.groupby('Readable Strategy')['latency'].mean().sort_values(ascending=True) # Najbrži prvi

    plt.figure(figsize=(10, 6))
    bars = plt.barh(latency_summary.index, latency_summary.values, color='salmon', alpha=0.7)
    
    plt.xlabel("Average Latency (s) - Lower is Better")
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.05, bar.get_y() + bar.get_height()/2, f'{width:.2f}s', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/5_latency_comparison.png", bbox_inches='tight')
    plt.show()

    print(f"\n[GOTOVO] Grafovi spremljeni u: {output_dir}")

csv_file_name = "rag_results_final.csv"
generate_custom_graphs(csv_file_name)