import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    'figure.dpi': 300, 
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (11, 7)
})

def format_strategy_name(name):
    """Pretvara tehnička imena u čitljiv format."""
    mapping = {
        "micro_overlap20": "128 Size, 20 Overlap",
        "standard_overlap50": "512 Size, 50 Overlap",
        "macro_overlap100": "1024 Size, 100 Overlap",
        "semantic": "Semantic Chunking"
    }
    return mapping.get(name, name)

def generate_paper_graphs(csv_path):
    if not os.path.exists(csv_path):
        print(f"Datoteka {csv_path} nije pronađena!")
        return

    df = pd.read_csv(csv_path)
    df['Strategy'] = df['strategy'].apply(format_strategy_name)
    
    df['Faithfulness'] = (df['oa_faithfulness'] + df['ge_faithfulness']) / 2
    df['Context_Relevancy'] = (df['oa_relevancy'] + df['ge_relevancy']) / 2
    df['Answer_Correctness'] = (df['oa_correctness'] + df['ge_correctness']) / 2
    df['Composite_Score'] = (df['Faithfulness'] + df['Context_Relevancy'] + df['Answer_Correctness']) / 3

    strategy_order =[
        "128 Size, 20 Overlap",
        "512 Size, 50 Overlap", 
        "1024 Size, 100 Overlap",
        "Semantic Chunking"
    ]
    existing_strategies =[s for s in strategy_order if s in df['Strategy'].unique()]

    output_dir = "Grafovi_Istrazivacki_Rad"
    os.makedirs(output_dir, exist_ok=True)
    print("Započinjem generiranje 10 akademskih grafova...\n")

    # ==========================================================
    # GRAF 1: OSNOVNE METRIKE
    # ==========================================================
    metrics_df = df.melt(id_vars=['Strategy'], value_vars=['Faithfulness', 'Context_Relevancy', 'Answer_Correctness'], 
                         var_name='Metric', value_name='Score')
    metrics_df['Metric'] = metrics_df['Metric'].str.replace('_', ' ')

    plt.figure()
    ax = sns.barplot(data=metrics_df, x='Strategy', y='Score', hue='Metric', order=existing_strategies, palette="viridis", errorbar=None)
    plt.title("Average Performance Metrics by Chunking Strategy", pad=15)
    plt.xlabel("")
    plt.ylabel("Score (0 - 1)")
    plt.xticks(rotation=30, ha='right')
    plt.ylim(0, 1.1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
    plt.savefig(f"{output_dir}/1_Metrics_Comparison.png", bbox_inches='tight')
    plt.close()

    # ==========================================================
    # GRAF 2: RASPODJELA TOČNOSTI (Boxplot)
    # ==========================================================
    plt.figure()
    sns.boxplot(data=df, x='Strategy', y='Answer_Correctness', order=existing_strategies, palette="Blues", 
                showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"})
    plt.title("Distribution of Answer Correctness (Boxplot)", pad=15)
    plt.xlabel("")
    plt.ylabel("Answer Correctness")
    plt.xticks(rotation=30, ha='right')
    plt.savefig(f"{output_dir}/2_Accuracy_Distribution.png", bbox_inches='tight')
    plt.close()

    # ==========================================================
    # GRAF 3: LATENCY VS ACCURACY 
    # ==========================================================
    summary_lat_acc = df.groupby('Strategy').agg({'latency': 'mean', 'Answer_Correctness': 'mean'}).reset_index()
    plt.figure()
    sns.scatterplot(data=summary_lat_acc, x='latency', y='Answer_Correctness', hue='Strategy', s=300, palette="deep", legend=False, edgecolor='black')
    
    # Približili smo tekst i centrirali ga po visini kruga
    for i in range(summary_lat_acc.shape[0]):
        plt.text(summary_lat_acc.latency[i] + 0.03, summary_lat_acc.Answer_Correctness[i], 
                 summary_lat_acc['Strategy'][i], fontsize=10, weight='bold', va='center')
                 
    plt.title("Trade-off Between Latency and Answer Correctness", pad=15)
    plt.xlabel("Average Latency (seconds)")
    plt.ylabel("Average Answer Correctness (0 - 1)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{output_dir}/3_Latency_vs_Accuracy.png", bbox_inches='tight')
    plt.close()

    # ==========================================================
    # GRAF 4: UKUPNI POREDAK 
    # ==========================================================
    overall_summary = df.groupby('Strategy')['Composite_Score'].mean().sort_values(ascending=True)
    plt.figure()
    bars = plt.barh(overall_summary.index, overall_summary.values, color='teal', alpha=0.8, edgecolor='black')
    plt.title("Overall Ranking of Strategies by Composite Score", pad=15)
    plt.xlabel("Average Composite Score (F+R+C)/3")
    plt.xlim(0, max(overall_summary.values) + 0.15)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', fontsize=11, weight='bold')
    plt.savefig(f"{output_dir}/4_Overall_Ranking.png", bbox_inches='tight')
    plt.close()

    # ==========================================================
    # GRAF 5: USPOREDBA LATENCIJE 
    # ==========================================================
    latency_summary = df.groupby('Strategy')['latency'].mean().sort_values(ascending=True)
    plt.figure()
    bars = plt.barh(latency_summary.index, latency_summary.values, color='salmon', alpha=0.8, edgecolor='black')
    plt.title("Average Processing Latency by Strategy", pad=15)
    plt.xlabel("Average Latency in seconds (Lower is better)")
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.05, bar.get_y() + bar.get_height()/2, f'{width:.2f}s', va='center', fontsize=11, weight='bold')
    plt.savefig(f"{output_dir}/5_Latency_Comparison.png", bbox_inches='tight')
    plt.close()

    # ==========================================================
    # GRAF 6: DATASET USPOREDBA - KOMPOZITNA OCJENA
    # ==========================================================
    if 'dataset' in df.columns and df['dataset'].nunique() > 1:
        plt.figure()
        ax_ds = sns.barplot(data=df, x='Strategy', y='Composite_Score', hue='dataset', order=existing_strategies, palette="Set2", errorbar=None)
        plt.title("Strategy Robustness Across Different Datasets", pad=15)
        plt.xlabel("")
        plt.ylabel("Composite Score")
        plt.xticks(rotation=30, ha='right')
        plt.ylim(0, 1.1)
        plt.legend(title="Dataset", loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
        for container in ax_ds.containers:
            ax_ds.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
        plt.savefig(f"{output_dir}/6_Dataset_Comparison.png", bbox_inches='tight')
        plt.close()

    # ==========================================================
    # GRAF 7: DATASET USPOREDBA - RASPODJELA TOČNOSTI 
    # ==========================================================
    if 'dataset' in df.columns and df['dataset'].nunique() > 1:
        plt.figure()
        sns.boxplot(data=df, x='Strategy', y='Answer_Correctness', hue='dataset', order=existing_strategies, palette="pastel")
        plt.title("Variance in Answer Correctness by Dataset", pad=15)
        plt.xlabel("")
        plt.ylabel("Answer Correctness")
        plt.xticks(rotation=30, ha='right')
        plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(f"{output_dir}/7_Dataset_Boxplot.png", bbox_inches='tight')
        plt.close()

    # ==========================================================
    # GRAF 8: VIOLIN PLOT - GUSTOĆA I STRUKTURA OCJENA 
    # ==========================================================
    plt.figure()
    sns.violinplot(data=df, x='Strategy', y='Composite_Score', order=existing_strategies, palette="muted", inner="quartile")
    plt.title("Density Distribution of Composite Scores (Violin Plot)", pad=15)
    plt.xlabel("Chunking Strategy")
    plt.ylabel("Composite Score Density")
    plt.xticks(rotation=30, ha='right')
    plt.savefig(f"{output_dir}/8_Violin_Density.png", bbox_inches='tight')
    plt.close()

    # ==========================================================
    # GRAF 9: GUSTOĆA TOČNOSTI (KDE PLOT) ZA NAJBOLJE STRATEGIJE
    # ==========================================================
    top_3_strategies = overall_summary.tail(3).index.tolist()
    df_top = df[df['Strategy'].isin(top_3_strategies)]
    plt.figure()
    sns.kdeplot(data=df_top, x='Answer_Correctness', hue='Strategy', fill=True, common_norm=False, palette="tab10", alpha=0.4)
    plt.title("Kernel Density Estimation of Correctness (Top 3 Strategies)", pad=15)
    plt.xlabel("Answer Correctness")
    plt.ylabel("Density")
    plt.savefig(f"{output_dir}/9_KDE_Top3_Accuracy.png", bbox_inches='tight')
    plt.close()

    # ==========================================================
    # GRAF 10: FAITHFULNESS VS RELEVANCY (Scatter)
    # ==========================================================
    summary_f_r = df.groupby('Strategy').agg({'Faithfulness': 'mean', 'Context_Relevancy': 'mean'}).reset_index()
    plt.figure()
    sns.scatterplot(data=summary_f_r, x='Context_Relevancy', y='Faithfulness', hue='Strategy', s=400, palette="Set1", legend=False, edgecolor='black')
    
    # Približili smo tekst i centrirali ga po visini kruga
    for i in range(summary_f_r.shape[0]):
        plt.text(summary_f_r.Context_Relevancy[i] + 0.003, summary_f_r.Faithfulness[i], 
                 summary_f_r['Strategy'][i], fontsize=10, weight='bold', va='center')
                 
    plt.title("Retrieval Quality (Relevancy) vs. Hallucination Prevention (Faithfulness)", pad=15)
    plt.xlabel("Context Relevancy (Did we find the text?)")
    plt.ylabel("Faithfulness (Did the model stick to the text?)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{output_dir}/10_Faithfulness_vs_Relevancy.png", bbox_inches='tight')
    plt.close()

    print(f"[GOTOVO] Svih 10 grafova spremno u mapi: {output_dir}/")
    
    generate_statistical_report(df, output_dir)


def generate_statistical_report(df, output_dir):
    """
    Generira detaljan izvještaj o standardnoj devijaciji i varijanci 
    koji je ključan za akademski rad.
    """
    print("\n" + "="*50)
    print(" UVOD U STATISTIČKU ANALIZU ")
    print("="*50)
    
    stats_df = df.groupby('Strategy')['Composite_Score'].agg(['mean', 'median', 'std', 'var', 'min', 'max']).reset_index()
    
    stats_df = stats_df.round(3)
    
    stats_csv_path = f"{output_dir}/Statisiticka_Analiza_Strategija.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    
    print(stats_df.to_string(index=False))
    print(f"\n[INFO] Tablica deskriptivne statistike spremljena u: {stats_csv_path}")

# POKRETANJE
csv_file_name = "./Rezultati/benchmark_multi_dataset_20260225_1336.csv" 
generate_paper_graphs(csv_file_name)