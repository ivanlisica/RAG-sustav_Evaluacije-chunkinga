import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Postavljanje stila
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

def generate_relevancy_vs_correctness(csv_path):
    if not os.path.exists(csv_path):
        print(f"Datoteka {csv_path} nije pronađena!")
        return

    # Učitavanje podataka
    df = pd.read_csv(csv_path)
    df['Readable Strategy'] = df['strategy'].apply(format_strategy_name)
    
    output_dir = "Grafovi_Evaluacije_Final"
    os.makedirs(output_dir, exist_ok=True)

    print("Generiram Graf 6: Context Relevancy vs Answer Correctness...")

    # Agregacija podataka (prosjeci)
    summary = df.groupby('Readable Strategy').agg({
        'context_relevancy': 'mean',
        'answer_correctness': 'mean'
    }).reset_index()

    plt.figure(figsize=(10, 7))
    
    # 1. Iscrtavanje točaka
    sns.scatterplot(
        data=summary,
        x='context_relevancy',
        y='answer_correctness',
        hue='Readable Strategy',
        s=400, # Veliki krugovi
        marker='o',
        legend=False, # Bez legende, koristimo direktne labele
        palette="viridis"
    )
    
    # 2. Dodavanje diagonalne linije (y=x) za referencu
    # Sve iznad linije znači da je model "pametniji" od konteksta
    # Sve ispod linije znači da model ne iskorištava puni potencijal konteksta
    lims = [
        min(summary.context_relevancy.min(), summary.answer_correctness.min()) - 0.05,
        max(summary.context_relevancy.max(), summary.answer_correctness.max()) + 0.05
    ]
    plt.plot(lims, lims, '--', color='gray', alpha=0.5, label='Ideal Correlation (y=x)')

    # 3. Dodavanje teksta pored točaka
    for i in range(summary.shape[0]):
        # Pomak teksta da ne gazi točku
        plt.text(
            x=summary.context_relevancy[i] + 0.005, 
            y=summary.answer_correctness[i] + 0.005, 
            s=summary['Readable Strategy'][i], 
            fontsize=10,
            weight='bold'
        )

    # 4. Oznake osi
    plt.xlabel("Average Context Relevancy (Kvaliteta Dohvata)")
    plt.ylabel("Average Answer Correctness (Točnost Odgovora)")
    
    # Prilagodba granica grafa
    plt.xlim(summary.context_relevancy.min() - 0.05, summary.context_relevancy.max() + 0.15) # Malo više mjesta desno za tekst
    plt.ylim(summary.answer_correctness.min() - 0.05, summary.answer_correctness.max() + 0.05)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    save_path = f"{output_dir}/6_relevancy_vs_correctness.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"[GOTOVO] Graf spremljen u: {save_path}")

# POKRETANJE
csv_file_name = "rag_results_final.csv"
generate_relevancy_vs_correctness(csv_file_name)