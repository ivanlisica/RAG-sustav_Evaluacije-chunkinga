import pandas as pd
import numpy as np
import os
from scipy import stats

def format_strategy_name(name):
    """Pretvara tehnička imena u čitljiv format."""
    mapping = {
        "micro_overlap20": "128 Size, 20 Overlap",
        "standard_overlap50": "512 Size, 50 Overlap",
        "macro_overlap100": "1024 Size, 100 Overlap",
        "semantic": "Semantic Chunking"
    }
    return mapping.get(name, name)

def run_advanced_statistical_analysis(csv_path):
    if not os.path.exists(csv_path):
        print(f"[GREŠKA] Datoteka {csv_path} nije pronađena!")
        return

    # Učitavanje i priprema podataka
    df = pd.read_csv(csv_path)
    df['Strategy'] = df['strategy'].apply(format_strategy_name)
    
    # Pojedinačni kompozitni rezultati za SVAKOG sudca
    df['OA_Overall'] = (df['oa_faithfulness'] + df['oa_relevancy'] + df['oa_correctness']) / 3
    df['GE_Overall'] = (df['ge_faithfulness'] + df['ge_relevancy'] + df['ge_correctness']) / 3
    
    # KONSENZUS (Prosjek oba sudca)
    df['Composite_Score'] = (df['OA_Overall'] + df['GE_Overall']) / 2

    output_dir = "Grafovi_Istrazivacki_Rad"
    os.makedirs(output_dir, exist_ok=True)
    report_path = f"{output_dir}/Napredna_Statisticka_Analiza.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write(" DUBINSKI IZVJEŠTAJ O STATISTIČKOJ ZNAČAJNOSTI I PRISTRANOSTI \n")
        f.write("="*70 + "\n\n")

        # =====================================================================
        # 1. ANALIZA STRATEGIJA CHUNKINGA
        # =====================================================================
        f.write("--- 1. ANALIZA STRATEGIJA CHUNKINGA ---\n")
        
        # Deskriptivna statistika
        f.write("Deskriptivna statistika (Consensus Score):\n")
        strat_stats = df.groupby('Strategy')['Composite_Score'].agg(['mean', 'std']).sort_values(by='mean', ascending=False)
        for strat, row in strat_stats.iterrows():
            f.write(f" - {strat}: Prosjek = {row['mean']:.3f} (SD = {row['std']:.3f})\n")
        f.write("\n")
        
        # ANOVA
        groups = [group['Composite_Score'].dropna().values for name, group in df.groupby('Strategy')]
        f_stat, p_value_anova = stats.f_oneway(*groups)
        f.write("Jednosmjerna ANOVA (Postoji li razlika među strategijama?):\n")
        f.write(f" -> F-statistika: {f_stat:.4f}, P-vrijednost: {p_value_anova:.4e}\n")
        if p_value_anova < 0.05:
            f.write(" -> ZAKLJUČAK: Postoji statistički značajna razlika u performansama strategija.\n\n")
        else:
            f.write(" -> ZAKLJUČAK: Nema statistički značajne razlike između strategija.\n\n")

        # T-Test (Najbolji vs Ostali)
        best_strategy = strat_stats.index[0]
        best_scores = df[df['Strategy'] == best_strategy]['Composite_Score'].dropna()
        f.write(f"T-test: Usporedba najbolje strategije ('{best_strategy}') s ostalima:\n")
        
        for other_strategy in strat_stats.index[1:]:
            other_scores = df[df['Strategy'] == other_strategy]['Composite_Score'].dropna()
            t_stat, p_val_t = stats.ttest_ind(best_scores, other_scores, equal_var=False)
            sig = "***" if p_val_t < 0.001 else "**" if p_val_t < 0.01 else "*" if p_val_t < 0.05 else "ns"
            f.write(f" - vs {other_strategy}: p = {p_val_t:.4f} ({sig})\n")
        f.write("\n")

        # =====================================================================
        # 2. ANALIZA DATASETOVA
        # =====================================================================
        f.write("--- 2. USPOREDBA DATASETOVA (Težina dokumenata) ---\n")
        if 'dataset' in df.columns and df['dataset'].nunique() > 1:
            datasets = df['dataset'].unique()
            ds1, ds2 = datasets[0], datasets[1]
            
            ds1_scores = df[df['dataset'] == ds1]['Composite_Score'].dropna()
            ds2_scores = df[df['dataset'] == ds2]['Composite_Score'].dropna()
            
            f.write(f"Prosjek za {ds1}: {ds1_scores.mean():.3f} (SD = {ds1_scores.std():.3f})\n")
            f.write(f"Prosjek za {ds2}: {ds2_scores.mean():.3f} (SD = {ds2_scores.std():.3f})\n")
            
            t_stat_ds, p_val_ds = stats.ttest_ind(ds1_scores, ds2_scores, equal_var=False)
            f.write(f"T-test: t = {t_stat_ds:.3f}, p = {p_val_ds:.4f}\n")
            
            if p_val_ds < 0.05:
                f.write(f" -> ZAKLJUČAK: Postoji značajna razlika. Modeli su se bolje snašli na datasetu '{ds1 if ds1_scores.mean() > ds2_scores.mean() else ds2}'.\n\n")
            else:
                f.write(" -> ZAKLJUČAK: Nema značajne razlike. Strategije su jednako robusne na oba dataseta.\n\n")
        else:
            f.write("Pronađen je samo jedan dataset. Analiza preskočena.\n\n")

        # =====================================================================
        # 3. ANALIZA PRISTRANOSTI SUDACA (LLM BIAS)
        # =====================================================================
        f.write("--- 3. ANALIZA PRISTRANOSTI SUDACA (LLM BIAS) ---\n")
        f.write("Ova sekcija analizira favorizira li sudac odgovore koje je generirao njegov vlastiti model (npr. ocjenjuje li GPT sudac više GPT generatora nego Gemini generatora).\n\n")
        
        pipelines = df['pipeline'].unique()
        openai_pipe = next((p for p in pipelines if 'OpenAI' in p or 'GPT' in p), None)
        gemini_pipe = next((p for p in pipelines if 'Gemini' in p), None)

        if openai_pipe and gemini_pipe:
            # 3.1. PRISTRANOST OPENAI SUDCA
            oa_scores_on_oa = df[df['pipeline'] == openai_pipe]['OA_Overall'].dropna()
            oa_scores_on_ge = df[df['pipeline'] == gemini_pipe]['OA_Overall'].dropna()
            
            mean_oa_on_oa = oa_scores_on_oa.mean()
            mean_oa_on_ge = oa_scores_on_ge.mean()
            t_oa, p_oa = stats.ttest_ind(oa_scores_on_oa, oa_scores_on_ge, equal_var=False)
            
            f.write("A) Kako OPENAI SUDAC ocjenjuje generatore?\n")
            f.write(f" - Ocjenjuje svoj model ({openai_pipe}): Prosjek = {mean_oa_on_oa:.3f}\n")
            f.write(f" - Ocjenjuje tuđi model ({gemini_pipe}): Prosjek = {mean_oa_on_ge:.3f}\n")
            f.write(f" - T-test: p = {p_oa:.4f}\n")
            if p_oa < 0.05 and mean_oa_on_oa > mean_oa_on_ge:
                f.write(" -> ZAKLJUČAK: Pronađena je STATISTIČKI ZNAČAJNA PRISTRANOST. OpenAI sudac značajno favorizira vlastite odgovore.\n\n")
            elif p_oa < 0.05 and mean_oa_on_oa < mean_oa_on_ge:
                f.write(" -> ZAKLJUČAK: Značajna razlika postoji, ali OpenAI sudac preferira Gemini odgovore (nema self-biasa).\n\n")
            else:
                f.write(" -> ZAKLJUČAK: Nema dokaza o pristranosti. OpenAI sudac jednako ocjenjuje oba modela.\n\n")

            # 3.2. PRISTRANOST GEMINI SUDCA
            ge_scores_on_ge = df[df['pipeline'] == gemini_pipe]['GE_Overall'].dropna()
            ge_scores_on_oa = df[df['pipeline'] == openai_pipe]['GE_Overall'].dropna()
            
            mean_ge_on_ge = ge_scores_on_ge.mean()
            mean_ge_on_oa = ge_scores_on_oa.mean()
            t_ge, p_ge = stats.ttest_ind(ge_scores_on_ge, ge_scores_on_oa, equal_var=False)
            
            f.write("B) Kako GEMINI SUDAC ocjenjuje generatore?\n")
            f.write(f" - Ocjenjuje svoj model ({gemini_pipe}): Prosjek = {mean_ge_on_ge:.3f}\n")
            f.write(f" - Ocjenjuje tuđi model ({openai_pipe}): Prosjek = {mean_ge_on_oa:.3f}\n")
            f.write(f" - T-test: p = {p_ge:.4f}\n")
            if p_ge < 0.05 and mean_ge_on_ge > mean_ge_on_oa:
                f.write(" -> ZAKLJUČAK: Pronađena je STATISTIČKI ZNAČAJNA PRISTRANOST. Gemini sudac značajno favorizira vlastite odgovore.\n\n")
            elif p_ge < 0.05 and mean_ge_on_ge < mean_ge_on_oa:
                f.write(" -> ZAKLJUČAK: Značajna razlika postoji, ali Gemini sudac preferira OpenAI odgovore (nema self-biasa).\n\n")
            else:
                f.write(" -> ZAKLJUČAK: Nema dokaza o pristranosti. Gemini sudac jednako ocjenjuje oba modela.\n\n")

            # 3.3. OPĆA STROGOĆA SUDACA (Paired T-test na istim pitanjima)
            f.write("C) Usporedba strogoće sudaca (Tko daje više ocjene za ISTE odgovore?)\n")
            oa_all = df['OA_Overall'].dropna()
            ge_all = df['GE_Overall'].dropna()
            
            # Provjera jesu li duljine iste (ako nema null vrijednosti)
            if len(oa_all) == len(ge_all):
                t_strict, p_strict = stats.ttest_rel(oa_all, ge_all) # Paired T-test
                f.write(f" - Prosjek SVIH ocjena koje je dao OpenAI sudac: {oa_all.mean():.3f}\n")
                f.write(f" - Prosjek SVIH ocjena koje je dao Gemini sudac: {ge_all.mean():.3f}\n")
                f.write(f" - Paired T-test: p = {p_strict:.4f}\n")
                if p_strict < 0.05:
                    harsher = "Gemini" if oa_all.mean() > ge_all.mean() else "OpenAI"
                    f.write(f" -> ZAKLJUČAK: Razlika je značajna. {harsher} sudac je statistički značajno STROŽI (daje niže ocjene) od konkurenta.\n")
                else:
                    f.write(" -> ZAKLJUČAK: Nema značajne razlike u strogoći između sudaca.\n")
            else:
                f.write(" -> Nije moguće provesti Paired T-test zbog nedostajućih podataka (NaN) u nekim retcima.\n")

        else:
            f.write("Nisu prepoznata oba pipelinea (OpenAI i Gemini) u podacima, analiza pristranosti preskočena.\n")

    # Ispis na ekran
    with open(report_path, "r", encoding="utf-8") as f:
        print(f.read())
        
    print(f"\n[GOTOVO] Dubinski tekstualni izvještaj je spremljen u: {report_path}")

# POKRETANJE
csv_file_name = "./Rezultati/benchmark_multi_dataset_20260225_1336.csv"  # Promijeni na svoj stvarni naziv!
run_advanced_statistical_analysis(csv_file_name)