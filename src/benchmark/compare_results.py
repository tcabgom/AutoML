"""
compare_automl.py

Script extendido para comparar resultados de tu AutoML (results.csv) con benchmarks
que incluyen múltiples AutoMLs en un mismo CSV (por ejemplo classification_1h8c*.csv).

Novedades:
- Detecta y separa los frameworks en los otros CSVs (columna 'framework').
- Compara tu AutoML con cada framework individualmente.
- Calcula cuántos frameworks gana tu AutoML por dataset y en total.
- Ordena frameworks por media de puntuación.
- Guarda tablas resumen (por dataset y por framework) y gráficos opcionales.

Ejemplo de uso:
python compare_automl.py -m results.csv -o classification_1h8c(1).csv --plots

Salida:
 - comparison_summary.csv        # resumen por dataset (con cuántos frameworks gana tu AutoML)
 - framework_summary.csv         # rendimiento medio por framework
 - merged_matched_rows.csv       # comparaciones por fold
 - overall_summary.json          # resumen global
 - plots/ (si se pide con --plots)
"""

import argparse
import os
import json
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COMMON_DATASET_KEYS = ["dataset","data","task","dataset_name","datasetname","problem","ds","nombre"]
COMMON_FOLD_KEYS = ["fold","cv","split","fold_number","foldnumber","fold_id"]
COMMON_SCORE_KEYS = ["score","auc","acc","accuracy","f1","balacc","balanced_accuracy","roc_auc","logloss","loss","metric"]


def guess_column(cols: List[str], targets: List[str]) -> Optional[str]:
    cols_low = [c.lower() for c in cols]
    for t in targets:
        for i,c in enumerate(cols_low):
            if t in c:
                return cols[i]
    return None


def read_and_pick(path: str, dataset_col=None, fold_col=None, score_col=None):
    df = pd.read_csv(path)
    cols = df.columns.tolist()

    d_col = dataset_col or guess_column(cols, COMMON_DATASET_KEYS)
    f_col = fold_col or guess_column(cols, COMMON_FOLD_KEYS)
    s_col = score_col or guess_column(cols, COMMON_SCORE_KEYS)

    if d_col is None or f_col is None or s_col is None:
        raise ValueError(f"Column detection failed in {path}. Available: {cols}")

    return df, d_col, f_col, s_col


def compare_against_frameworks(my_df, other_df, my_names, other_names):
    my_d, my_f, my_s = my_names
    o_d, o_f, o_s = other_names

    if 'framework' not in other_df.columns:
        raise ValueError("Other CSV must include a 'framework' column.")

    # Prepare my data
    my_small = my_df[[my_d, my_f, my_s]].copy()
    my_small.columns = ["dataset","fold","my_score"]

    other_df['framework'] = other_df['framework'].astype(str)

    results = []

    for fw, df_fw in other_df.groupby('framework'):
        other_small = df_fw[[o_d, o_f, o_s]].copy()
        other_small.columns = ["dataset","fold","other_score"]
        other_small["dataset"] = other_small["dataset"].astype(str).str.strip()
        my_small["dataset"] = my_small["dataset"].astype(str).str.strip()

        merged = pd.merge(my_small, other_small, on=["dataset","fold"], how="left")
        merged["my_score"] = pd.to_numeric(merged["my_score"], errors="coerce")
        merged["other_score"] = pd.to_numeric(merged["other_score"], errors="coerce")
        merged = merged.dropna(subset=["my_score","other_score"])

        merged['delta'] = merged['my_score'] - merged['other_score']
        merged['framework'] = fw
        results.append(merged)

    comp_all = pd.concat(results, ignore_index=True)
    return comp_all


def summarize_results(comp_all):
    # Mean score per framework
    fw_summary = comp_all.groupby('framework').agg(
        mean_score=('other_score','mean'),
        mean_delta=('delta','mean')
    ).reset_index().sort_values('mean_score', ascending=False)

    # Dataset-level comparison: count frameworks beaten by my AutoML
    ds_fw = comp_all.groupby(['dataset','framework']).agg(
        my_mean=('my_score','mean'),
        other_mean=('other_score','mean')
    ).reset_index()
    ds_summary = ds_fw.groupby('dataset').apply(lambda g: pd.Series({
        'frameworks_total': len(g),
        'frameworks_beaten': (g['my_mean'] > g['other_mean']).sum(),
        'frameworks_lost': (g['my_mean'] < g['other_mean']).sum(),
        'my_mean': g['my_mean'].mean(),
        'other_mean_avg': g['other_mean'].mean()
    })).reset_index()

    return fw_summary, ds_summary


def plot_frameworks(fw_summary, outdir):
    plt.figure(figsize=(8,4))
    plt.bar(fw_summary['framework'], fw_summary['mean_score'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Mean score')
    plt.title('Mean performance by framework')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,'framework_means.png'))
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare AutoML results vs multi-framework benchmark CSVs")
    parser.add_argument('-m','--my',required=True)
    parser.add_argument('-o','--others',nargs='+',required=True)
    parser.add_argument('--dataset-col')
    parser.add_argument('--fold-col')
    parser.add_argument('--score-col')
    parser.add_argument('-d','--outdir',default='./comparison_out')
    parser.add_argument('--plots',action='store_true')
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)

    my_df, my_d, my_f, my_s = read_and_pick(args.my, args.dataset_col, args.fold_col, args.score_col)

    all_comparisons = []
    for path in args.others:
        other_df, od, of, os_ = read_and_pick(path, None, None, None)
        comp = compare_against_frameworks(my_df, other_df, (my_d,my_f,my_s), (od,of,os_))
        comp['source_file'] = os.path.basename(path)
        all_comparisons.append(comp)

    comp_all = pd.concat(all_comparisons, ignore_index=True)

    fw_summary, ds_summary = summarize_results(comp_all)

    comp_all.to_csv(os.path.join(args.outdir,'merged_matched_rows.csv'), index=False)
    fw_summary.to_csv(os.path.join(args.outdir,'framework_summary.csv'), index=False)
    ds_summary.to_csv(os.path.join(args.outdir,'comparison_summary.csv'), index=False)

    overall = {
        'frameworks_total': int(fw_summary.shape[0]),
        'datasets_total': int(ds_summary.shape[0]),
        'avg_frameworks_beaten_per_dataset': float(ds_summary['frameworks_beaten'].mean()),
        'overall_my_mean': float(comp_all['my_score'].mean()),
        'overall_other_mean': float(comp_all['other_score'].mean())
    }
    with open(os.path.join(args.outdir,'overall_summary.json'),'w') as f:
        json.dump(overall, f, indent=2)

    if args.plots:
        plot_frameworks(fw_summary, args.outdir)

    print("Done. Results in", args.outdir)


if __name__ == '__main__':
    #  python compare_results.py --my results/test.csv --others results/benchmark_results.csv
    main()
