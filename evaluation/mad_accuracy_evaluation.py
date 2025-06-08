#!/usr/bin/env python3

"""
MAD (Mean Absolute Difference) Accuracy Evaluation Script

This script computes accuracy metrics by comparing responses between different waves
and LLM-imputed responses. It generates both Excel summaries and visualization plots.

Based on the MAD 0514 notebook analysis - following the exact same logic.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import sem, t
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import argparse
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def summary_mad(values):
    """
    Compute summary statistics for MAD values including confidence intervals.
    
    Args:
        values: Array-like of MAD values
        
    Returns:
        Tuple of (mean, stderr, ci_low, ci_high)
    """
    arr = np.array(values)
    mean_mad = arr.mean()
    stderr = sem(arr) if len(arr) > 1 else 0
    ci_low, ci_high = t.interval(0.95, len(arr) - 1, loc=mean_mad, scale=stderr) if len(arr) > 1 else (np.nan, np.nan)
    return round(mean_mad, 3), round(stderr, 3), round(ci_low, 3), round(ci_high, 3)


def assign_decile(value, thresholds):
    """
    Assign a value to its corresponding decile based on thresholds.
    
    Args:
        value: The value to assign
        thresholds: Array of threshold values for deciles
        
    Returns:
        Decile number (1-10) or NaN if value is NaN
    """
    if pd.isna(value):
        return np.nan
    for i, t in enumerate(thresholds):
        if value <= t:
            return i + 1
    return 10


def compute_mad_summary(csv_dir: str, output_excel_path: str, output_fig_path: str, fig_title: str):
    """
    Compute MAD accuracy summary from CSV files and generate outputs.
    
    Args:
        csv_dir: Directory containing the input CSV files
        output_excel_path: Path for the output Excel file
        output_fig_path: Path for the output figure
        fig_title: Title for the generated figure
    """
    # Starting MAD accuracy evaluation
    
    # Read required CSV files
    try:
        # Skip the description row (row index 1) when reading CSVs
        df_wave1_3 = pd.read_csv(os.path.join(csv_dir, "responses_wave1_3_formatted.csv"), skiprows=[1])
        df_wave4 = pd.read_csv(os.path.join(csv_dir, "responses_wave4_formatted.csv"), skiprows=[1])
        df_llm = pd.read_csv(os.path.join(csv_dir, "responses_llm_imputed_formatted.csv"), skiprows=[1])
    except Exception as e:
        logger.error(f"Error reading CSV files: {e}")
        return
    
    # Log data shapes for debugging if needed
    logger.debug(f"Loaded data shapes: wave1_3: {df_wave1_3.shape}, wave4: {df_wave4.shape}, llm: {df_llm.shape}")

    # ---------------------------
    # Clean data
    # ---------------------------
    for df in [df_wave1_3, df_wave4, df_llm]:
        df.columns = df.columns.str.upper()
        df.drop(columns=[c for c in df.columns if c == "WAVE"], inplace=True, errors='ignore')
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---------------------------
    # Set up respondent and column filtering
    # ---------------------------
    common_ids = set(df_wave1_3["TWIN_ID"]) & set(df_wave4["TWIN_ID"]) & set(df_llm["TWIN_ID"])
    df_wave1_3 = df_wave1_3[df_wave1_3["TWIN_ID"].isin(common_ids)].set_index("TWIN_ID")
    df_wave4 = df_wave4[df_wave4["TWIN_ID"].isin(common_ids)].set_index("TWIN_ID")
    df_llm = df_llm[df_llm["TWIN_ID"].isin(common_ids)].set_index("TWIN_ID")

    # Columns to decile-normalize
    decile_cols_164 = ["Q164", "Q166"]
    decile_cols_168 = ["Q168", "Q170"]

    # Ensure relevant columns are numeric
    for df in [df_wave1_3, df_wave4, df_llm]:
        for col in decile_cols_164 + decile_cols_168:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter columns that actually exist in the data
    existing_decile_cols_164 = [col for col in decile_cols_164 if col in df_wave1_3.columns]
    existing_decile_cols_168 = [col for col in decile_cols_168 if col in df_wave1_3.columns]
    
    # Only create combined data if columns exist
    if existing_decile_cols_164:
        combined_164_166 = pd.concat([df_wave1_3[col] for col in existing_decile_cols_164]).dropna()
    else:
        combined_164_166 = pd.Series(dtype=float)
        
    if existing_decile_cols_168:
        combined_168_170 = pd.concat([df_wave1_3[col] for col in existing_decile_cols_168]).dropna()
    else:
        combined_168_170 = pd.Series(dtype=float)

    grouped_deciles = {}
    if len(combined_164_166) > 0:
        grouped_deciles["QID164_GROUP"] = np.percentile(combined_164_166, np.arange(10, 100, 10))
    if len(combined_168_170) > 0:
        grouped_deciles["QID168_GROUP"] = np.percentile(combined_168_170, np.arange(10, 100, 10))

    for df in [df_wave1_3, df_wave4, df_llm]:
        if "QID164_GROUP" in grouped_deciles:
            for col in existing_decile_cols_164:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: assign_decile(x, grouped_deciles["QID164_GROUP"]))
        if "QID168_GROUP" in grouped_deciles:
            for col in existing_decile_cols_168:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: assign_decile(x, grouped_deciles["QID168_GROUP"]))

    cols_all = [col for col in df_wave1_3.columns if col in df_wave4.columns and col in df_llm.columns]

    valid_mask = pd.DataFrame({
        col: df_wave1_3[col].notna() & df_wave4[col].notna() & df_llm[col].notna()
        for col in cols_all
    })
    valid_mask = valid_mask.loc[:, valid_mask.any()]
    
    # Update cols_all to only include columns that have valid data
    cols_all = [col for col in cols_all if col in valid_mask.columns]

    # ────────────────────────────────────────────
    # Manual min/max definitions for every column
    manual_minmax = {}

    # false consensus (self)
    manual_minmax.update({f"FALSE CONS. SELF _{i}": (1, 5) for i in range(1, 11)})

    # false consensus (others)
    others_ids = [1,2,3,4,5,6,7,10,11,12]
    manual_minmax.update({f"FALSE CONS. OTHERS _{i}": (0, 100) for i in others_ids})

    # base‐rate / "Form A"
    manual_minmax["Q156_1"]    = (0, 100)
    manual_minmax["FORM A _1"] = (0, 100)

    # 5‐point questions (range 1–6)
    codes_5_6 = ["157","158"] + [f"160_{i}" for i in (1,2,3)] + [f"159_{i}" for i in (1,2,3)]
    manual_minmax.update({f"Q{c}": (1, 6) for c in codes_5_6})

    # outcome‐bias and similar (range 1–7)
    for c in ("161","162"):
        manual_minmax[f"Q{c}"] = (1, 7)

    # anchoring & adjustment (range 1–10)
    for c in ("164","166","168","170"):
        manual_minmax[f"Q{c}"] = (1, 10)

    # less‑is‑more & siblings (range 1–5 or 1–6)
    for c in ("171","172","173","174","175","176"):
        manual_minmax[f"Q{c}"] = (1, 5)
    for c in ("177","178","179"):
        manual_minmax[f"Q{c}"] = (1, 6)

    # sunk cost fallacy (0–20)
    manual_minmax["Q181"] = (0, 20)
    manual_minmax["Q182"] = (0, 20)

    # absolute vs relative savings (1–2)
    for c in ("183","184"):
        manual_minmax[f"Q{c}"] = (1, 2)

    # Allais (1–10)
    for c in ("189","190","191"):
        manual_minmax[f"Q{c}"] = (1, 10)

    # myside bias (1–2)
    for c in ("192","193"):
        manual_minmax[f"Q{c}"] = (1, 2)

    # WTA/WTP (1–6)
    for c in ("194","195"):
        manual_minmax[f"Q{c}"] = (1, 6)

    # prob‐matching vs max (1–2)
    manual_minmax.update({f"Q198_{i}": (1, 2) for i in range(1, 11)})
    manual_minmax.update({f"Q203_{i}": (1, 2) for i in range(1, 7)})

    # non‑separability (1–7)
    manual_minmax.update({f"NONSEPARABILTY BENE _{i}": (1, 7) for i in range(1,5)})
    manual_minmax.update({f"NONSEPARABILITY RIS _{i}": (1, 7) for i in range(1,5)})

    # omission‐bias & denominator neglect
    manual_minmax["OMISSION BIAS "]       = (1, 4)
    manual_minmax["DENOMINATOR NEGLECT "] = (1, 2)

    # pricing questions (1–2)
    manual_minmax.update({f"{i}_Q295": (1, 2) for i in range(1, 41)})

    # ────────────────────────────────────────────
    # Filter cols_all and build mins, maxs, ranges
    logger.debug(f"Before manual filtering: {len(cols_all)} columns")
    logger.debug(f"Manual min/max definitions available for: {len(manual_minmax)} columns")
    
    # Debug: Show some example columns from data
    logger.debug(f"First 10 columns in data: {cols_all[:10]}")
    logger.debug(f"First 10 expected columns: {list(manual_minmax.keys())[:10]}")
    
    # Check for case sensitivity issues
    cols_all_upper = [col.upper() for col in cols_all]
    manual_keys_upper = [key.upper() for key in manual_minmax.keys()]
    matching_upper = [col for col in cols_all_upper if col in manual_keys_upper]
    logger.debug(f"Columns matching when case-insensitive: {len(matching_upper)}")
    
    cols_all = [col for col in cols_all if col in manual_minmax]
    logger.info(f"After manual filtering: {len(cols_all)} columns for MAD analysis")
    
    if len(cols_all) == 0:
        logger.warning("No valid columns found for MAD analysis after filtering.")
        logger.warning("This might indicate data compatibility issues between the expected and actual column names.")
        # Create empty results
        mad_col_df = pd.DataFrame()
        mad_col_summary_df = pd.DataFrame()
        mad_task_level_df = pd.DataFrame()
        mad_task_summary_df = pd.DataFrame()
        
        # Save empty Excel file
        logger.info(f"Saving empty Excel summary to: {output_excel_path}")
        os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
        with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
            mad_task_summary_df.to_excel(writer, sheet_name="Accuracy Summary - task level", index=False)
            mad_task_level_df.to_excel(writer, sheet_name="Accuracy - task level", index=False) 
            mad_col_df.to_excel(writer, sheet_name="Accuracy - column level", index=False)
        
        # Create empty plot
        logger.info(f"Creating empty plot: {output_fig_path}")
        os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'No data available for plotting', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16)
        ax.set_title(fig_title)
        plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.warning("MAD accuracy evaluation completed with no data.")
        return

    mins   = pd.Series({col: mn for col, (mn, mx) in manual_minmax.items() if col in cols_all})
    maxs   = pd.Series({col: mx for col, (mn, mx) in manual_minmax.items() if col in cols_all})
    ranges = maxs - mins

    # Draw the random baseline once for every respondent × every column
    random_baseline = pd.DataFrame(
        { col: np.random.randint(int(mins[col]), int(maxs[col]) + 1, size=len(df_wave1_3))
          for col in cols_all },
        index=df_wave1_3.index
    )

    # ---------------------------
    # Compute per-column MADs
    # ---------------------------
    mad_col_rows = []
    for col in cols_all:
        mask = valid_mask[col]
        x1 = df_wave1_3.loc[mask, col]
        x4 = df_wave4.loc[mask, col]
        xl = df_llm.loc[mask, col]
        # random baseline: uniform in [min, max]
        xrand = random_baseline.loc[mask, col]
        
        r = ranges[col]
        norm_diff_14 = (x1 - x4).abs() / r
        norm_diff_1l = (x1 - xl).abs() / r
        norm_diff_rand = (x1 - xrand).abs() / r
        
        mean_14, se_14, ci_lo_14, ci_hi_14 = summary_mad(1-norm_diff_14)
        mean_1l, se_1l, ci_lo_1l, ci_hi_1l = summary_mad(1-norm_diff_1l)
        mean_rand, se_rand, ci_lo_rand, ci_hi_rand = summary_mad(1-norm_diff_rand)
        
        mad_col_rows.append({
            "Column": col,
            
            "wave4 vs. wave1_3": norm_diff_14.mean(),
            "wave4 vs. wave1_3 Accuracy": mean_14,
            "wave4 vs. wave1_3 Accuracy 95% CI Lower": ci_lo_14,
            "wave4 vs. wave1_3 Accuracy 95% CI Higher": ci_hi_14,
            
            "llm vs. wave1_3": norm_diff_1l.mean(),
            "llm vs. wave1_3 Accuracy": mean_1l,
            "llm vs. wave1_3 Accuracy 95% CI Lower": ci_lo_1l,
            "llm vs. wave1_3 Accuracy 95% CI Higher": ci_hi_1l,
            "llm accuracy / wave4 accuracy": (1 - norm_diff_1l.mean()) / (1 - norm_diff_14.mean()),
            
            "random_baseline vs wave1_3": norm_diff_rand.mean(),
            "random_baseline vs wave1_3 Accuracy": mean_rand,
            "random_baseline vs. wave1_3 Accuracy 95% CI Lower": ci_lo_rand,
            "random_baseline vs. wave1_3 Accuracy 95% CI Higher": ci_hi_rand,
            "random_baseline accuracy / wave4 accuracy": (1 - norm_diff_rand.mean()) / (1 - norm_diff_14.mean()),
            
            "number of respondents": mask.sum()
        })
    mad_col_df = pd.DataFrame(mad_col_rows)

    # ---------------------------
    # Summary of column-level MADs
    # ---------------------------
    summary = []
    for label in ["wave4 vs. wave1_3", "llm vs. wave1_3","random_baseline vs wave1_3"]:
        mad_vals = mad_col_df[label]
        acc_vals = mad_col_df[f"{label} Accuracy"]
        mean_mad, se_mad, ci_lo_mad, ci_hi_mad = summary_mad(mad_vals)
        mean_acc, se_acc, ci_lo_acc, ci_hi_acc = summary_mad(acc_vals)
        summary.append({
            "Comparison": label,
            "Mean Accuracy": mean_acc,
            "Accuracy Standard Error": se_acc,
            "Accuracy 95% CI Lower": ci_lo_acc,
            "Accuracy 95% CI Upper": ci_hi_acc,
            "TWIN_IDs": len(common_ids),
            "Min Columns Used": valid_mask.sum(axis=1).min(),
            "Max Columns Used": valid_mask.sum(axis=1).max(),
            "Mean Columns Used": round(valid_mask.sum(axis=1).mean(), 2)
        })
    mad_col_summary_df = pd.DataFrame(summary)

    # Column-level analyses completed

    # ---------------------------
    # Task-level MADs (including benchmark)
    # ---------------------------
    qid_to_task_raw = {
        **{f"False Cons. self _{i}": "false consensus" for i in range(1, 11)},
        **{f"False cons. others _{i}": "false consensus" for i in [1,2,3,4,5,6,7,10,11,12]},
        "Q156_1": "base rate", "Form A _1": "base rate",
        "Q157": "framing problem", "Q158": "framing problem",
        **{f"Q160_{i}": "conjunction problem (Linda)" for i in [1,2,3]},
        **{f"Q159_{i}": "conjunction problem (Linda)" for i in [1,2,3]},
        "Q161": "outcome bias", "Q162": "outcome bias",
        "Q164": "anchoring and adjustment", "Q166": "anchoring and adjustment",
        "Q168": "anchoring and adjustment", "Q170": "anchoring and adjustment",
        **{f"Q17{i}": "less is more" for i in range(1, 10)},
        "Q181": "sunk cost fallacy", "Q182": "sunk cost fallacy",
        "Q183": "absolute vs. relative savings", "Q184": "absolute vs. relative savings",
        "Q189": "WTA/WTP-Thaler", "Q190": "WTA/WTP-Thaler", "Q191": "WTA/WTP-Thaler",
        "Q192": "Allais", "Q193": "Allais",
        "Q194": "myside", "Q195": "myside",
        **{f"Q198_{i}": "prob matching vs. max" for i in range(1, 11)},
        **{f"Q203_{i}": "prob matching vs. max" for i in range(1, 7)},
        **{f"nonseparabilty bene _{i}": "non-separability of risks and benefits" for i in range(1, 5)},
        **{f"nonseparability ris _{i}": "non-separability of risks and benefits" for i in range(1, 5)},
        "Omission bias ": "omission",
        "Denominator neglect ": "denominator neglect",
        **{f"{i}_Q295": "pricing" for i in range(1, 41)}
    }
    qid_to_task = {k.upper(): v for k, v in qid_to_task_raw.items()}
    all_tasks = sorted(set(qid_to_task.values()))

    task_level_mads = []
    for task in all_tasks:
        task_cols = [col for col, t in qid_to_task.items() if t == task and col in cols_all]
        if not task_cols:
            continue

        mad_14_all, mad_1l_all, mad_rand_all = [], [], []
        respondent_counts = set()

        for col in task_cols:
            mask = df_wave1_3[col].notna() & df_wave4[col].notna() & df_llm[col].notna()
            if not mask.any():
                continue

            v1 = df_wave1_3.loc[mask, col]
            v4 = df_wave4.loc[mask, col]
            vl = df_llm.loc[mask, col]
            # random baseline
            vrand = random_baseline.loc[v1.index, col]
            
            r = ranges[col]
            mad_14_all.append((v1 - v4).abs() / r)
            mad_1l_all.append((v1 - vl).abs() / r)
            mad_rand_all.append((v1 - vrand).abs() / r)
            respondent_counts.update(v1.index)

        if mad_14_all and mad_1l_all and mad_rand_all:
            col_diffs_14 = pd.concat(mad_14_all)
            col_diffs_1l = pd.concat(mad_1l_all)
            col_diffs_rand = pd.concat(mad_rand_all)

            mean_14, se_14, ci_lo_14, ci_hi_14 = summary_mad(1-col_diffs_14)
            mean_1l, se_1l, ci_lo_1l, ci_hi_1l = summary_mad(1-col_diffs_1l)
            mean_rand, se_rand, ci_lo_rand, ci_hi_rand = summary_mad(1-col_diffs_rand)
            
            task_level_mads.append({
                "Task": task,
            
                "wave4 vs. wave1_3": col_diffs_14.mean(),
                "wave4 vs. wave1_3 Accuracy": mean_14,
                "wave4 vs. wave1_3 Accuracy 95% CI Lower": ci_lo_14,
                "wave4 vs. wave1_3 Accuracy 95% CI Higher": ci_hi_14,
            
                "llm vs. wave1_3": col_diffs_1l.mean(),
                "llm vs. wave1_3 Accuracy": mean_1l,
                "llm vs. wave1_3 Accuracy 95% CI Lower": ci_lo_1l,
                "llm vs. wave1_3 Accuracy 95% CI Higher": ci_hi_1l,
                "llm accuracy / wave4 accuracy": (1 - col_diffs_1l.mean()) / (1 - col_diffs_14.mean()),
            
                "random_baseline vs wave1_3": col_diffs_rand.mean(),            
                "random_baseline vs wave1_3 Accuracy": mean_rand,
                "random_baseline vs. wave1_3 Accuracy 95% CI Lower": ci_lo_rand,
                "random_baseline vs. wave1_3 Accuracy 95% CI Higher": ci_hi_rand,
                "random_baseline accuracy / wave4 accuracy": (1 - col_diffs_rand.mean()) / (1 - col_diffs_14.mean()),
                
                "number of respondents": len(respondent_counts)
            })

    mad_task_level_df = pd.DataFrame(task_level_mads)

    # ---------------------------
    # Task-level summary
    # ---------------------------
    respondent_task_mads = {
        "wave4 vs. wave1_3": [],
        "llm vs. wave1_3": [],
        "random_baseline vs wave1_3": []
    }
    task_usage_counts = []

    for rid in df_wave1_3.index:
        task_to_diffs_14 = {}
        task_to_diffs_1l = {}
        task_to_diffs_rand= {}
        
        for col in cols_all:
            task = qid_to_task.get(col)
            if not task:
                continue
            if pd.notna(df_wave1_3.at[rid, col]) and pd.notna(df_wave4.at[rid, col]) and pd.notna(df_llm.at[rid, col]):
                v1 = df_wave1_3.at[rid, col]
                v4 = df_wave4.at[rid, col]
                vl = df_llm.at[rid, col]
                # random baseline
                vr = random_baseline.at[rid, col]
                
                r = ranges[col]
                task_to_diffs_14.setdefault(task, []).append(abs(v1 - v4) / r)
                task_to_diffs_1l.setdefault(task, []).append(abs(v1 - vl) / r)
                task_to_diffs_rand.setdefault(task, []).append(abs(v1 - vr) / r)

        task_mads_14 = [np.mean(vals) for vals in task_to_diffs_14.values() if vals]
        task_mads_1l = [np.mean(vals) for vals in task_to_diffs_1l.values() if vals]
        task_mads_rand= [np.mean(vals) for vals in task_to_diffs_rand.values() if vals]

        if task_mads_14:
            respondent_task_mads["wave4 vs. wave1_3"].append(np.mean(task_mads_14))
        if task_mads_1l:
            respondent_task_mads["llm vs. wave1_3"].append(np.mean(task_mads_1l))
        if task_mads_rand:
            respondent_task_mads["random_baseline vs wave1_3"].append(np.mean(task_mads_rand))

        task_usage_counts.append(len(task_mads_14))

    task_summary = []
    for label in ["wave4 vs. wave1_3", "llm vs. wave1_3","random_baseline vs wave1_3"]:
        arr = np.array(respondent_task_mads[label])
        mean_mad = arr.mean()
        se = sem(arr)
        ci_low, ci_high = t.interval(0.95, len(arr)-1, loc=mean_mad, scale=se)
        task_summary.append({
            "Comparison": label,
            "Mean Accuracy": round(1 - mean_mad, 3),
            "Accuracy Standard Error": round(se, 3),
            "Accuracy 95% CI Lower": round(1 - ci_high, 3),
            "Accuracy 95% CI Upper": round(1 - ci_low, 3),
            "TWIN_IDs": len(arr),
            "Min Tasks Used": int(np.min(task_usage_counts)),
            "Max Tasks Used": int(np.max(task_usage_counts)),
            "Mean Tasks Used": round(np.mean(task_usage_counts), 2)
        })

    mad_task_summary_df = pd.DataFrame(task_summary)

    # ---------------------------
    # Save all to Excel
    # ---------------------------
    logger.info(f"Saving Excel summary to: {output_excel_path}")
    os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
    
    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        mad_task_summary_df.to_excel(writer, sheet_name="Accuracy Summary - task level", index=False)
        mad_task_level_df.to_excel(writer, sheet_name="Accuracy - task level", index=False)
        mad_col_df.to_excel(writer, sheet_name="Accuracy - column level", index=False)

    # Excel summary saved

    # ---------------------------
    # Generate plot
    # ---------------------------
    logger.info(f"Generating accuracy plot: {output_fig_path}")
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    
    if len(mad_task_level_df) == 0:
        logger.warning("No task-level data available for plotting. Creating empty plot.")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'No data available for plotting', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16)
        ax.set_title(fig_title)
        plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.warning("MAD accuracy evaluation completed with no plottable data.")
        return
    
    df = mad_task_level_df.sort_values("llm accuracy / wave4 accuracy", ascending=False)
    tasks   = df["Task"]
    x       = np.arange(len(tasks))

    width   = 0.8
    bar_h   = width / 4
    offsets = np.array([-1.5, -0.5, +0.5, +1.5]) * bar_h

    means    = [
        df["wave4 vs. wave1_3 Accuracy"]*100,
        df["llm vs. wave1_3 Accuracy"]*100,
        df["random_baseline vs wave1_3 Accuracy"]*100
    ]

    errs     = [
        np.vstack(( 
          means[0] - df["wave4 vs. wave1_3 Accuracy 95% CI Lower"]*100,
          df["wave4 vs. wave1_3 Accuracy 95% CI Higher"]*100 - means[0]
        )),
        np.vstack((
          means[1] - df["llm vs. wave1_3 Accuracy 95% CI Lower"]*100,
          df["llm vs. wave1_3 Accuracy 95% CI Higher"]*100 - means[1]
        )),
        np.vstack((
          means[2] - df["random_baseline vs. wave1_3 Accuracy 95% CI Lower"]*100,
          df["random_baseline vs. wave1_3 Accuracy 95% CI Higher"]*100 - means[2]
        ))
    ]

    labels   = [
        "Test-retest",
        "Digital twins",
        "Random"
    ]

    colors = ["tab:grey","tab:blue", "tab:green"]

    # compute averages
    avg = [m.replace([np.inf,-np.inf], np.nan).dropna().mean() for m in means]

    fig, ax = plt.subplots(figsize=(12, 8))

    # now zip offsets, means, errs, labels, colors in exactly the same order
    for off, mean, err, lbl, col in zip(offsets, means, errs, labels, colors):
        ax.barh(
            x + off,
            mean,
            height=bar_h,
            color=col,
            xerr=err,
            capsize=5,
            label=lbl
        )

    # horizontal formatting
    ax.set_yticks(x)
    ax.set_yticklabels(tasks)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(fig_title)

    # format x-axis as percentages
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f%%'))

    # average lines
    for a, lbl, col in zip(avg, labels, colors):
        ax.axvline(
            a,
            color=col,
            ls="--",
            lw=1.5,
            label=f"{lbl} - average = {a:.2f}%"
        )

    ax.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory
    
    # MAD accuracy evaluation completed successfully


def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(description="Compute MAD accuracy evaluation")
    
    # Support both old-style individual arguments and new config-based approach
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument("--csv-dir", help="Directory containing input CSV files")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--output-excel-filename", default="mad_accuracy_summary.xlsx", help="Output Excel filename")
    parser.add_argument("--output-plot-filename", default="accuracy_dist.png", help="Output plot filename")
    parser.add_argument("--plot-title", default="Task-Level Accuracy", help="Figure title")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    # Legacy support for positional arguments
    parser.add_argument("csv_dir", nargs="?", help="Directory containing input CSV files (legacy)")
    parser.add_argument("output_excel", nargs="?", help="Output Excel file path (legacy)")
    parser.add_argument("output_fig", nargs="?", help="Output figure file path (legacy)")
    parser.add_argument("--title", help="Figure title (legacy)")
    
    args = parser.parse_args()
    
    # Handle config-based execution
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract paths from config with variable substitution
        trial_dir = config.get('trial_dir', '')
        csv_dir = os.path.join(trial_dir, 'csv_comparison', 'csv_formatted') if trial_dir else None
        output_dir_template = config.get('evaluation', {}).get('output_dir', f"{trial_dir}/accuracy_evaluation")
        output_dir = output_dir_template.replace('${trial_dir}', trial_dir) if trial_dir else 'accuracy_evaluation'
        model_name = config.get('model_name', 'LLM')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set output paths
        output_excel = os.path.join(output_dir, args.output_excel_filename)
        output_fig = os.path.join(output_dir, args.output_plot_filename)
        title = args.plot_title.replace('LLM', model_name) if 'LLM' in args.plot_title else args.plot_title
        
        if args.verbose:
            print(f"Config-based execution:")
            print(f"  CSV directory: {csv_dir}")
            print(f"  Output directory: {output_dir}")
            print(f"  Model name: {model_name}")
        
        # Run main analysis
        compute_mad_summary(csv_dir, output_excel, output_fig, title)
    
    # Handle CLI-based execution
    elif args.csv_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_excel = os.path.join(args.output_dir, args.output_excel_filename)
        output_fig = os.path.join(args.output_dir, args.output_plot_filename)
        compute_mad_summary(args.csv_dir, output_excel, output_fig, args.plot_title)
    
    # Handle legacy positional arguments
    elif args.csv_dir and args.output_excel and args.output_fig:
        title = args.title or "Task-Level Accuracy"
        compute_mad_summary(args.csv_dir, args.output_excel, args.output_fig, title)
    
    else:
        parser.error("Either --config must be provided, or --csv-dir and --output-dir, or legacy positional arguments")


if __name__ == "__main__":
    main()