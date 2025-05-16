import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import sem, t, norm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter # Added for plot formatting


# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def summary_mad(values: pd.Series | np.ndarray):
    """Return mean, standard error and 95% CI (low, high) rounded to 3 decimals."""
    arr = np.asarray(values, dtype=float)
    mean_mad = arr.mean() # This is actually mean of (1-normalized_difference) which is an accuracy
    stderr = sem(arr) if len(arr) > 1 else 0
    if len(arr) > 1 and stderr > 0: # Added stderr > 0 check
        ci_low, ci_high = t.interval(0.95, len(arr) - 1, loc=mean_mad, scale=stderr)
    else:
        ci_low = ci_high = mean_mad # If no variance or single point, CI is just the mean
        if stderr == 0 and len(arr) > 1 : # If no variance, CI is just the mean
             ci_low = ci_high = mean_mad
        else: # if len(arr) <=1 or stderr is nan
             ci_low = ci_high = np.nan


    return tuple(round(x, 3) if pd.notna(x) else np.nan for x in (mean_mad, stderr, ci_low, ci_high))


def calculate_correlation(series1: pd.Series, series2: pd.Series, min_common_pairs: int = 3) -> tuple[float | np.ndarray, int]:
    """Calculate Pearson correlation and the number of pairs used."""
    if not isinstance(series1, pd.Series): series1 = pd.Series(series1)
    if not isinstance(series2, pd.Series): series2 = pd.Series(series2)

    # Align series and drop NaNs pair-wise
    df_corr = pd.concat([series1, series2], axis=1).dropna()
    n_pairs = len(df_corr)

    if n_pairs < min_common_pairs:
        return np.nan, n_pairs
    
    # Check for zero variance in either series after NaN removal
    if df_corr.iloc[:, 0].var() == 0 or df_corr.iloc[:, 1].var() == 0:
        return np.nan, n_pairs
        
    corr = df_corr.iloc[:, 0].corr(df_corr.iloc[:, 1], method='pearson')
    return corr, n_pairs

def get_single_correlation_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Calculate CI for a single Pearson r using Fisher's z-transformation."""
    if pd.isna(r) or n < 4: # n-3 must be > 0 for SE_z
        return np.nan, np.nan
    
    # Clip r to avoid inf with arctanh for r = +/-1
    r_clipped = np.clip(r, -1 + 1e-9, 1 - 1e-9)
    z = np.arctanh(r_clipped)
    se_z = 1 / np.sqrt(n - 3)
    
    z_critical = norm.ppf(1 - alpha / 2) # Using norm.ppf for z_critical
    
    ci_low_z = z - z_critical * se_z
    ci_high_z = z + z_critical * se_z
    
    return np.tanh(ci_low_z), np.tanh(ci_high_z)

def summary_fisher_z_transformed_correlations(r_values: pd.Series | np.ndarray, alpha: float = 0.05):
    """Summarize a list of correlations using Fisher's z-transformation."""
    if isinstance(r_values, list): r_values = np.array(r_values)
    r_values_clean = r_values[~np.isnan(r_values) & (r_values >= -1) & (r_values <= 1)]

    if len(r_values_clean) == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Clip r to avoid inf with arctanh for r = +/-1
    r_clipped = np.clip(r_values_clean, -1 + 1e-9, 1 - 1e-9)
    z_values = np.arctanh(r_clipped)
    
    mean_z = np.mean(z_values)
    stderr_z = sem(z_values) if len(z_values) > 1 else 0
    
    if len(z_values) > 1 and stderr_z > 0:
        # CI for the mean of z-scores
        df_t = len(z_values) - 1
        ci_low_z, ci_high_z = t.interval(1 - alpha, df_t, loc=mean_z, scale=stderr_z)
    else:
        ci_low_z = ci_high_z = mean_z
        if stderr_z == 0 and len(z_values) > 1 :
             ci_low_z = ci_high_z = mean_z
        else:
             ci_low_z = ci_high_z = np.nan # Should be mean_z if single value, but CI is undefined

    # Transform summary stats back to r-scale
    mean_r_transformed = np.tanh(mean_z)
    ci_low_r_transformed = np.tanh(ci_low_z)
    ci_high_r_transformed = np.tanh(ci_high_z)
    
    # SEM is usually reported on the z-scale or not reported for transformed mean r. We return sem_z.
    return tuple(round(x, 3) if pd.notna(x) else np.nan for x in (mean_r_transformed, stderr_z, ci_low_r_transformed, ci_high_r_transformed))


def read_required_csv(csv_dir: str, filename: str) -> pd.DataFrame:
    """Read a CSV from *csv_dir* ensuring it exists."""
    path = os.path.join(csv_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required input CSV not found: {path}")
    return pd.read_csv(path)


# -------------------------------------------------
# Core computation
# -------------------------------------------------

def compute_mad_summary(csv_dir: str, output_excel_path: str, output_fig_path: str, output_corr_fig_path: str, output_corr_simple_avg_fig_path: str, fig_title: str, corr_fig_title: str, corr_simple_avg_fig_title: str):
    """Compute MAD accuracy and Correlation metrics and write an Excel file and plots."""
    df_wave1_3 = read_required_csv(csv_dir, "responses_wave1_3_formatted.csv")
    df_wave4 = read_required_csv(csv_dir, "responses_wave4_formatted.csv")
    df_llm = read_required_csv(csv_dir, "responses_llm_imputed_formatted.csv")

    for df in (df_wave1_3, df_wave4, df_llm):
        df.columns = df.columns.str.upper()
        df.drop(columns=[c for c in df.columns if c == "WAVE"], inplace=True, errors="ignore")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    common_ids = set(df_wave1_3["TWIN_ID"]) & set(df_wave4["TWIN_ID"]) & set(df_llm["TWIN_ID"])
    df_wave1_3 = df_wave1_3[df_wave1_3["TWIN_ID"].isin(common_ids)].set_index("TWIN_ID")
    df_wave4 = df_wave4[df_wave4["TWIN_ID"].isin(common_ids)].set_index("TWIN_ID")
    df_llm = df_llm[df_llm["TWIN_ID"].isin(common_ids)].set_index("TWIN_ID")
    
    # Decile normalization (as in notebook, though not directly used in the final task-level plot's primary values)
    decile_cols_164 = ["Q164", "Q166"]
    decile_cols_168 = ["Q168", "Q170"]

    for df in [df_wave1_3, df_wave4, df_llm]:
        for col in decile_cols_164 + decile_cols_168:
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate deciles based on df_wave1_3 as in the notebook
    combined_164_166 = pd.concat([df_wave1_3[col] for col in decile_cols_164 if col in df_wave1_3.columns]).dropna()
    combined_168_170 = pd.concat([df_wave1_3[col] for col in decile_cols_168 if col in df_wave1_3.columns]).dropna()

    grouped_deciles = {}
    if not combined_164_166.empty:
        grouped_deciles["QID164_GROUP"] = np.percentile(combined_164_166, np.arange(10, 100, 10))
    if not combined_168_170.empty:
        grouped_deciles["QID168_GROUP"] = np.percentile(combined_168_170, np.arange(10, 100, 10))

    def assign_decile(value, thresholds):
        if pd.isna(value): return np.nan
        for i, t in enumerate(thresholds):
            if value <= t: return i + 1
        return 10

    for df in [df_wave1_3, df_wave4, df_llm]:
        for col in decile_cols_164:
            if col in df.columns and "QID164_GROUP" in grouped_deciles:
                df[col] = df[col].apply(lambda x: assign_decile(x, grouped_deciles["QID164_GROUP"]))
        for col in decile_cols_168:
            if col in df.columns and "QID168_GROUP" in grouped_deciles:
                 df[col] = df[col].apply(lambda x: assign_decile(x, grouped_deciles["QID168_GROUP"]))

    manual_minmax: dict[str, tuple[int, int]] = {}
    manual_minmax.update({f"FALSE CONS. SELF _{i}": (1, 5) for i in range(1, 11)})
    manual_minmax.update({f"FALSE CONS. OTHERS _{i}": (0, 100) for i in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]})
    manual_minmax["Q156_1"] = (0, 100)
    manual_minmax["FORM A _1"] = (0, 100)
    codes_5_6 = ["157", "158"] + [f"160_{i}" for i in (1, 2, 3)] + [f"159_{i}" for i in (1, 2, 3)]
    manual_minmax.update({f"Q{c}": (1, 6) for c in codes_5_6})
    for c in ("161", "162"): manual_minmax[f"Q{c}"] = (1, 7)
    for c in ("164", "166", "168", "170"): manual_minmax[f"Q{c}"] = (1, 10)
    for c in ("171", "172", "173", "174", "175", "176"): manual_minmax[f"Q{c}"] = (1, 5)
    for c in ("177", "178", "179"): manual_minmax[f"Q{c}"] = (1, 6)
    manual_minmax["Q181"] = (0, 20); manual_minmax["Q182"] = (0, 20)
    for c in ("183", "184"): manual_minmax[f"Q{c}"] = (1, 2)
    for c in ("189", "190", "191"): manual_minmax[f"Q{c}"] = (1, 10)
    for c in ("192", "193"): manual_minmax[f"Q{c}"] = (1, 2)
    for c in ("194", "195"): manual_minmax[f"Q{c}"] = (1, 6)
    manual_minmax.update({f"Q198_{i}": (1, 2) for i in range(1, 11)})
    manual_minmax.update({f"Q203_{i}": (1, 2) for i in range(1, 7)})
    manual_minmax.update({f"NONSEPARABILTY BENE _{i}": (1, 7) for i in range(1, 5)})
    manual_minmax.update({f"NONSEPARABILITY RIS _{i}": (1, 7) for i in range(1, 5)})
    manual_minmax["OMISSION BIAS "] = (1, 4); manual_minmax["DENOMINATOR NEGLECT "] = (1, 2)
    manual_minmax.update({f"{i}_Q295": (1, 2) for i in range(1, 41)})

    cols_all = [col for col in df_wave1_3.columns if col in df_wave4.columns and col in df_llm.columns and col in manual_minmax]
    mins = pd.Series({col: manual_minmax[col][0] for col in cols_all})
    maxs = pd.Series({col: manual_minmax[col][1] for col in cols_all})
    ranges = maxs - mins
    ranges[ranges == 0] = 1 # Avoid division by zero for columns with no range

    rng = np.random.default_rng(seed=42)
    random_baseline = pd.DataFrame({
        col: rng.integers(low=mins[col], high=maxs[col] + 1, size=len(df_wave1_3))
        for col in cols_all
    }, index=df_wave1_3.index)

    valid_mask = pd.DataFrame({
        col: df_wave1_3[col].notna() & df_wave4[col].notna() & df_llm[col].notna()
        for col in cols_all
    })
    valid_mask = valid_mask.loc[:, valid_mask.any()] # Keep only columns with at least one valid respondent

    mad_col_rows = []
    for col in cols_all:
        if col not in valid_mask.columns or not valid_mask[col].any(): # Skip if no valid data
            continue
        mask = valid_mask[col]
        v1 = df_wave1_3.loc[mask, col]
        v4 = df_wave4.loc[mask, col]
        vl = df_llm.loc[mask, col]
        vbm_mode = vl.mode()
        vbm_val = vbm_mode.iloc[0] if not vbm_mode.empty else np.nan
        vbm = pd.Series(vbm_val, index=v1.index)
        vrand = random_baseline.loc[mask, col]
        
        r = ranges[col]
        if r == 0: r = 1 # Avoid division by zero, already handled but defensive

        norm_diff_14 = (v1 - v4).abs() / r
        norm_diff_1l = (v1 - vl).abs() / r
        norm_diff_rand = (v1 - vrand).abs() / r

        acc_14, se_14, ci_lo_14, ci_hi_14 = summary_mad(1 - norm_diff_14)
        acc_1l, se_1l, ci_lo_1l, ci_hi_1l = summary_mad(1 - norm_diff_1l)
        acc_rand, se_rand, ci_lo_rand, ci_hi_rand = summary_mad(1 - norm_diff_rand)
        
        div_ratio_1l_14 = acc_1l / acc_14 if acc_14 != 0 else np.nan
        div_ratio_rand_14 = acc_rand / acc_14 if acc_14 != 0 else np.nan


        mad_col_rows.append({
            "Column": col,
            "wave4 vs. wave1_3": norm_diff_14.mean(), "wave4 vs. wave1_3 Accuracy": acc_14,
            "wave4 vs. wave1_3 Accuracy 95% CI Lower": ci_lo_14, "wave4 vs. wave1_3 Accuracy 95% CI Higher": ci_hi_14,
            "llm vs. wave1_3": norm_diff_1l.mean(), "llm vs. wave1_3 Accuracy": acc_1l,
            "llm vs. wave1_3 Accuracy 95% CI Lower": ci_lo_1l, "llm vs. wave1_3 Accuracy 95% CI Higher": ci_hi_1l,
            "llm accuracy / wave4 accuracy": div_ratio_1l_14,
            "random_baseline vs wave1_3": norm_diff_rand.mean(), "random_baseline vs wave1_3 Accuracy": acc_rand,
            "random_baseline vs. wave1_3 Accuracy 95% CI Lower": ci_lo_rand, "random_baseline vs. wave1_3 Accuracy 95% CI Higher": ci_hi_rand,
            "random_baseline accuracy / wave4 accuracy": div_ratio_rand_14,
            "number of respondents": mask.sum(),
            # Correlation item-level
            "Correlation W4vW13": calculate_correlation(v1, v4)[0],
            "Correlation LLMvW13": calculate_correlation(v1, vl)[0],
            "Correlation RANDvW13": 0.0 # Per user request
        })
    mad_col_df = pd.DataFrame(mad_col_rows)

    summary_rows = []
    comparison_labels = ["wave4 vs. wave1_3", "llm vs. wave1_3", "random_baseline vs wave1_3"]
    for label in comparison_labels:
        acc_vals = mad_col_df[f"{label} Accuracy"]
        mean_acc, se_acc, ci_lo_acc, ci_hi_acc = summary_mad(acc_vals.dropna()) # Ensure NaNs are dropped before CI calc
        summary_rows.append({
            "Comparison": label, "Mean Accuracy": mean_acc, "Accuracy Standard Error": se_acc,
            "Accuracy 95% CI Lower": ci_lo_acc, "Accuracy 95% CI Upper": ci_hi_acc,
            "TWIN_IDs": len(common_ids), # From notebook's mad_col_summary_df
            "Min Columns Used": valid_mask.sum(axis=1).min(), # From notebook
            "Max Columns Used": valid_mask.sum(axis=1).max(), # From notebook
            "Mean Columns Used": round(valid_mask.sum(axis=1).mean(), 2) # From notebook
        })
    mad_col_summary_df = pd.DataFrame(summary_rows)

    # --- Item-level Correlation Summary (New) ---
    corr_col_summary_rows = []
    correlation_labels = ["Correlation W4vW13", "Correlation LLMvW13", "Correlation RANDvW13"]
    display_labels_corr = ["Wave4 vs Wave1_3", "LLM vs Wave1_3", "Random Baseline vs Wave1_3"]

    for label, disp_label in zip(correlation_labels, display_labels_corr):
        r_vals = mad_col_df[label].dropna() # Ensure NaNs are dropped
        if disp_label in ["Random Baseline vs Wave1_3"]: # Handle user request for these to be 0
            mean_r, se_z, ci_lo_r, ci_hi_r = 0.0, 0.0, 0.0, 0.0
        elif not r_vals.empty:
            mean_r, se_z, ci_lo_r, ci_hi_r = summary_fisher_z_transformed_correlations(r_vals)
        else:
            mean_r, se_z, ci_lo_r, ci_hi_r = np.nan, np.nan, np.nan, np.nan
        
        corr_col_summary_rows.append({
            "Comparison": disp_label, "Mean Correlation (Fisher-Z avg)": mean_r, 
            "Fisher Z-scores SEM": se_z,
            "Mean Correlation 95% CI Lower": ci_lo_r, "Mean Correlation 95% CI Higher": ci_hi_r,
            "Number of Items": len(r_vals) if not disp_label in ["Random Baseline vs Wave1_3"] else mad_col_df[label].notna().sum()
        })
    corr_col_summary_df = pd.DataFrame(corr_col_summary_rows)


    # --- Task-level MADs (replicating notebook logic) ---
    qid_to_task_raw = {
        **{f"FALSE CONS. SELF _{i}": "false consensus" for i in range(1, 11)},
        **{f"FALSE CONS. OTHERS _{i}": "false consensus" for i in [1,2,3,4,5,6,7,10,11,12]},
        "Q156_1": "base rate", "FORM A _1": "base rate",
        "Q157": "framing problem", "Q158": "framing problem",
        **{f"Q160_{i}": "conjunction problem (Linda)" for i in [1,2,3]},
        **{f"Q159_{i}": "conjunction problem (Linda)" for i in [1,2,3]},
        "Q161": "outcome bias", "Q162": "outcome bias",
        "Q164": "anchoring and adjustment", "Q166": "anchoring and adjustment",
        "Q168": "anchoring and adjustment", "Q170": "anchoring and adjustment",
        **{f"Q17{i}": "less is more" for i in range(1, 10)}, # Q171..Q179
        "Q181": "sunk cost fallacy", "Q182": "sunk cost fallacy",
        "Q183": "absolute vs. relative savings", "Q184": "absolute vs. relative savings",
        "Q189": "WTA/WTP-Thaler", "Q190": "WTA/WTP-Thaler", "Q191": "WTA/WTP-Thaler",
        "Q192": "Allais", "Q193": "Allais", # Notebook had these reversed with myside, corrected here
        "Q194": "myside", "Q195": "myside",
        **{f"Q198_{i}": "prob matching vs. max" for i in range(1, 11)},
        **{f"Q203_{i}": "prob matching vs. max" for i in range(1, 7)},
        **{f"NONSEPARABILTY BENE _{i}": "non-separability of risks and benefits" for i in range(1, 5)},
        **{f"NONSEPARABILITY RIS _{i}": "non-separability of risks and benefits" for i in range(1, 5)},
        "OMISSION BIAS ": "omission",
        "DENOMINATOR NEGLECT ": "denominator neglect",
        **{f"{i}_Q295": "pricing" for i in range(1, 41)}
    }
    qid_to_task = {k.upper(): v for k, v in qid_to_task_raw.items()}
    all_tasks = sorted(list(set(qid_to_task.values())))

    task_level_mads_rows = []

    # --- Bootstrap setup for task-level average item correlation CIs ---
    N_BOOTSTRAPS = 100  # Reduced from 1000 to 100 for faster computation
    BOOTSTRAP_SEED = 42
    np.random.seed(BOOTSTRAP_SEED)
    original_twin_ids_for_boot = df_wave1_3.index.unique() # df_wave1_3 is already filtered for common_ids and indexed

    boot_task_avg_item_corrs_dist = {
        task: {'W4vW13': [], 'LLMvW13': []} for task in all_tasks
    }

    print(f"Starting bootstrap for task-level correlation CIs ({N_BOOTSTRAPS} iterations)...")
    for i in range(N_BOOTSTRAPS):
        if (i + 1) % 100 == 0:
            print(f"  Bootstrap iteration {i + 1}/{N_BOOTSTRAPS}")
        
        boot_twin_ids_sample = np.random.choice(original_twin_ids_for_boot, size=len(original_twin_ids_for_boot), replace=True)
        
        df_w13_b = df_wave1_3.loc[boot_twin_ids_sample].copy()
        df_w4_b  = df_wave4.loc[boot_twin_ids_sample].copy()
        df_llm_b = df_llm.loc[boot_twin_ids_sample].copy()
        # Note: random_baseline is used for MAD calculation, not directly in item-level correlations that are averaged.
        # If it were, it would need to be df_random_baseline_orig.loc[boot_twin_ids_sample].copy()

        # Create valid_mask_b for this bootstrap sample
        valid_mask_b_dict = {}
        for col_vm_b in cols_all: 
            s1_valid_b = df_w13_b[col_vm_b].notna()
            s4_valid_b = df_w4_b[col_vm_b].notna()
            sl_valid_b = df_llm_b[col_vm_b].notna()
            valid_mask_b_dict[col_vm_b] = s1_valid_b & s4_valid_b & sl_valid_b
        valid_mask_b = pd.DataFrame(valid_mask_b_dict)
        if not valid_mask_b.empty:
            valid_mask_b = valid_mask_b.loc[:, valid_mask_b.any()]
        else: # Handle empty valid_mask_b if all columns are all NaN after bootstrap
            # This case means no valid data for any item in this bootstrap sample, skip correlation part for this sample
            for task_name_b_empty in all_tasks:
                 # Optionally append NaNs or handle as per statistical requirements for empty bootstrap results
                 pass # Or append NaN to distributions if that makes sense for percentile calc later
            continue # Skip to next bootstrap iteration

        # Calculate item-level correlations for this bootstrap sample
        current_boot_item_corrs_w4 = {}
        current_boot_item_corrs_llm = {}

        for col_item_b in cols_all:
            if col_item_b not in valid_mask_b.columns or not valid_mask_b[col_item_b].any():
                current_boot_item_corrs_w4[col_item_b] = np.nan
                current_boot_item_corrs_llm[col_item_b] = np.nan
                continue
            
            mask_col_b_iter = valid_mask_b[col_item_b]
            v1_bc_iter = df_w13_b.loc[mask_col_b_iter, col_item_b]
            v4_bc_iter = df_w4_b.loc[mask_col_b_iter, col_item_b]
            vl_bc_iter = df_llm_b.loc[mask_col_b_iter, col_item_b]
            
            corr_14_b_iter, _ = calculate_correlation(v1_bc_iter, v4_bc_iter)
            corr_1l_b_iter, _ = calculate_correlation(v1_bc_iter, vl_bc_iter)
            current_boot_item_corrs_w4[col_item_b] = corr_14_b_iter
            current_boot_item_corrs_llm[col_item_b] = corr_1l_b_iter

        # Calculate task-level average item correlations for this bootstrap sample
        for task_name_b in all_tasks:
            task_cols_current_b = [c for c, t_name in qid_to_task.items() if t_name == task_name_b and c in current_boot_item_corrs_w4] # Check against available boot corrs
            if not task_cols_current_b: continue

            item_corrs_14_task_b_list = [current_boot_item_corrs_w4.get(tc) for tc in task_cols_current_b]
            item_corrs_1l_task_b_list = [current_boot_item_corrs_llm.get(tc) for tc in task_cols_current_b]
            
            # Filter out NaNs before Fisher-Z, as summary_fisher_z does this internally too but good practice here.
            item_corrs_14_task_b_clean = [r for r in item_corrs_14_task_b_list if pd.notna(r)]
            item_corrs_1l_task_b_clean = [r for r in item_corrs_1l_task_b_list if pd.notna(r)]

            if item_corrs_14_task_b_clean:
                avg_r_14_b_iter, _, _, _ = summary_fisher_z_transformed_correlations(pd.Series(item_corrs_14_task_b_clean))
                if pd.notna(avg_r_14_b_iter): # Only append valid averages
                    boot_task_avg_item_corrs_dist[task_name_b]['W4vW13'].append(avg_r_14_b_iter)
            
            if item_corrs_1l_task_b_clean:
                avg_r_1l_b_iter, _, _, _ = summary_fisher_z_transformed_correlations(pd.Series(item_corrs_1l_task_b_clean))
                if pd.notna(avg_r_1l_b_iter):
                     boot_task_avg_item_corrs_dist[task_name_b]['LLMvW13'].append(avg_r_1l_b_iter)
    print("Bootstrap finished.")

    # --- Populate task_level_mads_rows using full sample for point estimates and bootstrap for CIs ---
    for task in all_tasks:
        task_cols = [col for col, t_name in qid_to_task.items() if t_name == task and col in cols_all]
        if not task_cols: continue

        # --- Accuracy calculations (remain unchanged, based on pooled differences) ---
        mad_14_all_task, mad_1l_all_task, mad_bm_all_task, mad_rand_all_task = [], [], [], []
        respondent_counts_task = set()
        for col in task_cols:
            if col not in valid_mask.columns or not valid_mask[col].any(): continue
            mask = valid_mask[col]
            v1 = df_wave1_3.loc[mask, col]
            v4 = df_wave4.loc[mask, col]
            vl = df_llm.loc[mask, col]
            vbm_mode = vl.mode()
            vbm_val = vbm_mode.iloc[0] if not vbm_mode.empty else np.nan
            vbm = pd.Series(vbm_val, index=v1.index)
            vrand = random_baseline.loc[mask, col]
            r = ranges[col]
            if r == 0: r = 1
            mad_14_all_task.append((v1 - v4).abs() / r)
            mad_1l_all_task.append((v1 - vl).abs() / r)
            mad_bm_all_task.append((v1 - vbm).abs() / r)
            mad_rand_all_task.append((v1 - vrand).abs() / r)
            respondent_counts_task.update(v1.index)
        
        acc_14, se_14, ci_lo_14, ci_hi_14 = np.nan, np.nan, np.nan, np.nan
        acc_1l, se_1l, ci_lo_1l, ci_hi_1l = np.nan, np.nan, np.nan, np.nan
        acc_bm, se_bm, ci_lo_bm, ci_hi_bm = np.nan, np.nan, np.nan, np.nan
        acc_rand, se_rand, ci_lo_rand, ci_hi_rand = np.nan, np.nan, np.nan, np.nan
        div_ratio_1l_14_task, div_ratio_bm_14_task, div_ratio_rand_14_task = np.nan, np.nan, np.nan
        mean_diff_14, mean_diff_1l, mean_diff_bm, mean_diff_rand = np.nan, np.nan, np.nan, np.nan

        if mad_14_all_task:
            task_diffs_14 = pd.concat(mad_14_all_task)
            task_diffs_1l = pd.concat(mad_1l_all_task)
            task_diffs_bm = pd.concat(mad_bm_all_task)
            task_diffs_rand = pd.concat(mad_rand_all_task)
            mean_diff_14 = task_diffs_14.mean()
            mean_diff_1l = task_diffs_1l.mean()
            mean_diff_bm = task_diffs_bm.mean()
            mean_diff_rand = task_diffs_rand.mean()
            acc_14, se_14, ci_lo_14, ci_hi_14 = summary_mad(1 - task_diffs_14)
            acc_1l, se_1l, ci_lo_1l, ci_hi_1l = summary_mad(1 - task_diffs_1l)
            acc_bm, se_bm, ci_lo_bm, ci_hi_bm = summary_mad(1 - task_diffs_bm)
            acc_rand, se_rand, ci_lo_rand, ci_hi_rand = summary_mad(1 - task_diffs_rand)
            div_ratio_1l_14_task = acc_1l / acc_14 if acc_14 != 0 and pd.notna(acc_14) and pd.notna(acc_1l) else np.nan
            div_ratio_bm_14_task = acc_bm / acc_14 if acc_14 != 0 and pd.notna(acc_14) and pd.notna(acc_bm) else np.nan
            div_ratio_rand_14_task = acc_rand / acc_14 if acc_14 != 0 and pd.notna(acc_14) and pd.notna(acc_rand) else np.nan

        # --- Correlation: Point estimates from full sample, CIs from bootstrap ---
        # Point Estimates (from original mad_col_df)
        orig_item_corrs_14_for_task = [mad_col_df.loc[mad_col_df["Column"] == tc, "Correlation W4vW13"].iloc[0] for tc in task_cols if tc in mad_col_df["Column"].values]
        orig_item_corrs_1l_for_task = [mad_col_df.loc[mad_col_df["Column"] == tc, "Correlation LLMvW13"].iloc[0] for tc in task_cols if tc in mad_col_df["Column"].values]
        
        orig_item_corrs_14_clean = [r for r in orig_item_corrs_14_for_task if pd.notna(r)]
        orig_item_corrs_1l_clean = [r for r in orig_item_corrs_1l_for_task if pd.notna(r)]

        point_est_avg_item_r_14, _, _, _ = summary_fisher_z_transformed_correlations(pd.Series(orig_item_corrs_14_clean)) if orig_item_corrs_14_clean else (np.nan, np.nan, np.nan, np.nan)
        point_est_avg_item_r_1l, _, _, _ = summary_fisher_z_transformed_correlations(pd.Series(orig_item_corrs_1l_clean)) if orig_item_corrs_1l_clean else (np.nan, np.nan, np.nan, np.nan)

        # CIs from Bootstrap
        ci_lo_avg_item_r_14, ci_hi_avg_item_r_14 = np.nan, np.nan
        dist_14 = boot_task_avg_item_corrs_dist[task]['W4vW13']
        if dist_14:
            ci_lo_avg_item_r_14 = np.percentile([r for r in dist_14 if pd.notna(r)], 2.5)
            ci_hi_avg_item_r_14 = np.percentile([r for r in dist_14 if pd.notna(r)], 97.5)
        
        ci_lo_avg_item_r_1l, ci_hi_avg_item_r_1l = np.nan, np.nan
        dist_1l = boot_task_avg_item_corrs_dist[task]['LLMvW13']
        if dist_1l:
            ci_lo_avg_item_r_1l = np.percentile([r for r in dist_1l if pd.notna(r)], 2.5)
            ci_hi_avg_item_r_1l = np.percentile([r for r in dist_1l if pd.notna(r)], 97.5)

        task_level_mads_rows.append({
            "Task": task,
            "wave4 vs. wave1_3": mean_diff_14, "wave4 vs. wave1_3 Accuracy": acc_14,
            "wave4 vs. wave1_3 Accuracy 95% CI Lower": ci_lo_14, "wave4 vs. wave1_3 Accuracy 95% CI Higher": ci_hi_14,
            "llm vs. wave1_3": mean_diff_1l, "llm vs. wave1_3 Accuracy": acc_1l,
            "llm vs. wave1_3 Accuracy 95% CI Lower": ci_lo_1l, "llm vs. wave1_3 Accuracy 95% CI Higher": ci_hi_1l,
            "llm accuracy / wave4 accuracy": div_ratio_1l_14_task,
            "random_baseline vs wave1_3": mean_diff_rand, "random_baseline vs wave1_3 Accuracy": acc_rand,
            "random_baseline vs. wave1_3 Accuracy 95% CI Lower": ci_lo_rand, "random_baseline vs. wave1_3 Accuracy 95% CI Higher": ci_hi_rand,
            "random_baseline accuracy / wave4 accuracy": div_ratio_rand_14_task,
            "number of respondents": len(respondent_counts_task), 
            "Correlation W4vW13": point_est_avg_item_r_14, 
            "Correlation W4vW13 CI Lower": ci_lo_avg_item_r_14, "Correlation W4vW13 CI Higher": ci_hi_avg_item_r_14,
            "Correlation LLMvW13": point_est_avg_item_r_1l,
            "Correlation LLMvW13 CI Lower": ci_lo_avg_item_r_1l, "Correlation LLMvW13 CI Higher": ci_hi_avg_item_r_1l,
            "Correlation BMvW13": 0.0, "Correlation BMvW13 CI Lower": np.nan, "Correlation BMvW13 CI Higher": np.nan,
            "Correlation RANDvW13": 0.0, "Correlation RANDvW13 CI Lower": np.nan, "Correlation RANDvW13 CI Higher": np.nan,
        })
    mad_task_level_df = pd.DataFrame(task_level_mads_rows)

    # Task-level summary (Tab: MAD Summary - task level)
    # This part calculates mean accuracy across tasks for each respondent, then averages those respondent-level task-mean-accuracies.
    respondent_task_mads_collector = {label: [] for label in comparison_labels}
    # --- New: Respondent-level Task Correlation Summary ---
    respondent_task_corrs_collector = {"W4vW13": [], "LLMvW13": [], "BMvW13": [], "RANDvW13": []}

    task_usage_counts = []

    for rid in df_wave1_3.index:
        task_to_diffs_14_resp = {}
        task_to_diffs_1l_resp = {}
        task_to_diffs_bm_resp = {}
        task_to_diffs_rand_resp = {}
        
        # For respondent-level task correlations
        task_corrs_14_resp_list = []
        task_corrs_1l_resp_list = []
        
        num_valid_tasks_for_respondent = 0

        for task in all_tasks: # Iterate through tasks first
            task_cols_for_task = [col for col, t_name in qid_to_task.items() if t_name == task and col in cols_all]
            if not task_cols_for_task: continue

            item_diffs_14_list = []
            item_diffs_1l_list = []
            item_diffs_bm_list = []
            item_diffs_rand_list = []

            has_valid_item_for_task = False
            
            # For respondent-task correlation
            v1_items_resp_task, v4_items_resp_task, vl_items_resp_task = [], [], []

            for col in task_cols_for_task:
                if col not in valid_mask.columns or not valid_mask.loc[rid,col]: continue # Skip if this respondent-column is invalid

                v1_val = df_wave1_3.at[rid, col]
                v4_val = df_wave4.at[rid, col]
                vl_val = df_llm.at[rid, col]
                
                vbm_mode_col = df_llm[col].mode() # mode for the column
                vbm_val_col = vbm_mode_col.iloc[0] if not vbm_mode_col.empty else np.nan
                vb = vbm_val_col # Use column mode for benchmark

                vr = random_baseline.at[rid, col]
                r = ranges[col]
                if r == 0: r = 1
                
                item_diffs_14_list.append(abs(v1_val - v4_val) / r)
                item_diffs_1l_list.append(abs(v1_val - vl_val) / r)
                item_diffs_bm_list.append(abs(v1_val - vb) / r) # abs(v1-vb)/r not abs(v1-vbm_val_col)/r
                item_diffs_rand_list.append(abs(v1_val - vr) / r)
                has_valid_item_for_task = True
                
                v1_items_resp_task.append(v1_val)
                v4_items_resp_task.append(v4_val)
                vl_items_resp_task.append(vl_val)


            if has_valid_item_for_task: # if the respondent answered at least one item in this task
                num_valid_tasks_for_respondent +=1
                # Mean difference for this task, for this respondent
                task_to_diffs_14_resp[task] = np.mean(item_diffs_14_list) if item_diffs_14_list else np.nan
                task_to_diffs_1l_resp[task] = np.mean(item_diffs_1l_list) if item_diffs_1l_list else np.nan
                task_to_diffs_bm_resp[task] = np.mean(item_diffs_bm_list) if item_diffs_bm_list else np.nan
                task_to_diffs_rand_resp[task] = np.mean(item_diffs_rand_list) if item_diffs_rand_list else np.nan
                
                # Respondent-task correlation
                r_14_resp_task, _ = calculate_correlation(pd.Series(v1_items_resp_task), pd.Series(v4_items_resp_task))
                if pd.notna(r_14_resp_task): task_corrs_14_resp_list.append(r_14_resp_task)
                
                r_1l_resp_task, _ = calculate_correlation(pd.Series(v1_items_resp_task), pd.Series(vl_items_resp_task))
                if pd.notna(r_1l_resp_task): task_corrs_1l_resp_list.append(r_1l_resp_task)

        
        task_usage_counts.append(num_valid_tasks_for_respondent)

        # Mean of task-mean-differences for this respondent
        resp_mean_task_mad_14 = np.nanmean(list(task_to_diffs_14_resp.values()))
        resp_mean_task_mad_1l = np.nanmean(list(task_to_diffs_1l_resp.values()))
        resp_mean_task_mad_rand = np.nanmean(list(task_to_diffs_rand_resp.values()))

        if pd.notna(resp_mean_task_mad_14): respondent_task_mads_collector["wave4 vs. wave1_3"].append(1-resp_mean_task_mad_14)
        if pd.notna(resp_mean_task_mad_1l): respondent_task_mads_collector["llm vs. wave1_3"].append(1-resp_mean_task_mad_1l)
        if pd.notna(resp_mean_task_mad_rand): respondent_task_mads_collector["random_baseline vs wave1_3"].append(1-resp_mean_task_mad_rand)

        # Summarize respondent's task correlations (Fisher-Z average)
        if task_corrs_14_resp_list:
             # Use summary_fisher_z_transformed_correlations which returns (mean_r, sem_z, ci_low_r, ci_high_r)
             # We only need the mean_r for this collector.
            mean_r_14_resp, _, _, _ = summary_fisher_z_transformed_correlations(pd.Series(task_corrs_14_resp_list))
            if pd.notna(mean_r_14_resp): respondent_task_corrs_collector["W4vW13"].append(mean_r_14_resp)
        
        if task_corrs_1l_resp_list:
            mean_r_1l_resp, _, _, _ = summary_fisher_z_transformed_correlations(pd.Series(task_corrs_1l_resp_list))
            if pd.notna(mean_r_1l_resp): respondent_task_corrs_collector["LLMvW13"].append(mean_r_1l_resp)
        
        # For BM and RAND, correlations are 0 by request
        respondent_task_corrs_collector["BMvW13"].append(0.0) # All respondents get 0 for these
        respondent_task_corrs_collector["RANDvW13"].append(0.0)


    task_summary_rows = []
    for label in comparison_labels:
        # Here, respondent_task_mads_collector[label] contains respondent-level mean task accuracies
        acc_vals_resp_level = np.array(respondent_task_mads_collector[label])
        
        mean_acc, se_acc, ci_lo_acc, ci_hi_acc = summary_mad(acc_vals_resp_level[~np.isnan(acc_vals_resp_level)]) # Drop NaNs before CI
        
        task_summary_rows.append({
            "Comparison": label, "Mean Accuracy": mean_acc, "Accuracy Standard Error": se_acc,
            "Accuracy 95% CI Lower": ci_lo_acc, "Accuracy 95% CI Upper": ci_hi_acc,
            "TWIN_IDs": len(acc_vals_resp_level[~np.isnan(acc_vals_resp_level)]), # Count non-NaN respondents
            "Min Tasks Used": int(np.min(task_usage_counts)) if task_usage_counts else 0,
            "Max Tasks Used": int(np.max(task_usage_counts)) if task_usage_counts else 0,
            "Mean Tasks Used": round(np.mean(task_usage_counts), 2) if task_usage_counts else 0,
        })
    mad_task_summary_df = pd.DataFrame(task_summary_rows)
    
    # --- New: Task-level Correlation Summary (Respondent-wise) ---
    corr_task_summary_rows = []
    corr_summary_display_labels = {
        "W4vW13": "Wave4 vs Wave1_3",
        "LLMvW13": "LLM vs Wave1_3",
        "RANDvW13": "Random Baseline vs Wave1_3"
    }
    for key, disp_label in corr_summary_display_labels.items():
        r_vals_resp_level = pd.Series(respondent_task_corrs_collector[key]).dropna()

        if disp_label in ["Random Baseline vs Wave1_3"]: # Handle user request
            mean_r, se_z, ci_lo_r, ci_hi_r = 0.0, 0.0, 0.0, 0.0
            num_respondents_for_corr = len(df_wave1_3.index) # All respondents assigned 0
        elif not r_vals_resp_level.empty:
            mean_r, se_z, ci_lo_r, ci_hi_r = summary_fisher_z_transformed_correlations(r_vals_resp_level)
            num_respondents_for_corr = len(r_vals_resp_level)
        else:
            mean_r, se_z, ci_lo_r, ci_hi_r = np.nan, np.nan, np.nan, np.nan
            num_respondents_for_corr = 0
            
        corr_task_summary_rows.append({
            "Comparison": disp_label, "Mean Correlation (Fisher-Z avg of respondent means)": mean_r,
            "Fisher Z-scores SEM (of respondent means)": se_z,
            "Mean Correlation 95% CI Lower": ci_lo_r, "Mean Correlation 95% CI Higher": ci_hi_r,
            "TWIN_IDs (with valid task corrs)": num_respondents_for_corr,
            "Min Tasks Used (overall)": int(np.min(task_usage_counts)) if task_usage_counts else 0,
            "Max Tasks Used (overall)": int(np.max(task_usage_counts)) if task_usage_counts else 0,
            "Mean Tasks Used (overall)": round(np.mean(task_usage_counts), 2) if task_usage_counts else 0,
        })
    corr_task_summary_df = pd.DataFrame(corr_task_summary_rows)

    # --- Save outputs ---
    os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        mad_task_summary_df.to_excel(writer, sheet_name="Accuracy Summary - task level", index=False)
        corr_task_summary_df.to_excel(writer, sheet_name="Corr Summary - task level", index=False) # New
        mad_task_level_df.to_excel(writer, sheet_name="Accuracy & Corr - task level", index=False) # Renamed & updated
        mad_col_summary_df.to_excel(writer, sheet_name="Accuracy Summary - col level", index=False)
        corr_col_summary_df.to_excel(writer, sheet_name="Correlation Summary - col level", index=False) # New
        mad_col_df.to_excel(writer, sheet_name="Accuracy & Corr - column level", index=False) # Renamed & updated
    print(f"MAD accuracy and correlation evaluation written to {output_excel_path}")

    # --- Generate and Save Plot (using mad_task_level_df) ---
    df_plot = mad_task_level_df.sort_values("llm accuracy / wave4 accuracy", ascending=False).copy()
    tasks_plot = df_plot["Task"]
    x_plot = np.arange(len(tasks_plot))

    width_plot = 0.8
    bar_h_plot = width_plot / 3  # Changed from 4 to 3 bars
    offsets_plot = np.array([-1.0, 0.0, +1.0]) * bar_h_plot  # Adjusted offsets for 3 bars
    
    #Convert accuracy to percentage for plotting
    means_plot = [
        df_plot["wave4 vs. wave1_3 Accuracy"] * 100,
        df_plot["llm vs. wave1_3 Accuracy"] * 100,
        df_plot["random_baseline vs wave1_3 Accuracy"] * 100
    ]
    
    errs_plot = [
        np.vstack((
            means_plot[0] - df_plot["wave4 vs. wave1_3 Accuracy 95% CI Lower"] * 100,
            df_plot["wave4 vs. wave1_3 Accuracy 95% CI Higher"] * 100 - means_plot[0]
        )),
        np.vstack((
            means_plot[1] - df_plot["llm vs. wave1_3 Accuracy 95% CI Lower"] * 100,
            df_plot["llm vs. wave1_3 Accuracy 95% CI Higher"] * 100 - means_plot[1]
        )),
        np.vstack((
            means_plot[2] - df_plot["random_baseline vs. wave1_3 Accuracy 95% CI Lower"] * 100,
            df_plot["random_baseline vs. wave1_3 Accuracy 95% CI Higher"] * 100 - means_plot[2]
        ))
    ]
    # Replace NaNs in error arrays with 0 to avoid plotting issues, or handle appropriately
    for i in range(len(errs_plot)):
        errs_plot[i][np.isnan(errs_plot[i])] = 0

    labels_plot = [
        "Wave 4 vs. Wave 1_3", "LLM vs. Wave 1_3",
        "Random baseline vs. Wave 1_3"
    ]
    colors_plot = ["tab:grey", "tab:blue", "tab:green"]  # Removed orange color for homogeneous

    avg_plot = [m.replace([np.inf, -np.inf], np.nan).dropna().mean() for m in means_plot]

    fig, ax = plt.subplots(figsize=(12, 8))
    for off, mean_vals, err_vals, lbl, col in zip(offsets_plot, means_plot, errs_plot, labels_plot, colors_plot):
        # Ensure mean_vals and err_vals are numpy arrays for consistent handling with NaNs for plotting
        mean_vals_np = np.array(mean_vals)
        # For barh, xerr needs to be paired for each bar. If a mean is NaN, its error should also effectively be NaN or handled.
        # We'll plot bars where mean is not NaN. Errors for NaN means will be skipped by matplotlib if xerr is structured correctly.
        
        # Create a mask for valid (non-NaN) means
        valid_means_mask = ~np.isnan(mean_vals_np)
        
        # Filter data for plotting
        x_filtered = x_plot[valid_means_mask]
        off_filtered = off # offset is scalar
        mean_filtered = mean_vals_np[valid_means_mask]
        
        # Adjust error array: needs to be 2xN. For NaN means, error can be 0 or NaN.
        # Let's ensure error array matches dimensions of filtered means.
        # If err_vals is 2xM, and mean_filtered is N, we need 2xN.
        err_filtered_lower = np.array(err_vals[0])[valid_means_mask]
        err_filtered_higher = np.array(err_vals[1])[valid_means_mask]
        err_to_plot = np.vstack((err_filtered_lower, err_filtered_higher))
        
        # Handle cases where all means might be NaN for a category
        if len(mean_filtered) > 0:
            ax.barh(
                x_filtered + off_filtered, mean_filtered,
                height=bar_h_plot, color=col, xerr=err_to_plot,
                capsize=5, label=lbl
            )

    ax.set_yticks(x_plot)
    ax.set_yticklabels(tasks_plot)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(fig_title) # Use the passed figure title
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f%%'))

    for val_avg, lbl, col in zip(avg_plot, labels_plot, colors_plot):
        if pd.notna(val_avg): # Only plot average line if it's a valid number
            ax.axvline(val_avg, color=col, ls="--", lw=1.5, label=f"{lbl} average = {val_avg:.2f}%")

    ax.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    print(f"Task-level accuracy distribution plot saved to {output_fig_path}")

    # --- Generate and Save Correlation Plot (Fisher-Z Average) ---
    # df_corr_plot is mad_task_level_df which now contains the new average item-level correlations per task
    # with bootstrapped CIs
    df_corr_plot = mad_task_level_df.sort_values("Correlation LLMvW13", ascending=False).copy() 
    tasks_corr_plot = df_corr_plot["Task"]
    x_corr_plot = np.arange(len(tasks_corr_plot))

    # Bar properties (can reuse some from accuracy plot or define new)
    width_corr_plot = 0.6 # Adjust as needed
    bar_h_corr_plot = width_corr_plot / 2 # For two main bars: W4, LLM
    offsets_corr_plot = np.array([-0.5, +0.5]) * bar_h_corr_plot
    
    corr_means_plot = [
        df_corr_plot["Correlation W4vW13"],
        df_corr_plot["Correlation LLMvW13"]
    ]
    
    corr_errs_plot = [
        np.vstack(( # Error relative to the mean
            corr_means_plot[0] - df_corr_plot["Correlation W4vW13 CI Lower"],
            df_corr_plot["Correlation W4vW13 CI Higher"] - corr_means_plot[0]
        )),
        np.vstack((
            corr_means_plot[1] - df_corr_plot["Correlation LLMvW13 CI Lower"],
            df_corr_plot["Correlation LLMvW13 CI Higher"] - corr_means_plot[1]
        ))
    ]
    # Replace NaNs in error arrays with 0 to avoid plotting issues
    for i in range(len(corr_errs_plot)):
        corr_errs_plot[i][np.isnan(corr_errs_plot[i])] = 0

    corr_plot_labels = ["Wave 4 vs. Wave 1_3", "LLM vs. Wave 1_3"]
    corr_plot_colors = ["tab:grey", "tab:blue"] # Re-using some colors

    # Calculate averages for lines using Fisher-Z transformation
    avg_corr_plot_vals = []
    for r_series in corr_means_plot:
        # r_series here is a pandas Series of (task-level average item correlations)
        # To get the grand average for the line on the plot, we Fisher-Z average these task-level averages.
        avg_r, _, _, _ = summary_fisher_z_transformed_correlations(r_series.dropna())
        avg_corr_plot_vals.append(avg_r)
    
    # BM and RAND correlations are 0 --- REMOVING THESE FROM PLOT
    # avg_corr_plot_vals.extend([0.0, 0.0]) 
    # corr_plot_labels_with_baselines = corr_plot_labels + ["LLM_homogeneous vs. Wave 1_3", "Random baseline vs. Wave 1_3"]
    # corr_plot_colors_with_baselines = corr_plot_colors + ["tab:orange", "tab:green"]


    fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
    for off, mean_vals, err_vals, lbl, col in zip(offsets_corr_plot, corr_means_plot, corr_errs_plot, corr_plot_labels, corr_plot_colors):
        mean_vals_np = np.array(mean_vals)
        valid_means_mask = ~np.isnan(mean_vals_np)
        
        x_filtered = x_corr_plot[valid_means_mask]
        mean_filtered = mean_vals_np[valid_means_mask]
        
        err_filtered_lower = np.array(err_vals[0])[valid_means_mask]
        err_filtered_higher = np.array(err_vals[1])[valid_means_mask]
        err_to_plot = np.vstack((err_filtered_lower, err_filtered_higher))
        
        if len(mean_filtered) > 0:
            ax_corr.barh(
                x_filtered + off, mean_filtered,
                height=bar_h_corr_plot, color=col, xerr=err_to_plot,
                capsize=3, label=lbl
            )

    ax_corr.set_yticks(x_corr_plot)
    ax_corr.set_yticklabels(tasks_corr_plot)
    ax_corr.invert_yaxis()
    ax_corr.set_xlabel("Pearson Correlation Coefficient (r)")
    ax_corr.set_title(corr_fig_title) # Use the passed correlation figure title
    ax_corr.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_corr.set_xlim([-1.1, 1.1]) # Correlations are between -1 and 1

    # Plot average lines (including 0 for BM and RAND)
    for val_avg, lbl, col in zip(avg_corr_plot_vals, corr_plot_labels, corr_plot_colors):
        if pd.notna(val_avg):
            ax_corr.axvline(val_avg, color=col, ls="--", lw=1.5, label=f"{lbl} average r = {val_avg:.2f}")

    ax_corr.legend(loc='lower right')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_corr_fig_path), exist_ok=True)
    plt.savefig(output_corr_fig_path, dpi=300, bbox_inches="tight")
    print(f"Task-level correlation distribution plot saved to {output_corr_fig_path}")

    # --- Generate and Save Correlation Plot (Simple Average) (New) ---
    fig_corr_simple, ax_corr_simple = plt.subplots(figsize=(12, 8))
    
    # Re-use plotting setup from the Fisher-Z averaged correlation plot where appropriate
    # df_corr_plot, tasks_corr_plot, x_corr_plot are the same
    # corr_means_plot (task-level average item-level r values - full sample point estimate),
    # corr_errs_plot (CIs for each task's average item-level r - from bootstrap) are the same.
    # corr_plot_labels, corr_plot_colors are the same for the bars

    for off, mean_vals, err_vals, lbl, col in zip(offsets_corr_plot, corr_means_plot, corr_errs_plot, corr_plot_labels, corr_plot_colors):
        mean_vals_np = np.array(mean_vals)
        valid_means_mask = ~np.isnan(mean_vals_np)
        
        x_filtered = x_corr_plot[valid_means_mask]
        mean_filtered = mean_vals_np[valid_means_mask]
        
        err_filtered_lower = np.array(err_vals[0])[valid_means_mask]
        err_filtered_higher = np.array(err_vals[1])[valid_means_mask]
        err_to_plot = np.vstack((err_filtered_lower, err_filtered_higher))
        
        if len(mean_filtered) > 0:
            ax_corr_simple.barh(
                x_filtered + off, mean_filtered,
                height=bar_h_corr_plot, color=col, xerr=err_to_plot,
                capsize=3, label=lbl
            )

    ax_corr_simple.set_yticks(x_corr_plot)
    ax_corr_simple.set_yticklabels(tasks_corr_plot)
    ax_corr_simple.invert_yaxis()
    ax_corr_simple.set_xlabel("Pearson Correlation Coefficient (r)")
    ax_corr_simple.set_title(corr_simple_avg_fig_title) # Use the new title
    ax_corr_simple.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_corr_simple.set_xlim([-1.1, 1.1])

    # Calculate simple averages for lines
    avg_corr_simple_plot_vals = []
    for r_series_for_avg in corr_means_plot: # corr_means_plot contains the task-level average item-level r's
        # To get the grand average line, we take a simple mean of these task-level average item-level r's
        simple_avg_r = r_series_for_avg.dropna().mean()
        avg_corr_simple_plot_vals.append(simple_avg_r)

    # Plot simple average lines
    for val_avg, lbl, col in zip(avg_corr_simple_plot_vals, corr_plot_labels, corr_plot_colors):
        if pd.notna(val_avg):
            ax_corr_simple.axvline(val_avg, color=col, ls=":", lw=1.5, label=f"{lbl} simple avg r = {val_avg:.2f}") # Differentiate line style or label

    ax_corr_simple.legend(loc='lower right')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_corr_simple_avg_fig_path), exist_ok=True)
    plt.savefig(output_corr_simple_avg_fig_path, dpi=300, bbox_inches="tight")
    print(f"Task-level correlation distribution plot (simple average) saved to {output_corr_simple_avg_fig_path}")


# -------------------------------------------------
# Entry-point
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute MAD accuracy and Correlation metrics from formatted CSVs.")
    parser.add_argument("--csv-dir", required=True, help="Directory containing formatted response CSVs (csv_formatted folder).")
    parser.add_argument("--output-dir", required=True, help="Directory to write the Excel summary and plot into (accuracy_evaluation folder).")
    parser.add_argument(
        "--output-excel-filename",
        default="mad_accuracy_summary.xlsx",
        help="Filename for the Excel workbook (within output-dir).",
    )
    parser.add_argument(
        "--output-plot-filename",
        default="accuracy_dist.png",
        help="Filename for the accuracy distribution plot (within output-dir)."
    )
    parser.add_argument( # Argument for the plot title
        "--plot-title",
        default="Task-Level Accuracy with 95% CI", # Default title
        help="Title for the accuracy distribution plot."
    )
    parser.add_argument(
        "--output-corr-plot-filename",
        default="correlation_dist.png",
        help="Filename for the correlation distribution plot (within output-dir)."
    )
    parser.add_argument(
        "--corr-plot-title",
        default="Task-Level Correlation with 95% CI (Pearson r)",
        help="Title for the correlation distribution plot (Fisher-Z avg)."
    )
    parser.add_argument(
        "--output-corr-simple-avg-plot-filename",
        default="correlation_dist_simple_avg.png",
        help="Filename for the simple average correlation distribution plot (within output-dir)."
    )
    parser.add_argument(
        "--corr-simple-avg-plot-title",
        default="Task-Level Correlation (Simple Avg r) with 95% CI",
        help="Title for the simple average correlation distribution plot."
    )
    args = parser.parse_args()

    csv_dir = os.path.abspath(args.csv_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    output_excel_path = os.path.join(output_dir, args.output_excel_filename)
    output_fig_path = os.path.join(output_dir, args.output_plot_filename)
    output_corr_fig_path = os.path.join(output_dir, args.output_corr_plot_filename)
    output_corr_simple_avg_fig_path = os.path.join(output_dir, args.output_corr_simple_avg_plot_filename) # New

    compute_mad_summary(csv_dir, output_excel_path, output_fig_path, output_corr_fig_path, output_corr_simple_avg_fig_path, args.plot_title, args.corr_plot_title, args.corr_simple_avg_plot_title)


if __name__ == "__main__":
    main() 