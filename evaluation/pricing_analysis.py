#!/usr/bin/env python3
"""
New pricing_analysis.py script.

This script generates demand curves based on the outputs of convert_csv_format.py:
- A randdollar_breakdown.csv file containing TWIN_ID, Product_ID, Price, 
  mapping column names, and the actual choice values from Wave 1-3, Wave 4, and LLM data.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_label_formatted_csv(file_path):
    """Loads a _label_formatted.csv file, skipping the description row."""
    if not os.path.exists(file_path):
        # File not found - returning empty DataFrame
        return pd.DataFrame()
    try:
        # First row is header, second is description, data starts from third.
        # header=0 uses the first row as column names.
        # skiprows=[1] skips the second row (the description row).
        df = pd.read_csv(file_path, header=0, skiprows=[1], low_memory=False, dtype=str, keep_default_na=False, na_filter=False)
        df.columns = [col.strip() for col in df.columns]
        if "TWIN_ID" in df.columns:
            df["TWIN_ID"] = df["TWIN_ID"].astype(str).str.strip()
            df.set_index("TWIN_ID", inplace=True)
        else:
            # TWIN_ID column not found - data linkage might fail
            pass
        return df
    except Exception as e:
        # Error loading file - return empty DataFrame
        pass
        return pd.DataFrame()

def load_randdollar_breakdown(file_path):
    """Loads the randdollar_breakdown.csv file."""
    if not os.path.exists(file_path):
        # Error: randdollar_breakdown.csv not found - cannot proceed
        return None
    try:
        df = pd.read_csv(file_path, low_memory=False, dtype=str, keep_default_na=False, na_filter=False)
        df["TWIN_ID"] = df["TWIN_ID"].astype(str).str.strip()
        df["product_ID"] = pd.to_numeric(df["product_ID"], errors='coerce')
        df["price_numeric"] = pd.to_numeric(df["price_numeric"], errors='coerce')
        # Ensure new value columns are present, even if some are empty, to avoid KeyErrors later
        for col_val_wave in ['response_wave3', 'response_wave4', 'response_llm']:
            if col_val_wave not in df.columns:
                df[col_val_wave] = np.nan # Add as nan if missing
        df.dropna(subset=["product_ID", "price_numeric"], inplace=True)
        return df
    except Exception as e:
        # Error loading file - return empty DataFrame
        pass
        return None

def prepare_purchase_data(df_randdollar_breakdown, wave_name):
    """
    Prepares a long-form DataFrame of purchase observations for a given wave
    using the pre-loaded choice values in df_randdollar_breakdown.
    """
    if df_randdollar_breakdown is None or df_randdollar_breakdown.empty:
        # Skipping wave due to empty input randdollar_breakdown data
        return pd.DataFrame()

    all_observations = []
    
    value_column_map = {
        "Wave1-3": "response_wave3",
        "Wave4": "response_wave4",
        "LLM_Imputed": "response_llm"
    }
    
    value_col_for_wave = value_column_map.get(wave_name)
    if not value_col_for_wave:
        # Error: Invalid wave_name provided to prepare_purchase_data
        return pd.DataFrame()

    if value_col_for_wave not in df_randdollar_breakdown.columns:
        # Warning: Value column not found in randdollar_breakdown
        pass
        return pd.DataFrame()

    # Check for essential columns from the breakdown itself, plus the specific wave value column
    required_cols_breakdown = ["TWIN_ID", "product_ID", "price_numeric", "wave4_column_name", "input_column_name", "position"] + [value_col_for_wave]
    if not all(col in df_randdollar_breakdown.columns for col in required_cols_breakdown):
        missing_cols = [col for col in required_cols_breakdown if col not in df_randdollar_breakdown.columns]
        # Error: df_randdollar_breakdown is missing required columns
        return pd.DataFrame()

    for _, row_breakdown in df_randdollar_breakdown.iterrows():
        twin_id = row_breakdown["TWIN_ID"]
        product_id_val = row_breakdown["product_ID"]
        price_val = row_breakdown["price_numeric"]
        wave4_col_name = row_breakdown["wave4_column_name"] 
        original_input_col = row_breakdown["input_column_name"] 
        order_displayed_val = row_breakdown["position"]
        
        choice_text = row_breakdown.get(value_col_for_wave) # Get the pre-loaded value
        
        purchase = np.nan
        if isinstance(choice_text, str):
            choice_text_lower = choice_text.lower().strip()
            if "yes, i would purchase the product" in choice_text_lower:
                purchase = 1
            elif "no, i would not purchase the product" in choice_text_lower:
                purchase = 0
        
        if not pd.isna(purchase):
            all_observations.append({
                "TWIN_ID": twin_id,
                "product_ID": product_id_val,
                "price_numeric": price_val,
                "Purchase": purchase,
                "Wave": wave_name,
                "Original_Input_Col": original_input_col,
                "Position": order_displayed_val,
                "Wave4_Col_Name": wave4_col_name
            })

    return pd.DataFrame(all_observations)

def calculate_relative_prices(df_all_purchases):
    """Calculates relative price rank for each product observation."""
    if df_all_purchases.empty:
        return df_all_purchases, 0

    # Get unique sorted prices for each product
    product_price_distributions = df_all_purchases.groupby("product_ID")["price_numeric"].apply(
        lambda x: sorted(pd.Series(x).unique())
    ).reset_index(name="UniqueSortedPrices")

    df_all_purchases = pd.merge(df_all_purchases, product_price_distributions, on="product_ID", how="left")

    def get_rank(row):
        try:
            return row["UniqueSortedPrices"].index(row["price_numeric"]) + 1
        except (ValueError, TypeError):
            return np.nan

    df_all_purchases["Relative_Price_Rank"] = df_all_purchases.apply(get_rank, axis=1)
    df_all_purchases.dropna(subset=["Relative_Price_Rank"], inplace=True)
    df_all_purchases["Relative_Price_Rank"] = df_all_purchases["Relative_Price_Rank"].astype(int)

    nprices = 0
    if "Relative_Price_Rank" in df_all_purchases and not df_all_purchases["Relative_Price_Rank"].empty:
        nprices = int(df_all_purchases["Relative_Price_Rank"].max())
    
    return df_all_purchases, nprices

def main():
    parser = argparse.ArgumentParser(description="Generate demand curves from formatted CSV files.")
    
    # Support both config-based and individual argument approaches
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument("--wave13-label-csv", help="Path to Wave 1-3 label formatted CSV.")
    parser.add_argument("--wave4-label-csv", help="Path to Wave 4 label formatted CSV.")
    parser.add_argument("--llm-label-csv", help="Path to LLM imputed label formatted CSV.")
    parser.add_argument("--randdollar-breakdown-csv", help="Path to randdollar_breakdown.csv.")
    parser.add_argument("--output-plot", default="average_demand_curve_new.png", help="Path to save the output plot.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Handle config-based execution
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract paths from config
        trial_dir = config.get('trial_dir', '')
        if not trial_dir:
            parser.error("trial_dir must be specified in config file")
        
        # Build paths based on config
        csv_comp_dir = os.path.join(trial_dir, "csv_comparison")
        wave13_label_csv = os.path.join(csv_comp_dir, "csv_formatted_label", "responses_wave1_3_label_formatted.csv")
        wave4_label_csv = os.path.join(csv_comp_dir, "csv_formatted_label", "responses_wave4_label_formatted.csv")
        llm_label_csv = os.path.join(csv_comp_dir, "csv_formatted_label", "responses_llm_imputed_label_formatted.csv")
        randdollar_output_template = config.get('randdollar_output', f"{trial_dir}/csv_comparison/randdollar_breakdown.csv")
        randdollar_breakdown_csv = randdollar_output_template.replace('${trial_dir}', trial_dir) if trial_dir else os.path.join(csv_comp_dir, "randdollar_breakdown.csv")
        
        # Set output plot path
        pricing_results_dir = os.path.join(trial_dir, "pricing_analysis_results")
        os.makedirs(pricing_results_dir, exist_ok=True)
        output_plot = os.path.join(pricing_results_dir, "average_demand_curve.png")
        
        # Override with command line args if provided
        if args.wave13_label_csv:
            wave13_label_csv = args.wave13_label_csv
        if args.wave4_label_csv:
            wave4_label_csv = args.wave4_label_csv
        if args.llm_label_csv:
            llm_label_csv = args.llm_label_csv
        if args.randdollar_breakdown_csv:
            randdollar_breakdown_csv = args.randdollar_breakdown_csv
        if args.output_plot != "average_demand_curve_new.png":
            output_plot = args.output_plot
        
        # Set args for the rest of the function
        args.wave13_label_csv = wave13_label_csv
        args.wave4_label_csv = wave4_label_csv
        args.llm_label_csv = llm_label_csv
        args.randdollar_breakdown_csv = randdollar_breakdown_csv
        args.output_plot = output_plot
    
    # Handle CLI-based execution - require all arguments
    elif not all([args.wave13_label_csv, args.wave4_label_csv, args.llm_label_csv, args.randdollar_breakdown_csv]):
        parser.error("Either --config must be provided, or all CSV file arguments (--wave13-label-csv, --wave4-label-csv, --llm-label-csv, --randdollar-breakdown-csv) must be provided")

    # Load data
    # Loading data
    df_randdollar_breakdown = load_randdollar_breakdown(args.randdollar_breakdown_csv)
    if df_randdollar_breakdown is None or df_randdollar_breakdown.empty:
        # Could not load or empty randdollar_breakdown.csv - exiting
        return

    # df_wave3_label = load_label_formatted_csv(args.wave13_label_csv) # No longer needed
    # df_wave4_label = load_label_formatted_csv(args.wave4_label_csv) # No longer needed
    # df_llm_label = load_label_formatted_csv(args.llm_label_csv) # No longer needed

    # Prepare purchase data for each wave using only df_randdollar_breakdown
    # Preparing purchase data
    data_wave3 = prepare_purchase_data(df_randdollar_breakdown, "Wave1-3")
    data_wave4 = prepare_purchase_data(df_randdollar_breakdown, "Wave4")
    data_llm = prepare_purchase_data(df_randdollar_breakdown, "LLM_Imputed")

    all_purchase_data = pd.concat([data_wave3, data_wave4, data_llm], ignore_index=True)

    if all_purchase_data.empty:
        # No purchase data could be processed - exiting
        return

    # Calculate relative prices
    # Calculating relative prices
    all_purchase_data, nprices = calculate_relative_prices(all_purchase_data)

    if nprices == 0:
        # Could not determine price ranks or no price variation - cannot generate demand curves
        return
    
    # Number of relative price points and total valid observations determined
    if not all_purchase_data.empty:
        # Data validation - sample and purchase counts available
        pass

    # Compute demand curves
    # Computing demand curves
    demand_curves = {}
    for wave_name in ["Wave1-3", "Wave4", "LLM_Imputed"]:
        current_wave_data = all_purchase_data[all_purchase_data["Wave"] == wave_name]
        if not current_wave_data.empty:
            curve = current_wave_data.groupby("Relative_Price_Rank")["Purchase"].mean()
            # Reindex to ensure all price ranks from 1 to nprices are present
            demand_curves[wave_name] = curve.reindex(range(1, nprices + 1), fill_value=np.nan)
        else:
            demand_curves[wave_name] = pd.Series([np.nan] * nprices, index=range(1, nprices + 1))

    # Plotting
    # Plotting demand curves
    plt.figure(figsize=(10, 6))
    x_axis = np.arange(1, nprices + 1)

    if not demand_curves["Wave1-3"].isna().all():
        plt.plot(x_axis, demand_curves["Wave1-3"], linestyle='-', marker='o', label='Wave 1-3') # Matched old style
    if not demand_curves["Wave4"].isna().all():
        plt.plot(x_axis, demand_curves["Wave4"], linestyle=':', marker='s', label='Wave 4') # Matched old style, changed marker for distinction
    if not demand_curves["LLM_Imputed"].isna().all():
        plt.plot(x_axis, demand_curves["LLM_Imputed"], linestyle='-.', marker='^', label='LLM Imputed (Twins)') # Matched old style

    plt.ylim(0, 1)
    plt.xticks(x_axis, fontsize=12) # Ensure all price ranks are shown if not too many
    plt.yticks(fontsize=12)
    plt.xlabel('Relative Price Rank', fontsize=16)
    plt.ylabel('Purchase Probability', fontsize=16)
    plt.title('Average Demand Curve by Relative Price', fontsize=20)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
    # Demand curve plot saved
    # plt.show() # Uncomment to display plot if running interactively

if __name__ == "__main__":
    main() 