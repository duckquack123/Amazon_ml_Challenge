import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List

def compare_csv_files(file1_path, file2_path, output_file=None):
    """
    Compare two CSV files with respect to sample_id
    
    Args:
        file1_path (str): Path to first CSV file (submission.csv)
        file2_path (str): Path to second CSV file (v6.csv)
        output_file (str): Optional path to save comparison results
    """
    
    # Read both CSV files
    print("Reading CSV files...")
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    print(f"File 1 ({file1_path}) shape: {df1.shape}")
    print(f"File 2 ({file2_path}) shape: {df2.shape}")
    
    # Check if both files have sample_id column
    if 'sample_id' not in df1.columns or 'sample_id' not in df2.columns:
        print("Error: Both files must have 'sample_id' column")
        return
    
    # Basic statistics
    print("\n=== BASIC COMPARISON ===")
    print(f"File 1 unique sample_ids: {df1['sample_id'].nunique()}")
    print(f"File 2 unique sample_ids: {df2['sample_id'].nunique()}")
    
    # Check for common sample_ids
    common_ids = set(df1['sample_id']).intersection(set(df2['sample_id']))
    only_in_file1 = set(df1['sample_id']) - set(df2['sample_id'])
    only_in_file2 = set(df2['sample_id']) - set(df1['sample_id'])
    
    print(f"Common sample_ids: {len(common_ids)}")
    print(f"Only in file 1: {len(only_in_file1)}")
    print(f"Only in file 2: {len(only_in_file2)}")
    
    # If there are price columns, compare them
    if 'price' in df1.columns and 'price' in df2.columns:
        print("\n=== PRICE COMPARISON ===")
        
        # Merge on sample_id for comparison
        merged = pd.merge(df1, df2, on='sample_id', suffixes=('_file1', '_file2'), how='inner')
        
        if len(merged) > 0:
            # Calculate differences
            merged['price_diff'] = merged['price_file1'] - merged['price_file2']
            merged['price_diff_abs'] = np.abs(merged['price_diff'])
            merged['price_diff_pct'] = (merged['price_diff'] / merged['price_file2']) * 100
            # NEW ADDITION: Calculate Mean Absolute Percentage Error (MAPE)
            merged['price_diff_pct_abs'] = np.abs(merged['price_diff'] / merged['price_file2']) * 100
            
            print(f"Price comparison for {len(merged)} common sample_ids:")
            print(f"Mean price difference: {merged['price_diff'].mean():.4f}")
            print(f"Mean absolute price difference: {merged['price_diff_abs'].mean():.4f}")
            print(f"Mean percentage difference: {merged['price_diff_pct'].mean():.4f}%")
            # NEW ADDITION: Print MAPE
            print(f"Mean Absolute Percentage Error (MAPE): {merged['price_diff_pct_abs'].mean():.4f}%")
            print(f"Max absolute difference: {merged['price_diff_abs'].max():.4f}")
            print(f"Min absolute difference: {merged['price_diff_abs'].min():.4f}")
            
            # Statistics for each file
            print(f"\nFile 1 price stats:")
            print(f"  Mean: {df1['price'].mean():.4f}")
            print(f"  Std: {df1['price'].std():.4f}")
            print(f"  Min: {df1['price'].min():.4f}")
            print(f"  Max: {df1['price'].max():.4f}")
            
            print(f"\nFile 2 price stats:")
            print(f"  Mean: {df2['price'].mean():.4f}")
            print(f"  Std: {df2['price'].std():.4f}")
            print(f"  Min: {df2['price'].min():.4f}")
            print(f"  Max: {df2['price'].max():.4f}")
            
            # Show top differences
            print("\n=== TOP 10 LARGEST DIFFERENCES ===")
            top_diff = merged.nlargest(10, 'price_diff_abs')[['sample_id', 'price_file1', 'price_file2', 'price_diff', 'price_diff_abs']]
            print(top_diff.to_string(index=False))
            
            # Create visualizations
            create_comparison_plots(merged, file1_path, file2_path)
            
            # Save detailed comparison if output file is specified
            if output_file:
                merged.to_csv(output_file, index=False)
                print(f"\nDetailed comparison saved to: {output_file}")
        
        else:
            print("No common sample_ids found for price comparison")
    
    # Show sample_ids that are different
    if only_in_file1:
        print(f"\n=== SAMPLE_IDS ONLY IN FILE 1 (first 10) ===")
        print(list(only_in_file1)[:10])
    
    if only_in_file2:
        print(f"\n=== SAMPLE_IDS ONLY IN FILE 2 (first 10) ===")
        print(list(only_in_file2)[:10])

def create_comparison_plots(merged_df, file1_name, file2_name):
    """Create visualization plots for the comparison"""
    
    plt.figure(figsize=(18, 10)) # Increased figure size for better layout
    
    # Plot 1: Scatter plot of prices
    plt.subplot(2, 3, 1)
    sns.scatterplot(x='price_file1', y='price_file2', data=merged_df, alpha=0.6)
    min_val = min(merged_df['price_file1'].min(), merged_df['price_file2'].min())
    max_val = max(merged_df['price_file1'].max(), merged_df['price_file2'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel(f'Price ({os.path.basename(file1_name)})')
    plt.ylabel(f'Price ({os.path.basename(file2_name)})')
    plt.title('Price Scatter Plot')
    
    # Plot 2: Distribution of price differences
    plt.subplot(2, 3, 2)
    sns.histplot(merged_df['price_diff'], bins=50, kde=True)
    plt.xlabel('Price Difference (File1 - File2)')
    plt.title('Distribution of Price Differences')
    
    # Plot 3: Distribution of absolute differences
    plt.subplot(2, 3, 3)
    sns.histplot(merged_df['price_diff_abs'], bins=50, kde=True)
    plt.xlabel('Absolute Price Difference')
    plt.title('Distribution of Absolute Differences')
    
    # Plot 4: Price comparison by sample (first 100 samples)
    plt.subplot(2, 3, 4)
    sample_data = merged_df.head(100)
    plt.plot(sample_data.index, sample_data['price_file1'], 'o-', label='File 1', alpha=0.7, markersize=4)
    plt.plot(sample_data.index, sample_data['price_file2'], 'o-', label='File 2', alpha=0.7, markersize=4)
    plt.xlabel('Sample Index (First 100)')
    plt.ylabel('Price')
    plt.title('Price Comparison (First 100 Samples)')
    plt.legend()
    
    # NEW ADDITION: Create a summary statistics plot
    plt.subplot(2, 3, 5)
    correlation = merged_df['price_file1'].corr(merged_df['price_file2'])
    mad = merged_df['price_diff_abs'].mean()
    mape = merged_df['price_diff_pct_abs'].mean()
    
    summary_text = (
        f"Key Metrics:\n\n"
        f"Correlation: {correlation:.4f}\n\n"
        f"Mean Absolute\nDifference (MAD): {mad:.4f}\n\n"
        f"Mean Absolute\nPercentage Error (MAPE): {mape:.2f}%"
    )
    
    plt.text(0.5, 0.5, summary_text, 
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=14,
             bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9))
    plt.title('Comparison Summary')
    plt.axis('off')

    # Plot 6: Distribution of absolute percentage differences
    plt.subplot(2, 3, 6)
    sns.histplot(merged_df['price_diff_pct_abs'], bins=50, kde=True)
    plt.xlabel('Absolute Percentage Difference (%)')
    plt.title('Distribution of MAPE')
    
    plt.tight_layout()
    plt.savefig('price_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization plots created and saved as 'price_comparison_plots.png'")

if __name__ == "__main__":
    # --- Build Robust File Paths ---
    # Get the directory where this script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    print(f"Script is running from: {script_directory}")

    # Helper to resolve expected CSVs from common locations
    def resolve_file(filename: str, extra_candidates: List[str] = None) -> str:
        candidates = [
            os.path.join(script_directory, filename),
            os.path.join(os.path.dirname(script_directory), 'dataset', filename),
        ]
        if extra_candidates:
            candidates.extend(extra_candidates)
        for p in candidates:
            p_norm = os.path.normpath(p)
            if os.path.exists(p_norm):
                return p_norm
        # If not found, raise with a helpful message
        tried_str = "\n  - " + "\n  - ".join(os.path.normpath(c) for c in candidates)
        raise FileNotFoundError(
            f"Could not find '{filename}'. Tried locations:{tried_str}\n"
            f"Tip: Ensure the file exists in the dataset folder at the project root."
        )

    # Resolve paths for required files
    file1_path = resolve_file('submission.csv')
    file2_path = resolve_file('v6_2.csv')

    # Choose an output location in the dataset folder next to inputs if available
    dataset_dir = os.path.join(os.path.dirname(script_directory), 'dataset')
    output_dir = dataset_dir if os.path.isdir(dataset_dir) else script_directory
    output_file = os.path.join(output_dir, "comparison_results.csv")

    print(f"Resolved submission.csv -> {file1_path}")
    print(f"Resolved v6.csv         -> {file2_path}")
    print(f"Output will be saved to -> {output_file}")

    # Run comparison
    print("\nStarting file comparison...")
    compare_csv_files(file1_path, file2_path, output_file)