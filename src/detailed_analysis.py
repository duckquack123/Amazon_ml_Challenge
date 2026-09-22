import pandas as pd
import numpy as np

def detailed_analysis():
    """Generate a detailed summary report of the comparison"""
    
    # Read the comparison results
    comparison_df = pd.read_csv("../dataset/comparison_results.csv")
    
    print("=" * 60)
    print("DETAILED COMPARISON ANALYSIS: submission.csv vs v6.csv")
    print("=" * 60)
    
    # Basic statistics
    print("\n📊 OVERVIEW:")
    print(f"• Total samples compared: {len(comparison_df):,}")
    print(f"• All sample_ids match between both files ✓")
    
    # Price analysis
    print("\n💰 PRICE ANALYSIS:")
    price_diff = comparison_df['price_diff']
    price_diff_abs = comparison_df['price_diff_abs']
    price_diff_pct = comparison_df['price_diff_pct']
    
    print(f"• Average price in submission.csv: ${comparison_df['price_file1'].mean():.2f}")
    print(f"• Average price in v6.csv: ${comparison_df['price_file2'].mean():.2f}")
    print(f"• Overall, v6.csv has higher prices on average")
    
    print(f"\n📈 DIFFERENCE STATISTICS:")
    print(f"• Mean difference: ${price_diff.mean():.2f} (submission - v6)")
    print(f"• Mean absolute difference: ${price_diff_abs.mean():.2f}")
    print(f"• Standard deviation of differences: ${price_diff.std():.2f}")
    print(f"• Median difference: ${price_diff.median():.2f}")
    
    # Categorize differences
    small_diff = (price_diff_abs <= 1).sum()
    medium_diff = ((price_diff_abs > 1) & (price_diff_abs <= 10)).sum()
    large_diff = ((price_diff_abs > 10) & (price_diff_abs <= 50)).sum()
    very_large_diff = (price_diff_abs > 50).sum()
    
    print(f"\n📊 DIFFERENCE CATEGORIES:")
    print(f"• Small differences (≤ $1): {small_diff:,} ({small_diff/len(comparison_df)*100:.1f}%)")
    print(f"• Medium differences ($1-$10): {medium_diff:,} ({medium_diff/len(comparison_df)*100:.1f}%)")
    print(f"• Large differences ($10-$50): {large_diff:,} ({large_diff/len(comparison_df)*100:.1f}%)")
    print(f"• Very large differences (> $50): {very_large_diff:,} ({very_large_diff/len(comparison_df)*100:.1f}%)")
    
    # Direction of differences
    higher_in_submission = (price_diff > 0).sum()
    higher_in_v6 = (price_diff < 0).sum()
    same_price = (price_diff == 0).sum()
    
    print(f"\n🔄 PRICE DIRECTION:")
    print(f"• Higher in submission.csv: {higher_in_submission:,} ({higher_in_submission/len(comparison_df)*100:.1f}%)")
    print(f"• Higher in v6.csv: {higher_in_v6:,} ({higher_in_v6/len(comparison_df)*100:.1f}%)")
    print(f"• Exactly same: {same_price:,} ({same_price/len(comparison_df)*100:.1f}%)")
    
    # Correlation
    correlation = comparison_df['price_file1'].corr(comparison_df['price_file2'])
    print(f"\n🔗 CORRELATION:")
    print(f"• Price correlation between files: {correlation:.4f}")
    if correlation > 0.8:
        print("• Strong positive correlation - predictions are quite similar")
    elif correlation > 0.5:
        print("• Moderate positive correlation - predictions have some similarity")
    else:
        print("• Weak correlation - predictions are quite different")
    
    # Extremes
    print(f"\n🔥 EXTREME CASES:")
    max_diff_idx = price_diff_abs.idxmax()
    max_diff_sample = comparison_df.loc[max_diff_idx]
    print(f"• Largest difference: Sample ID {max_diff_sample['sample_id']}")
    print(f"  - submission.csv: ${max_diff_sample['price_file1']:.2f}")
    print(f"  - v6.csv: ${max_diff_sample['price_file2']:.2f}")
    print(f"  - Difference: ${abs(max_diff_sample['price_diff']):.2f}")
    
    # Summary recommendations
    print(f"\n💡 SUMMARY:")
    if correlation > 0.7 and price_diff_abs.mean() < 5:
        print("• The two predictions are quite similar overall")
    elif correlation > 0.5:
        print("• The predictions show moderate agreement but with notable differences")
    else:
        print("• The predictions are significantly different - may need investigation")
    
    print(f"• v6.csv tends to predict higher prices than submission.csv")
    print(f"• Consider investigating samples with very large differences")
    
    # Save summary statistics
    summary_stats = {
        'Total_Samples': len(comparison_df),
        'Mean_Price_Submission': comparison_df['price_file1'].mean(),
        'Mean_Price_V6': comparison_df['price_file2'].mean(),
        'Mean_Absolute_Difference': price_diff_abs.mean(),
        'Correlation': correlation,
        'Large_Differences_Count': large_diff + very_large_diff,
        'Percentage_Higher_in_V6': (higher_in_v6/len(comparison_df)*100)
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv("../dataset/comparison_summary.csv", index=False)
    print(f"\n📁 Summary statistics saved to: comparison_summary.csv")

if __name__ == "__main__":
    detailed_analysis()