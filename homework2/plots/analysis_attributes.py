import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def analyze_csv(file_path, if_save=False):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Get descriptive statistics for numerical attributes
    desc_stats = df.describe()
    print("Descriptive Statistics:")
    print(desc_stats)
    
    # Generate box plots for numerical attributes
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    if len(numerical_cols) > 0:
        # Create a box plot for each numerical attribute
        for col in numerical_cols:
            # save the box plot as a file
            if if_save:
                plt.figure()
                df.boxplot(column=col)
                plt.title(f'Box plot for {col}')
                plt.savefig(f'plots/box_plot_{col}_{os.path.basename(file_path).split(".")[0]}.png', dpi=300)
                plt.close()

    else:
        print("No numerical attributes found for box plots.")
    
     # Analyze categorical/ordinal attributes
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) > 0:
        categorical_stats = {}
        for col in categorical_cols:
            non_missing_count = df[col].notna().sum()
            unique_values_count = df[col].nunique()
            categorical_stats[col] = [non_missing_count, unique_values_count]
        
        # Create a DataFrame for categorical/ordinal attribute statistics
        categorical_stats_df = pd.DataFrame.from_dict(
            categorical_stats, orient='index', columns=['Non-missing Count', 'Unique Values Count']
        )
        print("\nCategorical/Ordinal Attribute Statistics:")
        print(categorical_stats_df)
        
        # Generate histograms for categorical/ordinal attributes
        for col in categorical_cols:
            if if_save:
                plt.figure()
                df[col].hist()
                plt.title(f'Histogram for {col}')
                # rotate the x-axis labels for better readability and zoom out them
                if len(df[col].unique()) > 10:
                    plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'plots/histogram_{col}_{os.path.basename(file_path).split(".")[0]}.png', dpi=300)
    else:
        print("No categorical/ordinal attributes found for histograms.")

    # For each categorical or ordinal attribute, generate a boxplot for each numerical attribute
    if not if_save:
        for cat_col in categorical_cols:
            for num_col in numerical_cols:
                df.boxplot(column=num_col, by=cat_col, figsize=(10, 6))
                plt.title(f'Box plot for {num_col} by {cat_col}')
                plt.suptitle('')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'box_plot_{num_col}_by_{cat_col}_{os.path.basename(file_path).split(".")[0]}.png', dpi=300)
            

if __name__ == "__main__":
    file = './tema2_SalaryPrediction/SalaryPrediction_full.csv'
    analyze_csv(file, if_save=True)
    print(f"\nAnalysis completed for {file}\n")

    file = './tema2_AVC/AVC_full.csv'
    analyze_csv(file, if_save=True)
    print(f"\nAnalysis completed for {file}\n")

