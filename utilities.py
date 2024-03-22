import pandas as pd 
import numpy as np
import os 
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime 
from scipy.stats import linregress


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''Clean the data by removing rows with NaN values and invalid datetime values'''

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Convert the 'Model' column to datetime, and drop rows with errors
    for value in df['Model']:
        try:
            if isinstance(value, datetime):
                # Replace datetime values with 'Model_unknown'
                df.loc[df['Model'] == value, 'Model'] = 'Model_unknown'
        except TypeError:
            pass

    return df



def bar_line_graph(col,df:pd.DataFrame):
    '''To make the Categorical Column and see the trend between Average Popularity and Average MSRP'''
    df[col] = df[col].astype('str')
    grouped_df = df.groupby(col)

    # Calculate the average of 'MSRP' and 'Popularity' columns within each group
    avg_msrp = grouped_df['MSRP'].mean()
    avg_popularity = grouped_df['Popularity'].mean()

    # Sort values based on Average MSRP in descending order
    avg_msrp_sorted = avg_msrp.sort_values(ascending=False)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot bar graph for average MSRP
    avg_msrp_sorted.plot(kind='bar', ax=ax1, color='tab:blue')
    ax1.set_xlabel(f'{col}')
    ax1.set_ylabel('Average MSRP', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title(f'Average MSRP/Popularity within each {col}')

    # Create a second y-axis for average Popularity and plot line chart
    ax2 = ax1.twinx()
    avg_popularity.reindex(avg_msrp_sorted.index).plot(kind='line', marker='o', ax=ax2, color='tab:orange', label='{Avg Popularity}')
    ax2.set_ylabel('Average Popularity', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.grid(True)
    ax2.legend(loc='upper right')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()





def scatter_plots(col, df: pd.DataFrame):
    '''Create Reg Plots to check the Data Correlations'''
    # Calculate correlation coefficient between specified column and 'MSRP'
    correlation_coefficient = df[col].corr(df['MSRP'])

    # Create a regplot between specified column and 'MSRP'
    plt.figure(figsize=(10, 6))
    sns.regplot(x=col, y='MSRP', data=df, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})

    # Get the regression coefficients
    reg_coef = np.polyfit(df[col], df['MSRP'], deg=1)

    # Calculate R-squared value
    slope, intercept, r_value, p_value, std_err = linregress(df[col], df['MSRP'])
    r_squared = r_value ** 2

    # Combine text for correlation coefficient, regression coefficients, and R-squared value
    annotation_text = f'Correlation Coefficient: {correlation_coefficient:.2f}\n' \
                      f'Regression Coefficients: {reg_coef}\n' \
                      f'R-squared: {r_squared:.2f}'

    # Add combined annotation text inside the graph area but above the plot
    plt.xlabel(f'{col}')
    plt.ylabel('MSRP')
    plt.text(x=np.min(df[col]), y=np.max(df['MSRP']) * 0.95, s=annotation_text, fontsize=12, ha='left', va='top')
    plt.title(f'Regression Plot of {col} vs MSRP')
    plt.show()

def cat_corr_plots(col, df, max_unique=10):
    '''Create the correlation heatmap for the categorical plots with unique values less than 10'''
    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Select relevant columns
    x_df = df[[col, 'MSRP']]
    
    # Limit the number of unique values considered for plotting
    unique_values = df[col].nunique()
    if unique_values > max_unique:
        print(f"Number of unique values for {col} exceeds the maximum limit. Skipping plot.")
        return
    
    # Create dummy variables
    x_df = pd.get_dummies(x_df)
    # Compute correlation matrix
    correlation_matrix = x_df.corr()
    
    # Limit figure size based on the number of unique values
    fig_size = (10, 6) if unique_values <= 10 else (unique_values * 0.9, unique_values * 0.9)
    plt.figure(figsize=fig_size)
    
    # Plot correlation heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Heatmap for {col}')
    plt.show()

