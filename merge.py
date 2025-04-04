# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:03:52 2025

@author: Admin
"""

def merge():    
    import pandas as pd  
    from sklearn.preprocessing import LabelEncoder
    
    # Load data
    static_data = pd.read_csv("static_client_data.csv")
    time_series_data = pd.read_csv("time_series_data.csv")
    target_data = pd.read_csv("target_data.csv")
    macro_scenarios = pd.read_csv("macro_scenarios.csv")

    # Merge datasets
    merged_df = time_series_data.merge(static_data, on="client_id", how="left")
    merged_df = merged_df.merge(target_data, on="client_id", how="left")

    merged_df.to_csv("merged.csv", index=False)
    print("Merge done!!")
    
    df = merged_df.copy()

    # Perform one-hot encoding on multiple columns
    encoded_columns = ['gender', 'employment_status', 'investment_goals']
    df_one_hot = pd.get_dummies(df, columns=encoded_columns, prefix=encoded_columns)
    
    # Convert only the newly created one-hot encoded columns to integer
    encoded_cols = [col for col in df_one_hot.columns if any(prefix in col for prefix in encoded_columns)]
    df_one_hot[encoded_cols] = df_one_hot[encoded_cols].astype(int)
    
    # Save the encoded dataframe to CSV
    df_one_hot.to_csv("encoded.csv", index=False)
    
    print("One-Hot Encoded DataFrame saved as 'encoded.csv'")

if __name__ == "__main__":
    merge()
