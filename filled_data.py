# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 08:54:05 2025

@author: Admin
"""

import pandas as pd
def fun1_fill():
# Load the dataset
    df = pd.read_csv("C:/Users/Admin/Downloads/hackstars/lag_features.csv")  
    
    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns  # Identify numeric columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns  # Identify non-numeric columns
    
    # Convert only numeric columns to proper format
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Fill NaNs in numeric columns with their column mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Save the updated dataset
    df.to_csv("filled_data.csv", index=False)
    
    print("Blank spaces filled with column mean successfully! Non-numeric columns are preserved.")

if __name__ == "__main__":
    fun1_fill()
    