# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:39:17 2025

@author: Admin
"""

import pandas as pd

def lag_fun():
    df = pd.read_csv("encoded.csv")
    
    df.sort_values(by=['month'], inplace=True)
    
    lag_features = ['portfolio_value', 'monthly_contribution', 'market_volatility_index', 'macroeconomic_score', 'sentiment_index']
    
    for lag in range(1, 4):  # Lagging up to 3 months
        for col in lag_features:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
         
    df.to_csv("lag_features.csv",index=False)
    
    print("Lag done!!")
if __name__ == "__main__":
    lag_fun()