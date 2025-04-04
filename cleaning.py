# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 09:47:17 2025

@author: Admin
"""

import numpy as np
import pandas as pd
import re

def clean_data():
    df=pd.read_csv("static_client_data.csv")
    
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    
    df["age"].replace(0, pd.NA, inplace=True)
    
    median_age_retired = df.loc[df["employment_status"] == "Retired", "age"].median()
    median_age_employed = df.loc[df["employment_status"].isin(["Salaried", "Self-Employed"]), "age"].median()
    median_age_unemployed = df["age"].median()
    
    
    df.loc[(df["age"].isnull()) & (df["employment_status"] == "Retired"), "age"] = median_age_retired
    df.loc[(df["age"].isnull()) & (df["employment_status"].isin(["Salaried", "Self-Employed"])), "age"] = median_age_employed
    df.loc[(df["age"].isnull()) & (df["employment_status"] == "Unemployed"), "age"] = median_age_unemployed
    
    
    
    df["annual_income"].replace(0, pd.NA, inplace=True)
    
    median_income_retired = df.loc[df["employment_status"] == "Retired", "annual_income"].median()
    median_income_employed = df.loc[df["employment_status"].isin(["Salaried", "Self-Employed"]), "annual_income"].median()
    median_income_unemployed = df.loc[df["employment_status"] == "unemployed", "annual_income"].median()
    
    df.loc[(df["annual_income"].isnull()) & (df["employment_status"] == "Unemployed"), "annual_income"] = median_age_unemployed
    df.loc[(df["annual_income"].isnull()) & (df["employment_status"] == "Retired"), "annual_income"] = median_income_retired
    df.loc[(df["annual_income"].isnull()) & (df["employment_status"].isin(["Salaried", "Self-Employed"])), "annual_income"] = median_income_employed
    
    
    df.dropna(subset=["client_id"], inplace=True)
    
    
    df.drop_duplicates(subset=["client_id"], inplace=True)
    
    
    def is_valid_client_id(client_id):
        return bool(re.match(r"^[A-Za-z0-9_-]+$", str(client_id)))
    
    df = df[df["client_id"].astype(str).apply(is_valid_client_id)]
    
    df["net_worth"] = df["net_worth"].apply(lambda x: max(0, x) if pd.notnull(x) else None)
    df["net_worth"].fillna(df["net_worth"].median(), inplace=True)
    
    
    df["savings_rate"] = df["savings_rate"].apply(lambda x: min(max(0, x), 1) if pd.notnull(x) else None)
    df["savings_rate"].fillna(df["savings_rate"].median(), inplace=True)
    
    median_horizon = df.groupby("risk_appetite")["investment_horizon_years"].median().astype(int)
    
    df["investment_horizon_years"] = df.apply(
        lambda row: median_horizon[row["risk_appetite"]] if pd.isnull(row["investment_horizon_years"]) or row["investment_horizon_years"] == 0 
        else row["investment_horizon_years"], axis=1
    )
    
    df["financial_knowledge_score"] = df["financial_knowledge_score"].apply(lambda x: min(max(1, x), 10) if pd.notnull(x) else None)
    
    df["financial_knowledge_score"].fillna(df["financial_knowledge_score"].median(), inplace=True)
    
    
    df3=pd.read_csv("C:\\Users\\Admin\\Downloads\\hackstars\\time_series_data.csv")
    
    df3["month"] = pd.to_datetime(df3["month"], errors='coerce')
    
    
    df3["portfolio_value"] = df3["portfolio_value"].apply(lambda x: max(0, x) if pd.notnull(x) else None)
    df3["portfolio_value"].fillna(df3["portfolio_value"].median(), inplace=True)
    
    
    df3["equity_allocation_pct"] = df3["equity_allocation_pct"].clip(0, 100)
    df3["fixed_income_allocation_pct"] = df3["fixed_income_allocation_pct"].clip(0, 100)
    
    
    df3["monthly_contribution"] = df3["monthly_contribution"].apply(lambda x: max(0, x) if pd.notnull(x) else None)
    df3["monthly_contribution"].fillna(df3["monthly_contribution"].median(), inplace=True)
    
    
    df3["market_volatility_index"].fillna(df3["market_volatility_index"].median(), inplace=True)
    
    
    df3["macroeconomic_score"] = df3["macroeconomic_score"].clip(1, 10)
    df3["sentiment_index"] = df3["sentiment_index"].clip(1, 10)
    
    df1 = pd.read_csv("C:\\Users\\Admin\\Downloads\\hackstars\\target_data.csv")
    
    
    df1['recommended_strategy'] = df1['recommended_strategy'].fillna(df1['recommended_strategy'].mode()[0])
    
    num_cols = ['forecasted_value_year_1', 'forecasted_value_year_2', 'forecasted_value_year_3']
    df1[num_cols] = df1[num_cols].apply(lambda x: x.fillna(x.median()))
    
    print("data cleaning!!")
    
if __name__ == "__main__":
    clean_data() 