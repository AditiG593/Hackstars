# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:05:25 2025

@author: SHRUTI-NIDHI
"""

import dash
from dash import Dash, html, dcc, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc

# Initialize the Dash app with a Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server  # for deployment

# Data generation functions
def generate_client_data():
    try:
        df = pd.read_csv('static_client_data.csv')
       # Debug print
        df = df.head(10)
        df['preferred_asset_classes'] = df['preferred_asset_classes'].apply(eval)
        return df.to_dict('records')
    except FileNotFoundError:
        print("Error: static_client_data.csv file not found.")
        return []
    except Exception as e:
        print(f"Error loading client data: {str(e)}")
        return []

def generate_time_series_data(client_id, months=36):
    try:
        df = pd.read_csv('time_series_data.csv')
        client_data = df[df['client_id'] == client_id].sort_values('month').tail(months)
        return client_data[[
            'month', 'portfolio_value', 'equity_allocation_pct', 'fixed_income_allocation_pct',
            'monthly_contribution', 'market_volatility_index', 'macroeconomic_score', 'sentiment_index'
        ]].to_dict('records')
    except FileNotFoundError:
        print(f"Error: time_series_data.csv file not found for client {client_id}")
        return []
    except Exception as e:
        print(f"Error loading time series data for client {client_id}: {str(e)}")
        return []

def generate_asset_allocation(client_id):
    try:
        df = pd.read_csv('static_client_data.csv')
        client_data = df[df['client_id'] == client_id].iloc[0]
        preferred_assets = eval(client_data['preferred_asset_classes'])
        allocation = []
        num_assets = len(preferred_assets)
        base_percent = 100.0 / num_assets
        remaining = 100.0
        for i, asset in enumerate(preferred_assets):
            percent = remaining if i == num_assets - 1 else base_percent
            remaining -= percent if i < num_assets - 1 else 0
            allocation.append({"asset": asset, "value": round(percent, 1)})
        return allocation
    except FileNotFoundError:
        print(f"Error: static_client_data.csv file not found for client {client_id}")
        return []
    except IndexError:
        print(f"Error: Client {client_id} not found in static_client_data.csv")
        return []
    except Exception as e:
        print(f"Error generating asset allocation for client {client_id}: {str(e)}")
        return []

def generate_forecast(client_id, time_series):
    try:
        # Load the predicted data
        encoded_data = pd.read_csv("merged_predictions.csv")
        # Debug: Print column names and first few rows
        print("Columns in merged_predictions.csv:", encoded_data.columns.tolist())
        print("First few rows of merged_predictions.csv:\n", encoded_data.head())
        
        # Filter for the given client_id (normalize client_id to handle mismatches)
        client_strategy = encoded_data[encoded_data["client_id"].str.strip() == client_id.strip()]
        # Debug: Check if client_id is found
        print(f"Client ID {client_id} found in dataset: {not client_strategy.empty}")
        
        if not client_strategy.empty:
            # Access the first row and extract relevant columns
            client_row = client_strategy.iloc[0]
            # Debug: Print the entire row for this client
            print(f"Data for client {client_id}:\n", client_row.to_dict())
            
            # Access the recommended strategy (update column name if needed)
            strategy_mapping = {
                "conservative": "Conservative",
                "balanced": "Balanced",
                "aggressive": "Aggressive",
                # Add more mappings as needed
            }
            raw_strategy = client_row.get("predicted_recommended_strategy", ["Balanced","Conservative","Agressive"]).lower()
            recommended_strategy = strategy_mapping.get(raw_strategy, "Balanced").capitalize()
            # Debug: Print the recommended strategy
            print(f"Recommended strategy for client {client_id}: {recommended_strategy}")
            
            year1 = client_row["predicted_value_year_1"]
            year2 = client_row["predicted_value_year_2"]
            year3 = client_row["predicted_value_year_3"]
        else:
            # Fallback if no data for client_id
            print(f"No prediction data found for client {client_id}. Using fallback.")
            recommended_strategy = "Balanced"
            latest_value = time_series[-1]["portfolio_value"]
            monthly_contribution = time_series[-1]["monthly_contribution"]
            annual_growth = 0.07  # Default growth rate for "Balanced"
            year1 = latest_value * (1 + annual_growth) + (monthly_contribution * 12)
            year2 = year1 * (1 + annual_growth) + (monthly_contribution * 12)
            year3 = year2 * (1 + annual_growth) + (monthly_contribution * 12)

        # Construct the forecast dictionary
        all_forecasts = {
            recommended_strategy: {
                "year1": round(year1, 2),
                "year2": round(year2, 2),
                "year3": round(year3, 2)
            }
        }
        return {
            "recommended_strategy": recommended_strategy,
            "forecasted_value_year_1": round(year1, 2),
            "forecasted_value_year_2": round(year2, 2),
            "forecasted_value_year_3": round(year3, 2),
            "all_forecasts": all_forecasts
        }

    except FileNotFoundError:
        print("Error: merged_predictions.csv file not found. Using fallback.")
        recommended_strategy = "Balanced"
        latest_value = time_series[-1]["portfolio_value"]
        monthly_contribution = time_series[-1]["monthly_contribution"]
        annual_growth = 0.07
        year1 = latest_value * (1 + annual_growth) + (monthly_contribution * 12)
        year2 = year1 * (1 + annual_growth) + (monthly_contribution * 12)
        year3 = year2 * (1 + annual_growth) + (monthly_contribution * 12)
        all_forecasts = {
            recommended_strategy: {
                "year1": round(year1, 2),
                "year2": round(year2, 2),
                "year3": round(year3, 2)
            }
        }
        return {
            "recommended_strategy": recommended_strategy,
            "forecasted_value_year_1": round(year1, 2),
            "forecasted_value_year_2": round(year2, 2),
            "forecasted_value_year_3": round(year3, 2),
            "all_forecasts": all_forecasts
        }
    except Exception as e:
        print(f"Error in generate_forecast: {str(e)}")
        return None

# Initial data
clients = generate_client_data()
client_ids = [client["client_id"] for client in clients]
default_client = clients[0]["client_id"] if clients else None

# App layout (removed Scenario Testing tab)
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1("Wealth Compass AI", className="dash-title"),
            html.P("Multi-Dimensional Wealth Management Strategy Engine", className="dash-subtitle")
        ], className="header-text"),
        html.Div([
            html.Label("Select Client:"),
            dcc.Dropdown(id="client-dropdown", options=[{"label": cid, "value": cid} for cid in client_ids], value=default_client, clearable=False, className="client-dropdown")
        ], className="header-controls")
    ], className="header"),
    html.Div([
        html.Div(id="strategy-banner", className="strategy-banner"),
        dcc.Tabs(id="main-tabs", value="overview", children=[
            dcc.Tab(label="Overview", value="overview"),
            dcc.Tab(label="Portfolio Analysis", value="portfolio"),
        ], className="custom-tabs"),
        html.Div(id="tab-content", className="tab-content")
    ], className="main-content"),
    html.Div([
        html.P("© 2025 Wealth Compass AI. All rights reserved."),
        html.P(f"Data updated: {datetime.now().strftime('%B %d, %Y')}")
    ], className="footer")
], className="app-container")

# Callback for tab content
@app.callback(
    Output("tab-content", "children"),
    Output("strategy-banner", "children"),
    Input("main-tabs", "value"),
    Input("client-dropdown", "value")
)
def render_tab_content(tab, client_id):
    client = next((c for c in clients if c["client_id"] == client_id), None)
    if not client:
        return html.Div("Client not found"), html.Div()

    time_series = generate_time_series_data(client_id)
    forecast = generate_forecast(client_id, time_series)
    if forecast is None:
        return html.Div("Error generating forecast"), html.Div()

    asset_allocation = generate_asset_allocation(client_id)

    banner_color = {"Conservative": "blue", "Balanced": "green", "Aggressive": "red"}.get(forecast["recommended_strategy"], "blue")
    strategy_banner = html.Div([
        html.Div([
            html.H2(f"{forecast['recommended_strategy']} Investment Strategy Recommended"),
            html.Div([
                html.Div([html.P("Current Portfolio Value"), html.H3(f"${time_series[-1]['portfolio_value']:,.0f}")], className="banner-stat"),
                html.Div([html.P("3-Year Forecast"), html.H3(f"${forecast['forecasted_value_year_3']:,.0f}")], className="banner-stat")
            ], className="banner-stats")
        ], className="banner-content")
    ], className=f"strategy-banner {banner_color}")

    if tab == "overview":
        return overview_tab(client, time_series, forecast, asset_allocation), strategy_banner
    elif tab == "portfolio":
        return portfolio_tab(client, time_series, forecast, asset_allocation), strategy_banner
    return html.Div("Tab content not found"), strategy_banner

def overview_tab(client, time_series, forecast, asset_allocation):
    client_summary = html.Div([
        html.H3("Client Profile"), html.P("Key client information and profile data", className="card-description"),
        html.Div([
            html.Div([html.H4(client["client_id"]), html.P(f"{client['age']} years • {client['gender']} • {client['employment_status']}")], className="client-header"),
            html.Div([
                html.Div([html.Span("Annual Income"), html.Span(f"${client['annual_income']:,.0f}")], className="client-data-item"),
                html.Div([html.Span("Net Worth"), html.Span(f"${client['net_worth']:,.0f}")], className="client-data-item"),
                html.Div([html.Span("Investment Goal"), html.Span(client["investment_goals"])], className="client-data-item"),
                html.Div([html.Span("Time Horizon"), html.Span(f"{client['investment_horizon_years']} years")], className="client-data-item")
            ], className="client-data-grid"),
            html.Div([
                html.Div(f"{client['risk_appetite']} Risk", className=f"badge {client['risk_appetite'].lower()}"),
                html.Div(f"{client['dependents']} Dependents", className="badge outline"),
                html.Div(f"{client['savings_rate']*100:.1f}% Savings Rate", className="badge outline"),
                html.Div(f"Knowledge: {client['financial_knowledge_score']}/10", className="badge secondary")
            ], className="client-badges"),
            html.Div([html.H5("Preferred Asset Classes"), html.Div([html.Div(asset, className="asset-badge") for asset in client["preferred_asset_classes"]], className="asset-badges")], className="preferred-assets")
        ], className="client-summary-content")
    ], className="dashboard-card")

    portfolio_chart = html.Div([
        html.H3("Portfolio Value Forecast"), html.P("Historical performance and 3-year projection", className="card-description"),
        dcc.Graph(id="portfolio-chart", figure=create_portfolio_chart(time_series, forecast), config={"displayModeBar": False}, className="chart-container")
    ], className="dashboard-card")

    return html.Div([
        html.Div([html.Div([client_summary], className="column-1-3"), html.Div([portfolio_chart], className="column-2-3")], className="row"),
        html.Div([
            html.Div([create_key_metrics_card(client, time_series)], className="column-1-3"),
            html.Div([html.H3("Asset Allocation"), dcc.Graph(figure=create_asset_allocation_chart(asset_allocation), config={"displayModeBar": False})], className="column-1-3"),
            html.Div([create_market_indicators_card(time_series)], className="column-1-3")
        ], className="row")
    ], className="dashboard-grid")

def portfolio_tab(client, time_series, forecast, asset_allocation):
    strategy_card = html.Div([
        html.H3("Strategy Recommendation"), html.P("Personalized investment strategy based on your profile", className="card-description"),
        html.Div([
            html.H4(forecast["recommended_strategy"], className=f"strategy-title {forecast['recommended_strategy'].lower()}"),
            html.P("This strategy is aligned with your risk profile and investment goals."),
            html.Div([
                html.H5("3-Year Forecast"),
                html.Div([
                    html.Div([html.P("Year 1"), html.H4(f"${forecast['forecasted_value_year_1']:,.0f}")], className="forecast-item"),
                    html.Div([html.P("Year 2"), html.H4(f"${forecast['forecasted_value_year_2']:,.0f}")], className="forecast-item"),
                    html.Div([html.P("Year 3"), html.H4(f"${forecast['forecasted_value_year_3']:,.0f}")], className="forecast-item")
                ], className="forecast-grid")
            ], className="forecast-section"),
            html.Div([
               
                html.Div(
                    
                 className="alternatives-grid")
            ], className="alternatives-section")
        ], className="strategy-content")
    ], className="dashboard-card")

    portfolio_chart = html.Div([
        html.H3("Portfolio Value Forecast"), html.P("Historical performance and 3-year projection", className="card-description"),
        dcc.Graph(id="portfolio-chart-tab2", figure=create_portfolio_chart(time_series, forecast), config={"displayModeBar": False}, className="chart-container")
    ], className="dashboard-card")

    return html.Div([
        html.Div([html.Div([strategy_card], className="column-1-3"), html.Div([portfolio_chart], className="column-2-3")], className="row"),
        html.Div([
            html.Div([html.H3("Portfolio Allocation"), dcc.Graph(figure=create_asset_allocation_chart(asset_allocation), config={"displayModeBar": False})], className="dashboard-card"),
            html.Div([html.H3("Risk Assessment"), create_market_indicators_content(time_series)], className="dashboard-card")
        ], className="row two-columns")
    ], className="dashboard-grid")

def create_portfolio_chart(time_series, forecast):
    months = [item["month"] for item in time_series]
    values = [item["portfolio_value"] for item in time_series]
    last_date = months[-1]
    date_obj = datetime.strptime(last_date, '%Y-%m-%d')
    year, month = date_obj.year, date_obj.month

    forecast_dates = [
        last_date,  # Start from the last historical date
        f"{year + 1}-{month:02d}-01",
        f"{year + 2}-{month:02d}-01",
        f"{year + 3}-{month:02d}-01"
    ]
    forecast_values = [
        values[-1],  # Connect to the last historical value
        forecast["forecasted_value_year_1"],
        forecast["forecasted_value_year_2"],
        forecast["forecasted_value_year_3"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=values, mode='lines', name='Historical Value', line=dict(color='#6366F1', width=2)))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, mode='lines+markers', name='Projected Value', line=dict(color='#10B981', width=2, dash='dash')))
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(title=None, gridcolor='#e2e8f0'), yaxis=dict(title=None, gridcolor='#e2e8f0', tickformat='$,.0f'),
        plot_bgcolor='white', height=350, hovermode='x unified'
    )
    return fig

def create_asset_allocation_chart(asset_allocation):
    labels = [item["asset"] for item in asset_allocation]
    values = [item["value"] for item in asset_allocation]
    colors = ['#6366F1', '#8B5CF6', '#EC4899', '#10B981', '#3B82F6', '#F59E0B', '#EF4444']
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, textinfo='label+percent', marker=dict(colors=colors))])
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), showlegend=False, height=250, plot_bgcolor='white')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_market_indicators_content(time_series):
    latest = time_series[-1]
    older = time_series[max(0, len(time_series) - 4)]
    volatility_trend = latest["market_volatility_index"] - older["market_volatility_index"]
    macro_trend = latest["macroeconomic_score"] - older["macroeconomic_score"]
    sentiment_trend = latest["sentiment_index"] - older["sentiment_index"]
    return html.Div([
        html.Div([html.Div([html.Span("Market Volatility", className="indicator-label"), html.Span(f"{latest['market_volatility_index']:.1f}", className="indicator-value")], className="indicator-header"), dbc.Progress(value=latest["market_volatility_index"]*10, className="indicator-progress"), html.P(f"{'Increased' if volatility_trend > 0 else 'Decreased'} by {abs(volatility_trend):.1f} points in the last 3 months", className="indicator-trend")], className="market-indicator"),
        html.Div([html.Div([html.Span("Macroeconomic Score", className="indicator-label"), html.Span(f"{latest['macroeconomic_score']:.1f}", className="indicator-value")], className="indicator-header"), dbc.Progress(value=latest["macroeconomic_score"]*10, className="indicator-progress"), html.P(f"{'Improved' if macro_trend > 0 else 'Declined'} by {abs(macro_trend):.1f} points in the last 3 months", className="indicator-trend")], className="market-indicator"),
        html.Div([html.Div([html.Span("Market Sentiment", className="indicator-label"), html.Span(f"{latest['sentiment_index']:.1f}", className="indicator-value")], className="indicator-header"), dbc.Progress(value=latest["sentiment_index"]*10, className="indicator-progress"), html.P(f"{'Improved' if sentiment_trend > 0 else 'Declined'} by {abs(sentiment_trend):.1f} points in the last 3 months", className="indicator-trend")], className="market-indicator")
    ], className="market-indicators-content")

def create_key_metrics_card(client, time_series):
    latest_data = time_series[-1]
    contributions = [data["monthly_contribution"] for data in time_series[-12:]]
    avg_contribution = sum(contributions) / len(contributions) if contributions else 0
    current_value = latest_data["portfolio_value"]
    one_year_ago_idx = max(0, len(time_series) - 13)
    value_one_year_ago = time_series[one_year_ago_idx]["portfolio_value"]
    growth_rate = ((current_value - value_one_year_ago) / value_one_year_ago) * 100 if value_one_year_ago else 0
    debt_to_income_ratio = client["debt_to_income_ratio"] * 100
    income_to_portfolio_ratio = (client["annual_income"] / current_value) * 100 if current_value else 0

    return html.Div([
        html.H3("Key Financial Metrics"), html.P("Current snapshot of important financial indicators", className="card-description"),
        html.Div([
            html.Div([html.Div([html.Span("Monthly Contribution", className="metric-label"), html.Div([html.Span(f"${avg_contribution:,.0f}", className="metric-value"), html.Span("/month", className="metric-unit")], className="metric-value-container")], className="metric-card")], className="metric-col"),
            html.Div([html.Div([html.Span("Annual Growth", className="metric-label"), html.Div([html.Span(f"{'+' if growth_rate >= 0 else ''}{growth_rate:.1f}%", className=f"metric-value {'positive' if growth_rate >= 0 else 'negative'}"), html.Span("/year", className="metric-unit")], className="metric-value-container")], className="metric-card")], className="metric-col"),
            html.Div([html.Div([html.Span("Debt to Income", className="metric-label"), html.Div([html.Span(f"{debt_to_income_ratio:.1f}%", className=f"metric-value {'positive' if debt_to_income_ratio <= 36 else 'caution'}"), html.Span("ratio", className="metric-unit")], className="metric-value-container")], className="metric-card")], className="metric-col"),
            html.Div([html.Div([html.Span("Income/Portfolio", className="metric-label"), html.Div([html.Span(f"{income_to_portfolio_ratio:.1f}%", className="metric-value"), html.Span("ratio", className="metric-unit")], className="metric-value-container")], className="metric-card")], className="metric-col")
        ], className="metrics-grid")
    ], className="dashboard-card")

def create_market_indicators_card(time_series):
    return html.Div([
        html.H3("Market Indicators"),
        html.P("Current market conditions and economic indicators", className="card-description"),
        create_market_indicators_content(time_series)
    ], className="dashboard-card")

# Add CSS for styling (removed scenario-related CSS)
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Wealth Compass AI</title>
        {%favicon%}
        {%css%}
        <style>
            /* Global Styles */
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f7f9fc;
                color: #333;
            }
           
            .app-container {
                display: flex;
                flex-direction: column;
                min-height: 100vh;
            }
           
            /* Header Styles */
            .header {
                background-color: #fff;
                padding: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
            }
           
            .header-text {
                flex: 1;
            }
           
            .dash-title {
                margin: 0;
                font-size: 24px;
                font-weight: 700;
                color: #1a1f2c;
            }
           
            .dash-subtitle {
                margin: 5px 0 0;
                color: #666;
                font-size: 14px;
            }
           
            .header-controls {
                width: 250px;
            }
           
            /* Main Content */
            .main-content {
                flex: 1;
                padding: 20px;
                max-width: 1400px;
                margin: 0 auto;
                width: 100%;
                box-sizing: border-box;
            }
           
            /* Strategy Banner */
            .strategy-banner {
                padding: 15px 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                color: white;
            }
           
            .strategy-banner.blue { background-color: #3B82F6; }
            .strategy-banner.green { background-color: #10B981; }
            .strategy-banner.red { background-color: #EF4444; }
           
            .banner-content {
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
            }
           
            .banner-content h2 {
                margin: 0;
                font-size: 20px;
                font-weight: 600;
            }
           
            .banner-s {
                display: flex;
                gap: 20px;
            }
           
            .banner-stat {
                text-align: right;
            }
           
            .banner-stat p {
                margin: 0;
                font-size: 12px;
                opacity: 0.9;
            }
           
            .banner-stat h3 {
                margin: 5px 0 0;
                font-size: 18px;
                font-weight: 700;
            }
           
            /* Dashboard Cards */
            .dashboard-card {
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                padding: 20px;
                height: 100%;
                box-sizing: border-box;
            }
           
            .dashboard-card h3 {
                margin-top: 0;
                margin-bottom: 5px;
                font-size: 18px;
                font-weight: 600;
            }
           
            .card-description {
                margin-top: 0;
                margin-bottom: 15px;
                font-size: 13px;
                color: #666;
            }
           
            /* Dashboard Grid Layout */
            .dashboard-grid {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
           
            .row {
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
            }
           
            .column-1-3 {
                flex: 1;
                min-width: 300px;
            }
           
            .column-2-3 {
                flex: 2;
                min-width: 600px;
            }
           
            .column-1-2 {
                flex: 1;
                min-width: 400px;
            }
           
            .two-columns {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
           
            .full-width {
                width: 100%;
            }
           
            /* Client Summary */
            .client-summary-content {
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
           
            .client-header {
                display: flex;
                align-items: center;
                gap: 15px;
            }
           
            .client-header h4 {
                margin: 0;
                font-size: 18px;
            }
           
            .client-header p {
                margin: 5px 0 0;
                font-size: 13px;
                color: #666;
            }
           
            .client-data-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }
           
            .client-data-item {
                display: flex;
                flex-direction: column;
                font-size: 14px;
            }
           
            .client-data-item span:first-child {
                color: #666;
                font-size: 13px;
            }
           
            .client-data-item span:last-child {
                font-weight: 500;
                margin-top: 3px;
            }
           
            .client-badges {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 10px;
            }
           
            .badge {
                font-size: 12px;
                padding: 4px 10px;
                border-radius: 20px;
                font-weight: 500;
            }
           
            .badge.outline {
                background-color: transparent;
                border: 1px solid #d1d5db;
                color: #666;
            }
           
            .badge.secondary {
                background-color: #e5e7eb;
                color: #4b5563;
            }
           
            .badge.low {
                background-color: #3B82F6;
                color: white;
            }
           
            .badge.medium {
                background-color: #8B5CF6;
                color: white;
            }
           
            .badge.high {
                background-color: #EF4444;
                color: white;
            }
           
            .preferred-assets h5 {
                margin: 0 0 8px;
                font-size: 14px;
                font-weight: 500;
            }
           
            .asset-badges {
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
            }
           
            .asset-badge {
                font-size: 12px;
                padding: 3px 10px;
                border-radius: 20px;
                background-color: #f3f4f6;
                color: #4b5563;
            }
           
            /* Chart Container */
            .chart-container {
                width: 100%;
                margin-top: 5px;
            }
           
            /* Key Metrics */
            .metrics-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }
           
            .metric-card {
                padding: 10px;
                border: 1px solid #e5e7eb;
                border-radius: 6px;
                background-color: #f9fafb;
            }
           
            .metric-label {
                font-size: 12px;
                color: #6b7280;
                display: block;
                margin-bottom: 5px;
            }
           
            .metric-value-container {
                display: flex;
                align-items: baseline;
            }
           
            .metric-value {
                font-size: 18px;
                font-weight: 600;
            }
           
            .metric-value.positive {
                color: #10B981;
            }
           
            .metric-value.negative {
                color: #EF4444;
            }
           
            .metric-value.caution {
                color: #F59E0B;
            }
           
            .metric-unit {
                font-size: 12px;
                color: #6b7280;
                margin-left: 3px;
            }
           
            /* Market Indicators */
            .market-indicators-content {
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
           
            .market-indicator {
                margin-bottom: 15px;
            }
           
            .indicator-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
            }
           
            .indicator-label {
                font-size: 14px;
                font-weight: 500;
            }
           
            .indicator-value {
                font-size: 14px;
                font-weight: 600;
            }
           
            .indicator-progress {
                height: 6px !important;
                border-radius: 3px;
                margin-bottom: 5px;
            }
           
            .indicator-trend {
                font-size: 12px;
                color: #6b7280;
                margin: 5px 0 0;
            }
           
            /* Strategy Recommendation */
            .strategy-content {
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
           
            .strategy-title {
                font-size: 20px;
                margin: 0 0 10px;
                padding-bottom: 10px;
                border-bottom: 1px solid #e5e7eb;
            }
           
            .strategy-title.conservative {
                color: #3B82F6;
            }
           
            .strategy-title.balanced {
                color: #10B981;
            }
           
            .strategy-title.aggressive {
                color: #EF4444;
            }
           
            .forecast-section, .alternatives-section {
                margin-top: 15px;
            }
           
            .forecast-section h5, .alternatives-section h5 {
                margin: 0 0 10px;
                font-size: 16px;
                font-weight: 500;
            }
           
            .forecast-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
            }
           
            .forecast-item {
                text-align: center;
                padding: 10px;
                background-color: #f9fafb;
                border-radius: 6px;
            }
           
            .forecast-item p {
                margin: 0 0 5px;
                font-size: 13px;
                color: #6b7280;
            }
           
            .forecast-item h4 {
                margin: 0;
                font-size: 16px;
                font-weight: 600;
            }
           
            .alternatives-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
            }
           
            .strategy-alternative {
                padding: 10px;
                border-radius: 6px;
                text-align: center;
            }
           
            .strategy-alternative p {
                margin: 0 0 5px;
                font-size: 12px;
            }
           
            .strategy-alternative p:nth-child(2) {
                font-weight: 600;
                font-size: 13px;
            }
           
            .strategy-alternative p:nth-child(3) {
                font-size: 11px;
                color: #6b7280;
            }
           
            .strategy-alternative.conservative {
                background-color: #EBF5FF;
                border: 1px solid #BFDBFE;
            }
           
            .strategy-alternative.balanced {
                background-color: #ECFDF5;
                border: 1px solid #A7F3D0;
            }
           
            .strategy-alternative.aggressive {
                background-color: #FEF2F2;
                border: 1px solid #FECACA;
            }
           
            /* Tabs Styling */
            .custom-tabs .tab {
                color: #4b5563 !important;
                border-bottom-style: solid !important;
                border-bottom-width: 2px !important;
                border-bottom-color: transparent !important;
                background-color: transparent !important;
                padding: 12px 24px !important;
                font-weight: 500 !important;
            }
           
            .custom-tabs .tab--selected {
                color: #4f46e5 !important;
                border-bottom-color: #4f46e5 !important;
            }
           
            /* Footer */
            .footer {
                background-color: #fff;
                border-top: 1px solid #e5e7eb;
                padding: 15px 20px;
                text-align: center;
            }
           
            .footer p {
                margin: 5px 0;
                font-size: 14px;
                color: #6b7280;
            }
           
            /* Responsive adjustments */
            @media (max-width: 768px) {
                .header {
                    flex-direction: column;
                    align-items: flex-start;
                }
               
                .header-controls {
                    width: 100%;
                    margin-top: 15px;
                }
               
                .banner-content {
                    flex-direction: column;
                }
               
                .banner-stats {
                    margin-top: 15px;
                }
               
                .two-columns {
                    grid-template-columns: 1fr;
                }
               
                .column-2-3, .column-1-3, .column-1-2 {
                    min-width: 100%;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
