import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ------------------------
# Load Data
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('consumer_products_dashboard_data.csv', parse_dates=['Month'])
    return df

df = load_data()

# ------------------------
# Dashboard Title
# ------------------------
st.title("Consumer Products Sales & Market Expansion Dashboard")

# ------------------------
# Sidebar Filters
# ------------------------
st.sidebar.header("Filters")
regions = st.sidebar.multiselect("Select Region(s)", df['Region'].unique(), default=df['Region'].unique())
categories = st.sidebar.multiselect("Select Category(s)", df['Category'].unique(), default=df['Category'].unique())
channels = st.sidebar.multiselect("Select Channel(s)", df['Channel'].unique(), default=df['Channel'].unique())

# Filter data
filtered_df = df[(df['Region'].isin(regions)) & 
                 (df['Category'].isin(categories)) & 
                 (df['Channel'].isin(channels))]

# ------------------------
# Sales Performance
# ------------------------
st.header("Sales Performance")

# Total Sales by Region
sales_region = filtered_df.groupby('Region')['Sales'].sum().reset_index()
fig_region = px.bar(sales_region, x='Region', y='Sales', title="Total Sales by Region")
st.plotly_chart(fig_region)

# Sales Distribution by Category
sales_category = filtered_df.groupby('Category')['Sales'].sum().reset_index()
fig_category = px.pie(sales_category, values='Sales', names='Category', title="Sales Distribution by Category")
st.plotly_chart(fig_category)

# Sales Distribution by Channel
sales_channel = filtered_df.groupby('Channel')['Sales'].sum().reset_index()
fig_channel = px.bar(sales_channel, x='Channel', y='Sales', title="Sales Distribution by Channel")
st.plotly_chart(fig_channel)

# ------------------------
# Monthly Sales Trend
# ------------------------
st.header("Monthly Sales Trend")
sales_trend = filtered_df.groupby('Month')['Sales'].sum().reset_index()
fig_trend = px.line(sales_trend, x='Month', y='Sales', title="Monthly Sales Trend")
st.plotly_chart(fig_trend)

# ------------------------
# Customer Segmentation
# ------------------------
st.header("Top Customers")
top_customers = filtered_df.groupby('Customer')['Total_Purchase'].sum().sort_values(ascending=False).head(10).reset_index()
fig_customers = px.bar(top_customers, x='Customer', y='Total_Purchase', color='Customer', title="Top 10 Customers by Total Purchase")
st.plotly_chart(fig_customers)

# ------------------------
# Economic Insights
# ------------------------
st.header("Economic Indicators by Region")
st.dataframe(df[['Region','GDP_Growth','Population_Million']].drop_duplicates())

# ------------------------
# Predictive Analytics: Sales Forecast
# ------------------------

