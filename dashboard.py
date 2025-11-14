import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
sales_region = filtered_df.groupby('Region')['Sales'].sum()
fig, ax = plt.subplots()
sales_region.plot(kind='bar', ax=ax, color='skyblue')
ax.set_ylabel("Sales")
ax.set_title("Total Sales by Region")
st.pyplot(fig)

# Sales Distribution by Category
sales_category = filtered_df.groupby('Category')['Sales'].sum()
fig, ax = plt.subplots()
sales_category.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90, legend=False)
ax.set_ylabel('')
ax.set_title("Sales Distribution by Category")
st.pyplot(fig)

# Sales Distribution by Channel
sales_channel = filtered_df.groupby('Channel')['Sales'].sum()
fig, ax = plt.subplots()
sales_channel.plot(kind='bar', ax=ax, color='lightgreen')
ax.set_ylabel("Sales")
ax.set_title("Sales Distribution by Channel")
st.pyplot(fig)

# ------------------------
# Monthly Sales Trend
# ------------------------
st.header("Monthly Sales Trend")
sales_trend = filtered_df.groupby('Month')['Sales'].sum()
fig, ax = plt.subplots()
sales_trend.plot(kind='line', marker='o', ax=ax, color='orange')
ax.set_ylabel("Sales")
ax.set_title("Monthly Sales Trend")
st.pyplot(fig)

# ------------------------
# Customer Segmentation
# ------------------------
st.header("Top Customers")
top_customers = filtered_df.groupby('Customer')['Total_Purchase'].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
top_customers.plot(kind='bar', ax=ax, color='purple')
ax.set_ylabel("Total Purchase")
ax.set_title("Top 10 Customers by Total Purchase")
st.pyplot(fig)

# ------------------------
# Economic Indicators
# ------------------------
st.header("Economic Indicators by Region")
st.dataframe(df[['Region','GDP_Growth','Population_Million']].drop_duplicates())

# ------------------------
# Predictive Analytics: Sales Forecast
# ------------------------
st.header("Sales Forecast for Next 3 Months")
monthly_sales = filtered_df.groupby('Month')['Sales'].sum()

# Fit Exponential Smoothing model
model = ExponentialSmoothing(monthly_sales, seasonal='add', seasonal_periods=12).fit()
future_months = pd.date_range(start=monthly_sales.index[-1] + pd.Timedelta(days=30), periods=3, freq='M')
forecast = model.forecast(3)

# Combine actual and forecast
forecast_df = pd.DataFrame({'Month': future_months, 'Sales': forecast})
sales_forecast = pd.concat([monthly_sales.reset_index(), forecast_df])

fig, ax = plt.subplots()
ax.plot(sales_forecast['Month'], sales_forecast['Sales'], marker='o', color='red')
ax.set_ylabel("Sales")
ax.set_title("Sales Forecast (Next 3 Months)")
st.pyplot(fig)
