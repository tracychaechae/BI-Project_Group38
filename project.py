# BI-Project_Group38

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, classification_report
import os

# --- 1. SET PATHS ---
input_path = r"C:\Users\TracyLam\Desktop\COMP7810\dataset"
output_path = r"C:\Users\TracyLam\Desktop\COMP7810\python\final output"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# --- 2. DATA LOADING ---
print("Loading data...")
orders = pd.read_csv(os.path.join(input_path, 'Fecom Inc Orders.csv'), sep=';')
items = pd.read_csv(os.path.join(input_path, 'Fecom Inc Order Items.csv'), sep=';')
customers = pd.read_csv(os.path.join(input_path, 'Fecom Inc Customer List.csv'), sep=';')
reviews = pd.read_csv(os.path.join(input_path, 'Fecom_Inc_Order_Reviews_No_Emojis.csv'), sep=';')

# --- 3. DATA CLEANING & FEATURE ENGINEERING ---
print("Processing features...")
orders['Order_Purchase_Timestamp'] = pd.to_datetime(orders['Order_Purchase_Timestamp'])
orders['Order_Delivered_Customer_Date'] = pd.to_datetime(orders['Order_Delivered_Customer_Date'])
orders['Order_Estimated_Delivery_Date'] = pd.to_datetime(orders['Order_Estimated_Delivery_Date'])

# Logistics features
orders['shipping_duration'] = (orders['Order_Delivered_Customer_Date'] - orders['Order_Purchase_Timestamp']).dt.days
orders['delivery_delay'] = (orders['Order_Delivered_Customer_Date'] - orders['Order_Estimated_Delivery_Date']).dt.days

# Fill missing data
orders['shipping_duration'] = orders['shipping_duration'].fillna(orders['shipping_duration'].median())
orders['delivery_delay'] = orders['delivery_delay'].fillna(orders['delivery_delay'].median())

# Aggregate Items (Revenue & Volume)
order_items_agg = items.groupby('Order_ID').agg({
    'Price': 'sum',
    'Freight_Value': 'sum',
    'Order_Item_ID': 'count'
}).rename(columns={'Order_Item_ID': 'item_count', 'Price': 'revenue'}).reset_index()

# Merge Master Dataframe
df = orders.merge(order_items_agg, on='Order_ID', how='inner')
df = df.merge(customers[['Customer_Trx_ID', 'Subscriber_ID', 'Age']], on='Customer_Trx_ID', how='left')
df = df.merge(reviews[['Order_ID', 'Review_Score']], on='Order_ID', how='left')
df['Review_Score'] = df['Review_Score'].fillna(df['Review_Score'].median())

# Historical Totals
rev_2023 = df[df['Order_Purchase_Timestamp'].dt.year == 2023]['revenue'].sum()
rev_2024 = df[df['Order_Purchase_Timestamp'].dt.year == 2024]['revenue'].sum()

# --- 4. CORRELATION HEATMAP (ADJUSTED LABELS) ---
print("Generating Heatmap...")
corr_vars = df[['revenue', 'Freight_Value', 'item_count', 'shipping_duration', 'delivery_delay', 'Age', 'Review_Score']]
plt.figure(figsize=(14, 10))
sns.heatmap(corr_vars.corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Final Correlation Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'correlation_heatmap_final.png'))

# --- 5. MLR REVENUE FORECAST (JAN 2023 ONWARDS) ---
print("Building MLR Forecast Model...")
df['month_year'] = df['Order_Purchase_Timestamp'].dt.to_period('M')
monthly = df.groupby('month_year').agg({
    'revenue': 'sum',
    'Freight_Value': 'mean',
    'item_count': 'mean'
}).reset_index()

monthly['timestamp'] = monthly['month_year'].dt.to_timestamp()
monthly_filtered = monthly[monthly['timestamp'] >= '2023-01-01'].copy()
monthly_filtered['month_index'] = np.arange(len(monthly_filtered))

# Model with Trend (month_index) + High Correlation Drivers
features = ['month_index', 'Freight_Value', 'item_count']
X_mlr = monthly_filtered[features]
y_mlr = monthly_filtered['revenue']

mlr = LinearRegression().fit(X_mlr, y_mlr)
mlr_r2 = r2_score(y_mlr, mlr.predict(X_mlr))

# 12-Month 2025 Forecast
last_month = monthly_filtered['timestamp'].max()
forecast_len = 12
future_indices = np.arange(len(monthly_filtered), len(monthly_filtered) + forecast_len)
future_X = pd.DataFrame({
    'month_index': future_indices,
    'Freight_Value': [monthly_filtered['Freight_Value'].mean()] * forecast_len,
    'item_count': [monthly_filtered['item_count'].mean()] * forecast_len
})
forecast_values = mlr.predict(future_X)
forecast_dates = pd.date_range(start=last_month + pd.DateOffset(months=1), periods=forecast_len, freq='MS')

# Calculate Forecast Totals
forecast_df = pd.DataFrame({'date': forecast_dates, 'forecast_rev': forecast_values})
rev_2025_forecast = forecast_df[forecast_df['date'].dt.year == 2025]['forecast_rev'].sum()

# Labeled Forecast Plot
plt.figure(figsize=(16, 9))
plt.plot(monthly_filtered['timestamp'], monthly_filtered['revenue'], marker='o', label=f'Actual (R²={mlr_r2:.3f})', color='royalblue')
for i, val in enumerate(monthly_filtered['revenue']):
    plt.text(monthly_filtered['timestamp'].iloc[i], val, f'${val/1e6:.1f}M', ha='center', va='bottom', fontsize=9)

plt.plot(forecast_dates, forecast_values, marker='x', linestyle='--', color='crimson', label='Forecast')
for i, val in enumerate(forecast_values):
    plt.text(forecast_dates[i], val, f'${val/1e6:.1f}M', ha='center', va='bottom', color='crimson', fontsize=9, fontweight='bold')

plt.title('Monthly Revenue Forecast (Trend + Logistics Drivers)', fontsize=16)
plt.ylabel('Revenue ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_path, 'revenue_forecast_labeled.png'))

# --- 6. CHURN ANALYSIS & EXPORTS ---
print("Running Churn Analysis...")
cust_stats = df.groupby('Subscriber_ID').agg({
    'Order_Purchase_Timestamp': ['count', 'max'],
    #'revenue': 'sum',
    'Freight_Value': 'sum',
    'shipping_duration': 'mean',
    'delivery_delay': 'mean',
    'Age': 'first'
})
cust_stats.columns = ['order_count', 'last_purchase',  'total_freight', 'avg_shipping', 'avg_delay', 'age']
cust_stats = cust_stats.reset_index()

# Churn logic: Repeat buyers who did not return in 2024
repeat_buyers = cust_stats[cust_stats['order_count'] > 1].copy()
repeat_buyers['is_churned'] = (repeat_buyers['last_purchase'].dt.year < 2024).astype(int)

X_churn = repeat_buyers[['order_count',  'total_freight', 'avg_shipping', 'avg_delay', 'age']]
y_churn = repeat_buyers['is_churned']
X_train, X_test, y_train, y_test = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# --- 7. FINAL OUTPUTS ---
print("\n" + "="*30)
print(f"Total Revenue 2023: ${rev_2023:,.2f}")
print(f"Total Revenue 2024: ${rev_2024:,.2f}")
print(f"Total Revenue 2025 (Forecast): ${rev_2025_forecast:,.2f}")
print(f"MLR R-squared: {mlr_r2:.4f}")
print(f"RF Accuracy: {rf_acc:.4f}")
print("="*30)

# Export Churn Drivers and Validation Matrix
importance = pd.DataFrame({'Feature': X_churn.columns, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
importance.to_csv(os.path.join(output_path, 'churn_drivers_importance.csv'), index=False)
monthly_filtered.to_csv(os.path.join(output_path, 'validation_matrix.csv'), index=False)

print(f"\nAll charts and data files saved to: {output_path}")
