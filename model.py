import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Load Dataset
file_path = 'cleaned_sales_data.xlsx'
data = pd.read_excel(file_path)

# Preprocessing
data['purchase-date'] = pd.to_datetime(data['purchase-date'])
data.fillna(0, inplace=True)
data['total-tax'] = data['shipping-tax'] + data['item-tax'] + data['gift-wrap-tax']
data['revenue'] = (
    data['item-price'] + data['shipping-price'] + data['gift-wrap-price'] - data['item-promotion-discount']
)
daily_revenue = data.groupby('purchase-date')['revenue'].sum().reset_index()

# Add features
daily_revenue['day_of_week'] = daily_revenue['purchase-date'].dt.day_name()

# Prepare data for model
X = daily_revenue['purchase-date'].map(datetime.toordinal).values.reshape(-1, 1)
y = daily_revenue['revenue'].values

# Train model
model = LinearRegression()
model.fit(X, y)

# Generate future dates
last_date = daily_revenue['purchase-date'].max()
future_dates_dt = [last_date + timedelta(days=x) for x in range(1, 31)]
future_dates_ordinal = np.array([d.toordinal() for d in future_dates_dt]).reshape(-1, 1)

# Make predictions
future_revenue = model.predict(future_dates_ordinal)

# Create visualization
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# Plot historical data
plt.plot(
    daily_revenue['purchase-date'],
    daily_revenue['revenue'],
    label="Observed Revenue",
    marker='o',
    color='blue',
    linewidth=2,
    markersize=6,
    linestyle='--'
)

# Plot forecast
plt.plot(
    future_dates_dt,
    future_revenue,
    label="Forecasted Revenue (Next 30 Days)",
    marker='x',
    color='red',
    linewidth=2,
    markersize=6,
    linestyle='-.'
)

# Highlight transition point
plt.axvline(last_date, color='green', linestyle='--', linewidth=1.5, label="Forecast Start")

# Formatting
plt.title("Sales Forecasting: Observed vs. Predicted Revenue", fontsize=18)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Revenue ($)", fontsize=14)
plt.legend(fontsize=12, loc="upper left")
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)

# Format dates on x-axis
plt.gcf().autofmt_xdate()  # Automatically format date labels

# Show the plot
plt.tight_layout()
plt.show()

# Print forecast results
forecast_df = pd.DataFrame({
    'Date': future_dates_dt,
    'Forecasted_Revenue': future_revenue
})
print("\nForecast for next 30 days:")
print(forecast_df.to_string(index=False))

# Print popular products
popular_products = data['sku'].value_counts().reset_index()
popular_products.columns = ['product', 'count']
print("\nTop Products:")
print(popular_products.head())



























































































# import pandas as pd
# import scipy.stats
# # methods tstd tmean
# import matplotlib.pyplot as plt
# import seaborn as sns
# from mlxtend.frequent_patterns import apriori, association_rules
# from scipy.stats import pearsonr
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.linear_model import LinearRegression
# import numpy as np
#
# # Load Dataset
# file_path = 'cleaned_sales_data.xlsx'
# data = pd.read_excel(file_path)
#
# # Preprocessing
# data['purchase-date'] = pd.to_datetime(data['purchase-date'])
# data.fillna(0, inplace=True)  # Example handling missing values
# data.head()
#
# data['total-tax'] = data['shipping-tax'] +  data['item-tax'] + data['gift-wrap-tax']
# data['revenue'] = data['item-price'] + data['shipping-price'] + data['gift-wrap-price'] - data['item-promotion-discount']
# daily_revenue = data.groupby('purchase-date')['revenue'].sum().reset_index()
#
# daily_revenue['day_of_week'] = daily_revenue['purchase-date'].dt.day_name()
# daily_revenue['ordinal_date'] = daily_revenue['purchase-date'].map(pd.Timestamp.toordinal)
#
# # Prepare data
# X = daily_revenue[['ordinal_date']]
# y = daily_revenue['revenue']
#
# # Train model
# model = LinearRegression()
# model.fit(X, y)
#
# # Predict for the next 30 days
# future_dates = np.arange(X['ordinal_date'].max() + 1, X['ordinal_date'].max() + 31).reshape(-1, 1)
# future_revenue = model.predict(future_dates)
# print("Future Revenue: ", future_revenue)
#
# popular_products = data['sku'].value_counts().reset_index()
# popular_products.columns = ['product', 'count']
# print(popular_products.head())

# Convert the dataset into a transactional format
# basket = pd.crosstab(data['asin'], data['sku'])
#
# # Apply Apriori
# frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True)
# rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
#
# print(rules.head())
