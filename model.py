import pandas as pd
from scipy.stats import tmean, tstd
import numpy as np  # Replace numpy.arange
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# Load Dataset
file_path = 'cleaned_sales_data.xlsx'
data = pd.read_excel(file_path)

# Preprocessing
data['purchase-date'] = pd.to_datetime(data['purchase-date'])
data.fillna(0, inplace=True)  # Example handling missing values
data['total-tax'] = data['shipping-tax'] + data['item-tax'] + data['gift-wrap-tax']
data['revenue'] = (
    data['item-price'] + data['shipping-price'] + data['gift-wrap-price'] - data['item-promotion-discount']
)
daily_revenue = data.groupby('purchase-date')['revenue'].sum().reset_index()

daily_revenue['day_of_week'] = daily_revenue['purchase-date'].dt.day_name()
daily_revenue['ordinal_date'] = daily_revenue['purchase-date'].map(pd.Timestamp.toordinal)

# Prepare data
X = daily_revenue[['ordinal_date']]
y = daily_revenue['revenue']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict for the next 30 days
future_dates = np.arange(X['ordinal_date'].max() + 1, X['ordinal_date'].max() + 31).reshape(-1, 1)
future_dates_actual = pd.to_datetime(future_dates.ravel(), origin='1899-12-30', unit='D', errors='coerce')

# Drop any invalid timestamps if they still exist (unlikely after correction)
future_dates_actual = future_dates_actual.dropna()
future_revenue = model.predict(future_dates)
print("Future Revenue: ", future_revenue)

popular_products = data['sku'].value_counts().reset_index()
popular_products.columns = ['product', 'count']
print(popular_products.head())


# Plot improved sales forecasting visualization
sns.set_theme(style="whitegrid")

plt.figure(figsize=(14, 8))

# Plot observed revenue (historical data)
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

# Plot forecasted revenue (predicted data for the next 30 days)
plt.plot(
    future_dates_actual,
    future_revenue,
    label="Forecasted Revenue (Next 30 Days)",
    marker='x',
    color='red',
    linewidth=2,
    markersize=6,
    linestyle='-.'
)

# Highlight the transition point between observed and forecasted data
transition_date = daily_revenue['purchase-date'].max()
plt.axvline(transition_date, color='green', linestyle='--', linewidth=1.5, label="Forecast Start")

# Add labels, title, and legend
plt.title("Sales Forecasting: Observed vs. Predicted Revenue", fontsize=18)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Revenue ($)", fontsize=14)
plt.legend(fontsize=12, loc="upper left")

# Enhance tick formatting for dates and revenue
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)

# Add a grid for clarity
plt.grid(color='gray', linestyle=':', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()




























































































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
