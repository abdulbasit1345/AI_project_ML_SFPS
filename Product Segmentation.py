import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
file_path = "Data.xlsx"
data = pd.read_excel(file_path, sheet_name="Sheet1")

# Data Cleaning
relevant_columns = ['product-name', 'quantity', 'item-price']
cleaned_data = data[relevant_columns].dropna()

# Ensure correct data types
cleaned_data['quantity'] = cleaned_data['quantity'].astype(int)
cleaned_data['item-price'] = cleaned_data['item-price'].astype(float)

# Aggregate data by product
aggregated_data = cleaned_data.groupby('product-name').agg(
    total_quantity=('quantity', 'sum'),
    total_sales=('item-price', 'sum')
).reset_index()

# Features for clustering
features = aggregated_data[['total_quantity', 'total_sales']]

# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the Elbow method
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Apply K-Means Clustering
optimal_k = 3  # Choose the optimal number of clusters based on the Elbow Curve
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
aggregated_data['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize Clusters
plt.figure(figsize=(8, 5))
for cluster in range(optimal_k):
    cluster_data = aggregated_data[aggregated_data['Cluster'] == cluster]
    plt.scatter(cluster_data['total_quantity'], cluster_data['total_sales'], label=f'Cluster {cluster}')
plt.title('Product Clusters')
plt.xlabel('Total Quantity Sold')
plt.ylabel('Total Sales')
plt.legend()
plt.show()

# Top and Lowest 10 Products by Total Sales
top_10_sales = aggregated_data.sort_values(by='total_sales', ascending=False).head(10)
lowest_10_sales = aggregated_data.sort_values(by='total_sales', ascending=True).head(10)

# Top and Lowest 10 Products by Quantity
top_10_quantity = aggregated_data.sort_values(by='total_quantity', ascending=False).head(10)
lowest_10_quantity = aggregated_data.sort_values(by='total_quantity', ascending=True).head(10)

# Display results
print("Top 10 Products by Total Sales:")
print(top_10_sales)

print("\nLowest 10 Products by Total Sales:")
print(lowest_10_sales)

print("\nTop 10 Products by Quantity:")
print(top_10_quantity)

print("\nLowest 10 Products by Quantity:")
print(lowest_10_quantity)