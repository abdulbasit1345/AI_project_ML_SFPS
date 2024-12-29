import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ProductSegmentation:

    def __init__(self, daily_revenue):
        self.daily_revenue = daily_revenue
        self.scaler = StandardScaler()

    def calculate_product_metrics(self) -> pd.DataFrame:
        product_metrics = self.data.groupby('sku').agg({
            'revenue': ['sum', 'mean'],
            'quantity': 'sum',
            'item-price': 'sum',
            'item-promotion-discount': 'mean'
        }).reset_index()

        product_metrics.columns = [
            'sku', 'total_revenue', 'total_quantity', 'total_sales', 'avg_discount'
        ]
        return product_metrics

    def segment_products(self, n_clusters: int = 4) -> pd.DataFrame:
        """Perform product segmentation using KMeans"""
        metrics = self.calculate_product_metrics()

        # Select features for clustering
        features = ['total_revenue', 'total_quantity', 'total_sales', 'avg_discount']

        # Scale features
        X = self.scaler.fit_transform(metrics[features])

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        metrics['segment'] = kmeans.fit_predict(X)

        return metrics
