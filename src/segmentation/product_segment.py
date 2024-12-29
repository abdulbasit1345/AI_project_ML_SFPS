# src/segmentation/product_segment.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ProductSegmentation:
    def __init__(self, data):
        """
        Initialize ProductSegmentation with sales data

        Parameters:
        data (pd.DataFrame): DataFrame containing sales data with columns:
            - sku
            - revenue
            - quantity
            - item-price
            - item-promotion-discount
        """
        self.data = data  # Store the full dataset, not daily_revenue
        self.scaler = StandardScaler()

    def calculate_product_metrics(self) -> pd.DataFrame:
        """Calculate metrics for each product"""
        try:
            product_metrics = self.data.groupby('sku').agg({
                'revenue': ['sum', 'mean'],
                'quantity': 'sum',
                'item-price': 'mean',  # Changed from 'sum' to 'mean'
                'item-promotion-discount': 'mean'
            }).reset_index()

            # Flatten column names
            product_metrics.columns = [
                'sku',
                'total_revenue',
                'avg_revenue_per_sale',
                'total_quantity',
                'avg_price',  # Changed from total_sales
                'avg_discount'
            ]

            return product_metrics
        except KeyError as e:
            raise KeyError(f"Missing required column in data: {e}")

    def segment_products(self, n_clusters: int = 4) -> pd.DataFrame:
        """
        Perform product segmentation using KMeans

        Parameters:
        n_clusters (int): Number of clusters to create

        Returns:
        pd.DataFrame: Product metrics with segment labels
        """
        try:
            metrics = self.calculate_product_metrics()

            # Select features for clustering
            features = [
                'total_revenue',
                'avg_revenue_per_sale',
                'total_quantity',
                'avg_price',
                'avg_discount'
            ]

            # Check for missing values
            if metrics[features].isnull().any().any():
                metrics[features] = metrics[features].fillna(0)

            # Scale features
            X = self.scaler.fit_transform(metrics[features])

            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            metrics['segment'] = kmeans.fit_predict(X)

            # Add segment characteristics
            metrics['segment_label'] = metrics['segment'].map({
                0: 'High Value Products',
                1: 'Medium Value Products',
                2: 'Phase Out Products',
                3: 'Promotional'
            })

            return metrics
        except Exception as e:
            raise Exception(f"Error in segmentation process: {e}")