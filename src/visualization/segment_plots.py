import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px


class SegmentationPlotter:

    @staticmethod
    def plot_elbow_method(data: pd.DataFrame):
        aggregated_data = data.groupby('product-name').agg(
            total_quantity=('quantity', 'sum'),
            total_sales=('item-price', 'sum')
        ).reset_index()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(aggregated_data[['total_quantity', 'total_sales']])

        inertia = []
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_features)
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, 10), inertia, marker='o', linestyle='--')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')

        return plt

    @staticmethod
    def plot_clusters(segments):
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.scatterplot(
            data = segments,
            x='total_revenue',
            y='total_quantity',
            hue='segment',
            style='segment',
            s=100,
            ax=ax
        )
        # Formatting
        plt.title('Product Segments by Revenue and Quantity', fontsize=14)
        plt.xlabel('Total Revenue', fontsize=12)
        plt.ylabel('Total Quantity', fontsize=12)

        # Adjust layout
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_sales_distribution(data: pd.DataFrame, segment_col: str, sales_col: str):
        segment_summary = data.groupby(segment_col)[sales_col].sum().reset_index()

        fig = px.bar(
            segment_summary,
            x=segment_col,
            y=sales_col,
            text=sales_col,
            title="Sales Distribution by Segment",
            color=segment_col
        )
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        return fig
