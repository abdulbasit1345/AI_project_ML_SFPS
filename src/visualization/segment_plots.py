import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px


class SegmentationPlotter:

    @staticmethod
    def plot_elbow_method(data: pd.DataFrame, features: list):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[features])

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
        plt.show()

    @staticmethod
    def plot_clusters(data: pd.DataFrame, x: str, y: str, cluster_col: str):
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=cluster_col,
            title="Cluster Visualization",
            labels={x: x.capitalize(), y: y.capitalize()},
        )
        fig.update_traces(marker=dict(size=10, opacity=0.8))
        fig.show()

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
        fig.show()