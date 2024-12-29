import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import streamlit as st
from src.segmentation.product_segment import ProductSegmentation
from src.visualization.segment_plots import SegmentationPlotter


def show_segmentation_page(data):
    st.title("Product Segmentation Insights")

    # Initialize segmentation instance
    segmentation = ProductSegmentation(data)
    metrics = segmentation.calculate_product_metrics()

    # Sidebar options
    st.sidebar.title("Segmentation Options")
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=4)

    # Perform segmentation
    segmented_data = segmentation.segment_products(n_clusters=n_clusters)

    # Overview section
    st.subheader("Overview of Segmented Data")
    st.dataframe(segmented_data.head())

    # Visualization section
    st.subheader("Visualizations")

    # Elbow Method Plot
    st.write("### Optimal Number of Clusters")
    st.pyplot(SegmentationPlotter.plot_elbow_method(metrics, ['total_revenue', 'total_quantity', 'total_sales', 'avg_discount']))

    # Cluster Visualization
    st.write("### Cluster Visualization")
    SegmentationPlotter.plot_clusters(segmented_data, x="total_quantity", y="total_sales", cluster_col="segment")

    # Sales Distribution Plot
    st.write("### Sales Distribution by Segment")
    SegmentationPlotter.plot_sales_distribution(segmented_data, segment_col="segment", sales_col="total_sales")
