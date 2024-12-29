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

    # Scatter Plot
    # Cluster Visualization
    st.write("### Cluster Visualization")
    plotter = SegmentationPlotter()
    fig = plotter.plot_clusters(segmented_data)
    st.pyplot(fig)


# Elbow Method Plot
    st.write("### Optimal Number of Clusters")
    elbow_method = plotter.plot_elbow_method(data)
    st.pyplot(elbow_method)

    aggregated_data = data.groupby('product-name').agg(
        total_quantity=('quantity', 'sum'),
        total_sales=('item-price', 'sum')
    ).reset_index()

    # Sales Distribution Plot
    # st.write("### Sales Distribution by Segment")
    # sales_distribution_plot = SegmentationPlotter.plot_sales_distribution(segmented_data, segment_col="segment",
    #                                                                       sales_col='avg_revenue_per_sale')
    # st.pyplot(sales_distribution_plot)

    # Show segment statistics
    st.subheader("Segment Statistics")
    segment_stats = segmented_data.groupby('segment_label').agg({
        'total_revenue': 'sum',
        'total_quantity': 'sum',
        'avg_price': 'mean',
        'sku': 'count'
    }).round(2)
    segment_stats.columns = ['Total Revenue', 'Total Quantity', 'Average Price', 'Number of Products']
    st.dataframe(segment_stats)



