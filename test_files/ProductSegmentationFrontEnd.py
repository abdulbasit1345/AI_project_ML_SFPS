import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Mock function to simulate backend data retrieval
# Replace this function with the actual backend API call or data retrieval function
def fetch_segmentation_data():
    # Example dataset from backend
    data = pd.DataFrame({
        'product': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
        'segment': ['Electronics', 'Clothing', 'Electronics', 'Books', 'Clothing'],
        'sales': [15000, 12000, 18000, 8000, 10000]
    })
    return data

# Streamlit front-end for Enhanced Product Segmentation
def main():
    # Page title with emoji
    st.title("🌟 Product Segmentation Insights 🎯")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    selected_option = st.sidebar.radio(
        "Choose an Option:", 
        options=["Overview", "Visualization", "Detailed Analysis"]
    )

    try:
        # Fetch data from the backend
        data = fetch_segmentation_data()
        
        if selected_option == "Overview":
            st.subheader("Overview of the Dataset 📊")
            st.write("Explore the first few rows of the data:")
            st.dataframe(data.head())

            # Display dataset information
            st.write("### Dataset Information:")
            st.write(f"**Number of Rows:** {data.shape[0]}")
            st.write(f"**Number of Columns:** {data.shape[1]}")
            st.write("### Sample Columns:")
            st.write(", ".join(data.columns))

        elif selected_option == "Visualization":
            st.subheader("Interactive Product Segmentation Visualization 🌈")
            # Group data for segmentation analysis
            segment_summary = data.groupby('segment')['sales'].sum().reset_index()

            # Create an animated bar chart using Plotly
            fig = px.bar(
                segment_summary, 
                x='segment', 
                y='sales', 
                text='sales', 
                title="Product Sales by Segment (Animated)",
                labels={'sales': 'Sales ($)', 'segment': 'Product Segment'},
                color='segment',
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            fig.update_layout(
                title_font_size=22,
                xaxis_title_font_size=18,
                yaxis_title_font_size=18,
                font=dict(size=14)
            )

            # Display the chart
            st.plotly_chart(fig)

            # Pie chart for proportion of sales by segment
            st.subheader("Proportion of Sales by Segment 🥧")
            fig_pie = px.pie(
                segment_summary, 
                values='sales', 
                names='segment', 
                title="Sales Distribution by Segment",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_pie)

        elif selected_option == "Detailed Analysis":
            st.subheader("Detailed Product Analysis by Segment 🔍")
            selected_segment = st.selectbox(
                "Choose a Segment to Analyze:", 
                options=data['segment'].unique()
            )

            # Filter and display detailed data
            filtered_data = data[data['segment'] == selected_segment]
            st.write(f"Products in Segment '{selected_segment}':")
            st.dataframe(filtered_data)

            # Add histogram for product sales in the selected segment
            st.subheader(f"Sales Distribution in Segment '{selected_segment}' 📈")
            fig_hist = px.histogram(
                filtered_data, 
                x='sales', 
                nbins=10, 
                title=f"Sales Distribution in Segment '{selected_segment}'",
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig_hist)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Run the application
if __name__ == '__main__':
    main()
