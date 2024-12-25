import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Function for sales forecasting
def sales_forecasting(data):
    try:
        # Ensure the dataset has 'Date' and 'Sales' columns
        if 'Date' not in data.columns or 'Sales' not in data.columns:
            st.error("Dataset must contain 'Date' and 'Sales' columns.")
            return None, None

        # Convert 'Date' column to datetime and set as index
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.dropna(subset=['Date'])  # Remove rows with invalid dates
        data = data.set_index('Date')

        # Ensure 'Sales' column is numeric
        data['Sales'] = pd.to_numeric(data['Sales'], errors='coerce')
        data = data.dropna(subset=['Sales'])  # Remove rows with invalid sales data

        # Feature Engineering
        data['Month'] = data.index.month
        data['Year'] = data.index.year

        # Prepare Data for Training
        X = data[['Month', 'Year']]
        y = data['Sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Forecast Future Sales
        future = pd.DataFrame({'Month': [1, 2, 3], 'Year': [2025, 2025, 2025]})
        predictions = model.predict(future)

        # Plot Results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data['Sales'], label='Historical Sales', color='blue')
        ax.plot(pd.date_range('2025-01-01', periods=3, freq='M'), predictions, label='Forecast', color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.set_title('Sales Forecasting')
        ax.legend()

        return predictions, fig

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None


# Streamlit front end
def main():
    st.title("Sales Forecasting and Product Segmentation")
    st.sidebar.header("Navigation")
    
    # Sidebar menu
    menu = ["Sales Forecasting", "Product Segmentation"]
    choice = st.sidebar.selectbox("Select an Option", menu)

    if choice == "Sales Forecasting":
        st.header("Upload Sales Data for Forecasting")
        sales_file = st.file_uploader("Upload CSV File", type=['csv'])

        if sales_file:
            try:
                data = pd.read_csv(sales_file)
                predictions, fig = sales_forecasting(data)

                if predictions is not None and fig is not None:
                    st.subheader("Forecast Results")
                    st.write("Predicted Sales for Future Periods:")
                    st.write(predictions)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")

    elif choice == "Product Segmentation":
        st.header("Upload Product Data for Segmentation")
        product_file = st.file_uploader("Upload CSV File", type=['csv'])

        if product_file:
            st.error("The 'product_segmentation' function is not defined yet.")
            st.info("Stay tuned for updates on this feature!")


if __name__ == '__main__':
    main()
