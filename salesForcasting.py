import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Function to process and forecast sales
def sales_forecasting(data):
    # Preprocessing
    data['purchase-date'] = pd.to_datetime(data['purchase-date'])
    data.fillna(0, inplace=True)  # Handle missing values
    data['total-tax'] = data['shipping-tax'] + data['item-tax'] + data['gift-wrap-tax']
    data['revenue'] = (
        data['item-price'] + data['shipping-price'] + data['gift-wrap-price'] - data['item-promotion-discount']
    )
    daily_revenue = data.groupby('purchase-date')['revenue'].sum().reset_index()

    daily_revenue['day_of_week'] = daily_revenue['purchase-date'].dt.day_name()
    daily_revenue['ordinal_date'] = daily_revenue['purchase-date'].map(pd.Timestamp.toordinal)

    # Prepare data for forecasting
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

    # Check if lengths match
    future_revenue = model.predict(future_dates)

    # Ensure future_dates_actual and future_revenue have the same length
    if len(future_dates_actual) != len(future_revenue):
        # Adjust future_dates_actual to match the length of future_revenue
        min_length = min(len(future_dates_actual), len(future_revenue))
        future_dates_actual = future_dates_actual[:min_length]
        future_revenue = future_revenue[:min_length]

    return daily_revenue, future_dates_actual, future_revenue

# Streamlit front-end
def main():
    st.title("Sales Forecasting and Popular Product Insights")
    
    # Load predefined data from Data.xlsx
    try:
        # Load the data file
        data = pd.read_excel('Data.xlsx')
        
        # Display DataFrame
        st.write("Data Loaded Successfully:")
        st.dataframe(data.head())  # Display first few rows
        
        # Call forecasting function
        daily_revenue, future_dates_actual, future_revenue = sales_forecasting(data)
        
        # Display Forecasting Results
        st.subheader("Forecasted Revenue for the Next 30 Days")
        forecast_df = pd.DataFrame({'Date': future_dates_actual, 'Forecasted Revenue': future_revenue})
        st.write(forecast_df)

        # Popular Products Analysis
        popular_products = data['sku'].value_counts().reset_index()
        popular_products.columns = ['Product', 'Count']
        st.subheader("Popular Products")
        st.write(popular_products.head())

        # Sales Forecasting Visualization
        st.subheader("Sales Forecasting Visualization")
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
        st.pyplot(plt)

    except FileNotFoundError:
        st.error("The file 'Data.xlsx' was not found. Please make sure it is in the same directory as the script.")

# Run the application
if __name__ == '__main__':
    main()






# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LinearRegression

# # Function to process and forecast sales
# def sales_forecasting(data):
#     # Preprocessing
#     data['purchase-date'] = pd.to_datetime(data['purchase-date'])
#     data.fillna(0, inplace=True)  # Handle missing values
#     data['total-tax'] = data['shipping-tax'] + data['item-tax'] + data['gift-wrap-tax']
#     data['revenue'] = (
#         data['item-price'] + data['shipping-price'] + data['gift-wrap-price'] - data['item-promotion-discount']
#     )
#     daily_revenue = data.groupby('purchase-date')['revenue'].sum().reset_index()

#     daily_revenue['day_of_week'] = daily_revenue['purchase-date'].dt.day_name()
#     daily_revenue['ordinal_date'] = daily_revenue['purchase-date'].map(pd.Timestamp.toordinal)

#     # Prepare data for forecasting
#     X = daily_revenue[['ordinal_date']]
#     y = daily_revenue['revenue']

#     # Train model
#     model = LinearRegression()
#     model.fit(X, y)

#     # Predict for the next 30 days
#     future_dates = np.arange(X['ordinal_date'].max() + 1, X['ordinal_date'].max() + 31).reshape(-1, 1)
#     future_dates_actual = pd.to_datetime(future_dates.ravel(), origin='1899-12-30', unit='D', errors='coerce')

#     # Drop any invalid timestamps if they still exist (unlikely after correction)
#     future_dates_actual = future_dates_actual.dropna()
#     future_revenue = model.predict(future_dates)

#     return daily_revenue, future_dates_actual, future_revenue

# # Streamlit front-end
# def main():
#     st.title("Sales Forecasting and Popular Product Insights")
    
#     # Load predefined data from Data.xlsx
#     try:
#         # Load the data file
#         data = pd.read_excel('Data.xlsx')
        
#         # Display DataFrame
#         st.write("Data Loaded Successfully:")
#         st.dataframe(data.head())  # Display first few rows
        
#         # Call forecasting function
#         daily_revenue, future_dates_actual, future_revenue = sales_forecasting(data)
        
#         # Display Forecasting Results
#         st.subheader("Forecasted Revenue for the Next 30 Days")
#         forecast_df = pd.DataFrame({'Date': future_dates_actual, 'Forecasted Revenue': future_revenue})
#         st.write(forecast_df)

#         # Popular Products Analysis
#         popular_products = data['sku'].value_counts().reset_index()
#         popular_products.columns = ['Product', 'Count']
#         st.subheader("Popular Products")
#         st.write(popular_products.head())

#         # Sales Forecasting Visualization
#         st.subheader("Sales Forecasting Visualization")
#         sns.set_theme(style="whitegrid")

#         plt.figure(figsize=(14, 8))

#         # Plot observed revenue (historical data)
#         plt.plot(
#             daily_revenue['purchase-date'],
#             daily_revenue['revenue'],
#             label="Observed Revenue",
#             marker='o',
#             color='blue',
#             linewidth=2,
#             markersize=6,
#             linestyle='--'
#         )

#         # Plot forecasted revenue (predicted data for the next 30 days)
#         plt.plot(
#             future_dates_actual,
#             future_revenue,
#             label="Forecasted Revenue (Next 30 Days)",
#             marker='x',
#             color='red',
#             linewidth=2,
#             markersize=6,
#             linestyle='-.'
#         )

#         # Highlight the transition point between observed and forecasted data
#         transition_date = daily_revenue['purchase-date'].max()
#         plt.axvline(transition_date, color='green', linestyle='--', linewidth=1.5, label="Forecast Start")

#         # Add labels, title, and legend
#         plt.title("Sales Forecasting: Observed vs. Predicted Revenue", fontsize=18)
#         plt.xlabel("Date", fontsize=14)
#         plt.ylabel("Revenue ($)", fontsize=14)
#         plt.legend(fontsize=12, loc="upper left")

#         # Enhance tick formatting for dates and revenue
#         plt.xticks(fontsize=12, rotation=45)
#         plt.yticks(fontsize=12)

#         # Add a grid for clarity
#         plt.grid(color='gray', linestyle=':', linewidth=0.5)

#         # Show the plot
#         plt.tight_layout()
#         st.pyplot(plt)

#     except FileNotFoundError:
#         st.error("The file 'Data.xlsx' was not found. Please make sure it is in the same directory as the script.")

# # Run the application
# if __name__ == '__main__':
#     main()










































# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# import matplotlib.pyplot as plt

# # Function for sales forecasting
# def sales_forecasting(data):
#     try:
#         # Ensure the dataset has 'Date' and 'Sales' columns
#         if 'Date' not in data.columns or 'Sales' not in data.columns:
#             st.error("Dataset must contain 'Date' and 'Sales' columns.")
#             return None, None

#         # Convert 'Date' column to datetime and set as index
#         data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
#         data = data.dropna(subset=['Date'])  # Remove rows with invalid dates
#         data = data.set_index('Date')

#         # Ensure 'Sales' column is numeric
#         data['Sales'] = pd.to_numeric(data['Sales'], errors='coerce')
#         data = data.dropna(subset=['Sales'])  # Remove rows with invalid sales data

#         # Feature Engineering
#         data['Month'] = data.index.month
#         data['Year'] = data.index.year

#         # Prepare Data for Training
#         X = data[['Month', 'Year']]
#         y = data['Sales']
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Train Model
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#         model.fit(X_train, y_train)

#         # Forecast Future Sales
#         future = pd.DataFrame({'Month': [1, 2, 3], 'Year': [2025, 2025, 2025]})
#         predictions = model.predict(future)

#         # Plot Results
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(data.index, data['Sales'], label='Historical Sales', color='blue')
#         ax.plot(pd.date_range('2025-01-01', periods=3, freq='M'), predictions, label='Forecast', color='red')
#         ax.set_xlabel('Date')
#         ax.set_ylabel('Sales')
#         ax.set_title('Sales Forecasting')
#         ax.legend()

#         return predictions, fig

#     except Exception as e:
#         st.error(f"An error occurred: {e}")
#         return None, None


# # Streamlit front end
# def main():
#     st.title("Sales Forecasting and Product Segmentation")
#     st.sidebar.header("Navigation")
    
#     # Sidebar menu
#     menu = ["Sales Forecasting", "Product Segmentation"]
#     choice = st.sidebar.selectbox("Select an Option", menu)

#     if choice == "Sales Forecasting":
#         st.header("Upload Sales Data for Forecasting")
#         sales_file = st.file_uploader("Upload CSV File", type=['csv'])

#         if sales_file:
#             try:
#                 data = pd.read_csv(sales_file)
#                 predictions, fig = sales_forecasting(data)

#                 if predictions is not None and fig is not None:
#                     st.subheader("Forecast Results")
#                     st.write("Predicted Sales for Future Periods:")
#                     st.write(predictions)
#                     st.pyplot(fig)
#             except Exception as e:
#                 st.error(f"An error occurred while processing the file: {e}")

#     elif choice == "Product Segmentation":
#         st.header("Upload Product Data for Segmentation")
#         product_file = st.file_uploader("Upload CSV File", type=['csv'])

#         if product_file:
#             st.error("The 'product_segmentation' function is not defined yet.")
#             st.info("Stay tuned for updates on this feature!")


# if __name__ == '__main__':
#     main()
