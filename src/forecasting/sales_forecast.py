import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression


class SalesForecaster:
    def __init__(self, daily_revenue):
        self.daily_revenue = daily_revenue
        self.model = LinearRegression()

    def prepare_data(self):
        """Prepare data for the model"""
        X = self.daily_revenue['purchase-date'].map(datetime.toordinal).values.reshape(-1, 1)
        y = self.daily_revenue['revenue'].values
        return X, y

    def generate_forecast(self, days=30):
        """Generate forecast for specified number of days"""
        # Prepare and fit model
        X, y = self.prepare_data()
        self.model.fit(X, y)

        # Generate future dates
        last_date = self.daily_revenue['purchase-date'].max()
        future_dates_dt = [last_date + timedelta(days=x) for x in range(1, days + 1)]
        future_dates_ordinal = np.array([d.toordinal() for d in future_dates_dt]).reshape(-1, 1)

        # Generate predictions
        future_revenue = self.model.predict(future_dates_ordinal)

        return future_dates_dt, future_revenue, last_date

    def create_forecast_dataframe(self, future_dates_dt, future_revenue):
        """Create DataFrame with forecast results"""
        return pd.DataFrame({
            'Date': future_dates_dt,
            'Forecasted Revenue': future_revenue
        })
