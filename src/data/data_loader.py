import pandas as pd


class SalesDataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_and_preprocess(self) -> pd.DataFrame:
        """Load and preprocess the sales data"""
        data = pd.read_excel(self.file_path)
        data['purchase-date'] = pd.to_datetime(data['purchase-date'])
        data.fillna(0, inplace=True)
        return self._calculate_metrics(data)

    def _calculate_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate necessary metrics for analysis"""
        data['total-tax'] = data['shipping-tax'] + data['item-tax'] + data['gift-wrap-tax']
        data['revenue'] = (
                data['item-price'] +
                data['shipping-price'] +
                data['gift-wrap-price'] -
                data['item-promotion-discount']
        )
        return data

    def get_daily_revenue(self) -> pd.DataFrame:
        """Get daily revenue aggregation"""
        data = self.load_and_preprocess()
        daily_revenue = data.groupby('purchase-date')['revenue'].sum().reset_index()
        daily_revenue['day_of_week'] = daily_revenue['purchase-date'].dt.day_name()
        return daily_revenue

    def prepare_for_segmentation(self) -> pd.DataFrame:
        """Prepare data for product segmentation"""
        data = self.load_and_preprocess()
        # Add segmentation-specific preprocessing here
        return data

