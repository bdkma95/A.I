class DataProcessor:
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean the data."""
        data['xg_contribution'] = data['expected_goals'] + data['expected_assists']
        data['pass_accuracy'] = data['successful_passes'] / data['total_passes']
        data.fillna(0, inplace=True)
        return data
