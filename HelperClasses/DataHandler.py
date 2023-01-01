import pandas as pd
from datetime import datetime, timedelta
class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_data(self):
        # read the data from the CSV file into a DataFrame
        df = pd.read_csv(self.file_path)
        return df

    def format_tickstory_data(self, df):
        # parse the date and timestamp strings into datetime objects
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Timestamp'])

        # create a reference datetime to determine the amount of seconds passed.
        reference = datetime(2022, 11, 29)

        # subtract the reference datetime from the DateTime column
        timedelta = df['DateTime'] - reference

        # convert the timedelta to seconds
        seconds = timedelta.dt.total_seconds()

        # add the seconds to the DataFrame
        df['Seconds'] = seconds

        #Drop NaN values
        df = df.dropna()

        return df
