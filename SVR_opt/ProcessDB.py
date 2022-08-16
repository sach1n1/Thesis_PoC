import pandas as pd
from datetime import datetime
import sqlite3 as db


class ProcessDB:
    def __init__(self, db_path, forecast_dt, training_duration):
        self.database = db_path
        self.forecast_start_dt = forecast_dt
        self.duration = training_duration
        self.train, self.test = self.load_required_value()

    def load_required_value(self):
        train_start_dt = str(pd.Timestamp(self.forecast_start_dt) - pd.DateOffset(hours=self.duration, seconds=4))
        forecast_end_dt = str(pd.Timestamp(self.forecast_start_dt) + pd.DateOffset(hours=1))
        df_train = self.create_data_frame(train_start_dt, self.forecast_start_dt)
        forecast_start_dt = str(pd.Timestamp(self.forecast_start_dt) - pd.DateOffset(seconds=4))
        df_test = self.create_data_frame(forecast_start_dt, forecast_end_dt)
        return df_train, df_test

    def create_data_frame(self, start_date, end_date):
        con = db.connect(self.database)
        start_dt_utc = datetime.timestamp(datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')) * 1000
        end_dt_utc = datetime.timestamp(datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')) * 1000
        df = pd.read_sql_query(f"SELECT time, value FROM Value WHERE sensor_id=2 AND "
                               f"time >= '{int(start_dt_utc)}' AND time < '{int(end_dt_utc)}'",
                               con)
        df["time"] = df["time"].apply(lambda utc: datetime.fromtimestamp(int(utc / 1000)))
        df.drop_duplicates(subset="time", keep="first", inplace=True)
        df.index = df['time']

        df = df.reindex(pd.date_range(start_date,
                                      end_date,
                                      freq='S'))
        df.drop('time', axis=1, inplace=True)
        df = df.diff()
        df = df.interpolate().fillna(method='bfill')
        df.drop(df.tail(1).index, inplace=True)
        con.close()
        return df
