import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def get_data(file_name: str) -> pd.DataFrame:
    """
    Reads data from csv file and returns it as a pandas DataFrame.
    :param file_name: name of the csv file
    :return: pandas DataFrame
    
    Fills NaNs.
    Creates additional features.
    """
    df = pd.read_csv(file_name, sep=';', parse_dates = ['Time'])
    df.columns = [q_name.replace(' ', '_').lower() for q_name in df.columns]
    df['time'] = pd.to_datetime(df['time'], utc=True)

    df['summer_time'] = df['time'].dt.tz_convert("Europe/Bratislava").dt.hour - df['time'].dt.hour - 1
    df['summer_time'] = df['summer_time'].apply(lambda x: 0 if x == -24  else (1 if x == -23 else x))  # -> only values 0/1

    df['time'] = df['time'].dt.tz_localize(None)
    df.set_index('time', inplace=True)
    df.loc[:, 'date'] = df.index.date

    df['battery_power'] = df['battery_charging'] - df['battery_discharging']
    df['grid_power'] = df['grid_consumption'] - df['grid_backflow']

    df.loc[:, 'consumption'] = df.loc[:, 'consumption'].fillna((df.loc[:, 'consumption'].shift(48) + df.loc[:, 'consumption'].shift(-48)) / 2)
    df.loc[:, 'pv_generation'] = df.loc[:, 'pv_generation'].fillna((df.loc[:, 'pv_generation'].shift(48) + df.loc[:, 'pv_generation'].shift(-48)) / 2)
    df.loc[:, 'battery_power'] = df.loc[:, 'battery_power'].fillna((df.loc[:, 'battery_power'].shift(48) + df.loc[:, 'battery_power'].shift(-48)) / 2)
    df['grid_power'] = df['grid_power'].fillna((df['consumption'] - df['pv_generation'] + df['battery_power']))

    df['period_of_day'] = df.groupby(['date'])[df.columns.to_list()].cumcount()
    df['batt_power_cumsum'] = df['battery_power'].cumsum()

    df.loc[:, 'grid_backflow'] = df.loc[:, 'grid_backflow'].fillna(0)
    df.loc[:, 'grid_consumption'] = df.loc[:, 'grid_consumption'].fillna(df.loc[:, 'grid_power'] - df.loc[:, 'grid_backflow']) 

    # fillna day rows
    df.loc[df['period_of_day'] >= 10, 'battery_charging'] = df.loc[df['period_of_day'] >= 10, 'battery_charging'].fillna(0)
    df.loc[df['period_of_day'] >= 10, 'battery_discharging'] = df.loc[df['period_of_day'] >= 10, 'battery_discharging'].fillna(df['battery_power'] - df['battery_charging'])
    # fillna night rows
    df.loc[df['period_of_day'] < 10, 'battery_discharging'] = df.loc[df['period_of_day'] < 10, 'battery_discharging'].fillna(0)
    df.loc[df['period_of_day'] < 10, 'battery_charging'] = df.loc[df['period_of_day'] < 10, 'battery_charging'].fillna(df['battery_power'] - df['battery_discharging'])
    
    # remove the trend from battery_power_cumulative_sum
    num_range = np.arange(len(df['batt_power_cumsum'])).reshape(-1, 1)
    batt_power_cumsum = df['batt_power_cumsum']

    m = LinearRegression()
    m.fit(num_range, batt_power_cumsum)
    df['batt_p_cumsum_detrend'] = df['batt_power_cumsum'] - (m.coef_ * num_range.reshape(-1))
    
    # time lagging features
    df['batt_p_cumsum_detrend_shift1'] = df['batt_p_cumsum_detrend'].shift(1) # available capacity to charge/discharge

    df['pv_generation_shift1'] = df['pv_generation'].shift(1)
    df['consumption_shift1'] = df['consumption'].shift(1)
    df['grid_power_shift1'] = df['grid_power'].shift(1)

    df['consumption_shift48'] = df['consumption'].shift(48)
    df['pv_generation_shift48'] = df['pv_generation'].shift(48)

    df = df.dropna()
    
    return df

