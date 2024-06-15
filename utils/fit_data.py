from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


class LGBMModel:
    """"""
    def __init__(self, df_train, df_test, quantity_name):
        """"""
        if quantity_name == 'consumption':
            self.X_train, self.X_test = [df[['period_of_day', 'pv_generation_shift1', 'consumption_shift1', 'consumption_shift48', 'summer_time']] for df in (df_train, df_test)]
            self.y_train, self.y_test = [df[['consumption']] for df in (df_train, df_test)]
        elif quantity_name == 'pv_generation':
            self.X_train, self.X_test = [df[['period_of_day', 'pv_generation_shift1', 'pv_generation_shift48']] for df in (df_train, df_test)]
            self.y_train, self.y_test = [df[['pv_generation']] for df in (df_train, df_test)]
        elif quantity_name == 'battery_charging':
            self.X_train, self.X_test = [df[['period_of_day', 'batt_p_cumsum_detrend_shift1', 'pv_generation_shift1', 'consumption_shift1', 'summer_time']] for df in (df_train, df_test)]
            self.y_train, self.y_test = [df[['battery_charging']] for df in (df_train, df_test)]
        elif quantity_name == 'battery_discharging':
            self.X_train, self.X_test = [df[['period_of_day', 'batt_p_cumsum_detrend_shift1', 'pv_generation_shift1', 'consumption_shift1', 'summer_time']] for df in (df_train, df_test)]
            self.y_train, self.y_test = [df[['battery_discharging']] for df in (df_train, df_test)]
        elif quantity_name == 'grid_consumption':
            self.X_train, self.X_test = [df[['period_of_day', 'batt_p_cumsum_detrend_shift1', 'grid_power_shift1', 'pv_generation_shift1', 'summer_time']] for df in (df_train, df_test)]
            self.y_train, self.y_test = [df[['grid_consumption']] for df in (df_train, df_test)]
        elif quantity_name == 'grid_backflow':
            self.X_train, self.X_test = [df[['period_of_day', 'batt_p_cumsum_detrend_shift1', 'grid_power_shift1', 'pv_generation_shift1', 'summer_time']] for df in (df_train, df_test)]
            self.y_train, self.y_test = [df[['grid_backflow']] for df in (df_train, df_test)]
        
        self.df_train = df_train
        self.df_test = df_test
        self.df_predict_train = df_train[['period_of_day', 'date']].copy()
        self.df_predict_test = df_test[['period_of_day', 'date']].copy()
        self.quantity_name = quantity_name
        
    def fit_data(self):
        """"""
        lgbm_reg = LGBMRegressor(objective='regression', verbosity=-1)
        lgbm_reg.fit(self.X_train, self.y_train)
        
        return lgbm_reg

    def score(self, model, test: bool=True):
        """"""
        if test:
            df_predictions = self.df_predict_test
            X = self.X_test
            y = self.y_test
            df = self.df_test
        else: 
            df_predictions = self.df_predict_train
            X = self.X_train
            y = self.y_train
            df = self.df_train
        
        prediction = model.predict(X)  
        
        df_predictions[f'{y.columns[0]}_true'] = y.values
        df_predictions[f'{y.columns[0]}_predicted'] = prediction
            
        if self.quantity_name == 'grid_consumption':
            mask_grid = df_predictions['grid_consumption_predicted'] < 0
            df_predictions['grid_consumption_predicted'] = df_predictions['grid_consumption_predicted'].mask(mask_grid)
            df_predictions['grid_consumption_true'] = df['grid_consumption']
        elif self.quantity_name == 'grid_backflow':
            mask_grid = df_predictions['grid_backflow_predicted'] < 0
            df_predictions['grid_backflow_predicted'] = df_predictions['grid_backflow_predicted'].mask(mask_grid)
            df_predictions['grid_backflow_true'] = df['grid_backflow']
        elif self.quantity_name == 'battery_charging':
            mask_battery = df_predictions['battery_charging_predicted'] < 0
            df_predictions['battery_charging_predicted'] = df_predictions['battery_charging_predicted'].mask(mask_battery)
            df_predictions['battery_charging_true'] = df['battery_charging']
        elif self.quantity_name == 'battery_discharging':
            mask_battery = df_predictions['battery_discharging_predicted'] < 0
            df_predictions['battery_discharging_predicted'] = df_predictions['battery_discharging_predicted'].mask(mask_battery)
            df_predictions['battery_discharging_true'] = df['battery_discharging']
        else:
            pass

        df_predictions = df_predictions.fillna(0)  
        df_predictions[f'{self.quantity_name}_residuals'] = df_predictions[f'{self.quantity_name}_true'] - df_predictions[f'{self.quantity_name}_predicted']
            
        return df_predictions