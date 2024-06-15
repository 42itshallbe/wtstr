import argparse

import pandas as pd

from utils.process_data import get_data
from utils.fit_data import LGBMModel
from utils.plots import plot_results


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, type=str, help="Input .csv file name (with path, if it is not in the same folder as the script possible_call_example.py)")
parser.add_argument("-q", "--quantity", required=True, type=str, help="The quantity to fit")
parser.add_argument("-split", "--split_train_test", default=False, action=argparse.BooleanOptionalAction, help="When set to False, the model fits and predicts on the whole provided dataset.")
parser.add_argument("-d", "--days_to_test", type=int, default=4, help="Number of days from the provided dataset to allocate for test, when split_train_test set to True")
args = parser.parse_args()

quantity_name = args.quantity.lower().replace('-', '_')
quantity_choices = ['consumption', 'pv_generation', 'battery_charging', 'battery_discharging', 'grid_consumption', 'grid_backflow']
if quantity_name not in quantity_choices:
    raise ValueError(f"The parameter 'quantity' only accepts one of the following values: {quantity_choices}")

df_data = get_data(args.input)

# fit whole dataset
if not args.split_train_test:
    fit_model = LGBMModel(df_data, df_data, quantity_name)
    model_all = fit_model.fit_data()
    df_out_fit_all = fit_model.score(model_all)

    plot_results(df_out_fit_all, quantity_name, "Fit & predict on the whole dataset (no train/test split)")

# split train/test (last 4 days for test)
else:
    test_days = args.days_to_test
    if test_days >= df_data['date'].count() - 2:
        raise ValueError("Number of days for test must be < (total days in dataset - 3)")

    split_date = df_data['date'].max() - pd.Timedelta(days=test_days)
    df_train = df_data[df_data['date'] < split_date]
    df_test = df_data[df_data['date'] > split_date]

    fit_model = LGBMModel(df_train, df_test, quantity_name)
    model_split = fit_model.fit_data()
    df_out_train = fit_model.score(model_split, test=False)
    df_out_test = fit_model.score(model_split)

    plot_results(df_out_train, quantity_name, "Train")
    plot_results(df_out_test, quantity_name, "Test")