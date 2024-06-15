import os
from pathlib import Path

from utils.process_data import get_data

def test_df_nans():
    dir_path = Path('..\\')
    data_file = os.path.join(dir_path, 'SG.csv')
    df = get_data(data_file)
    assert df.isnull().sum().sum() == 0