# Preliminaries:
- install/set python 3.11.9 (with pyenv)
- git clone this repo
- create & activate virtual environment
- pip install -r requirements.txt

- .csv data file is not included in the repo, add one into the same folder as the script 'possible_call_example.py'

# How to run:
The main script to run is 'possible_call_example.py'.  
- it takes 4 arguments:  
: "-i", "--input"                   (required); type: string              - the .csv file to be analyzed  
: "-q", "--quantity"                (required); type: string              - Quantity to fit / one of the column names  
: "-split", "--split_train_test"    (optional); type: bool, default=False - When True('-split'), splits the dataset into train and test subsets and shows 2 sets of graphs separately.  
                                                                          - When False('-no-split'), the model fits the whole dataset  
: "-d", "--days_to_test"            (optional); type: integer, default=4  - When '-split' argument is set, '-d' sets the number of days from the dataset to assign to the test subset  

## Call examples of 'possible_call_example.py'
python possible_call_example.py -i SG.csv -q Battery-charging  
python possible_call_example.py --input SG.csv --quantity battery_discharging --split_train_test -d 5  
python possible_call_example.py -i SG.csv --quantity Consumption  
python possible_call_example.py -i SG.csv --q PV-generation -split -days_to_test 10  

# PyTest
One simple test file is included, 'test_process_data.py'  
- it uses hardcoded file path to 'wtstr/SG.csv' (not included in the repo)
- tests for presence of NaNs in the final transformed dataset

# Model
Light gradient boosting model from 'lightgbm' python package is used to fit the data.  
It is configured as "one period ahead" prediction, i.e. it uses lagged inputs.  

# Example outputs
call: python possible_call_example.py -i SG.csv --quantity Grid-consumption  
1 output:  
![image](https://github.com/42itshallbe/wtstr/assets/172781090/63ea8ef5-8723-443f-80db-b6ad5170ab09)

call: python possible_call_example.py -i SG.csv --quantity Battery-discharging -split -d 5  
2 outputs (1 for train and 1 for test subsets - last 5 days are in test)  
![image](https://github.com/42itshallbe/wtstr/assets/172781090/2501dc7a-d8a8-4c10-adad-0e6552ed060f)
![image](https://github.com/42itshallbe/wtstr/assets/172781090/fe596ce0-c3eb-4829-a214-7efe611ce2f7)

