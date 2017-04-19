"""
Private Score: 0.12305, Public Score: 0.11276, Local Score: 0.11976
"""

import pickle
import sys

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor, plot_importance

pd.options.mode.chained_assignment = None

validate = False

# Run the script with validate as argument for validation (```python xgboostregressor-log5.py validate```)
if (len(sys.argv) > 1) and (sys.argv[1] == "validate"):
    validate = True


################################################################
# Import CSV Data into Pandas DataFrames                       #
################################################################

training_df = pd.read_csv("data/train.csv", parse_dates=[2])
store_df = pd.read_csv("data/store.csv")
test_df = pd.read_csv("data/test.csv", parse_dates=[3])


################################################################
# Process Data (Universal)                                     #
################################################################

def is_nan(val):
    return val != val


def less_than_ten(val):
    if int(val) < 10:
        return "0" + val
    else:
        return val

############################################
# training_df & test_df                    #
############################################

# Fill NaN values in test_df with Open = 1 if DayOfWeek != 7
test_df["Open"][is_nan(test_df["Open"])] = (test_df["DayOfWeek"] != 7).astype(int)
training_df["Open"][is_nan(training_df["Open"])] = (training_df["DayOfWeek"] != 7).astype(int)

# Create "Year" & "Month" columns
training_df["Year"] = training_df["Date"].dt.year
training_df["Month"] = training_df["Date"].dt.month

test_df["Year"] = test_df["Date"].dt.year
test_df["Month"] = test_df["Date"].dt.month

# Create "YearMonth" column
# training_df["YearMonth"] = training_df["Date"].apply(lambda x: str(dt.datetime.strptime(x, "%Y-%m-%d").year) + "-" + less_than_ten(str(dt.datetime.strptime(x, "%Y-%m-%d").month)))
# test_df["YearMonth"] = test_df["Date"].apply(lambda x: str(dt.datetime.strptime(x, "%Y-%m-%d").year) + "-" + less_than_ten(str(dt.datetime.strptime(x, "%Y-%m-%d").month)))

# "StateHoliday" has values "0" & 0
training_df["StateHoliday"].loc[training_df["StateHoliday"] == 0] = "0"
test_df["StateHoliday"].loc[test_df["StateHoliday"] == 0] = "0"

# Create "StateHolidayBinary" column
# training_df["StateHolidayBinary"] = training_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
# test_df["StateHolidayBinary"] = test_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})

# One-hot encoding of "DayOfWeek" & "StateHoliday" columns
# training_df = pd.get_dummies(training_df, columns=["DayOfWeek", "StateHoliday"])
# test_df = pd.get_dummies(test_df, columns=["DayOfWeek", "StateHoliday"])

############################################
# store_df                                 #
############################################

# Fill NaN values in store_df for "CompetitionDistance" = 0 (since no record exists where "CD" = NaN & "COS[Y/M]" = !NaN)
store_df["CompetitionDistance"][is_nan(store_df["CompetitionDistance"])] = 0

# Fill NaN values in store_df for "CompetitionSince[X]" with 1900-01
store_df["CompetitionOpenSinceYear"][(store_df["CompetitionDistance"] != 0) & (is_nan(store_df["CompetitionOpenSinceYear"]))] = 1900
store_df["CompetitionOpenSinceMonth"][(store_df["CompetitionDistance"] != 0) & (is_nan(store_df["CompetitionOpenSinceMonth"]))] = 1

# One-hot encoding of "StoreType" & "Assortment" columns
# store_df = pd.get_dummies(store_df, columns=["StoreType", "Assortment"])


################################################################
# Process Data (Custom)                                        #
################################################################

print("Starting Custom Preprocessing.")

# Computation for AvgSales, AvgCustomers, AvgSalesPerCustomer
totalSalesPerStore = training_df.groupby([training_df['Store']])['Sales'].sum()
totalCustomersPerStore = training_df.groupby([training_df['Store']])['Customers'].sum()
numberOfOpenStores = training_df.groupby([training_df['Store']])['Open'].count()

# Average Sales for Open Days
AvgSales = totalSalesPerStore / numberOfOpenStores
AvgCustomers = totalCustomersPerStore / numberOfOpenStores
AvgSalesPerCustomer = AvgSales / AvgCustomers

store_df = pd.merge(store_df, AvgSales.reset_index(name='AvgSales'), how='left', on=['Store'])
store_df = pd.merge(store_df, AvgCustomers.reset_index(name='AvgCustomers'), how='left', on=['Store'])
store_df = pd.merge(store_df, AvgSalesPerCustomer.reset_index(name='AvgSalesPerCustomer'), how='left', on=['Store'])

# Merging store with training and test data frames
training_df = pd.merge(training_df, store_df, on="Store", how="left")
test_df = pd.merge(test_df, store_df, on="Store", how="left")

# Select only Open Stores for training
training_df = training_df[training_df["Open"] == 1]

# Computing the DayOfMonth
training_df["DayOfMonth"] = training_df.Date.dt.day
test_df["DayOfMonth"] = test_df.Date.dt.day

# Computing DayOfYear
training_df["DayOfYear"] = training_df.Date.dt.dayofyear
test_df["DayOfYear"] = test_df.Date.dt.dayofyear

# Computing Week
training_df["Week"] = training_df.Date.dt.week
test_df["Week"] = test_df.Date.dt.week

# Label encoding using panda category datatype. Label encoded StateHoliday, Assortment and StoreType
training_df['StateHoliday'] = training_df['StateHoliday'].astype('category').cat.codes
test_df['StateHoliday'] = test_df['StateHoliday'].astype('category').cat.codes

training_df['Assortment'] = training_df['Assortment'].astype('category').cat.codes
test_df['Assortment'] = test_df['Assortment'].astype('category').cat.codes

training_df['StoreType'] = training_df['StoreType'].astype('category').cat.codes
test_df['StoreType'] = test_df['StoreType'].astype('category').cat.codes

# Log Standardization ==> Better for RMSPE
training_df['Sales'] = np.log1p(training_df['Sales'])


def yearMonthGenerator(df):
    # Helper function to create CompetitionOpenYearMonthInteger
    try:
        date = '{}-{}'.format(int(df['CompetitionOpenSinceYear']), int(df['CompetitionOpenSinceMonth']))
        return pd.to_datetime(date)
    except:
        return 0

training_df['CompetitionOpenYearMonthInteger'] = pd.to_numeric(training_df.apply(lambda df: yearMonthGenerator(df), axis=1), errors="coerce")
test_df['CompetitionOpenYearMonthInteger'] = pd.to_numeric(test_df.apply(lambda df: yearMonthGenerator(df), axis=1), errors="coerce")

features = ['Store', 'DayOfMonth', 'Week', 'Month', 'Year', 'DayOfYear', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'StateHoliday', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenYearMonthInteger', 'AvgSales', 'AvgCustomers', 'AvgSalesPerCustomer']

# Fill NaN values with 0
training_df = training_df.fillna(0)
test_df = test_df.fillna(0)

print("Completed Custom Preprocessing")


################################################################
# RMSPE Function                                               #
################################################################

def rmspe(y_true, y_pred):
    """
    RMSPE =  sqrt(1/n * sum( ( (y_true - y_pred)/y_true) ** 2 ) )
    """
    # multiplying_factor = 1/y_true when y_true != 0, else multiplying_factor = 0
    multiplying_factor = np.zeros(y_true.shape, dtype=float)
    indices = y_true != 0
    multiplying_factor[indices] = 1.0 / (y_true[indices])
    diff = y_true - y_pred
    diff_percentage = diff * multiplying_factor
    diff_percentage_squared = diff_percentage ** 2
    rmspe = np.sqrt(np.mean(diff_percentage_squared))
    return rmspe


################################################################
# Training the Model & Predicting Sales                        #
################################################################

"""
XGB Regression Model with log and exp standardization. Creates additional features for DayOfYear, AvgSales, CustomersPerDay, AvgSalesPerCustomer.

Features: Store, DayOfMonth, Week, Month, Year, DayOfYear, DayOfWeek, Open, Promo, SchoolHoliday, StateHoliday, StoreType, Assortment, CompetitionDistance, CompetitionOpenYearMonthInteger, AvgSales, AvgCustomers, AvgSalesPerCustomer

Assumptions:
- DayOfMonth has an effect on sales, for example, the sales is higher on paydays.
- The store's opening/closing dates does not affect the store's performance. For example, a store that was closed yesterday will not get more sales today because of that.
- Sales has a correlation with Average Sales per Customer.
- Sales is affected by CompetitionOpenSince[X]. Thus, a new feature merging the year and month of the opening date is created, CompetitionOpenYearMonthInteger.
"""

if (validate):
    print("Performing validation.")
    # validation using the last 6 weeks of training set as test data. We simulate the samples as close to the test data as possible
    timeDelta = test_df.Date.max() - test_df.Date.min()
    maxDate = training_df.Date.max()
    minDate = maxDate - timeDelta
    # indices is a list of boolean literals which are True when date is within the last 6 weeks.
    indices = training_df["Date"].apply(lambda x: (x >= minDate and x <= maxDate))
    # inverse indices flips True and False
    inverse_indices = indices.apply(lambda x: (not x))
    # This returns the training set values only when indices is True
    y_test = training_df["Sales"][indices]
    X_test = training_df[features][indices]
    y_train = training_df["Sales"][inverse_indices]
    X_train = training_df[features][inverse_indices]

    # Comment this block when not training
    ################ TRAINING ###############
    # print("Training...")
    # regressor = XGBRegressor(n_estimators=3000, nthread=-1, max_depth=12,
    #                      learning_rate=0.02, silent=True, subsample=0.9, colsample_bytree=0.7)
    # regressor.fit(np.array(X_train), y_train)
    # with open("models/xgboostregressor-log5-validate.pkl", "wb") as fid:
    #     pickle.dump(regressor, fid)

    # print("Model saved to models/xgboostregressor-log5-validate.pkl")
    ########### TRAINING COMPLETED ##########

    # Uncomment this block when not training
    with open("models/xgboostregressor-log5-validate.pkl", "rb") as fid:
        regressor = pickle.load(fid)
    print("Loaded the model.")

    xgbPredict = regressor.predict(np.array(X_test))
    result = pd.DataFrame({"Sales": np.expm1(xgbPredict), "True": np.expm1(y_test.values)})
    result.to_csv("predictions/xgboostregressor-log5-validate.csv", index=False)

    print("Predictions saved to predictions/xgboostregressor-log5-validate.csv.")
    print("RMSPE: " + str(rmspe(y_true=np.expm1(y_test.values), y_pred=np.expm1(xgbPredict))))

else:
    X_train = training_df[features]
    X_test = test_df[features]
    y_train = training_df["Sales"]

    # Comment this block when not training
    ################ TRAINING ###############
    print("Training...")
    regressor = XGBRegressor(n_estimators=3000, nthread=-1, max_depth=12,
                             learning_rate=0.02, silent=True, subsample=0.9, colsample_bytree=0.7)
    regressor.fit(np.array(X_train), y_train)
    with open("models/xgboostregressor-log5.pkl", "wb") as fid:
        pickle.dump(regressor, fid)

    print("Model saved to models/xgboostregressor-log5.pkl")
    ########### TRAINING COMPLETED ##########

    # Uncomment this block when not training
    # with open("models/xgboostregressor-log5.pkl", "rb") as fid:
    #     regressor = pickle.load(fid)
    # print("Loaded the model.")

    print("Making Predictions...")

    xgbPredict = regressor.predict(np.array(X_test))
    result = pd.DataFrame({"Id": test_df["Id"], "Sales": np.expm1(xgbPredict)})
    result.to_csv("predictions/xgboostregressor-log5.csv", index=False)

    print("Predictions saved to predictions/xgboostregressor-log5.csv.")

# Show Feature Importance
# mapper = {'f{0}'.format(i): v for i, v in enumerate(features)}
# mapped = {mapper[k]: v for k, v in regressor.booster().get_score(importance_type='weight').items()}
# plot_importance(mapped)
# plt.show()
