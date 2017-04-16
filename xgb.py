"""
Private Score: 0.15591, Public Score: 0.17379
"""

import datetime as dt
import pickle
import csv
import matplotlib.pyplot as plt

import pandas as pd
from pandas import Series, DataFrame
import numpy as np

from xgboost import DMatrix, train, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

pd.options.mode.chained_assignment = None

################################################################
# Import CSV Data into Pandas DataFrames                       #
################################################################

types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(float),
         'PromoInterval': np.dtype(str)}

training_df = pd.read_csv("data/train.csv", parse_dates=[2], dtype=types)
store_df = pd.read_csv("data/store.csv")
test_df = pd.read_csv("data/test.csv", parse_dates=[3], dtype=types)
# print(training_df.head())
# print(store_df.head())
# print(test_df.head())


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

print ("Starting Custom Preprocessing.")

# Preprocessing on store
# SalesPerDay, CustomersPerDay, SalesPerCustomerPerDay

store_data_sales = training_df.groupby([training_df['Store']])['Sales'].sum()
store_data_customers = training_df.groupby([training_df['Store']])['Customers'].sum()
store_data_open = training_df.groupby([training_df['Store']])['Open'].count()

store_data_sales_per_day = store_data_sales / store_data_open
store_data_customers_per_day = store_data_customers / store_data_open
store_data_sales_per_customer_per_day = store_data_sales_per_day /store_data_customers_per_day

store_df = pd.merge(store_df, store_data_sales_per_day.reset_index(name='SalesPerDay'), how='left', on=['Store'])
store_df = pd.merge(store_df, store_data_customers_per_day.reset_index(name='CustomersPerDay'), how='left', on=['Store'])
store_df = pd.merge(store_df, store_data_sales_per_customer_per_day.reset_index(name='SalesPerCustomersPerDay'), how='left', on=['Store'])

# Merging store with training and test data frames
training_df = pd.merge(training_df, store_df, on="Store", how="left")
test_df = pd.merge(test_df, store_df, on="Store", how="left")

training_df = training_df.fillna(0)
test_df = test_df.fillna(0)

training_df = training_df[training_df["Open"] == 1]

# Computing the day
training_df["Day"] = training_df.Date.dt.day

test_df["Day"] = test_df.Date.dt.day

# Computing DayOfYear
training_df["DayOfYear"] = training_df.Date.dt.dayofyear

test_df["DayOfYear"] = test_df.Date.dt.dayofyear

# Computing Week
training_df["Week"] = training_df.Date.dt.week

test_df["Week"] = test_df.Date.dt.week

# Computing DayOfWeek
training_df["DayOfWeek"] = training_df.Date.dt.dayofweek

test_df["DayOfWeek"] = test_df.Date.dt.dayofweek

# Label encoding 
training_df['StateHoliday'] = training_df['StateHoliday'].astype('category').cat.codes
test_df['StateHoliday'] = test_df['StateHoliday'].astype('category').cat.codes

training_df['Assortment'] = training_df['Assortment'].astype('category').cat.codes
test_df['Assortment'] = test_df['Assortment'].astype('category').cat.codes

training_df['StoreType'] = training_df['StoreType'].astype('category').cat.codes
test_df['StoreType'] = test_df['StoreType'].astype('category').cat.codes

# Competition Open Int
def convertCompetitionOpen(df):
    try:
        date = '{}-{}'.format(int(df['CompetitionOpenSinceYear']), int(df['CompetitionOpenSinceMonth']))
        return pd.to_datetime(date)
    except:
        return 0

training_df['CompetitionOpenInt'] = training_df.apply(lambda df: convertCompetitionOpen(training_df), axis=1).astype(np.int64)

test_df['CompetitionOpenInt'] = test_df.apply(lambda df: convertCompetitionOpen(test_df), axis=1).astype(np.int64)

# Find Closed Store IDs
# closed_store_ids = test_df["Id"][test_df["Open"] == 0].values

# Log Standardization ==> Better for RMSPE
training_df['Sales'] = np.log1p(training_df['Sales'])

features = ['Store', 'Day', 'Week', 'Month','Year','DayOfYear', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'StateHoliday', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenInt', 'SalesPerDay', 'CustomersPerDay', 'SalesPerCustomersPerDay']

training_df = training_df.fillna(0)
test_df = test_df.fillna(0)

print ("Completed Custom Preprocessing.")

################################################################
# RMSPE Function                                               #
################################################################

def rmspe(y_true, y_pred):
    w = np.zeros(y_true.shape, dtype=float)
    index = y_true != 0
    w[index] = 1.0/(y_true[index])
    diff = y_true - y_pred
    diff_percentage = diff * w
    diff_percentage_squared = diff_percentage ** 2
    rmspe = np.sqrt(np.mean( diff_percentage_squared ))
    return rmspe

################################################################
# Training the Model & Predicting Sales                        #
################################################################

"""
XGB Model with log and exp standardization. The model uses xgb.train as an ensemble booster.

Features: Store, Day, Week, Month, Year, DayOfYear, DayOfWeek, Open, Promo, SchoolHoliday, StateHoliday, StoreType, Assortment, CompetitionDistance, CompetitionOpenInt, SalesPerDay, CustomersPerDay, SalesPerCustomersPerDay
"""

print("Training...")

def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe

X_train = training_df[features]
y_train = training_df["Sales"]
X_test = test_df[features]

dtrain = DMatrix(X_train, y_train)
dtest = DMatrix(X_test)

# Uncomment to train
param = {'bst:max_depth':12,
         'bst:eta':0.0095,
         'subsample':0.8,
         'colsample_bytree':0.7,
         'silent':1, 
         'objective':'reg:linear',
         'nthread':4,
         'seed':42}

plst = param.items()

# model = train(params=plst, dtrain = dtrain, num_boost_round = 20000)

# with open("models/xgb.pkl", "wb") as fid:
#     pickle.dump(model, fid)

# print("Model saved to models/xgb.pkl")

with open("models/xgb.pkl", "rb") as fid:
    model = pickle.load(fid)
print ("Model Loaded.")

print("Making predictions...")
xgbPredict = model.predict(dtest)

result = pd.DataFrame({"Id": test_df["Id"], "Sales": np.expm1(xgbPredict)})

result.to_csv("predictions/xgb.csv", index=False)

print("Predictions saved to predictions/xgb.csv.")

# Show Feature Importance
# plot_importance(model)
# plt.show()