"""
Private Score: 0.12708, Public Score: 0.11219
"""

import datetime as dt
import pickle
import csv

import pandas as pd
from pandas import Series, DataFrame
import numpy as np

from xgboost import XGBRegressor
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

# Merging store with training and test data frames
training_df = pd.merge(training_df, store_df, on="Store", how="left")
test_df = pd.merge(test_df, store_df, on="Store", how="left")

training_df = training_df.fillna(0)
test_df = test_df.fillna(0)

training_df = training_df[training_df["Open"] == 1]

# Computing the day
training_df["Day"] = training_df.Date.dt.day

test_df["Day"] = test_df.Date.dt.day

# Computing DayOfWeek
training_df["DayOfWeek"] = training_df.Date.dt.dayofweek

test_df["DayOfWeek"] = test_df.Date.dt.dayofweek

# Computing WeekOfYear
training_df["WeekOfYear"] = training_df.Date.dt.weekofyear
test_df["WeekOfYear"] = test_df.Date.dt.weekofyear

# Computing CompetitionOpen
training_df["CompetitionOpen"] = 12 * (training_df.Year - training_df.CompetitionOpenSinceYear) + \
        (training_df.Month - training_df.CompetitionOpenSinceMonth)
test_df["CompetitionOpen"] = 12 * (test_df.Year - test_df.CompetitionOpenSinceYear) + \
        (test_df.Month - test_df.CompetitionOpenSinceMonth)

# Computing PromoOpen
training_df['PromoOpen'] = 12 * (training_df.Year - training_df.Promo2SinceYear) + \
        (training_df.WeekOfYear - training_df.Promo2SinceWeek) / 4.0
training_df['PromoOpen'] = training_df.PromoOpen.apply(lambda x: x if x > 0 else 0)
training_df.loc[training_df.Promo2SinceYear == 0, 'PromoOpen'] = 0

test_df['PromoOpen'] = 12 * (test_df.Year - test_df.Promo2SinceYear) + \
        (test_df.WeekOfYear - test_df.Promo2SinceWeek) / 4.0
test_df['PromoOpen'] = test_df.PromoOpen.apply(lambda x: x if x > 0 else 0)
test_df.loc[test_df.Promo2SinceYear == 0, 'PromoOpen'] = 0

# Computing IsPromoMonth
month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
training_df['monthStr'] = training_df.Month.map(month2str)
training_df.loc[training_df.PromoInterval == 0, 'PromoInterval'] = ''
training_df['IsPromoMonth'] = 0
for interval in training_df.PromoInterval.unique():
    if interval != '':
        for month in interval.split(','):
            training_df.loc[(training_df.monthStr == month) & (training_df.PromoInterval == interval), 'IsPromoMonth'] = 1

month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
test_df['monthStr'] = test_df.Month.map(month2str)
test_df.loc[test_df.PromoInterval == 0, 'PromoInterval'] = ''
test_df['IsPromoMonth'] = 0
for interval in test_df.PromoInterval.unique():
    if interval != '':
        for month in interval.split(','):
            test_df.loc[(test_df.monthStr == month) & (test_df.PromoInterval == interval), 'IsPromoMonth'] = 1

# Find Closed Store IDs
# closed_store_ids = test_df["Id"][test_df["Open"] == 0].values

# Log Standardization ==> Better for RMSPE
training_df['Sales'] = np.log1p(training_df['Sales'])

label_encoded_features = ["StoreType", "Assortment", "StateHoliday"]

print("Preprocessing by label encoding.")
for f in training_df[label_encoded_features]:
    if training_df[f].dtype == "object":
        labels = LabelEncoder()
        labels.fit(list(training_df[f].values) + list(test_df[f].values))
        training_df[f] = labels.transform(list(training_df[f].values))
        test_df[f] = labels.transform(list(test_df[f].values))

features = ["Store", "DayOfWeek", "Year", "Month", "Day", "Open", "Promo", "SchoolHoliday", "CompetitionDistance", "Promo2", "WeekOfYear", "CompetitionOpen", "PromoOpen", "IsPromoMonth"]

features.extend(label_encoded_features)


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
XGB Regression Model with log and exp standardization.

Features: Store, DayOfWeek, Year, Month, Day, Open, Promo, SchoolHoliday, CompetitionDistance, Promo2, WeekOfYear, CompetitionOpen, PromoOpen, IsPromoMonth, StoreType, Assortment, StateHoliday
"""

print("Training...")

# Uncomment to train
regressor = XGBRegressor(n_estimators=3000, max_depth=12,
                         learning_rate=0.02, silent=True, 
                         subsample=0.9, colsample_bytree=0.7)
regressor.fit(np.array(training_df[features]), training_df["Sales"])

with open("models/xgboostregressor5.pkl", "wb") as fid:
    pickle.dump(regressor, fid)

print("Model saved to models/xgboostregressor5.pkl")

# with open("models/xgboostregressor5.pkl", "rb") as fid:
#     regressor = pickle.load(fid)

print("Making predictions...")
xgbPredict = regressor.predict(np.array(test_df[features]))

result = pd.DataFrame({"Id": test_df["Id"], "Sales": np.expm1(xgbPredict)})

result.to_csv("predictions/xgboostregressor5.csv", index=False)

print("Predictions saved to predictions/xgboostregressor5.csv.")
