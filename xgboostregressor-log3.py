"""
Private Score: 0.12511, Public Score: 0.11315
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
training_df = pd.read_csv("data/train.csv", dtype={"StateHoliday": pd.np.string_})
store_df = pd.read_csv("data/store.csv")
test_df = pd.read_csv("data/test.csv", dtype={"StateHoliday": pd.np.string_})

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
training_df["Year"] = training_df["Date"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").year)
training_df["Month"] = training_df["Date"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").month)

test_df["Year"] = test_df["Date"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").year)
test_df["Month"] = test_df["Date"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").month)

# Create "YearMonth" column
training_df["YearMonth"] = training_df["Date"].apply(lambda x: str(dt.datetime.strptime(x, "%Y-%m-%d").year) + "-" + less_than_ten(str(dt.datetime.strptime(x, "%Y-%m-%d").month)))
test_df["YearMonth"] = test_df["Date"].apply(lambda x: str(dt.datetime.strptime(x, "%Y-%m-%d").year) + "-" + less_than_ten(str(dt.datetime.strptime(x, "%Y-%m-%d").month)))

# "StateHoliday" has values "0" & 0
training_df["StateHoliday"].loc[training_df["StateHoliday"] == 0] = "0"
test_df["StateHoliday"].loc[test_df["StateHoliday"] == 0] = "0"

# Create "StateHolidayBinary" column
training_df["StateHolidayBinary"] = training_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
test_df["StateHolidayBinary"] = test_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})

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

# Merge store_df with test_df and training_df
training_df = pd.merge(training_df, store_df, on="Store", how="left")
test_df = pd.merge(test_df, store_df, on="Store", how="left")

# Estimating DayOfMonth. Useful for trends such as pay day.
training_df["DayOfMonth"] = training_df.Date.apply(lambda x: x.split("-")[2])
training_df["DayofMonth"] = training_df["DayOfMonth"].astype(float)

test_df["DayOfMonth"] = test_df.Date.apply(lambda x: x.split("-")[2])
test_df["DayOfMonth"] = test_df["DayOfMonth"].astype(float)

# Filling all NaN values with 0
training_df = training_df.fillna(0)
test_df = test_df.fillna(0)

# Selecting only open stores and ignoring the closed stores for training
training_df = training_df[training_df["Open"] == 1]

# Create new DataFrames for Average Customers per Store & per Store per Month for "Customers" != 0
avg_cust = training_df.groupby(["Store"]).agg({"Customers": {"AvgCustStore": lambda y: np.mean([x for x in y if x != 0])}})
avg_cust.columns = avg_cust.columns.get_level_values(1)
avg_cust.reset_index(inplace=True)

avg_cust_month = training_df.groupby(["Store", "Month"]).agg({"Customers": {"AvgCustStoreMonth": lambda y: np.mean([x for x in y if x != 0])}})
avg_cust_month.columns = avg_cust_month.columns.get_level_values(1)
avg_cust_month.reset_index(inplace=True)

training_df = pd.merge(training_df, avg_cust, on=["Store"])
training_df = pd.merge(training_df, avg_cust_month, on=["Store", "Month"])

test_df = pd.merge(test_df, avg_cust, on=["Store"])
test_df = pd.merge(test_df, avg_cust_month, on=["Store", "Month"])

# Log factorization of Sales changes the distribution and makes the performance much better 
training_df['Sales'] = np.log(training_df['Sales']+1)

# List of features to be used in the model
features = ["Store", "DayOfWeek", "Year", "Month", "DayOfMonth", "Open", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance", "Promo2", "AvgCustStore", "AvgCustStoreMonth"]

# Label encoding of columns (eg. StoreType with "a", "b", "c" and "d" would become 1, 2, 3 and 4)
for f in training_df[features]:
    if training_df[f].dtype == "object":
        labels = LabelEncoder()
        labels.fit(list(training_df[f].values) + list(test_df[f].values))
        training_df[f] = labels.transform(list(training_df[f].values))
        test_df[f] = labels.transform(list(test_df[f].values))


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
    multiplying_factor[indices] = 1.0/(y_true[indices])
    diff = y_true - y_pred
    diff_percentage = diff * multiplying_factor
    diff_percentage_squared = diff_percentage ** 2
    rmspe = np.sqrt(np.mean( diff_percentage_squared ))
    return rmspe

################################################################
# Training the Model & Predicting Sales                        #
################################################################

"""
A XGB regression model for all stores. Adds two extra features (AvgCustStore, AvgCustStoreMonth) to improve the model. Uses log standardization for the Sales output.

Features: Store, DayOfWeek, Year, Month, DayOfMonth, Open, Promo, StateHoliday, SchoolHoliday, StoreType, Assortment, CompetitionDistance, Promo2, AvgCustStore, AvgCustStoreMonth
"""

# Comment this block when not training
################ TRAINING ###############
print("Training...")
regressor = XGBRegressor(n_estimators=3000, nthread=-1, max_depth=12,
                         learning_rate=0.02, silent=True, subsample=0.9, colsample_bytree=0.7)
regressor.fit(np.array(training_df[features]), training_df["Sales"])

with open("models/xgboostregressor-log3.pkl", "wb") as fid:
    pickle.dump(regressor, fid)

print("Model saved to models/xgboostregressor-log3.pkl")
########### TRAINING COMPLETED ##########

# Uncomment this block when not training
# with open("models/xgboostregressor-log3.pkl", "rb") as fid:
#     regressor = pickle.load(fid)
# print ("Loaded the model.")

print("Making predictions...")

predictions = []
for i in test_df["Id"].tolist():
    if test_df[test_df["Id"] == i]["Open"].item() == 0:
        # Appending 0 for closed stores
        predictions += [[i, 0]]
    else:
        # Appending prediction for open stores
        prediction = np.exp(regressor.predict(np.array(test_df[test_df["Id"] == i][features]))[0])-1
        predictions += [[i, prediction]]

# Using the csv library to save the file
with open("predictions/xgboostregressor-log3.csv", "w") as f:
    csv_writer = csv.writer(f, lineterminator="\n")
    csv_writer.writerow(["Id", "Sales"])
    csv_writer.writerows(predictions)

print("Predictions saved.")
