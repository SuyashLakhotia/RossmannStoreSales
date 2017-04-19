"""
Private Score: 0.13205, Public Score: 0.11356
"""

import datetime as dt
import pickle

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
train_df = pd.read_csv("data/train.csv", parse_dates=[2])
# parse_dates parses the date column into datetime datatype
store_df = pd.read_csv("data/store.csv")
testing_df = pd.read_csv("data/test.csv", parse_dates=[3]})

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
# training_df["YearMonth"] = training_df["Date"].apply(lambda x: str(x.year) + "-" + less_than_ten(str(x.month)))
# test_df["YearMonth"] = test_df["Date"].apply(lambda x: str(x.year) + "-" + less_than_ten(str(x.month)))

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
# store_df["CompetitionOpenSinceYear"][(store_df["CompetitionDistance"] != 0) & (is_nan(store_df["CompetitionOpenSinceYear"]))] = 1900
# store_df["CompetitionOpenSinceMonth"][(store_df["CompetitionDistance"] != 0) & (is_nan(store_df["CompetitionOpenSinceMonth"]))] = 1

# One-hot encoding of "StoreType" & "Assortment" columns
# store_df = pd.get_dummies(store_df, columns=["StoreType", "Assortment"])


################################################################
# Process Data (Custom)                                        #
################################################################

training_df = pd.merge(train_df, store_df, on="Store", how="left")
test_df = pd.merge(testing_df, store_df, on="Store", how="left")

training_df["DayOfMonth"] = training_df.Date.dt.day

test_df["DayOfMonth"] = test_df.Date.dt.day

training_df = training_df.fillna(0)
test_df = test_df.fillna(0)

training_df = training_df[training_df["Open"] == 1]

features = ["Store", "DayOfWeek", "Year", "Month", "DayOfMonth", "Open", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance", "Promo2"]

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
A XGBoost regression model. The model trains using all store data and runs once against all records. The feature set is preprocessed to extract Year, Month and DayOfMonth.

Features: Store, DayOfWeek, Year, Month, DayOfMonth, Open, Promo, StateHoliday, SchoolHoliday, StoreType, Assortment, CompetitionDistance, Promo2

Assumptions:
- DayOfMonth has an effect on sales, for example, the sales is higher on paydays.
- The store's opening/closing dates does not affect the store's performance. For example, a store that was closed yesterday will not get more sales today because of that.
"""

print("Making predictions...")

# Comment this block when not training

################ TRAINING ###############
regressor = XGBRegressor(n_estimators=3000, nthread=-1, max_depth=12,
                         learning_rate = 0.02, silent = True, subsample = 0.9, colsample_bytree = 0.7)
regressor.fit(np.array(training_df[features]), training_df["Sales"])

# The model is pickled and saved to a file. The file can be loaded later to retrieve the object.
with open("models/xgboostregressor2.pkl", "wb") as fid:
    pickle.dump(regressor, fid)

print("Model saved to models/xgboostregressor2.pkl")
########### TRAINING COMPLETED ##########

# Uncomment this section to load from an existing model
# with open("models/xgboostregressor2.pkl", "rb") as fid:
#     regressor = pickle.load(fid)

# print("Model loaded.")

print("Making Predictions...")

xgbPredict=regressor.predict(np.array(test_df[features]))

result=pd.DataFrame({"Id": test_df["Id"], "Sales": xgbPredict})

result.to_csv("predictions/xgboostregressor2.csv", index = False)

print("Predictions saved to predictions/xgboostregressor2.csv.")

# Uncomment this section to show the feature importance chart
# mapper = {'f{0}'.format(i): v for i, v in enumerate(features)}
# mapped = {mapper[k]: v for k, v in regressor.booster().get_score(importance_type='weight').items()}
# plot_importance(mapped)
# plt.show()
