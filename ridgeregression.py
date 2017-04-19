"""
Private Score: 0.16179, Public Score: 0.14526
"""

import datetime as dt

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from sklearn.linear_model import Ridge

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None


################################################################
# Import CSV Data into Pandas DataFrames                       #
################################################################
training_df = pd.read_csv("data/train.csv")
# store_df = pd.read_csv("data/store.csv")
test_df = pd.read_csv("data/test.csv")

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
# training_df["StateHolidayBinary"] = training_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
# test_df["StateHolidayBinary"] = test_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})

# One-hot encoding of "DayOfWeek" & "StateHoliday" columns
training_df = pd.get_dummies(training_df, columns=["DayOfWeek", "StateHoliday"])
test_df = pd.get_dummies(test_df, columns=["DayOfWeek", "StateHoliday"])

############################################
# store_df                                 #
############################################

# Fill NaN values in store_df for "CompetitionDistance" = 0 (since no record exists where "CD" = NaN & "COS[Y/M]" = !NaN)
# store_df["CompetitionDistance"][is_nan(store_df["CompetitionDistance"])] = 0

# Fill NaN values in store_df for "CompetitionSince[X]" with 1900-01
# store_df["CompetitionOpenSinceYear"][(store_df["CompetitionDistance"] != 0) & (is_nan(store_df["CompetitionOpenSinceYear"]))] = 1900
# store_df["CompetitionOpenSinceMonth"][(store_df["CompetitionDistance"] != 0) & (is_nan(store_df["CompetitionOpenSinceMonth"]))] = 1

# One-hot encoding of "StoreType" & "Assortment" columns
# store_df = pd.get_dummies(store_df, columns=["StoreType", "Assortment"])


################################################################
# Process Data (Custom)                                        #
################################################################

# Drop "StateHolidayBinary" as "StateHoliday_[X]" Exists
# training_df.drop(["StateHolidayBinary"], axis=1, inplace=True)
# test_df.drop(["StateHolidayBinary"], axis=1, inplace=True)

# Remove rows for stores not in test_df
training_df = training_df[training_df["Store"].isin(test_df.Store.unique())]

# Remove all Closed Stores ("Sales" = 0)
training_df = training_df[training_df["Open"] != 0]

# Drop "Open" from training_df
training_df.drop(["Open"], axis=1, inplace=True)

# Save IDs of Closed Stores ("Sales" = 0 assigned later) & Remove Rows
closed_store_ids = test_df["Id"][test_df["Open"] == 0].values
test_df = test_df[test_df["Open"] != 0]

# Drop "Open" from test_df
test_df.drop(["Open"], axis=1, inplace=True)

# Drop "Date" & "YearMonth" column
training_df.drop(["Date", "YearMonth"], axis=1, inplace=True)
test_df.drop(["Date", "YearMonth"], axis=1, inplace=True)

# Create new DataFrames for Average Customers per Store & per Store per Month for "Customers" != 0
avg_cust = training_df.groupby(["Store"]).agg({"Customers": {"AvgCustStore": lambda y: np.mean([x for x in y if x != 0])}})
avg_cust.columns = avg_cust.columns.get_level_values(1)
avg_cust.reset_index(inplace=True)

avg_cust_month = training_df.groupby(["Store", "Month"]).agg({"Customers": {"AvgCustStoreMonth": lambda y: np.mean([x for x in y if x != 0])}})
avg_cust_month.columns = avg_cust_month.columns.get_level_values(1)
avg_cust_month.reset_index(inplace=True)

# Merge the newly created DataFrames with training_df & test_df
training_df = pd.merge(training_df, avg_cust, on=["Store"])
training_df = pd.merge(training_df, avg_cust_month, on=["Store", "Month"])

test_df = pd.merge(test_df, avg_cust, on=["Store"])
test_df = pd.merge(test_df, avg_cust_month, on=["Store", "Month"])

# Drop "Customers" from training_df
training_df.drop(["Customers"], axis=1, inplace=True)


################################################################
# RMSPE Function                                               #
################################################################

def rmspe(y_true, y_pred):
    w = np.zeros(y_true.shape, dtype=float)
    index = y_true != 0
    w[index] = 1.0 / (y_true[index])
    diff = y_true - y_pred
    diff_percentage = diff * w
    diff_percentage_squared = diff_percentage ** 2
    rmspe = np.sqrt(np.mean(diff_percentage_squared))
    return rmspe

################################################################
# Training the Model & Predicting Sales                        #
################################################################

"""
This model follows the features from linearregression-independent4.py, with the regression model set to Ridge Regression.

Treating each store as an independent regression problem, loop through all stores training the model for the particular store and predicting its sales value.

Features: Promo, SchoolHoliday, Year, Month, DayOfWeek (one-hot encoded), StateHoliday (one-hot encoded), AvgCustStore, AvgCustStoreMonth

Assumptions:
- The store's opening/closing dates does not affect the store's performance. For example, a store that was closed yesterday will not get more sales today because of that.
- The competition of each store will affect it consistently, hence, it does not matter.
- Each store's sales value is independent of the other stores and can be treated as independent regression problems.
"""

print("Making predictions...")

training_dict = dict(list(training_df.groupby("Store")))
test_dict = dict(list(test_df.groupby("Store")))
predictions = Series()

for i in test_dict:
    store = training_dict[i]

    X_train = store.drop(["Sales", "Store"], axis=1)
    Y_train = store["Sales"]
    X_test = test_dict[i].copy()

    # X_tr, X_te, Y_tr, Y_te = train_test_split(X_train, Y_train, test_size=0.4)
    # lreg = LinearRegression()
    # lreg.fit(X_tr, Y_tr)
    # Y_pr = lreg.predict(X_te)
    # print(rmspe(y_true=Y_te, y_pred=Y_pr))

    store_ids = X_test["Id"]
    X_test.drop(["Id", "Store"], axis=1, inplace=True)

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    lreg = Ridge(alpha=1.0)
    lreg.fit(X_train, Y_train)
    Y_pred = lreg.predict(X_test)

    predictions = predictions.append(Series(Y_pred, index=store_ids))

predictions = predictions.append(Series(0, index=closed_store_ids))

submission = pd.DataFrame({"Id": predictions.index, "Sales": predictions.values})
submission.to_csv("predictions/ridgeregression.csv", index=False)

print("Predictions saved.")
