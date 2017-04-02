"""
Private Score: XYZ, Public Score: XYZ
"""

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import datetime as dt

pd.options.mode.chained_assignment = None


################################################################
# Import CSV Data into Pandas DataFrames                       #
################################################################
training_df = pd.read_csv("data/train.csv")
store_df = pd.read_csv("data/store.csv")
test_df = pd.read_csv("data/test.csv")

# print(training_df.info())
# print(store_df.info())
# print(test_df.info())


################################################################
# Process Data (Universal)                                     #
################################################################

def is_nan(val):
    return val != val

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
training_df["YearMonth"] = training_df["Date"].apply(lambda x: str(dt.datetime.strptime(x, "%Y-%m-%d").year) + "-" + str(dt.datetime.strptime(x, "%Y-%m-%d").month))
test_df["YearMonth"] = test_df["Date"].apply(lambda x: str(dt.datetime.strptime(x, "%Y-%m-%d").year) + "-" + str(dt.datetime.strptime(x, "%Y-%m-%d").month))

# "StateHoliday" has values "0" & 0
training_df["StateHoliday"].loc[training_df["StateHoliday"] == 0] = "0"
test_df["StateHoliday"].loc[test_df["StateHoliday"] == 0] = "0"

# Create "StateHolidayBinary" column
training_df["StateHolidayBinary"] = training_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
test_df["StateHolidayBinary"] = test_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})

# One-hot encoding of "DayOfWeek" & "StateHoliday" columns
training_df = pd.get_dummies(training_df, columns=["DayOfWeek", "StateHoliday"])
test_df = pd.get_dummies(test_df, columns=["DayOfWeek", "StateHoliday"])

############################################
# store_df                                 #
############################################

# Add "AvgSales" & "AvgCustomers" columns to store_df
avg_sales_customers = training_df.groupby("Store")[["Sales", "Customers"]].mean()
avg_sales_customers_df = DataFrame({"Store": avg_sales_customers.index, "AvgSales": avg_sales_customers["Sales"], "AvgCustomers": avg_sales_customers["Customers"]}, columns=["Store", "AvgSales", "AvgCustomers"])
store_df = pd.merge(avg_sales_customers_df, store_df, on="Store")

# Fill NaN values in store_df for "CompetitionDistance" with the median value
store_df["CompetitionDistance"].fillna(store_df["CompetitionDistance"].median())

# One-hot encoding of "StoreType" & "Assortment" columns
store_df = pd.get_dummies(store_df, columns=["StoreType", "Assortment"])


################################################################
# Process Data (Custom)                                        #
################################################################

# Any custom data processing goes here.

################################################################
# Training the Model & Predicting Sales                        #
################################################################

"""
Describe the model here.

Features: < List of Features Used >
"""

# Code for training & using actual model goes here.
