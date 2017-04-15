import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

pd.options.mode.chained_assignment = None


################################################################
# Import CSV Data into Pandas DataFrames                       #
################################################################
training_df = pd.read_csv("data/train.csv")
store_df = pd.read_csv("data/store.csv")

# print(training_df.info())
# print(store_df.info())


################################################################
# Process Data                                                 #
################################################################

def is_nan(val):
    return val != val


def less_than_ten(val):
    if int(val) < 10:
        return "0" + val
    else:
        return val

############################################
# training_df                              #
############################################

# Create "Year", "Month" & "DayOfMonth" columns
training_df["Year"] = training_df["Date"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").year)
training_df["Month"] = training_df["Date"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").month)
training_df["DayOfMonth"] = training_df["Date"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").day)

# Create "YearMonth" column
training_df["YearMonth"] = training_df["Date"].apply(lambda x: str(dt.datetime.strptime(x, "%Y-%m-%d").year) + "-" + less_than_ten(str(dt.datetime.strptime(x, "%Y-%m-%d").month)))

# "StateHoliday" has values "0" & 0
training_df["StateHoliday"].loc[training_df["StateHoliday"] == 0] = "0"

# Create "StateHolidayBinary" column
training_df["StateHolidayBinary"] = training_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})

############################################
# training_df_open                         #
############################################

# Create DataFrame for only open days
training_df_open = training_df[training_df["Open"] != 0]

# Drop "Open" column
training_df_open.drop(["Open"], axis=1, inplace=True)

############################################
# store_df                                 #
############################################

# Add "AvgSales" & "AvgCustomers" columns to store_df
avg_sales_customers = training_df.groupby("Store")[["Sales", "Customers"]].mean()
avg_sales_customers_df = DataFrame({"Store": avg_sales_customers.index, "AvgSales": avg_sales_customers["Sales"], "AvgCustomers": avg_sales_customers["Customers"]}, columns=["Store", "AvgSales", "AvgCustomers"])
store_df = pd.merge(avg_sales_customers_df, store_df, on="Store")

# Fill NaN values in store_df for "CompetitionDistance" = 0 (since no record exists where "CD" = NaN & "COS[Y/M]" = !NaN)
store_df["CompetitionDistance"][is_nan(store_df["CompetitionDistance"])] = 0


################################################################
# Plot Data                                                    #
################################################################

############################################
# "Open" Data Field                        #
############################################

# Generate plot for No. of Open or Closed Stores (by Day Of Week)
fig, (axis1) = plt.subplots(1, 1, figsize=(15, 8))
sns.countplot(x="Open", hue="DayOfWeek", data=training_df, palette="husl", ax=axis1)
fig.tight_layout()
fig.savefig("plots/No. of Open or Closed Stores (by Day Of Week).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted No. of Open or Closed Stores (by Day Of Week)")


############################################
# "Date" Data Field & Derivatives          #
############################################

# Generate plots for Avg. Sales & Percentage Change (by Year-Month)
average_sales = training_df.groupby("YearMonth")["Sales"].mean()
pct_change_sales = training_df.groupby("YearMonth")["Sales"].sum().pct_change()
fig, (axis1, axis2) = plt.subplots(2, 1, sharex=True, figsize=(15, 16))
ax1 = average_sales.plot(legend=True, ax=axis1, marker="o", title="Average Sales")
ax1.set_xticks(range(len(average_sales)))
ax1.set_xticklabels(average_sales.index.tolist(), rotation=90)
ax2 = pct_change_sales.plot(legend=True, ax=axis2, marker="o", rot=90, colormap="summer", title="Sales Percent Change")
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Percentage Change (by Year-Month).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Percentage Change (by Year-Month)")

# Generate plots for Avg. Sales & Customers (by Year)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
sns.barplot(x="Year", y="Sales", data=training_df, ax=axis1, ci=None)
sns.barplot(x="Year", y="Customers", data=training_df, ax=axis2, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Customers (by Year).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Customers (by Year)")

# Generate plots for Avg. Sales & Customers (by Month)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
sns.barplot(x="Month", y="Sales", data=training_df, ax=axis1, ci=None)
sns.barplot(x="Month", y="Customers", data=training_df, ax=axis2, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Customers (by Month).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Customers (by Month)")

fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
sns.barplot(x="DayOfMonth", y="Sales", data=training_df, ax=axis1, ci=None)
sns.barplot(x="DayOfMonth", y="Customers", data=training_df, ax=axis2, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Customers (by Day Of Month).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Customers (by Day Of Month)")


############################################
# "DayOfWeek" Data Field                   #
############################################

# Generate plots for Avg. Sales & Customers (by Day of Week)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
sns.barplot(x="DayOfWeek", y="Sales", data=training_df, order=[1, 2, 3, 4, 5, 6, 7], ax=axis1, ci=None)
sns.barplot(x="DayOfWeek", y="Customers", data=training_df, order=[1, 2, 3, 4, 5, 6, 7], ax=axis2, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Customers (by Day of Week).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Customers (by Day of Week)")


############################################
# "Promo" Data Field                       #
############################################

# Generate plots for Avg. Sales & Customers (by Promo)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
sns.barplot(x="Promo", y="Sales", data=training_df, ax=axis1, ci=None)
sns.barplot(x="Promo", y="Customers", data=training_df, ax=axis2, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Customers (by Promo).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Customers (by Promo)")


############################################
# "StateHoliday" Data Field & Derivatives  #
############################################

# Generate plot for No. of State Holidays
fig, (axis1) = plt.subplots(1, 1, figsize=(15, 8))
sns.countplot(x="StateHoliday", data=training_df)
fig.tight_layout()
fig.savefig("plots/No. of State Holidays.png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted No. of State Holidays")

# Generate plots for Avg. Sales & Customers (by State Holiday Binary)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
sns.barplot(x="StateHolidayBinary", y="Sales", data=training_df, ax=axis1, ci=None)
sns.barplot(x="StateHolidayBinary", y="Customers", data=training_df, ax=axis2, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Customers (by State Holiday Binary).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Customers (by State Holiday Binary)")

# Generate plots for Avg. Sales & Customers (by State Holiday Binary for Open Stores)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
sns.barplot(x="StateHolidayBinary", y="Sales", data=training_df_open, ax=axis1, ci=None)
sns.barplot(x="StateHolidayBinary", y="Customers", data=training_df_open, ax=axis2, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Customers (by State Holiday Binary for Open Stores).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Customers (by State Holiday Binary for Open Stores)")

# Generate plots for Avg. Sales & Customers (by State Holiday)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
sns.barplot(x="StateHoliday", y="Sales", data=training_df, ax=axis1, ci=None)
mask = (training_df["StateHoliday"] != "0") & (training_df["Sales"] > 0)
sns.barplot(x="StateHoliday", y="Sales", data=training_df[mask], ax=axis2, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Customers (by State Holiday).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Customers (by State Holiday)")

# Generate plots for Avg. Sales & Customers (by State Holiday for Open Stores)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
sns.barplot(x="StateHoliday", y="Sales", data=training_df_open, ax=axis1, ci=None)
mask = (training_df_open["StateHoliday"] != "0") & (training_df_open["Sales"] > 0)
sns.barplot(x="StateHoliday", y="Sales", data=training_df_open[mask], ax=axis2, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Customers (by State Holiday for Open Stores).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Customers (by State Holiday for Open Stores)")


############################################
# "SchoolHoliday" Data Field               #
############################################

# Generate plot for No. of School Holidays
fig, (axis1) = plt.subplots(1, 1, figsize=(15, 8))
sns.countplot(x="SchoolHoliday", data=training_df)
fig.tight_layout()
fig.savefig("plots/No. of School Holidays.png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted No. of School Holidays")

# Generate plots for Avg. Sales & Customers (by School Holiday)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
sns.barplot(x="SchoolHoliday", y="Sales", data=training_df, ax=axis1, ci=None)
sns.barplot(x="SchoolHoliday", y="Customers", data=training_df, ax=axis2, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Customers (by School Holiday).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Customers (by School Holiday)")

# Generate plots for Avg. Sales & Customers (by School Holiday for Open Stores)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
sns.barplot(x="SchoolHoliday", y="Sales", data=training_df_open, ax=axis1, ci=None)
sns.barplot(x="SchoolHoliday", y="Customers", data=training_df_open, ax=axis2, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Customers (by School Holiday for Open Stores).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Customers (by School Holiday for Open Stores)")


############################################
# "Sales" Data Field                       #
############################################

# Generate plot for Frequency of Sales Values
fig, (axis1) = plt.subplots(1, 1, figsize=(15, 8))
training_df["Sales"].plot(kind="hist", bins=70, xlim=(0, 20000), ax=axis1)
fig.tight_layout()
fig.savefig("plots/Frequency of Sales Values.png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Frequency of Sales Values")

# Generate plot for Frequency of Sales Values (for Open Stores)
fig, (axis1) = plt.subplots(1, 1, figsize=(15, 8))
training_df_open["Sales"].plot(kind="hist", bins=70, xlim=(0, 20000), ax=axis1)
fig.tight_layout()
fig.savefig("plots/Frequency of Sales Values (for Open Stores).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Frequency of Sales Values (for Open Stores)")


############################################
# "Customers" Data Field                   #
############################################

# Generate plot for Frequency of Customers Values
fig, (axis1) = plt.subplots(1, 1, figsize=(15, 8))
training_df["Customers"].plot(kind="hist", bins=70, xlim=(0, 4000), ax=axis1)
fig.tight_layout()
fig.savefig("plots/Frequency of Customers Values.png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Frequency of Customers Values")

# Generate plot for Frequency of Customers Values (for Open Stores)
fig, (axis1) = plt.subplots(1, 1, figsize=(15, 8))
training_df_open["Customers"].plot(kind="hist", bins=70, xlim=(0, 4000), ax=axis1)
fig.tight_layout()
fig.savefig("plots/Frequency of Customers Values (for Open Stores).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Frequency of Customers Values (for Open Stores)")


############################################
# "StoreType" Data Field                   #
############################################

# Generate plot for No. Of Stores (by Store Type)
fig, (axis1) = plt.subplots(1, 1, figsize=(15, 8))
sns.countplot(x="StoreType", data=store_df, order=["a", "b", "c", "d"])
fig.tight_layout()
fig.savefig("plots/No. Of Stores (by Store Type).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted No. Of Stores (by Store Type)")

# Generate plot for Avg. Sales & Customers (by Store Type)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
sns.barplot(x="StoreType", y="AvgSales", data=store_df, order=["a", "b", "c", "d"], ax=axis1, ci=None)
sns.barplot(x="StoreType", y="AvgCustomers", data=store_df, order=["a", "b", "c", "d"], ax=axis2, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Customers (by Store Type).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Customers (by Store Type)")


############################################
# "Assortment" Data Field                  #
############################################

# Generate plot for No. Of Stores (by Assortment)
fig, (axis1) = plt.subplots(1, 1, figsize=(15, 8))
sns.countplot(x="Assortment", data=store_df, order=["a", "b", "c"])
fig.tight_layout()
fig.savefig("plots/No. Of Stores (by Assortment).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted No. Of Stores (by Assortment)")

# Generate plot for Avg. Sales & Customers (by Assortment)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
sns.barplot(x="Assortment", y="AvgSales", data=store_df, order=["a", "b", "c"], ax=axis1, ci=None)
sns.barplot(x="Assortment", y="AvgCustomers", data=store_df, order=["a", "b", "c"], ax=axis2, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Customers (by Assortment).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Customers (by Assortment)")


############################################
# "Promo2" Data Field                      #
############################################

# Generate plot for No. Of Stores (by Promo2)
fig, (axis1) = plt.subplots(1, 1, figsize=(15, 8))
sns.countplot(x="Promo2", data=store_df)
fig.tight_layout()
fig.savefig("plots/No. Of Stores (by Promo2).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted No. Of Stores (by Promo2)")

# # Generate plot for Avg. Sales & Customers (by Promo2)
# fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
# sns.barplot(x="Promo2", y="AvgSales", data=store_df, ax=axis1, ci=None)
# sns.barplot(x="Promo2", y="AvgCustomers", data=store_df, ax=axis2, ci=None)
# fig.tight_layout()
# fig.savefig("plots/Avg. Sales & Customers (by Promo2).png", dpi=fig.dpi)
# fig.clf()
# plt.close(fig)
# print("Plotted Avg. Sales & Customers (by Promo2)")


############################################
# "CompetitionDistance" Data Field         #
############################################

# Generate plot for CompetitionDistance vs. Avg. Sales
fig, (axis1) = plt.subplots(1, 1, figsize=(15, 8))
store_df.plot(kind="scatter", x="CompetitionDistance", y="AvgSales", ax=axis1)
fig.tight_layout()
fig.savefig("plots/Competition Distance vs. Avg. Sales.png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Competition Distance vs. Avg. Sales")

# Generate plot for CompetitionDistance vs. Avg. Customers
fig, (axis1) = plt.subplots(1, 1, figsize=(15, 8))
store_df.plot(kind="scatter", x="CompetitionDistance", y="AvgCustomers", ax=axis1)
fig.tight_layout()
fig.savefig("plots/Competition Distance vs. Avg. Customers.png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Competition Distance vs. Avg. Customers")


############################################
# "CompetitionSince[X]" Data Field         #
############################################

store_id = 6
store_data = training_df[training_df["Store"] == store_id]
average_store_sales = store_data.groupby("YearMonth")["Sales"].mean()

y = store_df["CompetitionOpenSinceYear"].loc[store_df["Store"] == store_id].values[0]
m = store_df["CompetitionOpenSinceMonth"].loc[store_df["Store"] == store_id].values[0]

# Generate plot for Sales for Store <store_id>
fig, (axis1) = plt.subplots(1, 1, figsize=(15, 8))
ax = average_store_sales.plot(legend=True, marker="o", ax=axis1)
ax.set_xticks(range(len(average_store_sales)))
ax.set_xticklabels(average_store_sales.index.tolist(), rotation=90)

# Add vertical line where competition started
if y >= 2013 and y == y and m == m:
    plt.axvline(x=((y - 2013) * 12) + (m - 1), linewidth=3, color="grey")

fig.tight_layout()
fig.savefig("plots/Effect of Competition on Store " + str(store_id) + ".png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Effect of Competition on Store " + str(store_id) + "")


###########################################
# Correlation of Features in training_df  #
###########################################

# One-hot encoding of "DayOfWeek" & "StateHoliday" columns
training_df = pd.get_dummies(training_df, columns=["DayOfWeek", "StateHoliday"])

# Generate correlation matrix for training_df
corr = training_df.corr()
fig, (axis1) = plt.subplots(figsize=(15, 15))
sns.heatmap(corr, square=True, ax=axis1)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
fig.tight_layout()
fig.savefig("plots/Correlation Matrix (training_df).png")
fig.clf()
plt.close(fig)
print("Plotted Correlation Matrix (training_df)")


############################################
# Correlation of Stores                    #
############################################

# Generate table for 1,115 stores and their total sales every month
store_piv = pd.pivot_table(training_df, values="Sales", index="YearMonth", columns=["Store"], aggfunc="sum")

# Generate correlation matrix for all stores
start_store = 1
end_store = 1115
fig, (axis1) = plt.subplots(1, 1, figsize=(100, 100))
sns.heatmap(store_piv[list(range(start_store, end_store + 1))].corr(), square=True, ax=axis1)
fig.tight_layout()
fig.savefig("plots/Correlation Matrix (All Stores).png")
fig.clf()
plt.close(fig)
print("Plotted Correlation Matrix (All Stores)")

# Generate correlation matrix for Stores 1 - 100
start_store = 1
end_store = 100
fig, (axis1) = plt.subplots(1, 1, figsize=(100, 100))
sns.heatmap(store_piv[list(range(start_store, end_store + 1))].corr(), square=True, ax=axis1)
fig.tight_layout()
fig.savefig("plots/Correlation Matrix (Stores 1 - 100).png")
fig.clf()
plt.close(fig)
print("Plotted Correlation Matrix (Stores 1 - 100)")


############################################
# Sales Trends On a Per Day Basis          #
############################################

store_data_sales = training_df.groupby([training_df["Store"]])["Sales"].sum()
store_data_customers = training_df.groupby([training_df["Store"]])["Customers"].sum()
store_data_open = training_df.groupby([training_df["Store"]])["Open"].count()

store_data_sales_per_day = store_data_sales / store_data_open
store_data_customers_per_day = store_data_customers / store_data_open
store_data_sales_per_customer_per_day = store_data_sales_per_day / store_data_customers_per_day

merged_df = pd.merge(store_df, store_data_sales_per_day.reset_index(name="AvgSalesForOpenDays"), how="left", on=["Store"])
merged_df = pd.merge(merged_df, store_data_customers_per_day.reset_index(name="AvgCustomersForOpenDays"), how="left", on=["Store"])
merged_df = pd.merge(merged_df, store_data_sales_per_customer_per_day.reset_index(name="AvgSalesPerCustomerForOpenDays"), how="left", on=["Store"])

fig, axis1 = plt.subplots(1, 1, figsize=(15, 8))
sns.barplot(x="Store", y="AvgSalesForOpenDays", data=merged_df, ax=axis1, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales for Open Days (by Store).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)

fig, axis1 = plt.subplots(1, 1, figsize=(15, 8))
sns.barplot(x="Store", y="AvgCustomersForOpenDays", data=merged_df, ax=axis1, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Customers for Open Days (by Store).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)

fig, axis1 = plt.subplots(1, 1, figsize=(15, 8))
sns.barplot(x="Store", y="AvgSalesPerCustomerForOpenDays", data=merged_df, ax=axis1, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales per Customer for Open Days (by Store).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)

fig, axis1 = plt.subplots(1, 1, figsize=(15, 8))
sns.barplot(x="StoreType", y="AvgSalesForOpenDays", order=["a", "b", "c", "d"], data=merged_df, ax=axis1, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales for Open Days (by Store Type).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)

fig, axis1 = plt.subplots(1, 1, figsize=(15, 8))
sns.barplot(x="StoreType", y="AvgCustomersForOpenDays", order=["a", "b", "c", "d"], data=merged_df, ax=axis1, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Customers for Open Days (by Store Type).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)

fig, axis1 = plt.subplots(1, 1, figsize=(15, 8))
sns.barplot(x="StoreType", y="AvgSalesPerCustomerForOpenDays", order=["a", "b", "c", "d"], data=merged_df, ax=axis1, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales per Customer for Open Days (by Store Type).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
