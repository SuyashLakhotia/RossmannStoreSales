"""
Current Best Result:
- Private Score: 0.11880, Public Score: 0.10648, i = 0.67, j = 0.970

Past Results:
- Private Score: 0.12057, Public Score: 0.10934, i = 0.75, j = 0.99
- Private Score: 0.12015, Public Score: 0.10561, i = 0.5, j = 0.985
- Private Score: 0.11919, Public Score: 0.10514, i = 0.5, j = 0.975
- Private Score: 0.11904, Public Score: 0.10527, i = 0.5, j = 0.970
- Private Score: 0.11882, Public Score: 0.10616, i = 0.65, j = 0.970
"""

import pandas as pd
import numpy as np
import sys

pd.options.mode.chained_assignment = None

determineBestWeights = False

# Run the script with determine-best-weights as argument for determine the best set of weights using the validation set (```python xgboostensemble.py determine-best-weights```)
if (len(sys.argv) > 1) and (sys.argv[1] == "determine-best-weights"):
    determineBestWeights = True

################################################################
# Import CSV Data into Pandas DataFrames                       #
################################################################

if determineBestWeights:
    file1 = "predictions/xgboostregressor-log5-validate.csv"
    file2 = "predictions/xgboostregressor-log-validate.csv"
else:
    file1 = "predictions/xgboostregressor-log5.csv"
    file2 = "predictions/xgboostregressor-log.csv"

model1 = pd.read_csv(file1)
model2 = pd.read_csv(file2)

################################################################
# Process Data                                                 #
################################################################

# Get the predictions from the both the datasets and the true sales value
sales_model1 = model1["Sales"]
sales_model2 = model2["Sales"]


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
# Making predictions                                  		   #
################################################################
"""
This model ensembles two xgboost models using static combination with weighted averages.

The determineBestWeights flag can be set to True to determine best possible weights with a set of local predictions.
"""

if (determineBestWeights):
    # Determining the lowest RMSPE value possible with different sets of weights and correction factors
    # y_true is the true sales value
    y_true = model1["True"]
    # A dictionary to store rmspe values and corresponding weights
    rmspeDict = {}

    # i is the weight ratios and j is the correction factor
    for i in np.arange(0.0, 1.0, 0.05):
        for j in np.arange(0.9, 1.0, 0.005):
            y_pred = (sales_model1 * i + sales_model2 * (1.0 - i)) * j
            rmspeValue = rmspe(y_true, y_pred)
            weightTuple = (i, j)
            rmspeDict[weightTuple] = rmspeValue

    minRMSPE = min(rmspeDict.values())
    weight_correction = []
    for key, value in rmspeDict.items():
        if value == minRMSPE:
            weight_correction = key
            break
    bestRatios = [minRMSPE, weight_correction]
    print("Minimum RMSPE Score = " + str(bestRatios))
else:
    # Predictions using static combining
    i = 0.67
    j = 0.97
    sales = (sales_model1 * i + sales_model2 * (1.0 - i)) * j

    result = pd.DataFrame({"Id": model2["Id"], "Sales": sales})
    result.to_csv("predictions/xgboostensemble.csv", index=False)
    print("Predictions saved to predictions/xgboostensemble.csv.")
