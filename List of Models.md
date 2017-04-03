## List of Models

- `linearregression.py`
    - Private Score: 0.16940, Public Score: 0.14870
    - Features: Promo, SchoolHoliday, DayOfWeek (one-hot encoded), StateHoliday (one-hot encoded)
    - Treating each store as an independent regression problem, loop through all stores training the model for the particular store and predicting its sales value.
- `simplemedian.py`
    - Private Score: 0.14598, Public Score: 0.14001
    - Features: Store, DayOfWeek, Promo
    - This model simply calculates the median value for every ["Store", "DayOfWeek", "Promo"] combination and assigns that value as the prediction for every ["Store", "DayOfWeek", "Promo"] combination in the test data.
