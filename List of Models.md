## List of Models

- `linearregression.py`
    - Private Score: 0.16939, Public Score: 0.14867
    - Features: Promo, SchoolHoliday, DayOfWeek (one-hot encoded), StateHoliday (one-hot encoded)
    - Each store is an independent regression problem.
- `linearregression2.py`
    - Private Score: 0.16405, Public Score: 0.14499
    - Features: Promo, DayOfWeek(one-hot encoded)
    - Each store is an independent regression problem.
- `simplemedian.py`
    - Private Score: 0.14598, Public Score: 0.14001
    - Features: Store, DayOfWeek, Promo
    - This model simply calculates the median value for every ["Store", "DayOfWeek", "Promo"] combination and assigns that value as the prediction for every ["Store", "DayOfWeek", "Promo"] combination in the test data.
