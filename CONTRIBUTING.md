## Contributing Guidelines

When trying a new model:

1. Create a branch named `model-name`.
2. Duplicate `template.py` and rename it to `model-name.py`.
3. Make sure the new model outputs its predictions to `predictions/model-name.csv`.

After submitting the `.csv` file to Kaggle, add the score of the model to the top of `model-name.py` in the following format:

```python
"""
Private Score: XYZ, Public Score: XYZ
"""
```

Submit a pull request to `master` once done. After merging with `master`, add details about the model to `List of Models.md`.
