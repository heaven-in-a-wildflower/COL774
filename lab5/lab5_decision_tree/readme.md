# COL774 Assignment 5 - Decision Trees

### Contributors
- Satwik
- Aneeket

## Methods Implemented
1. **Pruning**: We utilized `train.csv` for pruning instead of `val.csv`.
2. **Parameter Tuning**: We adjusted only the `max_depth` parameter, setting it to 9.

## Results
- Achieved a final accuracy of **79.4%** on the provided test set.

## Experimentation
We explored several techniques:
- **Cost Complexity Pruning**: Implemented to avoid overfitting on the small dataset.
- **Random Forest**: Experimented with, but ultimately discarded due to assignment restrictions on ensemble methods.
- **XGBoost**: Attempted initially, but discarded after realizing that boosting and ensemble-based methods were not permitted.

## Notes
- Since the test case was relatively small, we focused on keeping the model general to prevent overfitting.
