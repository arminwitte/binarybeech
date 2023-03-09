# binarybeech
Simplistic algorithms to train decision trees for regression and classification

> **NOTE:**  These pure python (and a bit of numpy) algorithms are many times slower than, e.g., `sklearn` or `xgboost`.

## Principle
Decision trees are, by design, data type agnostic. With only a few methods like a 'spliter' for input variables and meaningful quantification for 'losses', any data type can be perused. In this code, this is implemented using a factory pattern for data handling and metrics making decision tree learing simple and versatile.

## Example

Load the Classification And Regression Tree model class

```
from binarybeech.binarybeech import CART
```
get the data from a csv file
```
df_titanic = pd.read_csv("data/titanic.csv")
```
grow a decision tree
```
c = CART(df_titanic,"Survived", metrics_type="classification")
c.create_tree()
```
predict
```
p = c.predict(df_titanic)
```
validation metrics
```
val = c.validate()
```

Please have a look at the jupyter notebooks in this repository for more examples. To try them out online you can use [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arminwitte/binarybeech/HEAD?labpath=notebooks%2Ftitanic.ipynb).

## Usage
### binarybeech.binarybeech.CART
**`CART()`**

```CART(df, y_name, X_names=None, min_leaf_samples=1, min_split_samples=1, max_depth=10, metrics_type="regression", handle_missings="simple", data_handlers=None)```

* Parameters
    - **df**: pandas dataframe woth training data
    - **y_name**: name of the column with the output data/labels
    - **X_names**: list of names with the inputs to use for the modelling. If `None`, all columns except y_name are chosen. Default is `None`.
    - **min_leaf_samples**: If the number of training samples is lower than this, a terminal node (leaf) is created. Default is 1.
    - **min_split_samples**: If a split of the training data is proposed with at least one branch containing less samples than this, the split is rejected. Default is 1.
    - **max_depth**: Maximum number of sequential splits. This corresponds to the number of vertical layers of the tree. Default is 10, which corresponds to a maximum number of 1024 terminal nodes.
    - **metrics_type**: Metrics to use for the evaluation of split loss, etc. Can be either 'classification', 'logistic', 'regression', or None. Default is 'regression'. If None is chosen, the `metrics_type` is deduced from the training dataframe.
    - **handle_missings**: Specify the way how missing data is handeled. Can be eiter None or `simple`.
    - **data_handlers**: dict with data handler instances for each variable. The data handler determins, e.g., how splits of the dataset are made.
* Methods
    - `train()`:
* Attributes
    - **tree**:

## Performance
### Kaggle

## Sources

[Decision tree](https://en.m.wikipedia.org/wiki/Decision_tree)

[CART](https://de.m.wikipedia.org/wiki/CART_(Algorithmus))

[Gradient Boosted Tree](https://en.m.wikipedia.org/wiki/Gradient_boosting)

[Random Forest](https://de.m.wikipedia.org/wiki/Random_Forest)

[pruning](https://online.stat.psu.edu/stat508/lesson/11/11.8/11.8.2)

## Contributions
