# binarybeech
Simplistic algorithms to train decision trees for regression and classification

> **NOTE:**  These pure python (and a bit of numpy) algorithms are many times slower than, e.g., `sklearn` or `xgboost`.

## Principle
Decision trees are, by design, data type agnostic. With only a few methods like _spliter_ for input variables and meaningful quantification for the _loss_, any data type can be perused. In this code, this is implemented using a factory pattern for _data handling_ and _metrics_ making decision tree learing simple and versatile.

## Example

Load the Classification And Regression Tree model class

```
import pandas as pd
from binarybeech.binarybeech import CART
from binarybeech.utils import k_fold_split
```
get the data from a csv file
```
df = pd.read_csv("data/titanic.csv")
[(df_train, df_test)] = k_fold_split(df,frac=0.75,random=True,replace=False)
```
grow a decision tree
```
c = CART(df_train,"Survived", metrics_type="classification")
c.create_tree()
```
predict
```
c.predict(df_test)
```
validation metrics
```
c.validate(df=df_test)
```

Please have a look at the jupyter notebooks in this repository for more examples. To try them out online, you can use [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arminwitte/binarybeech/HEAD?labpath=notebooks%2Ftitanic.ipynb).

## Usage
### binarybeech.binarybeech.CART
**CART(df, y_name, X_names=None, min_leaf_samples=1, min_split_samples=1, max_depth=10, metrics_type="regression", handle_missings="simple", data_handlers=None)**

Class for a Classification and Regression Tree (CART) model.

* Parameters
    - **df**: pandas _dataframe_ with training data
    - **y_name**: name of the column with the output data/labels
    - **X_names**: _list_ of names with the inputs to use for the modelling. If _None_, all columns except y_name are chosen. Default is _None_.
    - **min_leaf_samples**: If the number of training samples is lower than this, a terminal node (leaf) is created. Default is 1.
    - **min_split_samples**: If a split of the training data is proposed with at least one branch containing less samples than this, the split is rejected. Default is 1.
    - **max_depth**: Maximum number of sequential splits. This corresponds to the number of vertical layers of the tree. Default is 10, which corresponds to a maximum number of 1024 terminal nodes.
    - **metrics_type**: Metrics to use for the evaluation of split loss, etc. Can be either "classification", "logistic", "regression", or _None_. Default is "regression". If _None_ is chosen, the `metrics_type` is deduced from the training _dataframe_.
    - **handle_missings**: Specify the way how missing data is handeled. Can be eiter _None_ or "simple".
    - **data_handlers**: _dict_ with data handler instances for each variable. The data handler determins, e.g., how splits of the dataset are made.
* Methods
    - **predict(df)**:
        + Parameters:
            * **df**: _dataframe_ with inputs for predictions.
        + Returns:
            * array with predicted values/labels.
    - **train(k=5, plot=True, slack=1.0)**:
        + Parameters:
            * **k**: number of different splits of the _dataframe_ into training and test sets for k-fold cross-validation.
            * **plot**: flag for plotting a diagram of the loss over cost complexity parameter alpha using _matplotlib_.
            * **slack**: the amount of slack granted in chosing the best cost complexity parameter alpha. It is given as multiplier for the standard deviation of the alpha at minimum loss and allows thus to chose an alpha that is probably larger to account for the uncertainty in the k-fold cross validation procedure.
        + Returns:
    - **create_tree(leaf_loss_threshold=1e-12)**
        + Returns
    - **prune(alpha_max=None, test_set=None, metrics_only=False)**
        + Parameters:
            * **alpha_max**: Stop the pruning procedure at this value of the cost complexity parameter alpha. If _None_, the tree is pruned down to its root giving the complete relationship between alpha and the loss. Default is _None_.
            * **test_set**: data set to use for the evaluation off the losses. If _None_, the training set is used. Default is _None_.
            * **metrics_only**: If _True_, pruning is performed on a copy of the tree, leaving the actual tree intact. Default is _False_
    - **validate(df=None)**
        + Parameters:
            * **df**: _dataframe_ to use for (cross-)validation. If _None_, the training set is used. Default is _None_.
        + Returns:
            * _dict_ with metrics, e.g. accuracy or RSquared.
* Attributes
    - **tree**:

### binarybeech.binarybeech.GradientBoostedTree

**GradientBoostedTree(df, y_name, X_names=None, sample_frac=1, n_attributes=None, learning_rate=0.1, cart_settings={}, init_metrics_type="logistic", gamma=None, handle_missings="simple", data_handlers=None)**

Class for a Gradient Boosted Tree model.

* Parameters
    - **df**: pandas _dataframe_ with training data
    - **y_name**: name of the column with the output data/labels
    - **X_names**: _list_ of names with the inputs to use for the modelling. If _None_, all columns except y_name are chosen. Default is _None_.
    - **sample_frac**: fraction (0, 1] of the training data to use for the training of an individual tree of the ensemble. Default is 1.
    - **n_attributes**: number of attributes (elements of the X_names list) to use for the training of an individual tree of the ensemble. Default is _None_ which corresponds to all available attributes.
    - **learning_rate**: the shinkage parameter used to "downweight" individual trees of the ensemble. Default is 0.1.
    - **cart_settings**: _dict_ that is passed on to the constuctor of the individual tree (binarybeech.binarybeech.CART). For details cf. above.
    - **init_metrics_type**: Metrics to use for the evaluation of split loss, etc if the initial tree (stump). Can be either "classification", "logistic", "regression", or _None_. Default is "regression". If _None_ is chosen, the `metrics_type` is deduced from the training _dataframe_.
    - **gamma**: weight for individual trees of the ensemble. If _None_, the weight for each tree is chosen by line search minimizing the loss given by _init_metrics_type_.
    - **handle_missings**: Specify the way how missing data is handeled. Can be eiter _None_ or "simple".
    - **data_handlers**: _dict_ with data handler instances for each variable. The data handler determins, e.g., how splits of the dataset are made.
* Methods
    - **predict(df)**
        + Parameters:
            * **df**: _dataframe_ with inputs for predictions.
        + Returns:
            * array with predicted values/labels.
    - **train(M)**
        + Parameters:
            * **M**: Number of individual trees to create for the ensemble.
        + Returns:
    - **validate(df=None)**
        + Parameters:
            * **df**: _dataframe_ to use for (cross-)validation. If _None_, the training set is used. Default is _None_.
        + Returns:
            * _dict_ with metrics, e.g. accuracy or RSquared.
* Attributes
    - **trees**

### binarybeech.binarybeech.RandomForest

**RandomForest(df, y_name, X_names=None, verbose=False, sample_frac=1, n_attributes=None, cart_settings={}, metrics_type="regression", handle_missings="simple", data_handlers=None)**

Class for a Random Forest model.

* Parameters
    - **df**: pandas _dataframe_ with training data
    - **y_name**: name of the column with the output data/labels
    - **X_names**: _list_ of names with the inputs to use for the modelling. If _None_, all columns except y_name are chosen. Default is _None_.
    - **verbose**: if set to _True_, status messages are sent to stdout. Default is _False_.
    - **sample_frac**: fraction (0, 1] of the training data to use for the training of an individual tree of the ensemble. Default is 1.
    - **n_attributes**: number of attributes (elements of the X_names list) to use for the training of an individual tree of the ensemble. Default is _None_ which corresponds to all available attributes.
    - **cart_settings**: _dict_ that is passed on to the constuctor of the individual tree (binarybeech.binarybeech.CART). For details cf. above.
    - **metrics_type**: Metrics to use for the evaluation of split loss, etc. Can be either "classification", "logistic", "regression", or _None_. Default is "regression". If _None_ is chosen, the `metrics_type` is deduced from the training _dataframe_.
    - **handle_missings**: Specify the way how missing data is handeled. Can be eiter _None_ or "simple".
    - **data_handlers**: _dict_ with data handler instances for each variable. The data handler determins, e.g., how splits of the dataset are made.
* Methods
    - **predict(df)**
        + Parameters:
            * **df**: _dataframe_ with inputs for predictions.
        + Returns:
            * array with predicted values/labels.
    - **train(M)**
        + Parameters:
            * **M**: Number of individual trees to create for the ensemble.
        + Returns:
    - **validate(df=None)**
        + Parameters:
            * **df**: _dataframe_ to use for (cross-)validation. If _None_, the training set is used. Default is _None_.
        + Returns:
            * _dict_ with metrics, e.g. accuracy or RSquared.
    - **validate_oob()**:
        + Returns:
            * _dict_ with metrics, e.g. accuracy or RSquared.
    - **variable_importance()**:
        + Returns:
            * _dict_ with normalized importance values.
* Attributes

For more information please feel free to take a look at the code.

## Performance

### Kaggle

## Sources

[Decision tree](https://en.m.wikipedia.org/wiki/Decision_tree)

[CART](https://de.m.wikipedia.org/wiki/CART_(Algorithmus))

[Gradient Boosted Tree](https://en.m.wikipedia.org/wiki/Gradient_boosting)

[Random Forest](https://de.m.wikipedia.org/wiki/Random_Forest)

[pruning](https://online.stat.psu.edu/stat508/lesson/11/11.8/11.8.2)

## Contributions
Contributions in the form of pull requests are always welcome.