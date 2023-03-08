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
* Parameters
* Methods

## Performance
### Kaggle

## Sources

[Decision tree](https://en.m.wikipedia.org/wiki/Decision_tree)

[CART](https://de.m.wikipedia.org/wiki/CART_(Algorithmus))

[Gradient Boosted Tree](https://en.m.wikipedia.org/wiki/Gradient_boosting)

[Random Forest](https://de.m.wikipedia.org/wiki/Random_Forest)

[pruning](https://online.stat.psu.edu/stat508/lesson/11/11.8/11.8.2)

## Contributions
