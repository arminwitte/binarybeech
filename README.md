# binarybeech
Simplistic algorithms to train decision trees for regression and classification

> **NOTE:**  These pure python (and a bit of numpy) algorithms are many times slower than, e.g., `sklearn` or `xgboost`.

## Example

```
from binarybeech.binarybeech import CART
df_titanic = pd.read_csv("data/titanic.csv")
c = CART(df_titanic,"Survived", metrics_type="classification")
c.create_tree()
p = c.predict(df_titanic)
val = c.validate()
```

Please have a look at the jupyter notebooks in this repository for more examples. To try them out online you can use [mybinder](https://mybinder.org/v2/gh/arminwitte/binarybeech/HEAD?labpath=treeGradientBoost.ipynb).

## Usage

## Performance
### Kaggle

## Sources

[Decision tree](https://en.m.wikipedia.org/wiki/Decision_tree)

[CART](https://de.m.wikipedia.org/wiki/CART_(Algorithmus))

[Gradient Boosted Tree](https://en.m.wikipedia.org/wiki/Gradient_boosting)

[Random Forest](https://de.m.wikipedia.org/wiki/Random_Forest)

[pruning](https://online.stat.psu.edu/stat508/lesson/11/11.8/11.8.2)

## Contributions