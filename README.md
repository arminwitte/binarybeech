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

Please have a look at the jupyter notebooks in this repository for more examples.

## Usage

## Performance
### Kaggle

## Sources

## Contributions