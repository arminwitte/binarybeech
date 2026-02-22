import json
import sys
import pandas as pd

from binarybeech.binarybeech import CART


def dump_prostate(mode="create"):
    df_prostate = pd.read_csv("data/prostate.data", sep="\t")
    train = df_prostate["train"].isin(["T"])
    df_prostate = df_prostate.drop(columns=["Unnamed: 0", "train"])

    if mode == "create":
        c = CART(df=df_prostate[train], y_name="lpsa", method="regression:regularized", seed=42)
    elif mode == "l1":
        c = CART(df=df_prostate, y_name="lpsa", method="regression:regularized", seed=42, lambda_l1=0.1, lambda_l2=0.0)
    elif mode == "l2":
        c = CART(df=df_prostate, y_name="lpsa", method="regression:regularized", seed=42, lambda_l1=0.0, lambda_l2=0.1)
    else:
        raise ValueError(mode)

    c.create_tree()
    t = c.tree
    out = {
        "leaf_count": t.leaf_count(),
        "nodes": [],
    }
    for n in t.nodes():
        d = {
            "is_leaf": n.is_leaf,
            "attribute": n.attribute,
            "threshold": None if n.threshold is None else str(n.threshold),
            "value": None if n.value is None else float(n.value) if hasattr(n.value, 'item') or isinstance(n.value, (int,float)) else str(n.value),
            "pinfo": n.pinfo,
        }
        out["nodes"].append(d)
    print(json.dumps(out, default=str))


def dump_titanic():
    df_titanic = pd.read_csv("data/titanic.csv")
    c = CART(df=df_titanic, y_name="Survived", method="classification")
    c.create_tree()
    t = c.tree
    out = {
        "leaf_count": t.leaf_count(),
        "nodes": [],
    }
    for n in t.nodes():
        d = {
            "is_leaf": n.is_leaf,
            "attribute": n.attribute,
            "threshold": None if n.threshold is None else str(n.threshold),
            "value": None if n.value is None else n.value,
            "pinfo": n.pinfo,
        }
        out["nodes"].append(d)
    print(json.dumps(out, default=str))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: tmp_dump_cart.py [prostate-create|prostate-l1|prostate-l2|titanic]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == 'prostate-create':
        dump_prostate('create')
    elif cmd == 'prostate-l1':
        dump_prostate('l1')
    elif cmd == 'prostate-l2':
        dump_prostate('l2')
    elif cmd == 'titanic':
        dump_titanic()
    else:
        print('unknown')
