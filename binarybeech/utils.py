import numpy as np

def print_bars(d, max_width=70):
    max_val = max(d.values)
    for key, val in d.items:
        l = int(round(50 * val / max_val))
        print(f"{key:10}|{"#"*l}{}{val:4.2})
    
