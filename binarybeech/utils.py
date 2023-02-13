import numpy as np

def print_bars(d, max_width=70):
    for key, val in d.items():
        l = int(round(50 * val))
        print(f"{key:10}|{"#"*l}{}{val:4.2}
    
