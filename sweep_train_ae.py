import numpy as np
import os

def main():
    for kld_weight in np.logspace(-7, -5, 20, endpoint=True):
        os.system(f"python train_ae.py --kld_weight {kld_weight}")



if __name__ == "__main__":
    main()

