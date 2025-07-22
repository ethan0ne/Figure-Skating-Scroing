

import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", type=str, required=True, help="Path to the .npy file")
    args = parser.parse_args()

    feature = np.load(args.npy)
    print(f"✅ Loaded: {args.npy}")
    print(f"Shape: {feature.shape}")
    print(f"Dtype: {feature.dtype}")
    print(f"First 5 rows:\n{feature[:5]}")

if __name__ == "__main__":
    main()