import argparse
import pandas as pd
import os
from PIL import Image

def main():
    print("DEBUG")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to the input data")
    args = parser.parse_args()
    input_data = args.input_data

    print(f"Test Datastore")
    print('-' * 10)
    print(f"Datastore Path: ${input_data}")

    test_url = "S1/B04/B04_R1/S1_B04_R1_PICT0001.JPG"

    path = os.path.join(input_data, test_url)
    print(path)
    image = Image.open(path)
    print(image.size)


if __name__ == '__main__':
    main()