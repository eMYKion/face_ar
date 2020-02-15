import pandas as pd
import argparse

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="save some sample data to a file")
    parser.add_argument("-f", action="store", dest="filename", type=str, required=True)
    args = parser.parse_args()

    data = [
        ["Chris Seiler", "male", 2],
        ["Audrey Tzeng", "female", 2],
        ["George Ralph", "male", 2],
        ["Mayank Mali", "male", 3]
    ]

    df = pd.DataFrame(data=data, columns=["name", "gender", "year"])
    df = df.set_index("name")

    df.to_csv(args.filename)

    print("file saved to preview:\n", df.head())




