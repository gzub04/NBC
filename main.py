import arff
import pandas as pd

from nbc import NBC

DATASETS_PATH = "./datasets/"


def main():
    with open(DATASETS_PATH + "2d-4c.arff", "r") as fp:
        dataset = arff.load(fp)
    df = pd.DataFrame(dataset['data'])

    nbc_obj = NBC(df, 3)
    nbc_obj.fit(0, 1)
    pass


if __name__ == '__main__':
    main()
