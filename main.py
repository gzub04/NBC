import arff
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import rand_score

from nbc import NBC

DATASETS_PATH = "./datasets/"


def main():
    with open(DATASETS_PATH + "2d-4c.arff", "r") as fp:
        dataset = arff.load(fp)
    df = pd.DataFrame(dataset['data'])

    nbc_obj = NBC(df, 131)
    nbc_obj.fit(0, 1)

    print(f'Number of groups: {nbc_obj.df['grouping'].max() + 1}')
    print(f'Rand metric: {rand_score(nbc_obj.df["class_no"], nbc_obj.df["grouping"])}')
    # print(f'Rand metric: {nbc_obj.rand_metric()*100:.2f}%')
    colors = {-1: 'black', 0: 'blue', 1: 'green', 2: 'orange', 3: 'purple', 4: 'gray',
              5: 'brown', 6: 'pink', 7: 'gray', 8: 'olive', 9: 'cyan'}
    nbc_obj.df.iloc[:, :2].plot.scatter(x=0, y=1, c=nbc_obj.df['grouping'].map(colors))
    plt.show()
    pass


if __name__ == '__main__':
    main()
