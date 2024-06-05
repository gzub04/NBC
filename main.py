import arff
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import adjusted_rand_score

from nbc import NBC

DATASETS_PATH = "./datasets/"


def main():
    with open(DATASETS_PATH + "2d-20c-no0.arff", "r") as fp:
        dataset = arff.load(fp)
    df = pd.DataFrame(dataset['data'])
    nbc = NBC(df)
    k_vals = []
    rand_scores = []
    adjusted_rand_scores = []
    # for i in range(18):
    #     k = 3 + i*10
    #     k_vals.append(k)
    #     nbc.set_df(df)
    #     nbc.set_k(k)
    #     nbc.fit()
    #     rand_scores.append(rand_score(nbc.df['class_no'], nbc.df['grouping']))
    #     adjusted_rand_scores.append(adjusted_rand_score(nbc.df['class_no'], nbc.df['grouping']))
    #     print(f'k = {k}:\n'
    #           f'Number of groups: {nbc.df['grouping'].max() + 1}\n'
    #           f'Rand score: {rand_scores[i]}\n'
    #           f'Adjusted rand score: {adjusted_rand_scores[i]}\n')
    #
    # plt.plot(k_vals, adjusted_rand_scores)
    # plt.title("k vs. Adjusted rand score")
    # plt.xlabel("k")
    # plt.ylabel("Adjusted rand score")
    # plt.show()

    nbc.set_k(23)
    nbc.set_df(df)
    nbc.fit()



    # nbc_obj = NBC(df, 131)
    # nbc_obj.fit()
    #
    # print(f'Number of groups: {nbc_obj.df['grouping'].max() + 1}')
    # print(f'Rand metric: {rand_score(nbc_obj.df["class_no"], nbc_obj.df["grouping"])}')
    # # print(f'Rand metric: {nbc_obj.rand_metric()*100:.2f}%')
    # colors = {-1: 'black', 0: 'blue', 1: 'green', 2: 'orange', 3: 'purple', 4: 'gray',
    #           5: 'brown', 6: 'pink', 7: 'gray', 8: 'olive', 9: 'cyan'}
    # nbc_obj.df.iloc[:, :2].plot.scatter(x=0, y=1, c=nbc_obj.df['grouping'].map(colors))
    plt.show()
    pass


if __name__ == '__main__':
    main()
