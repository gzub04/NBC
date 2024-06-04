import pandas as pd
import numpy as np


class NBC:
    def __init__(self, df=None, k=None):
        self.features_no = 0
        if isinstance(df, pd.DataFrame) or df is None:
            self.df = df
        else:
            raise TypeError('Given variable is not of pandas DataFrame type')

        if isinstance(k, int) or k is None:
            self.k = k
        else:
            raise TypeError('k must be an integer')

    def set_df(self, df):
        if isinstance(df, pd.DataFrame):
            self.df = df
        else:
            raise TypeError('Given variable is not of pandas DataFrame type')

    def set_k(self, k):
        if isinstance(k, int):
            self.k = k
        else:
            raise TypeError('K must be an integer')

    def check_data(self):
        """
        Checks if all data was provided and sets constants for fitting
        :return: True if data is correct
        """
        is_correct = True
        if self.df is None:
            print('Error: DataFrame is not declared')
            is_correct = False
        if self.k is None or self.k > len(self.df.index):
            print('Error: k is not declared or is larger than the data size')
            is_correct = False
        self.features_no = len(self.df.columns) - 1
        return is_correct

    def _tidy_data(self):
        """
        Takes a pandas dataframe and:
        - renames the column with true classifications to class_no
        - normalize data
        - reorders columns according to manhattan distance and saves them in new column 'distance'
        -
        :return: Nothing
        """
        self.df.rename(columns={self.features_no + 1: 'class_no'}, inplace=True)
        data_df = self.df.iloc[:, :-1]
        self.df.iloc[:, :-1] = (data_df - data_df.min()) / (data_df.max() - data_df.min())

        manhattan_distances = self.df.iloc[:, :-1].sum(axis=1)
        self.df['distance'] = manhattan_distances
        self.df.sort_values(by='distance', ascending=True, inplace=True)

        self.df['r-k-nearest'] = 0

    def gower_distance(self, idx_1, idx_2, ):
        x1 = self.df.iloc[idx_1]
        x2 = self.df.iloc[idx_2]
        numerator = 0
        for feature_idx in range(self.features_no):
            feature_type = self.df.iloc[:, feature_idx].dtype
            if feature_type == 'object' and x1[feature_idx] != x2[feature_idx]:
                numerator += 1
            else:
                if x1[feature_idx] != x2[feature_idx]:
                    numerator += abs(x1[feature_idx] - x2[feature_idx])
        return numerator / self.features_no  # -2, bo dwie (ostatnie) kolumny nie są danymi

    def _k_nearest_neighbors(self, data_idx):
        nearest_neighbors = np.empty(shape=self.k, dtype=int)
        gower_distances = np.empty(shape=self.k, dtype=float)

        offset_lower = 1
        offset_higher = 1
        lowest_index_reached = False
        highest_index_reached = False
        eps = [0, 0]  # [local index, epsilon]

        # Initial values for nearest neighbours
        for i in range(self.k):
            if lowest_index_reached:
                nearest_neighbors[i] = data_idx + offset_higher
                offset_higher += 1
            elif highest_index_reached:
                nearest_neighbors[i] = data_idx - offset_lower
                offset_lower += 1

            elif i % 2 == 0:
                if data_idx + offset_higher < len(self.df.index):
                    nearest_neighbors[i] = data_idx + offset_higher
                    offset_higher += 1
                else:
                    highest_index_reached = True
                    nearest_neighbors[i] = data_idx - offset_lower
                    offset_lower += 1

            elif data_idx - offset_lower >= 0:
                nearest_neighbors[i] = data_idx - offset_lower
                offset_lower += 1
            else:  # There are still higher indexes, but it was "i % 2 != 0"
                lowest_index_reached = True
                nearest_neighbors[i] = data_idx + offset_higher
                offset_higher += 1

            gower_distances[i] = self.gower_distance(data_idx, nearest_neighbors[i])
            if eps[1] < gower_distances[i]:
                eps[0] = i
                eps[1] = gower_distances[i]

        # check if there aren't any closer k's
        while True:
            if (data_idx + offset_higher < len(self.df.index)
                    and np.abs(self.df.loc[data_idx + offset_higher, 'distance'] - self.df.loc[data_idx, 'distance']) <
                    eps[1]):
                potential_idx = data_idx + offset_higher
                offset_higher += 1
            elif (data_idx - offset_lower >= 0
                  and np.abs(self.df.loc[data_idx - offset_lower, 'distance'] - self.df.loc[data_idx, 'distance']) <
                  eps[1]):
                potential_idx = data_idx - offset_lower
                offset_lower += 1
            else:
                break

            distance = self.gower_distance(data_idx, potential_idx)
            if distance < eps[1]:
                nearest_neighbors[eps[0]] = potential_idx
                gower_distances[eps[0]] = distance
                eps[0] = np.argmax(gower_distances)
                eps[1] = gower_distances[eps[0]]

        return nearest_neighbors

    def fit(self, x, y):
        if self.check_data() is False:
            return

        self._tidy_data()

        k_nearest = pd.Series([np.array] * len(self.df.index))
        for i in range(1261):
            k_nearest[i] = self._k_nearest_neighbors(i)
            # reverse k-neighbours
            for index in k_nearest[i]:
                self.df.loc[index, 'r-k-nearest'] += 1
        self.df.insert = k_nearest
