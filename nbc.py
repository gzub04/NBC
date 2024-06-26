import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import scipy


class NBC:
    """
    Assumes true values are in the last row
    """
    def __init__(self, df=None, k=None):
        self.features_no = 0
        if isinstance(df, pd.DataFrame):
            self.df = df.copy()
        elif df is None:
            self.df = None
        else:
            raise TypeError('Given variable is not of pandas DataFrame type')

        if isinstance(k, int) or k is None:
            self.k = k
        else:
            raise TypeError('k must be an integer')

    def set_df(self, df):
        if isinstance(df, pd.DataFrame):
            self.df = df.copy()
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

    def calculate_manhattan_distance(self):
        """
        :return: Manhattan distance
        """
        distances = pd.Series(np.zeros(shape=len(self.df.index)), name='distance')
        for row in self.df.index:
            for col in self.df.columns:
                if is_numeric_dtype(self.df[col].dtype):
                    distances[row] += self.df.loc[row, col]
        self.df = pd.concat([self.df, distances.T], axis=1)

    def _tidy_data(self):
        """
        Takes a pandas dataframe and:
        - renames the column with true classifications to class_no
        - normalize data
        - reorders columns according to manhattan distance and saves them in new column 'distance'
        - adds necessary columns
        - resets index
        :return: Nothing
        """
        self.df.rename(columns={self.df.columns[self.features_no]: 'class_no'}, inplace=True)
        df_num = self.df.iloc[:, :-1].select_dtypes(include='number')
        df_norm = (df_num - df_num.min()) / (df_num.max() - df_num.min())
        if not df_norm.empty:
            self.df[df_norm.columns] = df_norm

        self.calculate_manhattan_distance()
        self.df.sort_values(by='distance', ascending=True, inplace=True)

        r_k_nearest = pd.Series(np.full(len(self.df.index), 0), name='r-k-nearest')
        NDF = pd.Series(np.full(len(self.df.index), np.nan), name='NDF')
        grouping = pd.Series(np.full(len(self.df.index), -1), name='grouping')

        self.df = pd.concat([self.df, r_k_nearest.T, NDF.T, grouping.T], axis=1)

        self.df.reset_index(drop=True, inplace=True)

    def _gower_distance(self, idx_1, idx_2):
        x1 = self.df.iloc[idx_1]
        x2 = self.df.iloc[idx_2]
        numerator = 0
        for feature_idx in range(self.features_no):
            feature_type = self.df.iloc[:, feature_idx].dtype
            if not is_numeric_dtype(feature_type):
                if x1[feature_idx] != x2[feature_idx]:
                    numerator += 1
            else:
                numerator += np.abs(x1.iloc[feature_idx] - x2.iloc[feature_idx])
        return numerator / self.features_no

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

            gower_distances[i] = self._gower_distance(data_idx, nearest_neighbors[i])
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

            distance = self._gower_distance(data_idx, potential_idx)
            if distance < eps[1]:
                nearest_neighbors[eps[0]] = potential_idx
                gower_distances[eps[0]] = distance
                eps[0] = np.argmax(gower_distances)
                eps[1] = gower_distances[eps[0]]
            # elif distance == eps[1]:  # k+ neighbours can cause recursion limit error
            #     nearest_neighbors = np.append(nearest_neighbors, potential_idx)

        return nearest_neighbors

    def _assign_group(self, group_num, idx):
        """
        Assigns group number to a data point
        :param group_num: Group number that will be assigned
        :param idx: index of currently processed data point
        :return: True if group was assigned, False if not
        """
        if self.df.loc[idx, 'grouping'] == -1:
            self.df.loc[idx, 'grouping'] = group_num

            if self.df.loc[idx, 'NDF'] >= 1:
                for neighbour in self.df.loc[idx, 'k-nearest']:
                    self._assign_group(group_num, neighbour)
            return True
        return False

    def rand_metric(self):
        """
        :return: Rand metric
        """
        true_positives = 0
        true_negatives = 0
        for i in range(len(self.df.index)):
            for j in range(i + 1, len(self.df.index)):
                if (self.df.loc[i, 'grouping'] == self.df.loc[j, 'grouping'] and
                        self.df.loc[i, 'class_no'] == self.df.loc[j, 'class_no']):
                    true_positives += 1
                elif (self.df.loc[i, 'grouping'] != self.df.loc[j, 'grouping'] and
                        self.df.loc[i, 'class_no'] != self.df.loc[j, 'class_no']):
                    true_negatives += 1
        return (true_positives + true_negatives) / scipy.special.binom(len(self.df.index), 2)

    def fit(self):
        if self.check_data() is False:
            return

        self._tidy_data()

        k_nearest = pd.Series([np.array] * len(self.df.index))
        for i in range(len(self.df.index)):
            k_nearest[i] = self._k_nearest_neighbors(i)
            # reverse k-neighbours
            for index in k_nearest[i]:
                self.df.loc[index, 'r-k-nearest'] += 1
        self.df['k-nearest'] = k_nearest

        # calculate NDF
        for i in range(len(self.df.index)):
            self.df.loc[i, 'NDF'] = self.df.loc[i, 'r-k-nearest'] / self.df.loc[i, 'k-nearest'].size

        # grouping
        current_group = 0
        for i in range(len(self.df.index)):
            if self.df.loc[i, 'NDF'] >= 1 and self._assign_group(current_group, i):
                current_group += 1

