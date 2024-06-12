import arff
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score

from nbc import NBC

DATASETS_PATH = "./datasets/"


def read_config(config_path):
    config_dict = {}

    with open(config_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            if key == 'entry multiplier' or key == 'attribute multiplier':
                if '-' in value:
                    start, end = map(int, value.split('-'))
                    value = list(range(start, end + 1))
                    if start <= 0 or end <= 0:
                        raise ValueError('Invalid entry or attribute multiplier: cannot be below 1!')
                else:
                    value = int(value)
            if key == 'k value':
                value = int(value)

            config_dict[key] = value

    return config_dict


def main():
    load_data_time = 0
    algorithm_times = []
    load_data_timestamp = time.time()
    config = read_config("./config.txt")
    if isinstance(config['entry multiplier'], list) and isinstance(config['attribute multiplier'], list):
        raise ValueError('Cannot run with both entry and attribute multiplier at once!')

    with open(config['dataset path'], 'r') as fp:
        dataset = arff.load(fp)
    df_init = df = pd.DataFrame(dataset['data'])

    nbc = NBC()
    adjusted_rand_scores = []
    compare_name = ""   # entry multiplier or attribute multiplier

    load_data_time += time.time() - load_data_timestamp
    if isinstance(config['entry multiplier'], list):
        compare_name = 'entry multiplier'
        for multiplier in config['entry multiplier']:
            multiplier -= 1
            load_data_timestamp = time.time()
            df_tmp = df_init
            for i in range(multiplier):
                df_tmp = pd.concat([df_init, df_tmp], ignore_index=True)
            df = df_tmp
            nbc.set_df(df)
            nbc.set_k(config['k value'])
            load_data_time += time.time() - load_data_timestamp

            algorithm_timestamp = time.time()
            print(f"Iteration {multiplier + 1}:", end=" ")
            nbc.fit()
            algorithm_times.append(time.time() - algorithm_timestamp)
            print(f"{algorithm_times[-1]:.2f} seconds")
            adjusted_rand_scores.append(adjusted_rand_score(nbc.df['class_no'], nbc.df['grouping']))

    elif isinstance(config['attribute multiplier'], list):
        compare_name = 'attribute multiplier'
        for multiplier in config['attribute multiplier']:
            multiplier -= 1
            load_data_timestamp = time.time()
            df_tmp = df_init.iloc[:, :-1]
            for i in range(multiplier):
                df_tmp = df_tmp.join(df_init.iloc[:, :-1], lsuffix=f'_0{i}', rsuffix=f'_1{i}')
            df = pd.concat([df_tmp, df_init.iloc[:, -1]], axis=1)
            nbc.set_df(df)
            nbc.set_k(config['k value'])
            load_data_time += time.time() - load_data_timestamp

            algorithm_timestamp = time.time()
            print(f"Iteration {multiplier + 1}:", end=" ")
            nbc.fit()
            algorithm_times.append(time.time() - algorithm_timestamp)
            print(f"{algorithm_times[-1]:.2f} seconds")
            adjusted_rand_scores.append(adjusted_rand_score(nbc.df['class_no'], nbc.df['grouping']))
    else:
        nbc.set_df(df)
        nbc.set_k(config['k value'])
        algorithm_timestamp = time.time()
        print(f"Iteration 1:", end=" ")
        nbc.fit()
        algorithm_times.append(time.time() - algorithm_timestamp)
        print(f"{algorithm_times[-1]:.2f} seconds")
        adjusted_rand_scores.append(adjusted_rand_score(nbc.df['class_no'], nbc.df['grouping']))


    # plot results
    if compare_name == 'entry multiplier':
        config['entry multiplier'] = [i * df_init.shape[0] for i in config['entry multiplier']]
        plt.plot(config['entry multiplier'], adjusted_rand_scores)
        plt.title("Liczba obiektów vs. Adjusted rand score")
        plt.xlabel("Liczba obiektów")
        plt.ylabel("Adjusted rand score")
        plt.show()

        plt.figure()
        plt.plot(config['entry multiplier'], algorithm_times)
        plt.title("Liczba obiektów vs. Algorithm time")
        plt.xlabel("Liczba obiektów")
        plt.ylabel("Algorithm time (sec)")
        plt.show()
    elif compare_name == 'attribute multiplier':
        config['attribute multiplier'] = [i * df_init.shape[1] for i in config['attribute multiplier']]
        plt.plot(config['attribute multiplier'], adjusted_rand_scores)
        plt.title("Liczba cech vs. Adjusted rand score")
        plt.xlabel("Liczba cech")
        plt.ylabel("Adjusted rand score")
        plt.show()

        plt.figure()
        plt.plot(config['attribute multiplier'], algorithm_times)
        plt.title("Liczba cech vs. Algorithm time")
        plt.xlabel("Liczba cech")
        plt.ylabel("Algorithm time (sec)")
        plt.show()


    # save results
    data = {'entry multiplier': config['entry multiplier'], 'k value': config['k value'],
            'load data time': load_data_time, 'adjusted rand score': adjusted_rand_scores,
            'algorithm time': algorithm_times}
    df = pd.DataFrame(data)
    df.to_csv(f'./results/{compare_name}_{time.time()}.csv', index=False)


if __name__ == '__main__':
    main()
