import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.geo import N_CHICAGO_THREAT_GRID_LIST, \
    START_DATE, END_DATE
from utils.threat_models import generate_threat_datasets
from utils.datasets_generation import generate_one_step_datasets


def generate_surveillance_data(train_dataset, evaluation_dataset):

    surveillance_data = np.zeros((5, N_CHICAGO_THREAT_GRID_LIST))

    threat_datasets = generate_threat_datasets(train_dataset)

    crime_counts = evaluation_dataset.groupby(['latitude_index', 'longitude_index']).size()
    crime_counts = crime_counts.sort_values(ascending=False)

    # real crime occurence is our gold dataset
    threat_datasets['GOLD'] = {'cells': list(crime_counts.index)}
    threat_datasets.move_to_end('GOLD')

    for threat_model_index, (threat_model_name, threat_dataset) in enumerate(threat_datasets.items()):
        for cell_index, (latitude_index, longitude_index) in enumerate(threat_dataset['cells']):
            surveillance_data[threat_model_index][cell_index] = crime_counts.get(
                (latitude_index, longitude_index), 0)

    return surveillance_data, threat_datasets


def generate_one_step_surveillance_data(crimes_data, tweets_data, start_train_date, n_train_days):

    train_dataset, evaluation_dataset = generate_one_step_datasets(crimes_data,
                                                                   tweets_data,
                                                                   start_train_date,
                                                                   n_train_days)

    surveillance_data, threat_datasets = generate_surveillance_data(train_dataset,
                                                                    evaluation_dataset)

    return surveillance_data, threat_datasets


def generate_all_data_surveillance_data(crimes_data, tweets_data, n_train_days):
    agg_surveillance_data = np.zeros((5, N_CHICAGO_THREAT_GRID_LIST))
    all_threat_datasets = []

    start_train_dates = pd.date_range(START_DATE, END_DATE)[:-(n_train_days+1)][:1]

    for start_train_date in tqdm(start_train_dates):

        surveillance_data, threat_datasets = generate_one_step_surveillance_data(crimes_data,
                                                                                 tweets_data,
                                                                                 start_train_date,
                                                                                 n_train_days)

        agg_surveillance_data += surveillance_data
        all_threat_datasets.append((start_train_date, threat_datasets))

    agg_surveillance_data = agg_surveillance_data.cumsum(
        axis=1) / agg_surveillance_data.sum(axis=1)[:, None]

    return agg_surveillance_data, all_threat_datasets


def plot_surveillance_data(surveillance_data):
    pass
