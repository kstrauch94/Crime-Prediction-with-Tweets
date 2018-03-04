import numpy as np
from sklearn.neighbors.kde import KernelDensity

from utils.consts import KDE_BANDWITH


def train_KDE_model(train_df, bandwith=KDE_BANDWITH):
    '''
    Train KDE model.

    Input:
    train_df: train data frame with Latitude Logitude. 3 months prior data for the day of surveillance..

    Output:
    KDE Model
    '''

    kde = KernelDensity(bandwidth=bandwith,
                        metric='haversine',
                        kernel='gaussian',
                        algorithm='ball_tree')

    kde.fit(train_df[['latitude', 'longitude']] * np.pi / 180)

    return kde
