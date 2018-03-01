CHICAGO_COORDS = {'ll': {'longitude': -87.94011,
                         'latitude': 41.64454},  # Lower Left Corner
                  'ur': {'longitude': -87.52413,
                         'latitude': 42.02303}}  # Upper Right Corner

START_DATE = '2017-12-08'
END_DATE = '2018-02-19'

CRIME_TYPE = 'THEFT'
GEO_CELL_SIZE = 1000  # meter
FALSE_LABLE_DATASET_CELL_SIZE = 200
N_LATITUDE_CELLS = 35
N_LONGITUDE_CELLS = 42

KDE_BANDWITH = 0.00017

UTM_ZONE_NUMBER = 16
UTM_ZONE_LETTER = 'T'

LDA_PARAMS = {
    'n_components': 500,
    'max_iter': 10,
    'learning_method': 'online',
    'learning_offset': 50.,
    'random_state': 42,
    'verbose': 1
}

LDA_TOPICS = ['T{:03}'.format(i) for i in range(LDA_PARAMS['n_components'])]
