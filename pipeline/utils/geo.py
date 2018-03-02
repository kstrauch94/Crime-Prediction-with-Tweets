import functools
import itertools

import numpy as np

import pandas as pd
import utm


from utils.consts import CHICAGO_COORDS, GEO_CELL_SIZE, \
    UTM_ZONE_NUMBER, UTM_ZONE_LETTER, \
    FALSE_LABLE_DATASET_CELL_SIZE, LDA_PARAMS


def filter_by_geo_coord(df, bounderies):
    return df[(df['latitude'] >= bounderies['ll']['latitude']) &
              (df['latitude'] <= bounderies['ur']['latitude']) &
              (df['longitude'] >= bounderies['ll']['longitude']) &
              (df['longitude'] <= bounderies['ur']['longitude'])]


def _latlng2utm(coords):
    utm_coord = utm.from_latlon(coords['latitude'], coords['longitude'])[:2]
    return dict(
        zip(
            ('latitude', 'longitude'),
            utm_coord
        )
    )


def _utm2latlng(coords):
    utm_coord = utm.to_latlon(coords['latitude'], coords['longitude'],
                              UTM_ZONE_NUMBER, UTM_ZONE_LETTER)
    return dict(
        zip(
            ('latitude', 'longitude'),
            utm_coord
        )
    )


def _generate_utm_columns(row):
    return pd.Series(_latlng2utm(row))


def latlng2grid_cords(latitude, longitude, bounderies_utm, cell_size):
    utm_cords = _latlng2utm({'latitude': latitude,
                             'longitude': longitude})

    latitude_index = int(((utm_cords['latitude'] - bounderies_utm['ll']['latitude'])
                          / cell_size))

    longitude_index = int(((utm_cords['longitude'] - bounderies_utm['ll']['longitude'])
                           / cell_size))

    return latitude_index, longitude_index


def bounderis_latlng2utm(bounderies):
    return {'ll': _latlng2utm(bounderies['ll']),
            'ur': _latlng2utm(bounderies['ur'])}


def enrich_with_grid_coords(df, bounderies, cell_size):
    '''
    The accepts a data frame which has atleast 2 columns with names 'Latitude' and
    'Longitude'. It will be converted into UTM(Universal Transverse Mercator) co-odrinates for obtaining
    grid of a locality.

    input:
    Data frame with 'Latitude' and 'Longitude'

    output:
    Data frame with additional column which represents Grid Numbers
    '''

    bounderies_utm_cords = bounderis_latlng2utm(bounderies)

    # n_latitude_cells = int(math.ceil((bounderies_utm_cords['ur']['latitude'] -
    #                                  bounderies_utm_cords['ll']['latitude'])
    #                                 / cell_size))

    utm_coords = df[['latitude', 'longitude']].apply(lambda row: _generate_utm_columns(row), axis=1)

    df['latitude_index'] = (((utm_coords['latitude'] - bounderies_utm_cords['ll']['latitude'])
                             / cell_size)
                            .astype(int))

    df['longitude_index'] = (((utm_coords['longitude'] - bounderies_utm_cords['ll']['longitude'])
                              / cell_size)
                             .astype(int))

    # df['cell_index'] = df['longitude_index'] * n_latitude_cells + df['latitude_index']

    return df


def generate_grid_list(bounderies_utm, cell_size):

    utm_latitude_dim = np.arange(bounderies_utm['ll']['latitude'],
                                 bounderies_utm['ur']['latitude'],
                                 cell_size)

    utm_longitude_dim = np.arange(bounderies_utm['ll']['longitude'],
                                  bounderies_utm['ur']['longitude'],
                                  cell_size)

    utm_grid_list = [{'latitude': lat,
                      'longitude': lng}
                     for lat, lng in itertools.product(utm_latitude_dim,
                                                       utm_longitude_dim)]

    grid_list = pd.DataFrame([_utm2latlng(cord) for cord in utm_grid_list])

    return grid_list


def latlng2LDA_topics_chicago(latitude, longitude, doc_topics, docs):
    latitude_index, longitude_index = latlng2grid_cords_chicago(latitude, longitude)
    if (latitude_index, longitude_index) in docs.index:
        doc_index = docs.index.get_loc((latitude_index, longitude_index))
        return doc_topics[doc_index]
    else:
        #raise KeyError
        return np.zeros(LDA_PARAMS['n_components'])


CHICAGO_UTM_COORDS = bounderis_latlng2utm(CHICAGO_COORDS)

enrich_with_chicago_grid_1000 = functools.partial(enrich_with_grid_coords,
                                                  bounderies=CHICAGO_COORDS,
                                                  cell_size=GEO_CELL_SIZE)

enrich_with_chicago_grid_200 = functools.partial(enrich_with_grid_coords,
                                                 bounderies=CHICAGO_COORDS,
                                                 cell_size=FALSE_LABLE_DATASET_CELL_SIZE)

filter_by_chicago_coord = functools.partial(filter_by_geo_coord,
                                            bounderies=CHICAGO_COORDS)

latlng2grid_cords_chicago = functools.partial(latlng2grid_cords,
                                              bounderies_utm=CHICAGO_UTM_COORDS,
                                              cell_size=GEO_CELL_SIZE)

generate_chicago_threat_grid_list = functools.partial(generate_grid_list,
                                                      bounderies_utm=CHICAGO_UTM_COORDS,
                                                      cell_size=FALSE_LABLE_DATASET_CELL_SIZE)

N_CHICAGO_THREAT_GRID_LIST = len(generate_chicago_threat_grid_list())
