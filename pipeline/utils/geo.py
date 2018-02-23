import math

import pandas as pd
import utm


def filter_by_geo_coord(df, bounderies):
    return df[(df['latitude']  >= bounderies['ll']['latitude'])  &
              (df['latitude']  <= bounderies['ur']['latitude'])  &
              (df['longitude'] >= bounderies['ll']['longitude']) &
              (df['longitude'] <= bounderies['ur']['longitude'])]

def _generate_utm_columns(row):
    utm_coord = utm.from_latlon(row['latitude'], row['longitude'])[:2]
    return pd.Series(
        dict(
            zip(
                ('latitude', 'longitude'),
                utm_coord
           )
        )
    )

def enrich_with_grid_coords(df, bounderies, cell_size):
    """ The accepts a data frame which has atleast 2 columns with names 'Latitude' and
        'Longitude'. It will be converted into UTM(Universal Transverse Mercator) co-odrinates for obtaining
        grid of a locality.
        
        input:
        Data frame with 'Latitude' and 'Longitude'
    
        output:
        Data frame with additional column which represents Grid Numbers
        
        requirements:

    """
    bounderies_utm_cords = {'ll': _generate_utm_columns(bounderies['ll']),
                            'ur': _generate_utm_columns(bounderies['ur'])}

    n_latitude_cells = int(math.ceil((bounderies_utm_cords['ur']['latitude'] -
                            bounderies_utm_cords['ll']['latitude'])
                        / cell_size))
    
    utm_coords = df[['latitude', 'longitude']].apply(lambda row: _generate_utm_columns(row), axis=1)

    df.['latitude_index'] = (((utm_coords['latitude'] -  bounderies_utm_cords['ll']['latitude'])
                                / cell_size)
                            .astype(int))
    
    df['longitude_index'] = (((utm_coords['longitude'] -  bounderies_utm_cords['ll']['longitude'])
                                / cell_size)
                            .astype(int))
 
    df['cell_index'] = df['longitude_index'] * n_latitude_cells + df['latitude_index']
    
    return df