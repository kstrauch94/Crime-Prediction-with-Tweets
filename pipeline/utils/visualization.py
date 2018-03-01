import os

import numpy as np
import matplotlib.pyplot as plt
import shapefile as shp


""" Below are the set of constants that are being used in the Project
    includes
    - offset for generating the grid
    - lower left and upper right co-ordinates of a locality(in this case, Chicago)
"""
y, x = -87.94011, 41.64454
Y, X = -87.52413, 42.02303
coors_ll = {'low_left_x': x, 'low_left_y': y, 'up_right_x': X, 'up_right_y': Y}

# shapefile with neighborhood boundaries.
shapeFile = os.path.join(os.path.dirname(__file__),
                         'geo_export_0d18d288-fb07-4743-8d14-780bc1108034.shp')

# shapefile with just the city boundary.
# shapeFile = 'pipeline/utils/geo_export_72bbf21c-442f-41e2-bd40-433d7fb7928d.shp'


def getCityBase():
    """ Take the shapeFile of the city to extract all the points of the boundary.
        Flattens the polygons and returns the Latitudes and Longitudes

        Input:
        shapefile

        Output:
        Latitudes and Longitudes.
    """
    shapes = shp.Reader(shapeFile).shapeRecords()
    X, Y = [], []
    for shape in shapes:
        x, y = list(zip(*shape.shape.points))
        X.append(x)
        Y.append(y)
    X_flat = [item for sublist in X for item in sublist]
    Y_flat = [item for sublist in Y for item in sublist]
    return X_flat, Y_flat


def plotContour(kde_model):
    """ This function plots a Contour plot for the data and kde_model given.

        Input:
        data and kde model

        Output:
        displays the contour plot
    """
    xgrid = np.linspace(coors_ll['low_left_x'], coors_ll['up_right_x'], 200)
    ygrid = np.linspace(coors_ll['low_left_y'], coors_ll['up_right_y'], 240)
    X, Y = np.meshgrid(xgrid[::5], ygrid[::5][::-1])
    grid = np.vstack([X.ravel(), Y.ravel()]).T
    grid *= np.pi/180
    kde_values = kde_model.score_samples(grid)
    kde_values = np.exp(kde_values)
    kde_values = kde_values.reshape(X.shape)
    levels = np.linspace(kde_values.min(), kde_values.max(), 40)
    x, y = getCityBase()
    fig = plt.figure(figsize=(13, 15))
    plt.contourf(Y, X, kde_values, levels, cmap='Reds')
    plt.scatter(x, y, color='black', s=0.5, zorder=2)


def plotScatter(data):
    """ This function plots the city basemap and a scatter plot of provided points in Latitude and Longitude.

        Input:
        data with Latitude and Longitude

        Output:
        displays a plot with data on city map
    """
    x, y = getCityBase()
    fig = plt.figure(figsize=(13, 15))
    plt.scatter(data['Longitude'], data['Latitude'], color='red', s=0.5, zorder=1)
    plt.scatter(x, y, color='black', s=0.5, zorder=2)
