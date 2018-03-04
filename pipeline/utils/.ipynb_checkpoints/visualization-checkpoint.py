import os

import numpy as np
import matplotlib.pyplot as plt
import shapefile as shp

from utils.consts import CHICAGO_COORDS, CHICAGO_NEIGHBORHOOD, \
        SCATTER_SIZE_OF_CRIME_POINTS, \
        SCATTER_SIZE_OF_CHICAGO_CITY, CITY_MAP_ORDER,\
        CONTOUR_PLOT_COLOUR, CITY_MAP_COLOR, FIGURE_SIZE,\
        KDE_LEVELS, CRIME_POINTS_COLOR


def get_city_base(city_map = CHICAGO_NEIGHBORHOOD):
    """ 
    Take the shapeFile of the city to extract all the points of the boundary. 
    Flattens the polygons and returns the Latitudes and Longitudes
        
    Input:
    shapefile
        
    Output:
    Latitudes and Longitudes.
    """
    shapes = shp.Reader(city_map).shapeRecords()
    x,y = [],[]
    for shape in shapes:
        inner_x,inner_y = list(zip(*shape.shape.points))
        x.append(inner_x)
        y.append(inner_y)
    x_flat = [item for sublist in x for item in sublist]
    y_flat = [item for sublist in y for item in sublist]
    return x_flat,y_flat

def plot_contour(kde_model):
    """ 
    This function plots a Contour plot for the data and kde_model given.
    
    Input:
    data and kde model
        
    Output:
    displays the contour plot
    """
    xgrid = np.linspace(CHICAGO_COORDS['ll']['latitude']-0.04, CHICAGO_COORDS['ur']['latitude'], 200)
    ygrid = np.linspace(CHICAGO_COORDS['ll']['longitude']-0.04, CHICAGO_COORDS['ur']['longitude'], 240)
    kde_mesh_x, kde_mesh_y = np.meshgrid(xgrid[::5], ygrid[::5][::-1])
    grid = np.vstack([kde_mesh_x.ravel(), kde_mesh_y.ravel()]).T
    grid *= np.pi/180
    kde_values = kde_model.score_samples(grid)
    kde_values = np.exp(kde_values)
    kde_values = kde_values.reshape(kde_mesh_x.shape)
    levels = np.linspace(kde_values.min(), kde_values.max(), 40)
    city_x,city_y = get_city_base()
    fig = plt.figure(figsize = FIGURE_SIZE)
    plt.contourf(kde_mesh_y, kde_mesh_x, kde_values, levels, cmap = CONTOUR_PLOT_COLOUR)
    plt.scatter(city_x,city_y, color = CITY_MAP_COLOR, s = SCATTER_SIZE_OF_CHICAGO_CITY, zorder = CITY_MAP_ORDER)
    
def plot_scatter(data):
    """ 
    This function plots the city basemap and a scatter plot of provided points in Latitude and Longitude.
        
    Input:
    data with Latitude and Longitude
        
    Output:
    displays a plot with data on city map
    """
    city_x,city_y = get_city_base()
    fig = plt.figure(figsize = FIGURE_SIZE)
    plt.scatter(data['longitude'],data['latitude'], color = CRIME_POINTS_COLOR, s = SCATTER_SIZE_OF_CRIME_POINTS)
    plt.scatter(city_x,city_y, color = CITY_MAP_COLOR, s = SCATTER_SIZE_OF_CHICAGO_CITY, zorder = CITY_MAP_ORDER)
    
def plot_imshow(data, col_name):
    """
    This plots data with a X and Y axis with a specified column of a aggregated data
    """
    city_x,city_y = get_city_base()
    fig = plt.figure(figsize = FIGURE_SIZE)
    plt.imshow(data[['lat_ind','lng_ind',col_name]], cmap=CONTOUR_PLOT_COLOUR)
    plt.scatter(city_x,city_y, color = CITY_MAP_COLOR, s = SCATTER_SIZE_OF_CHICAGO_CITY, zorder = CITY_MAP_ORDER)