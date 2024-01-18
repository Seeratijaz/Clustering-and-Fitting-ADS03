from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import cluster_tools as ct
import errors as err
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def read_file(fn):
    """
    Reads a CSV file and returns a pandas DataFrame.

    Parameters:
    ------------
    fn (str): The filename of the CSV file to be read.

    Returns:
    ---------
    df (pandas.DatFrame): The DataFrame containing the data
    read from the CSV file.
    """
    address = fn
    df = pd.read_csv(address, skiprows=4)
    df = df.drop(
        columns=[
            'Country Code',
            'Indicator Name',
            'Indicator Code',
            'Unnamed: 67'])
    return df


def make_clusters(
        df,
        Feature1,
        Feature2,
        xlabel,
        ylabel,
        tit,
        no_clusters,
        df_fit,
        df_min,
        df_max):
    """
    Funtion to make Clusters
    Parameters
    ----------
    df : Pandas DataFrame
        Original DataFrame For Clustering.
    Feature1 : String
        First Feature.
    Feature2 : String
        Second Feature.
    xlabel : String
        X-axis Value for clustering.
    ylabel : String
        Y-axis Value for clustering.
    tit : String
        Title of the graph.
    no_clusters : int
        No of clusters.
    df_fit : Pandas DataFrame
        Normalize Dataframe for clustering.
    df_min : int
        minimum value of original Data Frame.
    df_max : int
        Maximum value of original Data Frame.

    Returns
    -------
    None.

    """
    nc = no_clusters  # number of cluster centres
    kmeans = KMeans(n_clusters=nc, n_init=10, random_state=0)
    kmeans.fit(df_fit)
    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    plt.figure(figsize=(8, 8))
    # scatter plot with colours selected using the cluster numbers
    # now using the original dataframe
    scatter = plt.scatter(df[Feature1], df[Feature2], c=labels, cmap="tab10")
    # colour map Accent selected to increase contrast between colours
    # rescale and show cluster centres
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:, 0]
    yc = scen[:, 1]
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title(tit)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('Clustering.png', dpi=300)
    plt.show()


def poly(x, a, b, c):
    """
    Calulates polynominal
    """
    x = x - 1990
    f = a + b * x + c * x**2

    return f


def get_data_for_country(df, country_name, start_year, end_year):
    """
    Get Data For Specific Country with start and end year
    Parameters
    ----------
    df : Pandas Data Frame
        Data Frame to Filter Data From.
    country_name : String
        Name of the country who's Data is Required.
    start_year : int
        start year of range.
    end_year : int
        ending year of range.

    Returns
    -------
    df : Pandas Data Frame
        Filtered DataFrame.

    """
    # Taking the Transpose
    df = df.T
    df.columns = df.iloc[0]
    df = df.drop(['Country Name'])
    df = df[[country_name]]
    df.index = df.index.astype(int)
    # Filtering Data For Year Range
    df = df[(df.index > start_year) & (df.index <= end_year)]
    df[country_name] = df[country_name].astype(float)
    return df


def plot_silhouette_score(data, max_clusters=10):
    """
    Evaluate and plot silhouette scores for different numbers of clusters.

    Parameters:
    - data: The input data for clustering.
    - max_clusters: The maximum number of clusters to evaluate.

    Returns:
    """

    silhouette_scores = []

    for n_clusters in range(2, max_clusters + 1):
        # Perform clustering using KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Plot the silhouette scores
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

# Reading the Co2 Emissions File
CO2_emissions_metric_tons_per_capita = read_file(
    'CO2_emissions_metric_tons_per_capita.csv')
Forest_area_of_land_area = read_file('Forest_area_%_of_land_area.csv')
country = 'United States'
df_co2 = get_data_for_country(
    CO2_emissions_metric_tons_per_capita,
    country,
    1990,
    2020)
df_FA = get_data_for_country(Forest_area_of_land_area, country, 1990, 2020)

# Merging the Data Frames
df = pd.merge(df_co2, df_FA, left_index=True, right_index=True)
df = df.rename(
    columns={
        country +
        "_x": 'Co2 Emissions Per Capita',
        country +
        "_y": 'Forest_Area'})
df_fit, df_min, df_max = ct.scaler(df)
plot_silhouette_score(df_fit, 12)

# Calling the Clustering Function
make_clusters(
    df,
    'Co2 Emissions Per Capita',
    'Forest_Area',
    'Co2 Emissions Per Capita',
    'Forest Area',
    'Co2 Emissions Per Capita vs GDP Forest Area in United States',
    2,
    df_fit,
    df_min,
    df_max)

# Fitting and Forecasting
# for the Forest Area
popt, pcorr = opt.curve_fit(poly, df_FA.index, df_FA[country])
# much better
df_FA["pop_poly"] = poly(df_FA.index, *popt)
plt.figure()
plt.plot(df_FA.index, df_FA[country], label="data")
plt.plot(df_FA.index, df_FA["pop_poly"], label="fit")
plt.legend()
plt.xlabel('Years')
plt.ylabel('Forest Area')
plt.title('Forest Area in United States 1990-2020')
plt.savefig(country + '_.png', dpi=300)
years = np.linspace(1990, 2030)
# Using the Ploy Funtion to get Forecast Values
pop_ploy = poly(years, *popt)
sigma = err.error_prop(years, poly, popt, pcorr)
low = pop_ploy - sigma
up = pop_ploy + sigma
plt.figure()
plt.plot(df_FA.index, df_FA[country], label="data")
plt.plot(years, pop_ploy, label="Forecast")
# plot error ranges with transparency
plt.fill_between(years, low, up, alpha=0.5, color="y")
plt.legend(loc="upper left")
plt.xlabel('Years')
plt.ylabel('Forest Area')
plt.title('Forest Area in United States Forecast')
plt.savefig(country + '__forecast.png', dpi=300)
plt.show()

# Fitting and Forecasting
# For Co2 Emissions Per Capita
popt, pcorr = opt.curve_fit(poly, df_co2.index, df_co2[country])
# Fitting with Poly Funtion
df_co2["pop_poly"] = poly(df_co2.index, *popt)
plt.figure()
plt.plot(df_co2.index, df_co2[country], label="data")
plt.plot(df_co2.index, df_co2["pop_poly"], label="fit")
plt.legend()
plt.xlabel('Years')
plt.ylabel('Co2 Emissions Per Capita')
plt.title('Co2 Emissions Per Capita in United States 1990-2020')
plt.savefig(country + '_.png', dpi=300)
years = np.linspace(1990, 2030)
pop_poly = poly(years, *popt)
sigma = err.error_prop(years, poly, popt, pcorr)
# Calculating the Error
low = pop_poly - sigma
up = pop_poly + sigma
plt.figure()
plt.plot(df_co2.index, df_co2[country], label="data")
plt.plot(years, pop_poly, label="Forecast")
# plot error ranges with transparency
plt.fill_between(years, low, up, alpha=0.5, color="y")
plt.legend(loc="upper left")
plt.xlabel('Years')
plt.ylabel('Co2 Emissions Per Capita')
plt.title('Co2 Emissions Per Capita in United States Forecast')
plt.savefig(country + '__forecast.png', dpi=300)
plt.show()
