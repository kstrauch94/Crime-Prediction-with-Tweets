import collections

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from utils.consts import LDA_TOPICS


def generate_threat_kde_dataset(train_dataset):
    """
    Generate list of city grid cells orderd by crime threat by a KDE model.
    """

    threat_grid_cells = train_dataset['X'][~train_dataset['Y']]
    kde_values = threat_grid_cells[['latitude_index', 'longitude_index', 'KDE']]

    threat_kde_df = kde_values.set_index(['latitude_index', 'longitude_index'])['KDE']
    threat_kde_df = threat_kde_df.sort_values(ascending=False)

    return list(threat_kde_df.index)


def generate_threat_logreg_dataset(train_dataset, additional_features):
    """
    Generate list of city grid cells orderd by crime threat by
    a model that is based on Logistic Regression.
    """

    is_crime_count = train_dataset['Y'].value_counts()
    logreg_C = is_crime_count[False] / is_crime_count[True]
    logreg = make_pipeline(StandardScaler(), LogisticRegression(C=logreg_C))
    logreg.fit(train_dataset['X'][['KDE'] + additional_features], train_dataset['Y'])

    threat_grid_cells = train_dataset['X'][~train_dataset['Y']]
    threat_grid_cells['logreg'] = logreg.predict_log_proba(
        threat_grid_cells[['KDE'] + additional_features])[:, 1]

    logreg_values = threat_grid_cells[['latitude_index', 'longitude_index', 'logreg']]
    threat_logreg_df = logreg_values.set_index(['latitude_index', 'longitude_index'])['logreg']
    threat_logreg_df = threat_logreg_df.sort_values(ascending=False)

    return list(threat_logreg_df.index), logreg


def generate_threat_datasets(train_dataset):
    """
    Generate all threat datasets by all four models:
    KDE, KDE+SENTIMENT, KDE, KDE+LDA, KDE+SENTIMENT+LDA
    """

    threat_datasets_list = []
    threat_datasets_list.append(('KDE', {'cells': generate_threat_kde_dataset(train_dataset)}))

    for model_name, additional_features in [('KDE+SENTIMENT', ['SENTIMENT']),
                                            ('KDE+LDA', LDA_TOPICS),
                                            ('KDE+SENTIMENT+LDA', ['SENTIMENT'] + LDA_TOPICS)]:

        cells, logreg = generate_threat_logreg_dataset(train_dataset, additional_features)

        threat_datasets_list.append((model_name, {'cells': cells, 'logreg': logreg}))

        threat_datasets = collections.OrderedDict(threat_datasets_list)

    return threat_datasets
