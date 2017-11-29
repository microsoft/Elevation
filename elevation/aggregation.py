import numpy as np
import sklearn.ensemble
import cPickle as pickle

from elevation import settings


def compute_stats_from_pred(pred, genic):
    stats = np.zeros((1, 30))

    functions = {'sum': np.sum, 'mean': np.mean, 'std': np.std,
                 'var': np.var, 'median': np.median, '90th percentile': np.percentile,
                 '95th percentile': np.percentile, '99th percentile': np.percentile}

    names = []
    i = 0
    for f in functions.keys():
        if " " in f:
            args = int(f.split(' ')[0].replace('th', ''))
        else:
            args = None

        if args != None:
            stats[0, i] = functions[f](pred, args)
            if np.sum(genic) > 2:
                stats[0, i+1] = functions[f](pred[genic], args)
            if np.sum(~genic) > 2:
                stats[0, i+2] = functions[f](pred[~genic], args)
        else:
            stats[0, i] = functions[f](pred)
            if np.sum(genic) > 2:
                stats[0, i+1] = functions[f](pred[genic])
            if np.sum(~genic) > 2:
                stats[0, i+2] = functions[f](pred[~genic])

        names.extend([f + p for p in [' ', ' genic', ' non-genic']])
        i += 3

    if np.sum(genic) > 1:
        stats[0, i] = np.sum(genic)/float(len(pred))
    if np.sum(~genic) > 1:
        stats[0, i+1] = np.sum(~genic)/float(len(pred))
    names.extend(['#genic/total', '#non-genic/total'])

    if np.sum(genic) > 1 and np.sum(~genic) > 1:
        stats[0, i+2] = np.sum(genic)/float(np.sum(~genic))
        stats[0, i+3]  = np.mean(pred[genic])/np.mean(pred[~genic])
        names.extend(['#genic/#non-genic', 'mean genic/mean non-genic'])
    stats[0, i+4]  = np.sum(genic)
    stats[0, i+5]  = np.sum(~genic)
    names.extend(['#genic', '#non-genic'])
    return stats, names


def get_stats(preds_g_elevation, preds_g_cfd, isgenic):
    # We are ignoring CFD stats because our experiment suggests the ensemble
    # is worse than linear-raw-stacker 6/14/2017 NF
    elevation_stats, names = compute_stats_from_pred(preds_g_elevation, isgenic)
    # cfd_stats, names = compute_stats_from_pred(preds_g_cfd, isgenic)
    return elevation_stats, names#np.concatenate((elevation_stats, cfd_stats), axis=1), names


def get_stats_dnase(preds_g_elevation, preds_g_cfd, dnase, isgenic, statsmode=1):
    if statsmode == 1:
        elevation_stats, names = compute_stats_from_pred(preds_g_elevation*dnase, isgenic)
        cfd_stats, names = compute_stats_from_pred(preds_g_cfd*dnase, isgenic)
        return elevation_stats, names#np.concatenate((elevation_stats, cfd_stats), axis=1), names
    elif statsmode == 2:
        elevation_stats, names = compute_stats_from_pred(preds_g_elevation*dnase, isgenic)
        cfd_stats, names = compute_stats_from_pred(preds_g_cfd*dnase, isgenic)
        elevation_stats_2, names = compute_stats_from_pred(preds_g_elevation, isgenic)
        cfd_stats_2, names = compute_stats_from_pred(preds_g_cfd, isgenic)
        return np.concatenate((elevation_stats,  elevation_stats_2), axis=1), names
    elif statsmode == 3:
        dnase_stats, names = compute_stats_from_pred(dnase, isgenic)
        elevation_stats, names = compute_stats_from_pred(preds_g_elevation, isgenic)
        cfd_stats, names = compute_stats_from_pred(preds_g_cfd, isgenic)
        return np.concatenate((elevation_stats,  dnase_stats), axis=1), names
    elif statsmode == 4:
        elevation_stats, names = compute_stats_from_pred(preds_g_elevation*dnase, isgenic)
        cfd_stats, names = compute_stats_from_pred(preds_g_cfd*dnase, isgenic)
        elevation_stats_2, names = compute_stats_from_pred(preds_g_elevation, isgenic)
        cfd_stats_2, names = compute_stats_from_pred(preds_g_cfd, isgenic)
        dnase_stats, names = compute_stats_from_pred(dnase, isgenic)
        return np.concatenate((elevation_stats, elevation_stats_2, dnase_stats), axis=1), names


def train_model(features, truth):
    # New model options resulting from random search 6/14/2017 NF
    model_options = {'loss': 'ls', 'learning_rate': 0.12121299999999999, 'n_estimators': 20, 'criterion': 'mae', 'min_samples_split': 3, 'max_depth': 3}
    clf = sklearn.ensemble.GradientBoostingRegressor(**model_options)
    clf.fit(features, truth)
    return clf


def get_aggregated_score(preds_stacker, preds_cfd, isgenic, model=None):
    """
    Aggregate score from guide seq predictions. See README.md for example usage.
    :param preds_stacker: 'linear-raw-stacker' predictions.
    :param preds_cfd: 'CFD' predictions.
    :param isgenic: list-like of bools.
    :param model: The aggregation model.
    :return: A single aggregated score.
    """
    if model is None:
        model = get_aggregation_model()
    return model.predict(get_stats(preds_stacker, preds_cfd, isgenic)[0])


def get_aggregated_score_dnase(preds_stacker, preds_cfd, dnase, isgenic, model=None):
    """
    Aggregate score from guide seq predictions. See README.md for example usage.
    :param preds_stacker: 'linear-raw-stacker' predictions.
    :param preds_cfd: 'CFD' predictions.
    :param isgenic: list-like of bools.
    :param model: The aggregation model.
    :return: A single aggregated score.
    """
    if model is None:
        model = get_aggregation_model()
    return model.predict(get_stats_dnase(preds_stacker, preds_cfd, dnase, isgenic, statsmode=1)[0])


def get_aggregation_model():
    with open(settings.agg_model_file, "r") as fh:
        model_data = pickle.load(fh)
    return model_data[0]


if __name__ == '__main__':
    import pickle
    from elevation import settings

    stacker = np.array([0.1, 0.2, 0.03])
    scores = np.array([0.05, 0.06, 0.07])
    dnase = np.array([2.0, 0.0, 352.0])
    genicValues = np.array([True, False, False])
    with open(settings.agg_model_file) as fh:
        final_model, other = pickle.load(fh)

    print get_aggregated_score_dnase(stacker, scores, dnase, genicValues, final_model)

    pass
