import elevation
import pandas
import azimuth
import joblib
import logging
from joblib import Memory
from elevation.model_comparison import *
import copy
import scipy.stats as ss
from sklearn.grid_search import ParameterGrid
import sklearn.linear_model
import scipy as sp
import scipy.stats
import elevation.models
import elevation.features
#import GPy
import socket
from elevation.stacker import *
import elevation.util as ut
from sklearn.metrics import auc, roc_curve
from elevation import settings
import sklearn.isotonic
from sklearn.cross_validation import StratifiedKFold
import sklearn.pipeline
import sklearn.preprocessing


memory = Memory(cachedir=settings.cachedir, verbose=0)
debug_base_model_predictions = False
logging.info("prediction_pipeline loaded.")


@memory.cache
def cross_validate_base_model(learn_options):
    learn_options = copy.deepcopy(learn_options)
    results, all_learn_options = mc.run_models(models=learn_options['models'], orders=[learn_options['order']], adaboost_learning_rates=[0.1],
                                                adaboost_max_depths=[3], adaboost_num_estimators=[100],
                                                learn_options_set={'final': learn_options},
                                                test=False, CV=True, setup_function=setup_elevation,
                                                set_target_fn=set_target_elevation, pam_audit=False, length_audit=False)

    if debug_base_model_predictions:
        y_true = results.values()[0][1][0][0]['dummy_for_no_cv']['raw']
        y_pred = results.values()[0][1][0][1]['dummy_for_no_cv']
        plt.figure()
        plt.hexbin(y_true, y_pred)
        plt.xlabel('True')
        plt.ylabel('Predicted')

    model = results.values()[0][3][0]
    feature_names = np.array(results[results.keys()[0]][6], dtype=str)
    return model, feature_names

# @memory.cache
def train_base_model(learn_options):
    learn_options = copy.deepcopy(learn_options)
    learn_options['cv'] = 'gene'
    results, all_learn_options = mc.run_models(models=learn_options['models'], orders=[learn_options['order']], adaboost_learning_rates=[0.1],
                                                adaboost_max_depths=[3], adaboost_num_estimators=[100],
                                                learn_options_set={'final': learn_options},
                                                test=False, CV=False, setup_function=setup_elevation,
                                                set_target_fn=set_target_elevation, pam_audit=False, length_audit=False)

    if debug_base_model_predictions:
        y_true = results.values()[0][1][0][0]['dummy_for_no_cv']['raw']
        y_pred = results.values()[0][1][0][1]['dummy_for_no_cv']
        plt.figure()
        plt.hexbin(y_true, y_pred)
        plt.xlabel('True')
        plt.ylabel('Predicted')
        import ipdb; ipdb.set_trace()

    model = results.values()[0][3][0]
    feature_names = np.array(results[results.keys()[0]][6], dtype=str)
    return model, feature_names

def filter_PAMs_index(data, allowed_PAMs):
    assert allowed_PAMs in ['all', 'doench', 'none']

    ind = np.zeros((data.shape[0]))
    ind = (ind==0)

    all_pams = ['AA',
                'AC',
                'AG',
                'GG',
                'AT',
                'CA',
                'CC',
                'CG',
                'CT',
                'GA',
                'GC',
                'GT',
                'TA',
                'TC',
                'TG',
                'TT']

    if allowed_PAMs == 'all':
        return ind

    elif allowed_PAMs == 'doench':
        allowed_PAMs_array = ['AG', 'GG']

    elif allowed_PAMs == 'none':
        allowed_PAMs_array = []

    not_allowed_PAMs_array = set.difference(set(all_pams), set(allowed_PAMs_array))

    for pam in not_allowed_PAMs_array:
        ind_pam_i = data['Annotation'].apply(lambda x: pam in x)
        ind[ind_pam_i] = False
        print("Found %d PAMs %s. Dropping them." % (ind_pam_i.sum(), pam))

    return ind

def filter_PAMs(data):
    return data[data['Annotation'].apply(lambda x: not ('AA' in x or
                                         'AC' in x or
                                         'AG' in x or
                                         'AT' in x or
                                         'CA' in x or
                                         'CC' in x or
                                         'CG' in x or
                                         'CT' in x or
                                         'GA' in x or
                                         'GC' in x or
                                         'GT' in x or
                                         'TA' in x or
                                         'TC' in x or
                                         'TG' in x or
                                         'TT' in x))]

def load_guideseq(learn_options, filterPAMs=False, subsample_zeros=False):
    learn_options = copy.deepcopy(learn_options)
    learn_options["V"] = "guideseq"; learn_options["left_right_guide_ind"] = [0,23,23]
    data, Y, target_genes = elevation.load_data.load_guideseq(learn_options)

    if filterPAMs:      
        raise Exception("should not be doing this here, rather when training the stacker, because some pickling happens outside of here")  
        data = filter_PAMs(data)

    if subsample_zeros:
        raise Exception("should not be doing this here, rather when training the stacker, because some pickling happens outside of here")
        ind_nonzero = (data['GUIDE-SEQ Reads'] != 0).values
        extra_zeros = np.random.permutation(np.where(ind_nonzero == False)[0])[:ind_nonzero.sum()]
        ind_keep = ind_nonzero.copy()
        ind_keep[extra_zeros] = True
        data = data[ind_keep]

    return data

@memory.cache
def load_mouse(learn_options):
    learn_options = copy.deepcopy(learn_options)
    learn_options["V"] = "mouse"# ; learn_options["left_right_guide_ind"] = [0,21,23]
    data, Y, target_genes = elevation.load_data.load_mouse_data()
    return data

@memory.cache
def load_Hsu(learn_options):
    learn_options = copy.deepcopy(learn_options)
    learn_options["V"] = "hsu-zhang-single"
    data, Y, target_genes = elevation.load_data.load_HsuZang_data(learn_options["V"])
    return data

def load_Frocketal(learn_options, which_table=3, filterPAMs=False):
    learn_options = copy.deepcopy(learn_options)
    learn_options["V"] = "Frocketal-Stable%d" % which_table
    data, Y, target_genes = elevation.load_data.load_Frock_et_al(which_table)

    if filterPAMs:
        data = filter_PAMs(data)

    return data

def predict(model, data, learn_options, learn_options_override=None, verbose=False):
    if learn_options_override is None:
        learn_options_override = learn_options
    predictions, model, learn_options, _tmpdata, feature_names, all_predictions_ind = predict_elevation(data=data, model=(model, learn_options), model_file=None, pam_audit=False, learn_options_override=learn_options_override,force_zero_intercept=False, naive_bayes_combine=True, verbose=verbose)
    return predictions, all_predictions_ind

@memory.cache
def predict_guideseq(model, data, learn_options, naive_bayes_combine=True):
    learn_options = copy.deepcopy(learn_options)
    # learn_options["V"] = "guideseq"; learn_options["left_right_guide_ind"] = [0,22,23]
    learn_options_override = {'left_right_guide_ind' : None}
    # data, Y, target_genes = elevation.load_data.load_guideseq(learn_options)
    N = data.shape[0]
    learn_options_override = {'left_right_guide_ind' : None}
    predictions, all_predictions_ind = predict(model, data, learn_options, learn_options_override)

    if naive_bayes_combine:
        num_annot = np.array([len(t) for t in data["Annotation"].values])
        num_cd33_pred = np.array([len(t) for t in all_predictions_ind])
        # assert np.all(num_annot==num_cd33_pred), "# annot don't match # predictions"
        max_num_mut = np.max(num_cd33_pred)
        preds_cd33_model = all_predictions_ind.copy()

        # preds_cd33_model = np.nan*np.zeros((N, max_num_mut))
        # for n in range(N):
        #     for i in range(num_annot[n]):
        #         preds_cd33_model[n, i] = all_predictions_ind[n][i]

        Y_cd33 = elevation.load_data.load_cd33(learn_options)[1]

        if learn_options['post-process Platt']:
            Y = np.array(Y_cd33['Day21-ETP'].values, dtype=float)
            Y_bin = (Y >= 1.)*1
            clf = sklearn.linear_model.LogisticRegressionCV(n_jobs=20, cv=20, penalty='l2', solver='liblinear')
            # clf = sklearn.linear_model.LogisticRegression()
            clf.fit(Y[:, None], Y_bin[:, None])
            final_preds = np.zeros_like(preds_cd33_model) * np.nan
            for i in range(preds_cd33_model.shape[0]):
                for j in range(preds_cd33_model.shape[1]):
                    if np.isnan(preds_cd33_model[i, j]):
                        continue
                    final_preds[i, j] = clf.predict_proba(preds_cd33_model[i, j])[:, 1][0]
        else:
            final_preds = preds_cd33_model
    else:
        final_preds = predictions
    # p_Y = (Y_cd33['Day21-ETP'] >= 1.0).sum()/float(Y_cd33.shape[0])
    # final_preds *= p_Y

    return final_preds

def generate_result_string(predictions, truth, fold, metric=sp.stats.spearmanr, name='Spearman r'):
    results_str = "Fold %d, %s: " % (fold+1, name)
    for model in predictions.keys():
        if metric is azimuth.metrics.ndcg_at_k_ties:
            results_str += '%s=%.3f ' % (model, azimuth.metrics.ndcg_at_k_ties(truth[fold].flatten(), predictions[model][fold].flatten(), truth[fold].shape[0]))
        else:
            results_str += '%s=%.3f ' % (model, metric(predictions[model][fold], truth[fold])[0])
    return results_str

def filter_pam_out_of_muts(data, i):
    tmp_muts = data['mut positions'].iloc[i]
    # because Hsu-Zhang ignores alternate PAMs which we have encoded with '22'
    pam_pos = 22
    if pam_pos in tmp_muts:
        tmp_muts.remove(pam_pos)
    tmp_muts = np.array(tmp_muts)
    num_m = len(tmp_muts)
    return num_m, tmp_muts

def normalizeX(X, strength, feature="product"):
    #assert X.shape[1] == len(feature_names)
    Xscale = ((X - X.mean(0)) / X.std(0))
    X = Xscale
    #if feature is not None:
    #    prod_ind = 21
    #    assert feature_names[prod_ind] == feature
    #    XscaleProd = Xscale.copy()
    #    XscaleProd[:,prod_ind] = Xscale[:,prod_ind]*strength
    #    X = XscaleProd
    #else:
    #    X = Xscale
    return X


def stacked_predictions(data, preds_base_model, models=['product', 'CFD', 'constant-power', 'linear-raw-stacker', 'linreg-stacker', 'RF-stacker', 'GP-stacker', 'raw GP'],
                        truth=None, guideseq_data=None, preds_guideseq=None, prob_calibration_model=None, learn_options=None, return_model=False, trained_model=None,
                        models_to_calibrate=None, return_residuals=False):#, dnase_train=None, dnase_test=None):

    predictions = dict([(m, None) for m in models])

    num_mismatches = np.array([len(t) for t in data["Annotation"].values])

    # if ('use_mut_distances' in learn_options.keys() and learn_options['use_mut_distances']):
    data = elevation.features.extract_mut_positions_stats(data)

    if guideseq_data is not None:
        y = guideseq_data['GUIDE-SEQ Reads'].values[:, None]
        num_annot = np.array([len(t) for t in guideseq_data["Annotation"].values])

    if 'logistic stacker' in models:
        X = preds_guideseq.copy()
        Xtest = preds_base_model.copy()
        m = Stacker(y, X, warp_out=False)
        m.maximize()
        predictions['logistic stacker'] = m.predict(Xtest)

    if 'CFD' in models:
        # predicting
        if 'cfd_table_file' not in learn_options.keys():
            learn_options['cfd_table_file'] = settings.pj(settings.offtarget_data_dir, "STable 19 FractionActive_dlfc_lookup.xlsx")

        cfd = elevation.models.CFDModel(cfd_table_file=learn_options['cfd_table_file'])
        
        predictions['CFD'] = cfd.predict(data["Annotation"].values, learn_options["num_proc"])[:, None]

    if 'product' in models:
        predictions['product'] = np.nanprod(preds_base_model, axis=1)[:,None]

    if 'constant-power' in models:
        predictions['constant-power'] = np.power(0.5, num_mismatches)

    if 'CCTOP' in models:
        # predicting
        term1 = np.zeros((data.shape[0], 1))
        for i in range(len(term1)):
            num_m, tmp_muts = filter_pam_out_of_muts(data, i)
            term1[i] = np.sum(1.2**np.array(tmp_muts))
        predictions['CCTOP'] = -term1.flatten()

    if 'HsuZhang' in models:

        # predicting
        W = [0.0,0.0,0.014,0.0,0.0,0.395,0.317,0,0.389,0.079,0.445,0.508,0.613,0.851,0.732,0.828,0.615,0.804,0.685,0.583]
        pred = np.zeros((data.shape[0], 1))

        for i in range(len(pred)):
            num_m, tmp_muts = filter_pam_out_of_muts(data, i)

            if len(tmp_muts) == 0:
                pred[i] = 1.0
            else:
                d = ut.get_pairwise_distance_mudra(tmp_muts)
                term1 = np.prod(1. - np.array(W)[tmp_muts - 1])

                if num_m > 1:
                    term2 = 1./(((19-d)/19)*4 + 1)
                else:
                    term2 = 1

                term3 = 1./(num_m)**2
                pred[i] = term1*term2*term3

        predictions['HsuZhang'] = pred.flatten()

    if 'linear-raw-stacker' in models or 'GBRT-raw-stacker' in models:

        if trained_model is None:
            # put together the training data
            X = preds_guideseq.copy()
            X[np.isnan(X)] = 1.0
            feature_names = ['pos%d' % (i+1) for i in range(X.shape[1])]
            # adding product, num. annots and sum to log of itself
            X = np.concatenate((np.log(X), np.prod(X, axis=1)[:, None], num_annot[:, None], np.sum(X, axis=1)[:, None]), axis=1)
            feature_names.extend(['product', 'num. annotations', 'sum'])
            # X = np.log(X)

            # Only product
            # X = np.prod(X, axis=1)[:, None]
            # feature_names = ['product']

        Xtest = preds_base_model.copy()
        Xtest[np.isnan(Xtest)] = 1.0
        Xtest = np.concatenate((np.log(Xtest), np.prod(Xtest, axis=1)[:, None],  num_mismatches[:, None],  np.sum(Xtest, axis=1)[:, None]), axis=1)
        # Xtest = np.log(Xtest)
        # Xtest = np.prod(Xtest, axis=1)[:, None]

        if ('use_mut_distances' in learn_options.keys() and learn_options['use_mut_distances']):
            guideseq_data = elevation.features.extract_mut_positions_stats(guideseq_data)
            X_dist = guideseq_data[['mut mean abs distance', 'mut min abs distance', 'mut max abs distance', 'mut sum abs distance',
                                    'mean consecutive mut distance', 'min consecutive mut distance', 'max consecutive mut distance',
                                    'sum consecutive mut distance']].values
            Xtest_dist = data[['mut mean abs distance', 'mut min abs distance', 'mut max abs distance', 'mut sum abs distance',
                          'mean consecutive mut distance', 'min consecutive mut distance', 'max consecutive mut distance',
                          'sum consecutive mut distance']].values
            X = np.concatenate((X, X_dist), axis=1)
            Xtest = np.concatenate((Xtest, Xtest_dist), axis=1)


        if 'azimuth_score_in_stacker' in learn_options.keys() and learn_options['azimuth_score_in_stacker']:
            azimuth_score = elevation.model_comparison.get_on_target_predictions(guideseq_data, ['WT'])[0]
            X = np.concatenate((X, azimuth_score[:, None]), axis=1)

            azimuth_score_test = elevation.model_comparison.get_on_target_predictions(data, ['WT'])[0]
            Xtest = np.concatenate((Xtest, azimuth_score_test[:, None]), axis=1)

        if 'linear-raw-stacker' in models:
                        
            dnase_type = [key for key in learn_options.keys() if 'dnase' in key]
            assert len(dnase_type) <= 1            
            if len(dnase_type) == 1:
                dnase_type = dnase_type[0]
                use_dnase = learn_options[dnase_type]
            else:
                use_dnase = False                           

            if use_dnase:
                
                dnase_train = guideseq_data["dnase"].values
                dnase_test = data["dnase"].values
                assert dnase_train.shape[0] == X.shape[0]
                assert dnase_test.shape[0] == Xtest.shape[0]
                     
                if dnase_type == 'dnase:default':
                    # simple appending (Melih)     
                    X = np.concatenate((X, dnase_train[:, None]), axis=1)
                    Xtest = np.concatenate((Xtest, dnase_test[:, None]), axis=1)

                elif dnase_type == 'dnase:interact':
                # interaction with original features
                    X = np.concatenate((X, X*dnase_train[:, None]), axis=1)
                    Xtest = np.concatenate((Xtest, Xtest*dnase_test[:, None]), axis=1)

                elif dnase_type == 'dnase:only':
                    # use only the dnase                                
                    X = dnase_train[:, None]
                    Xtest = dnase_test[:, None]                

                elif dnase_type == 'dnase:onlyperm':
                    # use only the dnase                                
                    pind = np.random.permutation(dnase_train.shape[0])
                    pind_test = np.random.permutation(dnase_test.shape[0])
                    X = dnase_train[pind, None]
                    Xtest = dnase_test[pind_test, None]                
                else:
                    raise NotImplementedError("no such dnase type: %s" % dnase_type)

            normX = True
            strength = 1.0

            # train the model
            if trained_model is None:

                # subsample the data for more balanced training
                                
                ind_zero = np.where(y==0)[0]
                ind_keep = (y!=0).flatten()  
                nn = ind_keep.sum()              
                # take every kth' zero
                increment = int(ind_zero.shape[0]/float(nn))
                ind_keep[ind_zero[::increment]] = True

                #----- debug
                #ind_zero = np.where(y==0)[0]
                #ind_keep2 = (y!=0).flatten()                
                #ind_keep2[np.random.permutation(ind_zero)[0:nn]] = True
                #-----

                # from IPython.core.debugger import Tracer; Tracer()()
                # what been using up until 9/12/2016
                #clf = sklearn.linear_model.LassoCV(cv=10, fit_intercept=True, normalize=True)

                # now using this:
                num_fold = 10

                kfold = StratifiedKFold(y[ind_keep].flatten()==0, num_fold, random_state=learn_options['seed'])
                #kfold2 = StratifiedKFold(y[ind_keep2].flatten()==0, num_fold, random_state=learn_options['seed'])

                clf = sklearn.linear_model.LassoCV(cv=kfold, fit_intercept=True, normalize=(~normX),n_jobs=num_fold, random_state=learn_options['seed'])
                #clf2 = sklearn.linear_model.LassoCV(cv=kfold2, fit_intercept=True, normalize=(~normX),n_jobs=num_fold, random_state=learn_options['seed'])
                
                if normX:
                    clf = sklearn.pipeline.Pipeline([['scaling', sklearn.preprocessing.StandardScaler()], ['lasso', clf]])
                    #clf2 = sklearn.pipeline.Pipeline([['scaling', sklearn.preprocessing.StandardScaler()], ['lasso', clf2]])

                #y_transf = st.boxcox(y[ind_keep] - y[ind_keep].min() + 0.001)[0]                
                
                # scale to be between 0 and 1 first
                y_new = (y - np.min(y)) / (np.max(y) - np.min(y))
                #plt.figure(); plt.plot(y_new[ind_keep], '.'); 
                y_transf = st.boxcox(y_new[ind_keep] - y_new[ind_keep].min() + 0.001)[0]                
                
                # when we do renormalize, we konw that these values are mostly negative (see Teams on 6/27/2017),
                # so lets just make them go entirely negative(?)                
                #y_transf = y_transf - np.max(y_transf)
                
                #plt.figure(); plt.plot(y_transf, '.'); #plt.title("w out renorm, w box cox, then making all negative"); plt.show()
                #import ipdb; ipdb.set_trace()


                #y_transf = np.log(y[ind_keep] - y[ind_keep].min() + 0.001)               
                #y_transf = y[ind_keep]

                # debugging
                #y_transf2 = st.boxcox(y[ind_keep2] - y[ind_keep2].min() + 0.001)[0]
                #y_transf2 = y[ind_keep2]

                print "train data set size is N=%d" % len(y_transf)
                clf.fit(X[ind_keep], y_transf)                 
                #clf2.fit(X[ind_keep2], y_transf2) 
                #clf.fit(X_keep, tmpy) 
                
                #tmp = clf.predict(X)
                #sp.stats.spearmanr(tmp[ind_keep],y_transf.flatten())[0]
                #sp.stats.spearmanr(tmp[ind_keep], y[ind_keep])[0]
                #sp.stats.spearmanr(tmp, y)[0]
                #sp.stats.pearsonr(tmp[ind_keep],y_transf.flatten())[0]
                                       

                # clf.fit(X, y.flatten())
                # clf.fit(X, y, sample_weight=weights)
            else:
                clf = trained_model

            # if normX:
            #    predictions['linear-raw-stacker'] = clf.predict(normalizeX(Xtest, strength, None))
            # else:
            predictions['linear-raw-stacker'] = clf.predict(Xtest)
            # residuals = np.log(y[ind_keep].flatten()+0.001) - clf.predict(X[ind_keep])

    if 'linreg-stacker' in models:
        m_stacker = StackerFeat()
        m_stacker.fit(preds_guideseq, y, model='linreg', normalize_feat=False)
        predictions['linreg-stacker'] = m_stacker.predict(preds_base_model)

    if 'RF-stacker' in models:
        m_stacker = StackerFeat()
        m_stacker.fit(preds_guideseq, y, model='RFR', normalize_feat=False)
        predictions['RF-stacker'] = m_stacker.predict(preds_base_model)

    if 'GP-stacker'in models:
        m_stacker = StackerFeat()
        m_stacker.fit(preds_guideseq, y, model='GP', normalize_feat=False)
        predictions['GP-stacker'] = m_stacker.predict(preds_base_model)

    if 'raw GP' in models:
        X = preds_guideseq.copy()
        X[np.isnan(X)] = 1.0
        D_base_predictions = X.shape[1]
        X = np.concatenate((np.prod(X, axis=1)[:, None],
                            num_annot[:, None],
                            np.sum(X, axis=1)[:, None],
                            X),  axis=1)

        Xtest = preds_base_model.copy()
        Xtest[np.isnan(Xtest)] = 1.0
        Xtest = np.concatenate((np.prod(Xtest, axis=1)[:, None],
                                num_mismatches[:, None],
                                np.sum(Xtest, axis=1)[:, None],
                                Xtest), axis=1)

        K = GPy.kern.RBF(1, active_dims=[0]) + GPy.kern.RBF(1, active_dims=[1]) + GPy.kern.Linear(1, active_dims=[2]) + GPy.kern.RBF(D_base_predictions, active_dims=range(3, D_base_predictions+3))
        m = GPy.models.GPRegression(X, np.log(y), kernel=K)
        m.optimize_restarts(5, messages=0)
        predictions['raw GP'] = m.predict(Xtest)[0]

    if 'combine' in models:
        predictions['combine'] = np.ones_like(predictions[predictions.keys()[0]])

        for c_model in models:
            if c_model != 'combine':
                predictions['combine'] += predictions[c_model].flatten()[:, None]
        predictions['combine'] /= len(models)-1

    if 'ensemble' in models:
        predictions['ensemble'] = (predictions['product'].flatten() + predictions['linear-raw-stacker'].flatten())/2.


    if prob_calibration_model is not None:

        if models_to_calibrate is None:
            models_to_calibrate = ['linear-raw-stacker']

        for m in models:

            if False:# m == 'linear-raw-stacker':
                pred = np.exp(predictions[m].flatten()[:, None]) - 0.001 # undo log transformation
            else:
                pred = predictions[m].flatten()[:, None]

            if m in models_to_calibrate:

                cal_pred = prob_calibration_model[m].predict_proba(pred)[:, 1]
                #cal_pred = prob_calibration_model[m].predict_proba(pred)[:, 0]

                if len(pred) > 10:
                    assert np.allclose(sp.stats.spearmanr(pred, cal_pred)[0], 1.0)# or np.allclose(sp.stats.spearmanr(pred, cal_pred)[0], -1.0)

                predictions[m] = cal_pred

    if truth is not None:
        res_str = "Spearman r: "
        for m in models:
            res_str += "%s=%.3f " % (m, sp.stats.spearmanr(truth, predictions[m])[0])
        print res_str

        res_str = "NDCG: "
        for m in models:
            res_str += "%s=%.3f " % (m, azimuth.metrics.ndcg_at_k_ties(truth.values.flatten(), predictions[m].flatten(), truth.shape[0]))
        print res_str

    if return_model:
        if return_residuals:
            return predictions, clf, feature_names, residuals
        else:
            return predictions, clf, feature_names

    return predictions


def cross_validate_guideseq(data, preds_base_model, learn_options, models=['product', 'GP-stacker', 'linreg-stacker', 'RF-stacker', 'CFD', 'constant-power', 'constant-power-sanity-check'], seed=1234, n_folds=20):
    np.random.seed(seed)

    predictions = {}
    concatenated_predictions = {}
    for model in models:
        predictions[model] = [None for i in range(n_folds)]
        concatenated_predictions[model] = np.zeros_like(data['GUIDE-SEQ Reads'].values[:, None], dtype=np.float64)

    truth = [None for i in range(n_folds)]
    mismatches = [None for i in range(n_folds)]

    performance = pandas.DataFrame(data=np.zeros((len(models), n_folds)), columns=['Fold %d' % (i+1) for i in range(n_folds)], index=[model for model in models])


    # this extract the #mismatch classes for stratification
    label_encoder = sklearn.preprocessing.LabelEncoder()
    cv_classes = label_encoder.fit_transform(data['GUIDE-SEQ Reads'].values != 0)
    cv = sklearn.cross_validation.StratifiedKFold(cv_classes, n_folds=n_folds, shuffle=True)

    data = elevation.features.extract_mut_positions_stats(data)
    y = data['GUIDE-SEQ Reads'].values[:, None]
    num_annot = np.array([len(t) for t in data["Annotation"].values])
    all_test_ind = []

    for fold, [train, test] in enumerate(cv):
        test_ind = np.zeros_like(y).flatten() == 1
        train_ind = np.zeros_like(test_ind) == 1
        train_ind[train] = True
        test_ind[test] = True
        all_test_ind.extend(test.tolist())

        pred_all_models = stacked_predictions(data[test_ind], preds_base_model[test_ind], # test data
                                              preds_guideseq=preds_base_model[train_ind],  guideseq_data=data[train_ind], #train data
                                              models=models, learn_options=learn_options)

        truth[fold] = y[test]
        mismatches[fold] = num_annot[test][:, None]

        for model in pred_all_models:
            predictions[model][fold] = pred_all_models[model]

        print generate_result_string(predictions, truth, fold)
        print generate_result_string(predictions, truth, fold, metric=azimuth.metrics.ndcg_at_k_ties, name='NDCG k ties')
        for model in models:
            # sr =  sp.stats.spearmanr(predictions[model][fold], truth[fold])[0]
            sr = azimuth.metrics.ndcg_at_k_ties(truth[fold].flatten(), predictions[model][fold].flatten(), truth[fold].shape[0])
            performance['Fold %d' % (fold+1)].T[model] =sr
            assert not np.isnan(sr), "found nan performance metric"

            concatenated_predictions[model][test] = predictions[model][fold].flatten()[:, None]

    final_results_str = "Median NDCG across all folds: "
    for model in models:
        final_results_str += '%s=%.3f ' % (model, performance.T[model].median())

    print final_results_str

    return predictions, performance, mismatches, truth, concatenated_predictions, all_test_ind

def get_corr_by_mismatches(pred, truth, mism, cfd_pred=None):
    results = []
    sorted_mism = np.sort(np.unique(mism))
    for m in sorted_mism:
        if m == 1:
            continue
        pred_sub = pred[mism==m]
        truth_sub = truth[mism==m]
        corr = sp.stats.spearmanr(pred_sub, truth_sub)[0]
        pv = azimuth.util.get_pval_from_predictions(cfd_pred[mism==m], pred_sub, truth_sub, twotailed=False, method='steiger')[1]

        results.append([m, corr, (mism==m).sum(), sp.stats.spearmanr(pred, truth)[0], pv])
    return np.array(results)

def computeDoench_AUC(truth, predictions):
    """
    this is the AUC defined by John Doench in NBT 2016, on regression data
    to plot the AUC, use plt.plot(x, y, '-', label=label)
    """
    #ideal_norm = (truth - truth.min())/(truth.max()- truth.min())
    #pred_norm = (predictions - predictions.min())/(predictions.max() - predictions.min())
    sort_ind = np.argsort(predictions)[::-1]
    # uniform xvals:
    xvals = np.arange(len(truth)) / float(len(truth))
    truth_sort = truth[sort_ind]

    cumsumnorm = np.cumsum(truth_sort)/float(truth_sort.sum())
    auc_cfd = auc(xvals, cumsumnorm)
    return auc_cfd, xvals, cumsumnorm


def find_b(truth, pred, mism):
    from sklearn.cross_validation import KFold
    betas = np.linspace(0.0001, 1., 1000)

    kf = KFold(n=truth.shape[0], n_folds=20, shuffle=True)

    predictions = []
    all_truth = []
    base_pred = []

    for train, test in kf:
        results = []

        for b in betas:
            # print b
            results.append(sp.stats.spearmanr(truth[train], np.power(b, mism[train])*pred[train])[0])

        best_beta = betas[np.argmax(results)]

        preds = np.power(best_beta, mism[test])*pred[test]
        predictions.extend(preds.tolist())
        all_truth.extend(truth[test].tolist())
        base_pred.extend(pred[test].tolist())
        print best_beta, sp.stats.spearmanr(predictions, all_truth)[0]

    return sp.stats.spearmanr(predictions, all_truth), sp.stats.spearmanr(base_pred, all_truth)

def bootstrap_predictor(clf, Y_train, X_train, Y_test, X_test, replicates=100, perc_samples=0.9):
    predictions=[]

    for r in range(replicates):
        rand_sub = np.random.permutation(X_train.shape[0])[:int(X_train.shape[0]*0.9)]
        clf.fit(X_train[rand_sub], Y_train[rand_sub].flatten())
        predictions.append(clf.predict(X_test))

    predictions = np.array(predictions)
    pred_means = predictions.mean(0)
    pred_stds = predictions.std(0)

    # prob_predictions = np.zeros_like(predictions[0])
    # for i in range(prob_predictions.shape[0]):
    #     prob_predictions[i] = 1 - sp.stats.norm.cdf(1., loc=pred_means[i], scale=pred_stds[i])

    return pred_means# prob_predictions

@memory.cache
def train_final_model(learn_options=None):
    if learn_options is None:
        learn_options = {    'num_proc': 10,
                     'nuc_features_WT': False, 'include_pi_nuc_feat': False,
                     'annotation position one-hot': False,
                     'mutation_type' : False,
                     'mutation_details' : False,
                     'annotation_onehot' : False, # featurize like CFD
                     'annotation_decoupled_onehot' : True, # decouple the CFD features into letters and position
                     "include_Tm": False,
                     'include_azimuth_score': None, # all of them ["WT","MUT","DELTA"]
                     'azimuth_feat' : ['WT'], # ['WT'], #['WT'],#["MUT", "WT"],
                     "include_gene_position": False,
                     "cv": "stratified",
                     'adaboost_loss' : 'ls',
                     'adaboost_CV': False, "algorithm_hyperparam_search" : "grid",
                     'n_folds' : 10,
                     'allowed_category' : None,#"Mismatch",#"Insertion",
                     "include_NGGX_interaction": False,
                     'normalize_features' : False, 'class_weight': None,
                     "phen_transform": 'kde_cdf', #  'kde_cdf',
                     "training_metric": 'spearmanr',
                     "skip_pam_feat" : True, "letpos_indep_ft": False, "letpos_inter_ft": True,
                     "fit_intercept" : True,
                     "seed" : 12345,
                     "num_proc": 1,
                     "alpha": np.array([1.0e-3]),
                     "V": "CD33",
                     "left_right_guide_ind": [4,27,30], # 23-mer
                     "order": 1,
                     "testing_non_binary_target_name": 'ranks',
                     'models': ['AdaBoost'],
                     'post-process Platt': False,
                     'azimuth_score_in_stacker': False,
                     'use_mut_distances': False,
             }

    base_model, _ = train_base_model(learn_options)
    guideseq_data = load_guideseq(learn_options)
    preds_guideseq = predict_guideseq(base_model, guideseq_data, learn_options, naive_bayes_combine=True)

    # Train calibration model for probabilities
    cd33_data = elevation.load_data.load_cd33(learn_options)[0]
    cd33_data['Annotation'] = cd33_data['Annotation'].apply(lambda x: [x])
    prob_calibration_model = train_prob_calibration_model(cd33_data, guideseq_data, preds_guideseq, base_model, learn_options)

    return base_model, guideseq_data, preds_guideseq, learn_options, prob_calibration_model

def train_prob_calibration_model(cd33_data, guideseq_data, preds_guideseq, base_model, learn_options, which_stacker_model='linear-raw-stacker', other_calibration_models=None):
    assert which_stacker_model == 'linear-raw-stacker', "only LRS can be calibrated right now"
    # import ipdb; ipdb.set_trace()

    # if cd33_data is not None:
    Y_bin = cd33_data['Day21-ETP-binarized'].values
    Y = cd33_data['Day21-ETP'].values
    # else:
    #     ind = np.zeros_like(guideseq_data['GUIDE-SEQ Reads'].values)
    #     ind[guideseq_data['GUIDE-SEQ Reads'].values > 0] = True
    #     ind_zero = np.where(guideseq_data['GUIDE-SEQ Reads'].values==0)[0]
    #     ind[ind_zero[::ind_zero.shape[0]/float(ind.sum())]] = True
    #     ind = ind==True
    #     Y = guideseq_data[ind]['GUIDE-SEQ Reads'].values
    #     cd33_data = guideseq_data[ind]

    #X_guideseq = predict(base_model, cd33_data, learn_options)[0]
    nb_pred, individual_mut_pred_cd33 = predict(base_model, cd33_data, learn_options)

    # # This the models in the ensemble have to be calibrated as well, so we rely on
    # # having previously-calibrated models available in a dictionary
    # if which_model == 'ensemble':
    #     models = ['CFD', 'HsuZhang', 'product', 'linear-raw-stacker', 'ensemble']
    #     models_to_calibrate = ['product', 'linear-raw-stacker']
    #     calibration_models = other_calibration_models
    # else:
    #     models = [which_model]
    #     models_to_calibrate = None
    #     calibration_models = None

    # get linear-raw-stacker (or other model==which_model) predictions, including training of that model if appropriate (e.g. linear-raw-stacker)
    X_guideseq, clf_stacker_model, feature_names_stacker_model = stacked_predictions(cd33_data, individual_mut_pred_cd33,
                                     models=[which_stacker_model],
                                     guideseq_data=guideseq_data,
                                     preds_guideseq=preds_guideseq,
                                     learn_options=learn_options,
                                     models_to_calibrate=None,
                                     prob_calibration_model=None,
                                     return_model=True)
    X_guideseq = X_guideseq[which_stacker_model]

    clf = sklearn.linear_model.LogisticRegression(fit_intercept=True, solver='lbfgs')

    # fit the linear-raw-stacker (or whatever model is being calibrated) predictions on cd33 to the actual binary cd33 values    
    clf.fit(X_guideseq[:, None], Y_bin)    
    y_pred = clf.predict_proba(X_guideseq[:, None])[:, 1]
    #y_pred = clf.predict_proba(X_guideseq[:, None])[:, 0]

    #import ipdb; ipdb.set_trace()

    expected_sign = np.sign(sp.stats.spearmanr(X_guideseq, Y_bin)[0])
    assert np.allclose(sp.stats.spearmanr(y_pred, X_guideseq)[0], 1.0*expected_sign, atol=1e-2)

    return clf


def guideseq_zero_experiment(data, models=['CFD', 'HsuZhang'], learn_options=None):
    col = 'GUIDE-SEQ Reads'
    nonzero = np.where(data[col] != 0.0)[0]
    zero = np.where(data[col] == 0.0)[0]

    settings = np.linspace(0.0, 0.2, 1000)
    results = dict([(m, []) for m in models])

    for perc_zero in settings:
        selected = np.concatenate((nonzero, np.random.permutation(zero)[:int(len(zero)*perc_zero)]))
        print len(selected)
        pred_all_models = stacked_predictions(data.iloc[selected], None, # test data
                                              preds_guideseq=None,  guideseq_data=None,
                                              models=models, learn_options=learn_options)
        for m in models:
            results[m].append(sp.stats.spearmanr(data.iloc[selected][col].values, pred_all_models[m])[0])

    return results, settings


def plot_ndcg_at_k(predictions, truth, K=np.linspace(100, 30000, 100), normalize_from_below_too=False, flatten=True, method=3):
    if flatten:
        truth_all = np.concatenate(truth).flatten()
    else:
        truth_all = truth
    plt.figure()
    for m in predictions.keys():
        if flatten:
            all_pred = np.concatenate(predictions[m]).flatten()
        else:
            all_pred = predictions[m].flatten()

        vals = []
        for k in K:
            vals.append(azimuth.metrics.ndcg_at_k_ties(truth_all.copy(), all_pred.copy(), k, method=method, normalize_from_below_too=normalize_from_below_too))
        print method, vals
        plt.plot(K, vals, '.-', label=m)
    plt.legend(loc=0)



def plot_AUC_at_k(predictions, truth, K=np.linspace(100, 30000, 100), flatten=True):

    if flatten:
        truth_all = np.concatenate(truth).flatten()
    else:
        truth_all = truth.flatten()

    plt.figure()
    for method in predictions.keys():
        if flatten:
            all_pred = np.concatenate(predictions[method]).flatten()
        else:
            all_pred = predictions[method].flatten()

        vals = []
        for k in K:
            top_k = np.argsort(truth_all)[::-1][:k]
            vals.append(computeDoench_AUC(truth_all[top_k].copy(), all_pred[top_k].copy())[0])
        print method, vals
        plt.plot(K, vals, '.-', label=method)

    plt.xlabel('Top n values (sorted by truth)')
    plt.ylabel('AUC')
    plt.legend(loc=0)


def Haeussler_ROC(predictions, truth):
    plt.figure()
    for model in predictions.keys():
        fpr, tpr, thresholds = roc_curve(truth.flatten(), predictions[model].flatten())
        model_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=model + " AUC=%.3f" % model_auc)
    plt.legend(loc=0)

def plot_ndcg_by_mismatches(predictions, truth, mismatches, concatenate=True):
    if concatenate:
        truth_all = np.concatenate(truth).flatten()
        mism_all = np.concatenate(mismatches).flatten()
    plt.figure()
    mismatch_range = [2, 3, 4, 5, 6]
    for model in predictions.keys():
        model_results = []
        if concatenate:
            preds = np.concatenate(predictions[model]).flatten()
        for mi in mismatch_range:
            model_results.append(azimuth.metrics.ndcg_at_k_ties(truth_all[mism_all==mi], preds[mism_all==mi], truth_all[mism_all==mi].shape[0], normalize_from_below_too=True, method=3))
        plt.plot(mismatch_range, model_results, 'o-', label=model)
    plt.legend(loc=0)

def plot_spearman_with_different_weights(predictions, truth, weights, figsize=(12,10), plot=True):

    if plot:
        plt.figure(figsize=figsize)
    all_results = {}
    effective_sample_sizes = np.nan*np.zeros(len(weights))
    import elevation.metrics

    for model in predictions.keys():
        print model
        model_results = []

        for i, w in enumerate(weights):
            weights_array = truth.copy()
            weights_array += w

            # normalize between 0 and 1
            assert np.min(weights_array) >= 0
            # force max value to be 1.0 so we can take sum as effective sample size
            weights_array = weights_array / np.max(weights_array)            
            assert np.min(weights_array) >= 0 and np.max(weights_array) <= 1.0
            effective_sample_sizes[i] = np.sum(weights_array)
            model_results.append(elevation.metrics.spearman_weighted(truth, predictions[model].flatten(), w=weights_array))            
        if plot:
            plt.plot(weights, model_results, 'o-', label=model)
        all_results[model] = model_results
        
    
    if plot:
        plt.legend(loc=0)
        plt.xlabel('weight')
        plt.ylabel('correlation')
    return all_results, effective_sample_sizes


# JL 5/24/2017 --I have double checked that this yields the same results as the non-parallelized version
def plot_spearman_with_different_weights_parallel(predictions, truth, weights,
                                         figsize=(6, 5), plot=True, num_procs=None):
    from util import execute_parallel
    import elevation.metrics

    effective_sample_sizes = np.nan*np.zeros(len(weights))

    if plot:
        plt.figure(figsize=figsize)
    all_results = {}

    for model in predictions.keys():
        print model
        farg_pairs = []
        for i, w in enumerate(weights):
            weights_array = truth.copy()
            weights_array += w

            # normalize between 0 and 1
            assert np.min(weights_array) >= 0
            # force max value to be 1.0 so we can take sum as effective sample size
            weights_array = weights_array / np.max(weights_array)            
            assert np.min(weights_array) >= 0 and np.max(weights_array) <= 1.0
            effective_sample_sizes[i] = np.sum(weights_array)

            farg_pairs.append((elevation.metrics.spearman_weighted, (truth, predictions[model].flatten(), weights_array)))
        model_results = execute_parallel(farg_pairs, verbose=False, num_procs=num_procs)
        if plot:
            min_count = 10
            keep_ind = np.where(effective_sample_sizes > min_count)[0]
            #plt.plot(weights, model_results, 'o-', label=model)
            plt.semilogx(weights[keep_ind], np.array(model_results)[keep_ind], 'o-', label=model)
        all_results[model] = model_results
        
    if plot:
        plt.legend(loc=0)
        plt.xlabel('weight')
        plt.ylabel('correlation')
    return all_results, effective_sample_sizes

def plot_ndcg_with_different_discounts(predictions, truth, concatenate=False, thetas=np.logspace(np.log10(0.01), np.log10(1.0), 10), item=None):
    if concatenate:
        truth_all = np.concatenate(truth).flatten()
    else:
        if item is not None:
            truth_all = truth[item].flatten().copy()
        else:
            truth_all = truth.copy()

    plt.figure()
    all_results = {}
    for model in predictions.keys():
        model_results = []
        if concatenate:
            preds = np.concatenate(predictions[model]).flatten()
        else:
            if item is not None:
                preds = predictions[model][item].flatten().copy()
            else:
                preds = predictions[model].flatten().copy()

        for theta in thetas:
            model_results.append(azimuth.metrics.ndcg_at_k_ties(truth_all, preds, truth_all.shape[0], normalize_from_below_too=True, method=4, theta=theta))
        all_results[model] = model_results
        plt.plot(thetas, model_results, 'o-', label=model)
    plt.legend(loc=0)
    plt.xlabel("theta")
    plt.ylabel("NDCG")
    return all_results, thetas

def average_ndcg_across_folds(predictions, truth, thetas=np.logspace(np.log10(0.01), np.log10(1.0), 10)):
    results = dict([(m, np.zeros((len(thetas), len(truth)))) for m in predictions.keys()])

    for fold in range(len(truth)):
        ndcg_i, thetas = plot_ndcg_with_different_discounts(predictions, truth, thetas=thetas, item=fold, concatenate=False)
        for m in ndcg_i.keys():
            results[m][:, fold] = ndcg_i[m]

    plt.figure()
    for m in ndcg_i.keys():
        plt.errorbar(thetas, results[m].mean(1), 'o-', label=m)
    plt.legend(loc=0)



if __name__ == '__main__':


    np.random.seed(12345)
    learn_options = {'num_proc': 10,
    # 'replace_nan_with_base_pred': True,
                     'nuc_features_WT': False, 'include_pi_nuc_feat': False,
                     'annotation position one-hot': False,
                     'mutation_type' : False,
                     'mutation_details' : False,
                     'annotation_onehot' : True, # featurize like CFD
                     'annotation_decoupled_onehot' : ['pos', 'let', 'transl'], # decouple the CFD features into letters and position
                     "include_Tm": True,
                     'include_azimuth_score': None, # all of them ["WT","MUT","DELTA"]
                     'azimuth_feat' :['WT'],#["MUT", "WT"],
                     "include_gene_position": False,
                     "cv": "stratified",
                     'adaboost_loss' : 'ls',
                     'adaboost_CV': False, "algorithm_hyperparam_search" : "grid",
                     'n_folds' : 10,
                     'allowed_category' : None,#"Mismatch",#"Insertion",
                     "include_NGGX_interaction": False,
                     'normalize_features' : False, 'class_weight': None,
                     "phen_transform": 'kde_cdf', #  'kde_cdf',
                     "training_metric": 'spearmanr',
                     "skip_pam_feat" : True, "letpos_indep_ft": False, "letpos_inter_ft": True,
                     "fit_intercept" : True,
                     "seed" : 12345,
                     "num_proc": 1,
                     "alpha": np.array([1.0e-3]),
                     "V": "CD33", #TODO
                     # "target_name": 'Yranks', # TODO
                     # "ground_truth_label": 'Yranks', #TODO
                     # 'rank-transformed target name': 'Yranks',
                     # 'binary target name': 'Yranks',
                     "left_right_guide_ind": [4,27,30], # 21-mer
                     "order": 1,
                     "testing_non_binary_target_name": 'ranks',
                     'models': ["AdaBoost"],
                     'post-process Platt': False,
                     'azimuth_score_in_stacker': False,
                     'guide_seq_full': True,
                     'use_mut_distances': False,
                     'reload guideseq': False,
                     "renormalize_guideseq": True,
             }


    base_model, fnames = train_base_model(learn_options)

    guideseq_data = load_guideseq(learn_options, filterPAMs=False, subsample_zeros=False)

    # allsites_data = pandas.read_hdf('../../data/offtarget/allsites.pdhdf', start=0, stop=10)
    # allsites_data = pandas.read_pickle(settings.pj(settings.offtarget_data_dir, 'allsites.pd'))[:10]
    # nb_pred_allsites, individual_mut_pred_allsites = predict(base_model, allsites_data, learn_options)

    # with open('allsites_pred.pickle', 'wb') as f:
    #     pickle.dump(individual_mut_pred_allsites, f)
    cd33_data = elevation.load_data.load_cd33(learn_options)[0]
    cd33_data['Annotation'] = cd33_data['Annotation'].apply(lambda x: [x])

    with open('guideseq_all_zeros_pred.pickle', 'rb') as f:
        preds_guideseq, learn_options_p = pickle.load(f)



    calibration_models = {}
    calibration_models['LSR'] = train_prob_calibration_model(None, guideseq_data, preds_guideseq,
                                    base_model, learn_options, which_stacker_model='linear-raw-stacker',
                                    other_calibration_models=calibration_models)

    stop














    if False:
        print "generating guideseq base model predictions to pickle"
        preds_guideseq = predict_guideseq(base_model, guideseq_data, learn_options,  naive_bayes_combine=True)
        with open('guideseq_all_zeros_pred_cd33Hsu.pickle', 'wb') as f:
            pickle.dump([preds_guideseq, learn_options], f)
        import ipdb; ipdb.set_trace()
    else:
        print "loading guideseq base model predictions from pickle"
        with open('guideseq_all_zeros_pred.pickle', 'rb') as f:
            preds_guideseq, learn_options_p = pickle.load(f)

    preds_allsites = stacked_predictions(allsites_data, individual_mut_pred_allsites, guideseq_data=guideseq_data,
                                         preds_guideseq=preds_guideseq, learn_options=learn_options,
                                         models=['CFD', 'HsuZhang', 'CCTOP', 'product', 'linear-raw-stacker', 'ensemble'])


    import ipdb; ipdb.set_trace()
    if False:
        pred_guideseq = stacked_predictions(guideseq_data, base_preds_guideseq, guideseq_data=None,
                                            preds_guideseq=None,
                                            prob_calibration_model=None,
                                            learn_options=learn_options,
                                            models=['CFD', 'HsuZhang', 'CCTOP', 'product'],
                                            )
        muts = np.zeros((pred_guideseq['CCTOP'].shape[0],))
        truth = guideseq_data['GUIDE-SEQ Reads'].values.flatten()
        pred_guideseq['ensemble'] = (pred_guideseq['CFD'].flatten() + pred_guideseq['product'].flatten())/2.
        # guideseq_data = elevation.features.extract_mut_positions_stats(guideseq_data)
        # for i in range(len(guideseq_data)):
        #     muts[i], _ = filter_pam_out_of_muts(guideseq_data, i)

        # for i in range(2, 7):
        #     print i, computeDoench_AUC(truth[muts==i], pred_guideseq['CFD'][muts==i].flatten())[0], computeDoench_AUC(truth[muts==i], pred_guideseq['HsuZhang'][muts==i])[0]

        # K = np.logspace(np.log10(5), np.log10(guideseq_data.shape[0]), 20)
        # plot_AUC_at_k(pred_guideseq, guideseq_data["GUIDE-SEQ Reads"].values.flatten(), K, flatten=False)
        # plt.title("AUC on normalized Guideseq")
        # plot_ndcg_with_different_discounts(pred_guideseq, guideseq_data["GUIDE-SEQ Reads"].values.flatten(), concatenate=False)
        # plt.title("NDCG on normalized Guideseq")
        stop


    if False:
        assert learn_options['renormalize_guideseq'], "check this"
        predictions, performance, mismatches, truth, c_pred, t_ind =  cross_validate_guideseq(guideseq_data,
                                                                                                         base_preds_guideseq,
                                                                                                         learn_options,
        models= ['CFD','HsuZhang', 'CCTOP', 'product', 'linear-raw-stacker'], n_folds=5)
        # all_results, thetas = plot_ndcg_with_different_discounts(predictions, truth, concatenate=True)
        predictions_flat = dict([(k, np.concatenate(predictions[k]).flatten()) for k in predictions.keys()])
        truth_all = np.concatenate(truth).flatten()
        # K = np.linspace(100, guideseq_data.shape[0], 10)
        # plot_AUC_at_k(predictions, truth, K=K)
        # plot_ndcg_at_k(predictions, truth, K=K)
        # plot_ndcg_at_k(predictions, truth, K=K, normalize_from_below_too=True)
        # plot_ndcg_at_k(predictions, truth, K=K, normalize_from_below_too=False)
    if False:
        with open('roc.pickle', 'rb') as f:
            pred_roc, roc_Y_vals = pickle.load(f)

        plot_ndcg_with_different_discounts(pred_roc, roc_Y_vals.values)
    if False:
        hmg = elevation.load_data.load_hauessler_minus_guideseq()
        hmg['GUIDE-SEQ Reads'] = hmg['readFraction'].copy()
        # nb_pred_hmg, individual_mut_pred_hmg = predict(base_model, hmg, learn_options)
        with open('/tmp/trainhu_gs.pickle', 'rb') as f:
            individual_mut_pred_hmg = pickle.load(f)

        pred_guideseq = stacked_predictions(guideseq_data, base_preds_guideseq, guideseq_data=hmg,
                                            preds_guideseq=individual_mut_pred_hmg,
                                            prob_calibration_model=None,
                                            learn_options=learn_options,
                                            models=['product', 'linear-raw-stacker']#, 'CFD', 'HsuZhang', 'CCTOP'],
                                            )
        plot_ndcg_with_different_discounts(pred_guideseq, guideseq_data['GUIDE-SEQ Reads'].values, concatenate=False)
        import ipdb; ipdb.set_trace()
    if True:
        roc_data, roc_Y_bin, roc_Y_vals = elevation.load_data.load_HauesslerFig2();
        nb_pred_roc, individual_mut_pred_roc = predict(base_model, roc_data, learn_options)
        pred_roc = stacked_predictions(roc_data, individual_mut_pred_roc, learn_options=learn_options, guideseq_data=guideseq_data, preds_guideseq=base_preds_guideseq, models=['product', 'linear-raw-stacker', 'CFD', 'HsuZhang', 'CCTOP'])
        plot_ndcg_with_different_discounts(pred_roc, roc_Y_vals.values)
        stop
