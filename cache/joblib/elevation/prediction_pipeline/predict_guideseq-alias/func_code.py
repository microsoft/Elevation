# first line: 184
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
