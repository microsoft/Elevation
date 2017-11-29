import numpy as np
import scipy.stats as st


def pearson_weighted(x, y, w=None):
    assert len(x.shape) == 1
    assert len(y.shape) == 1

    if w is None:
        w = np.ones_like(y)

    assert x.shape == y.shape and y.shape==w.shape, "must have same shape"
    
    m_x_w = np.sum(x * w)/np.sum(w)
    m_y_w = np.sum(y * w)/np.sum(w)

    cov_x_y = np.sum(w*(x - m_x_w)*(y - m_y_w))/np.sum(w)
    cov_x_x = np.sum(w*(x - m_x_w)*(x - m_x_w))/np.sum(w)
    cov_y_y = np.sum(w*(y - m_y_w)*(y - m_y_w))/np.sum(w)

    pearson = cov_x_y/np.sqrt(cov_x_x * cov_y_y)

    return pearson

def spearman_weighted(x, y, w=None):
    x_ranked = st.mstats.rankdata(x)
    y_ranked = st.mstats.rankdata(y)

    return pearson_weighted(x_ranked, y_ranked, w=w)

def spearman_weighted_swap_perm_test(preds1, preds2, true_labels, nperm, weights_array):
        
    if isinstance(preds1, list):
        preds1 = np.array(preds1)
    else:
        preds1 = preds1.flatten()

    if isinstance(preds2, list):
        preds2 = np.array(preds2)
    else:
        preds2 = preds2.flatten()

    if isinstance(true_labels, list):
        true_labels = np.array(true_labels)
    else:
        true_labels = true_labels.flatten()

    assert len(preds1) == len(preds2), "need same number of preditions from each model"
    assert len(preds1) == len(true_labels), "need same number of preditions in truth and predictions"
    N = len(preds1)

    # re-sort all by truth ordering so that when swap they are aligned
    sorted_ind = np.argsort(true_labels)[::-1]
    true_labels = true_labels[sorted_ind]
    preds1 = preds1[sorted_ind]
    preds2 = preds2[sorted_ind]

    ranks1 = st.mstats.rankdata(preds1)
    ranks2 = st.mstats.rankdata(preds2)
        
    corr1 = spearman_weighted(true_labels, ranks1, w=weights_array)
    corr2 = spearman_weighted(true_labels, ranks2, w=weights_array)
                
    real_corr_diff = np.abs(corr1 - corr2)                
    perm_corr_diff = np.nan*np.zeros(nperm)

    for t in range(nperm):
        pair_ind_to_swap = np.random.rand(N) < 0.5

        ranks1_perm = ranks1.copy();
        ranks1_perm[pair_ind_to_swap] = ranks2[pair_ind_to_swap]

        ranks2_perm = ranks2.copy();
        ranks2_perm[pair_ind_to_swap] = ranks1[pair_ind_to_swap]

        corr1_p = spearman_weighted(true_labels, ranks1_perm, w=weights_array)
        corr2_p = spearman_weighted(true_labels, ranks2_perm, w=weights_array)
        perm_corr_diff[t] = np.abs(corr1_p - corr2_p)
                        
    num_stat_greater = np.max((((perm_corr_diff > real_corr_diff).sum() + 1), 1.0))
    pval = num_stat_greater / nperm

    if False:
        plt.figure();
        plt.plot(np.sort(perm_corr_diff), '.')
        plt.plot(real_corr_diff*np.ones(perm_corr_diff.shape), 'k-')
        plt.show()
                        
    return pval, real_corr_diff, perm_corr_diff, corr1, corr2



if __name__ == '__main__':

    for i in range(10):
        x = np.random.randn(100)
        y = np.random.randn(100)

        assert np.allclose(pearson_weighted(x, y), st.pearsonr(x, y)[0])
        assert np.allclose(spearman_weighted(x, y), st.spearmanr(x, y)[0])


        x = y.copy()
        x += np.random.randn(*x.shape) * 0.05
        assert np.allclose(spearman_weighted(x, y), st.spearmanr(x, y)[0])
        assert np.allclose(pearson_weighted(x, y), st.pearsonr(x, y)[0]) 

