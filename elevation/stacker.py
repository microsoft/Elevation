import numpy as np
import scipy as sp
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sklearn.linear_model as lm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#import GPy
import scipy.stats as ss


class StackerFeat(object):
    def __init__(self):
        pass

    def featurize(self, X, training=False):
        f_dict = {}
        f_dict["f_prod"] = np.nanprod(X, axis=1)[:, None]
        f_dict["f_sum"] = np.nansum(X, axis=1)[:, None]
        f_dict["f_mean"] = np.nanmean(X, axis=1)[:, None]
        f_dict["f_max"] = np.nanmax(X, axis=1)[:, None]
        f_dict["f_min"] = np.nanmin(X, axis=1)[:, None]
        f_dict["f_count"] = X.shape[1] - np.sum(np.isnan(X), axis=1)[:, None]

        # now make each feature interact with the counts:
        if False:
            for key in f_dict.keys():
                if not (key=="count"):
                    f_dict[key + "_count"] = np.multiply(f_dict["f_count"], f_dict[key])
        self.ordered_keys = []
        f_concat = None
        for key in f_dict.keys():
            self.ordered_keys.append(key)
            dat = f_dict[key]
            if self.normalize_feat:
                #print "normalizing features"
                if training:
                    self.mean = np.mean(dat)
                    self.std = np.mean(dat)
                dat = (dat - self.mean) / self.std
            if f_concat is None:
                f_concat = dat
            else:
                f_concat = np.concatenate((f_concat, dat), axis=1)
        return f_concat


    def fit(self, X, y, model, normalize_feat=False, phen_transform="log"):
        self.model=model
        self.normalize_feat = normalize_feat
        y = np.asarray(y, dtype=float)
        assert not np.any(np.isnan(y)), "found nan in y for stacker"

        # z = st.mstats.rankdata(y)/y.shape[0]
        # z =  (y - y.min())/(y.max()-y.min())
        # z = z -np.mean(z)
        # z = y

        X_feat = self.featurize(X, training=True)

        # z = np.log(y)
        # z = np.sqrt(y) + np.sqrt(y+1)
        if phen_transform=="sqrt":
            z = np.sqrt(y)
        elif phen_transform=="identity":
            z = y
        elif phen_transform=="log":
            z = np.log(y)

        assert not np.any(np.isnan(z)), "found nan in z for stacker"

        if model=="linreg":
            self.m = lm.LinearRegression()
            self.m.fit(X_feat, z)
        if model=="L1":
            alpha = np.array([1e-6*pow(1.3,x) for x in range(0,100)])
            self.m = lm.LassoCV(alphas=None, fit_intercept=True, normalize=True, max_iter=1000, verbose=False, cv=10)
            self.m.fit(X_feat, z.flatten())
        elif model=="GBR":
            self.m = GradientBoostingRegressor(min_samples_split=40, min_samples_leaf=40, max_depth=1, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)
            #loss=None
            self.m.fit(X_feat, z)
        elif model=="RFR":
            self.m = RandomForestRegressor(n_estimators=2000)
            self.m.fit(X_feat, z.flatten())
        elif model=="GP":
            # for i in range(X_feat.shape[1]):
            #     plt.figure()
            #     plt.plot(X_feat[:, i], np.log(y), 'bo', alpha=.4)
            #     plt.figure()
            #     plt.plot(X_feat[:, i], np.sqrt(y), 'ro', alpha=.4)

            # z -= z.mean()
            # z /= z.std()
            # X_feat -= X_feat.mean(0)
            # X_feat /= X_feat.std(0)
            kern = GPy.kern.RBF(X_feat.shape[1], ARD=False)
            # kern = GPy.kern.Linear(X_feat.shape[1], ARD=False)
            # kern = GPy.kern.Linear(4, ARD=True, active_dims=[0,1,2,3]) + GPy.kern.RBF(1, active_dims=[4]) + GPy.kern.RBF(1, active_dims=[5])

            self.m = GPy.models.GPRegression(X_feat, z, kernel=kern, normalizer=True) # NOTE: this is normalized (mean+var)
            # self.m = GPy.models.WarpedGP(X_feat, y, kernel=kern, warping_terms=1)
            #self.m.optimize_restarts(3, messages=0)
            self.m.optimize()

    def predict(self, X):
        X_feat = self.featurize(X)
        # X_feat -= X_feat.mean(0)
        # X_feat /= X_feat.std(0)
        if self.model=="GP":
            return self.m.predict(X_feat)[0]
        else:
            return self.m.predict(X_feat)

class Stacker(object):
    def __init__(self, y, X, warp_out=False, loss='spearman', opt_method="optimize", combiner="nb"):
        """
        combiner is either 'nb' for Naive Bayes (like CFD), or "nb_modulated" for a variation on it
        """
        self.y = np.array(y, dtype=float)
        self.X = X
        self.warp_out = warp_out # also warp the output in addition to the default warping of base predicton inputs
        self.loss = loss # 'spearman' or otherwise defaults to 'rmse'
        # renormalize the target variable to lie in [0, 1]:
        self.z = (self.y - self.y.min())/(self.y.max()-self.y.min())
        # self.z = st.mstats.rankdata(self.y)/self.y.shape[0]
        self.maximum = {'a':np.nan, 'b':np.nan}
        self.opt_method = opt_method
        self.combiner = combiner

    def __str__(self):
        return "a=%f, b=%f, k=%f, loss=%s, opt_method=%s" % (self.maximum['a'], self.maximum['b'], self.maximum['k'], self.loss, self.opt_method)

    @staticmethod
    def warp_func(X, a, b):
        if False:
            #return (1 + np.tanh(a*(b+X)))/2.
            fx = np.tanh(a*(b + X))
            uniquevals = np.unique(fx[~np.isnan(fx)])
            assert len(uniquevals) > 1, "only val=%d found, probably saturation of tanh" % uniquevals[0]
            fxmax = np.tanh(a*(b + 1.0))
            fxmin = np.tanh(a*(b + 0.0))
            fx = (fx-fxmin)/(fxmax - fxmin)
        else:
            fx = ss.logistic.cdf(X, loc=a, scale=np.exp(b))
        assert not np.all(np.isnan(fx)), "all nans found in fx"
        return fx

    def warp_output(self, c, d):
        return Stacker.warp_func(self.y, c, d)

    def warp_inputs(self, a, b, X=None):
        if X is None:
            X = self.X
        Xwarp = Stacker.warp_func(X, a, b)
        assert not np.all(np.isnan(Xwarp)), "all nans found in Xwarp"
        return Xwarp

    def negf(self, params):
        return -1.0*self.f(params)

    def f(self, params):
        """
        Objective function to minimize/maximize
        """
        verbose = False

        if len(params)==3:
            [a,b,k] = params
            c = None
            d = None
        elif len(params)==5:
            [a,b,k,c,d] = params
        else:
            raise Exception()

        transform_type = "rank_transform"
        #transform_type = "tanh"

        if self.warp_out:
            if transform_type == "rank_transform":
                z = self.z
            elif transform_type == "tanh":
                z = self.warp_output(c, d)
            else:
                raise Exception()
        else:
            z = self.y.copy()

        #preds = np.nanprod(self.warp_inputs(a,b), axis=1).flatten()
        preds = self.predict(self.X, a, b, k, c, d)

        if self.loss == 'spearman':
            loss_val = st.spearmanr(z, preds)[0]
        elif self.loss == 'rmse':
            loss_val = -np.sqrt(np.mean((z.flatten() - preds)**2))
        else:
            raise Exception("loss not found: %s, valid loss in [spearman, rmse]" % self.loss)

        assert not np.isnan(loss_val), "loss_val is nan for a=%f, b=%f, loss=%s" % (a, b, self.loss)
        return loss_val

    def maximize(self):
        from bayes_opt import BayesianOptimization

        contour=None
        A=None
        B=None
        K=None
        if self.opt_method=="bo":
            raise NotImplementedError("have not updated this in a while so needs work most likely")
            if self.warp_out:
                bo = BayesianOptimization(self.f, {'a': (1e-3, 10.),
                                                   'b': (-10.0, 10.0),
                                                   'c': (1e-3, 10.),
                                                   'd': (-10.0, 10.0)})
            else:
                bo = BayesianOptimization(self.f, {'a': (1e-3, 5.),
                                                   'b': (-3.0, 3.0)})

            bo.maximize(init_points=20, n_iter=30)
            self.maximum = bo.res['max']['max_params']
        elif self.opt_method=="grid":
            S = 1000 # number of points to try in the grid
            if False:
                A = np.linspace(1e-3, 1.5, S)
                B = np.linspace(-10.0, 10.0, S)
                K = np.logspace(0, 1, S)
            else:
                A = [0.01]
                B = [10]
                K = np.linspace(0, 0.03, S)
            contour = np.zeros((len(A), len(B), len(K))) # objective function surface
            for i,a in enumerate(A):
                for j,b in enumerate(B):
                    for m,k in enumerate(K):
                        contour[i,j,m] = self.f([a,b,k])
            opt = np.where(contour==np.nanmax(contour))
            self.maximum = {'a': A[opt[0][0]], 'b': B[opt[1][0]], 'k': K[opt[2][0]]}
        elif self.opt_method=="optimize":
            #a=0.01; b=10; k=0
            a=0; b=1; k=0
            res = minimize(self.negf, [a,b,k], method='Nelder-Mead', options={'disp': False, 'xtol' : 1e-6})
            self.maximum = {'a': res.x[0], 'b': res.x[1], 'k' : res.x[2]}
        return contour, A, B, K

    def predict(self, X, a=None, b=None, k=None, c=None, d=None):
        """
        X is N x M where N is # of guides, and M is max # annotations
        """
        if a is None: a = self.maximum['a']
        if b is None: b = self.maximum['b']
        if k is None: k = self.maximum['k']
        assert c is None
        assert d is None

        warpedX = self.warp_inputs(a, b, X)

        if self.combiner == "nb":
            pass
        elif self.combiner == "nb_modulated":
            num_annot = np.sum(~np.isnan(X), axis=1)
            modulation = 1.0/num_annot**k
            modulation = np.tile(modulation, [6,1]).T
            warpedX = warpedX**modulation
        else:
            raise Exception()

        assert np.nanmin(warpedX) > 0.0
        assert np.nanmax(warpedX) < 1.0
        return np.nanprod(warpedX, axis=1)[:, None]


class TransformBaseStacker(Stacker):

    def __init__(self, y, X, num_mut, warp_out = False, loss = 'spearman', opt_method = 'optimize', combiner = 'nb', phenotype_transformation=np.log, clf=None):
        self.y = phenotype_transformation(y)
        self.clf = clf
        self.X_num_mut = num_mut
        return super(TransformBaseStacker, self).__init__(y, X, warp_out, loss, opt_method, combiner)

    def featurize(self, X, X_num_mut):
        Xfeat = X.copy()
        Xfeat[np.isnan(Xfeat)] = 1.0
        Xfeat = np.concatenate((Xfeat, X_num_mut, np.prod(Xfeat, axis=1)[:, None], np.sum(Xfeat, axis=1)[:, None]), axis=1)
        return Xfeat

    def fit(self):
        self.maximize()

    def f(self, params):
        """
        Objective function to minimize/maximize
        """


        if len(params)==3:
            [a,b,k] = params
            c = None
            d = None
        elif len(params)==5:
            [a,b,k,c,d] = params
        else:
            raise Exception()

        z = self.y.copy()

        warpedX = self.warp_inputs(a, b, self.X)
        Xfeat = self.featurize(warpedX, self.X_num_mut)
        self.clf.fit(Xfeat, z.flatten())

        preds = self.predict(self.X, self.X_num_mut, a, b, k, c, d)

        if self.loss == 'spearman':
            loss_val = st.spearmanr(z, preds)[0]
        elif self.loss == 'rmse':
            loss_val = -np.sqrt(np.mean((z.flatten() - preds)**2))
        else:
            raise Exception("loss not found: %s, valid loss in [spearman, rmse]" % self.loss)

        assert not np.isnan(loss_val), "loss_val is nan for a=%f, b=%f, loss=%s" % (a, b, self.loss)
        return loss_val

    def predict(self, X, X_num_mut, a=None, b=None, k=None, c=None, d=None):
        """
        X is N x M where N is # of guides, and M is max # annotations
        """
        if a is None: a = self.maximum['a']
        if b is None: b = self.maximum['b']
        if k is None: k = self.maximum['k']
        assert c is None
        assert d is None

        warpedX = self.warp_inputs(a, b, X)
        Xfeat = self.featurize(warpedX, X_num_mut)

        assert np.nanmin(warpedX) > 0.0
        assert np.nanmax(warpedX) < 1.0

        return self.clf.predict(Xfeat)

if __name__ == '__main__':
    np.random.seed(1)
    y = np.linspace(0, 1, 100)[:, None] + np.random.randn(100, 1)*0.1
    X = np.logspace(0, 2, 100)[:, None] + np.random.randn(100, 1)*0.1 # np.random.randn(100, 5)
    X[:, 1:] = np.nan
    m = Stacker(y, X)
    m.maximize()
    m.predict(X)

    exit

    x=np.linspace(0,1,100)
    a_range = [1, 0.5]#np.logspace(1e-5, 100, 100)
    for a in a_range:
        fx = a*np.tanh(x/a)
        plt.plot(x, fx, '.')
        plt.title("diff values of a")
    #plt.legend(a_range, loc=0)
    plt.xlabel("x")
    plt.ylabel("f(x)")


    S = 3 # number of points to try in the grid
    A = [0.01]#np.linspace(1e-3, 1.5, S)
    B = [10]#np.linspace(-10.0, 10.0, S)
    x=np.linspace(0,1,100)
    plt.figure()
    myleg = []
    for i,a in enumerate(A):
        for j,b in enumerate(B):
            myleg.append((a,b))
            fx = Stacker.warp_func(x, a,b)
            plt.plot(x, fx, '.')
    plt.legend(myleg, fontsize='xx-small')
    plt.show()
