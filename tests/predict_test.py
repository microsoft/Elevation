import os
import sys
import shutil

import unittest
# from mock import patch, Mock, PropertyMock, MagicMock

import pandas as pd
import numpy as np
from warnings import warn


class PredictTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PredictTest, self).__init__(*args, **kwargs)

    # @unittest.skip("ignore")
    def test_predict_hmg(self):
        sys.stdout = sys.__stdout__

        import elevation.load_data
        from elevation import settings, options
        from elevation.cmds import predict

        hmg = predict.Predict(init_models=False).get_hmg_data()
        wildtype = list(hmg['30mer'])[:settings.pred_test_num]
        offtarget = list(hmg['30mer_mut'])[:settings.pred_test_num]

        predictions = predict.Predict().execute(wildtype, offtarget)
        pred_df_data = {key: val.reshape(val.shape[0]) for key, val in predictions.iteritems()}
        pred_df = pd.DataFrame(data=pred_df_data)

        truth_df = pd.read_excel(settings.pred_default_fixture_file)[:settings.pred_test_num]
        for column in pred_df:
            if np.any(np.abs(pred_df[column] - truth_df[column]) > 0):
                warn("predictions don't exactly match expected for %s" % column)
                idx = np.abs(pred_df[column] - truth_df[column]) > 0
                x = pred_df[column][idx] - truth_df[column][idx]
                # for i, value in enumerate(x):
                    # warn("Inequality %s %s: %s" % (column, i, value))
            assert np.allclose(pred_df[column], truth_df[column], atol=1e-08, rtol=0.0), "%s doesn't match" % column

    # @unittest.skip("ignore")
    def test_agg_nicolo(self):
        import pickle
        from elevation import settings
        from elevation import aggregation

        with open(settings.agg_nicolo_fixture_file, "r") as fh:
            nicolo_results = pickle.load(fh)

        model = aggregation.get_aggregation_model()
        y_pred = model.predict(nicolo_results[0])
        assert np.allclose(y_pred, nicolo_results[1])


    @unittest.skip("ignore")
    def test_predict_nicolo(self):
        import pickle
        from elevation import settings
        from elevation.cmds.predict import Predict
        preds_file = settings.pj(settings.repo_root, 'tests', 'fixtures', 'preds.lrs.hmg_v1v2.gsgr1.boxcox1.pkl')
        with open(preds_file, 'r') as f:
            preds = pickle.load(f)
            p = Predict() # updated (new) Hauessler & GUIDE-seq
            p.hmg_data = p.get_hmg_data(force_compute=True)
            guides, offtargets = p.hmg_data['30mer'].values, p.hmg_data['30mer_mut'].values
            hmg_preds = p.execute(guides, offtargets)['linear-raw-stacker']
            assert np.allclose(preds, hmg_preds)


    # @unittest.skip("ignore")
    def test_agg_hauessler(self):
        sys.stdout = sys.__stdout__

        import pickle
        from elevation import settings
        from elevation import aggregation

        with open(settings.agg_model_file) as fh:
            final_model, other = pickle.load(fh)

        inputs = pd.read_excel(settings.pred_default_fixture_file)
        results = []
        rs = np.random.RandomState(settings.default_random_seed)
        perm = rs.permutation(inputs.shape[0])
        stacker = inputs["linear-raw-stacker"].values[perm]
        cfd = inputs["CFD"].values[perm]
        isgenic = rs.random_sample(inputs.shape[0]) > 0.5
        pos = 0
        while pos < perm.shape[0]:
            end = pos + rs.randint(1, 2000)
            if end > perm.shape[0]:
                end = perm.shape[0]
            result = aggregation.get_aggregated_score(
                stacker[pos:end],
                cfd[pos:end],
                isgenic[pos:end],
                final_model)
            results += list(result)
            pos = end

        pred_df = pd.DataFrame(data={"agg_score": results})
        truth_df = pd.read_excel(settings.agg_default_fixture_file)
        for column in pred_df:
            if np.any(np.abs(pred_df[column] - truth_df[column]) > 0):
                warn("aggregate predictions don't exactly match expected for %s" % column)
                idx = np.abs(pred_df[column] - truth_df[column]) > 0
                x = pred_df[column][idx] - truth_df[column][idx]
                for i, value in enumerate(x):
                    warn("Inequality %s %s: %s" % (column, i, value))
            assert np.allclose(pred_df[column], truth_df[column], atol=1e-10, rtol=0.0), "%s doesn't match" % column


# class FitTest(unittest.TestCase):
#
#    def __init__(self, *args, **kwargs):
#        super(FitTest, self).__init__(*args, **kwargs)
#
#    def setUp(self):
#        from elevation import settings
#
#        self.cachedir = settings.pj(settings.repo_root, "tests", "cache")
#        self.cachedir_patch = patch('elevation.settings.cachedir',  self.cachedir)
#        self.cachedir_patch.start()
#
#        self.tmpdir = settings.pj(settings.repo_root, "tests", "tmp")
#        self.tmpdir_patch = patch('elevation.settings.tmpdir', self.tmpdir)
#        self.tmpdir_patch.start()
#
#        print self.tmpdir
#        if os.path.exists(self.cachedir):
#            shutil.rmtree(self.cachedir)
#        os.mkdir(self.cachedir)
#
#        if os.path.exists(self.tmpdir):
#            shutil.rmtree(self.tmpdir)
#        os.mkdir(self.tmpdir)
#
#    def tearDown(self):
#        self.cachedir_patch.stop()
#        self.tmpdir_patch.stop()
#
#    @unittest.skip("ignore")
#    def test_settings_mock(self):
#        sys.stdout = sys.__stdout__
#
#        from elevation import settings, prediction_pipeline, load_data
#        from elevation.cmds import fit, predict
#        import elevation
#
#        assert self.cachedir == settings.cachedir
#        assert self.cachedir == prediction_pipeline.settings.cachedir
#        assert self.cachedir == load_data.settings.cachedir
#        assert self.cachedir == fit.settings.cachedir
#        assert self.cachedir == predict.settings.cachedir
#        assert self.cachedir == elevation.settings.cachedir
#        assert self.cachedir == elevation.prediction_pipeline.settings.cachedir
#        assert self.cachedir == elevation.load_data.settings.cachedir
#        assert self.cachedir == elevation.cmds.fit.settings.cachedir
#        assert self.cachedir == elevation.cmds.predict.settings.cachedir
#
#        assert self.tmpdir == settings.tmpdir
#        assert self.tmpdir == prediction_pipeline.settings.tmpdir
#        assert self.tmpdir == load_data.settings.tmpdir
#        assert self.tmpdir == fit.settings.tmpdir
#        assert self.tmpdir == predict.settings.tmpdir
#        assert self.tmpdir == elevation.settings.tmpdir
#        assert self.tmpdir == elevation.prediction_pipeline.settings.tmpdir
#        assert self.tmpdir == elevation.load_data.settings.tmpdir
#        assert self.tmpdir == elevation.cmds.fit.settings.tmpdir
#        assert self.tmpdir == elevation.cmds.predict.settings.tmpdir

# @unittest.skip("ignore")
# def test_retrain_predict_hauessler(self):
#     from elevation.cmds import predict, fit
#
#     learn_options_override = {
#         "seed": 12345
#     }
#
#     fit.Fit().execute(learn_options_override=learn_options_override, force_rebuild=True)
#
# @unittest.skip("ignore")
# def test_retrain_new_seed_predict_hauessler(self):
#     pass
