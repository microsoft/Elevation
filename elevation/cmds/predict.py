import sys
import io
import time
import numpy as np

from command_base import Command
from elevation import settings

import pandas as pd
import elevation
import elevation.load_data
import elevation.util
import elevation.prediction_pipeline as pp
import matplotlib.pyplot as plt
import scipy.stats as st
import pickle

from elevation import options

class Predict(Command):

    def __init__(self, learn_options_override=None, init_models=True):
        # np.random.seed(123)

        self.learn_options = options.learn_options

        if isinstance(learn_options_override, dict):
            self.learn_options.update(learn_options_override)
        else:
            assert learn_options_override is None

        self.base_model = None
        self.guideseq_data = None
        self.preds_guideseq = None
        self.cd33_data = None
        self.calibration_models = None

        # only for prediction tests
        self.haeussler_data = None
        self.haeussler_preds = None
        self.hmg_data = None
        self.hmg_preds = None

        if init_models:
            start = time.time()
            self.base_model = self.get_base_model()
            self.guideseq_data = self.get_guideseq_data()
            self.preds_guideseq = self.get_preds_guideseq()
            self.cd33_data = self.get_cd33()
            self.calibration_models = self.get_calibrated()
            print "Time spent loading pickles: ", time.time() - start

    def execute(self, wildtype, offtarget):
        start = time.time()
        wt = wildtype
        mut = offtarget
        df = pd.DataFrame(columns=['30mer', '30mer_mut', 'Annotation'], index=range(len(wt)))
        df['30mer'] = wt
        df['30mer_mut'] = mut

        # df['Annotation'] = df.apply(lambda x: elevation.load_data.annot_from_seqs(x['30mer'], x['30mer_mut'], x['Num mismatches']), axis=1)
        annot = []
        for i in range(len(wt)):
            annot.append(elevation.load_data.annot_from_seqs(wt[i], mut[i]))
        df['Annotation'] = annot
        print "Time spent parsing input: ", time.time() - start

        base_model_time = time.time()
        nb_pred, individual_mut_pred = elevation.prediction_pipeline.predict(self.base_model, df, self.learn_options)
        print "Time spent in base model predict(): ", time.time() - base_model_time

        start = time.time()
        pred = pp.stacked_predictions(df, individual_mut_pred,
                                                     learn_options=self.learn_options,
                                                     guideseq_data=self.guideseq_data,
                                                     preds_guideseq=self.preds_guideseq,
                                                     prob_calibration_model=self.calibration_models,
                                                     models=['linear-raw-stacker', 'CFD'])
        print "Time spent in stacked_predictions: ", time.time() - start

        # pred = {'linear-raw-stacker': [...], 'CFD': [...]}
        return pred

    def get_base_model(self, force_compute=False):
        base_model, fnames = elevation.util.get_or_compute(
            elevation.settings.base_model_file,
            (pp.train_base_model, (self.learn_options,)),
            force_compute=force_compute
        )
        return base_model

    def get_guideseq_data(self, force_compute=False):
        guideseq_data = elevation.util.get_or_compute(
            elevation.settings.guideseq_data,
            (pp.load_guideseq, (self.learn_options, False, False)),
            force_compute=force_compute
        )
        return guideseq_data

    def get_preds_guideseq(self, force_compute=False):
        preds_guideseq = elevation.util.get_or_compute(
            elevation.settings.gspred_filename,
            (pp.predict_guideseq, (self.base_model, self.guideseq_data, self.learn_options, True)),
            force_compute=force_compute
        )
        return preds_guideseq

    def get_haeussler_data(self, force_compute=False):
        return elevation.util.get_or_compute(
            elevation.settings.hauessler_data,
            (elevation.load_data.load_HauesslerFig2, ()),
            force_compute=force_compute
        )

    def get_preds_haeussler(self, force_compute=False):
        return elevation.util.get_or_compute(
            elevation.settings.hpred_filename,
            (pp.predict, (self.base_model, self.haeussler_data[0], self.learn_options)),
            force_compute=force_compute
        )

    def get_hmg_data(self, force_compute=False):
        hmg_v1v2 = elevation.util.get_or_compute(
            elevation.settings.hmg_data,
            (elevation.load_data.load_hauessler_minus_guideseq, (self.learn_options,)),
            force_compute=force_compute
        )
        hmg_v1v2["GUIDE-SEQ Reads"] = hmg_v1v2["readFraction"]
        return hmg_v1v2

    def get_preds_hmg(self, force_compute=False):
        return elevation.util.get_or_compute(
            elevation.settings.hmgpred_filename,
            (pp.predict, (self.base_model, self.hmg_data, self.learn_options)),
            force_compute=force_compute
        )

    def get_cd33(self, force_compute=False):
        cd33_data = elevation.util.get_or_compute(
            elevation.settings.cd33_file,
            (elevation.load_data.load_cd33, (self.learn_options,)),
            force_compute=force_compute
        )
        cd33_data = cd33_data[0]
        cd33_data['Annotation'] = cd33_data['Annotation'].apply(lambda x: [x])
        return cd33_data

    def get_calibrated(self, force_compute=False):
        calibration_models = elevation.util.get_or_compute(
            elevation.settings.calibration_file,
            (self.generate_calibrated, ()),
            force_compute=force_compute
        )
        return calibration_models

    def generate_models(self):
        # TODO: no direct dependencies on the below...
        # hmg = elevation.load_data.load_hauessler_minus_guideseq()
        # hmg['GUIDE-SEQ Reads'] = hmg['readFraction'].copy()
        #
        # nb_pred_hmg, individual_mut_pred_hmg = elevation.util.get_or_compute(
        #     elevation.settings.trainhu_gs_file,
        #     (pp.predict, (self.base_model, hmg, self.learn_options)))
        pass

    def generate_calibrated(self):
        to_be_calibrated = ['linear-raw-stacker']
        calibration_models = {}
        for m in to_be_calibrated:
            calibration_models[m] = pp.train_prob_calibration_model(self.cd33_data, self.guideseq_data, self.preds_guideseq,
                                                                    self.base_model, self.learn_options, which_stacker_model=m,
                                                                    other_calibration_models=calibration_models)
        return calibration_models

    @classmethod
    def execute_file(cls, filename, ontarget_column, offtarget_column, delimiter=","):
        save_stdout = sys.stdout
        save_stderr = sys.stderr
        sys.stderr = sys.stdout = io.BytesIO()

        # compute preds
        data = pd.read_csv(filename, delimiter=delimiter)
        wildtype, offtarget = data[[ontarget_column, offtarget_column]].values.T
        preds = cls().execute(wildtype, offtarget)
        columns = preds.keys()
        num_items = data.shape[0]
        preds = {k: v.flatten() for k, v in preds.iteritems()}

        sys.stdout = save_stdout
        sys.stderr = save_stderr
        print delimiter.join(columns)
        for i in range(num_items):
            print delimiter.join(["%.8f" % preds[col][i] for col in columns])

    @classmethod
    def cli_execute(cls):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', '--file', help="File to execute.")
        parser.add_argument('--ontarget_column', help="Column containing ontarget sequences.")
        parser.add_argument('--offtarget_column', help="Column containing offtarget sequences.")
        parser.add_argument('--delimiter', default=",", help="file delimiter.")
        args = parser.parse_args()
        # print "\n", "ARGS", args, "\n"
        return cls.execute_file(**vars(args))

if __name__ == "__main__":
    # this will cache files and models, in a directory specified settings.py, namely: tmpdir = pj(repo_root, "tmp")
    
    sys.stdout = sys.__stdout__
    # initialize predictor
    p = Predict(learn_options_override={'num_proc': 15}) # udpated (new) Hauessler & GUIDE-seq
    # p = Predict(learn_options_override={'num_proc': 30, 'guideseq_version': 1}) # old version of those data, for comparison
    
    #p.hmg_data = p.get_hmg_data()
    #p.hmg_preds = p.get_preds_hmg()
    
    # it will have generated pickle files from everything (base model, guideseq, predicitons, etc.)

    p.haeussler_data = p.get_haeussler_data()
    p.haeussler_preds = p.get_preds_haeussler()
