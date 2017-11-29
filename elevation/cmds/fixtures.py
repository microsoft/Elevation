import sys

import pickle

import pandas as pd
import numpy as np

import elevation
import elevation.load_data
import elevation.util
from elevation.cmds.predict import Predict
from elevation import aggregation
from elevation import settings
from command_base import Command
from elevation import options


class Fixtures(Command):

    def get_hmg(self):
        hmg = elevation.load_data.load_hauessler_minus_guideseq(options.learn_options)
        wildtype = list(hmg['30mer'])
        offtarget = list(hmg['30mer_mut'])
        return wildtype, offtarget

    def generate_prediction_fixture(self):
        wildtype, offtarget = self.get_hmg()

        p = Predict()
        pred_calibrated = p.execute(wildtype, offtarget)
        pred_calibrated_writable = {key: val.reshape(val.shape[0]) for key, val in pred_calibrated.iteritems()}
        df = pd.DataFrame(data=pred_calibrated_writable)

        # write all for more extensive testing
        df.to_excel(elevation.settings.pred_default_fixture_file)

    def generate_aggregation_fixtures(self):
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
        count = 0
        total_count = 0
        while pos < perm.shape[0]:
            end = pos + rs.randint(1, 2000)
            if end > perm.shape[0]:
                end = perm.shape[0]
            # print pos, end, perm.shape[0]
            result = aggregation.get_aggregated_score(
                stacker[pos:end],
                cfd[pos:end],
                isgenic[pos:end],
                final_model)
            results += list(result)
            total_count += len(stacker[pos:end])
            count += 1
            pos = end
        assert count == len(results)
        assert total_count == len(stacker)
        df = pd.DataFrame(data={"agg_score": results})
        df.to_excel(settings.agg_default_fixture_file)

    def execute(self):
        self.generate_prediction_fixture()
        self.generate_aggregation_fixtures()


if __name__ == "__main__":
    sys.stdout = sys.__stdout__
    Fixtures().execute()

