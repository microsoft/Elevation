# !IMPORTANT: Do not import anything that depends on prediction_pipeline here.

import os, shutil
import sys
import time
import numpy as np

from command_base import Command
from elevation import settings


class Fit(Command):

    def execute(self, crispr_repo_dir="", learn_options_override=None, force_rebuild=False):
        if crispr_repo_dir:
            if not os.path.exists(crispr_repo_dir):
                print("%s not found. Exiting." % crispr_repo_dir)
                sys.exit(1)
            settings.update_CRISPR(crispr_repo_dir)

        if force_rebuild:
            shutil.rmtree(settings.tmpdir)
            os.mkdir(settings.tmpdir)
            shutil.rmtree(settings.cachedir)
            os.mkdir(settings.cachedir)
        else:
            user_input = raw_input('Existing models may need to be removed. Delete contents of '
                                   '%s and %s? (yes/no):' % (settings.tmpdir, settings.cachedir))

            if user_input != "yes" and user_input != "no":
                print("unrecognized input %s. Exiting." % user_input)
                return
            if user_input == "yes":
                shutil.rmtree(settings.tmpdir)
                os.mkdir(settings.tmpdir)
                shutil.rmtree(settings.cachedir)
                os.mkdir(settings.cachedir)

        import elevation.prediction_pipeline
        reload(elevation.prediction_pipeline)
        from predict import Predict
        # from fixtures import Fixtures

        print("Forcing re-computation of models using CRISPR repo %s" % settings.CRISPR_dir)
        p = Predict(init_models=False, learn_options_override=learn_options_override)
        p.base_model = p.get_base_model(True)
        p.guideseq_data = p.get_guideseq_data(True)
        p.preds_guideseq = p.get_preds_guideseq(True)
        p.cd33_data = p.get_cd33(True)
        p.calibration_models = p.get_calibrated(True)
        # print("Regenerating Fixtures.")
        # Fixtures().execute()
        print("Models have been computed and saved. Please run py.test tests")

    @classmethod
    def cli_execute(cls):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--crispr_repo_dir', default="", help="Specify path to CRISPR repository.")
        args = parser.parse_args()
        # print "\n", "ARGS", args, "\n"
        return cls().execute(**vars(args))

if __name__ == "__main__":
    sys.stdout = sys.__stdout__
