from flask import Flask, request
from flask_restful import reqparse, Resource, Api
from json import dumps
import elevation.load_data
import elevation.prediction_pipeline
import numpy as np
import pandas
import time

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
parser.add_argument('wildtype', type=str, action='append')
parser.add_argument('offtarget', type=str, action='append')

base_model, guideseq_data, preds_guideseq, learn_options, prob_calibration_model = elevation.prediction_pipeline.train_final_model()


class ElevationPrediction(Resource):
    def post(self):
        start = time.time()
        args = parser.parse_args()
        wt = args['wildtype']
        mut = args['offtarget']
        df = pandas.DataFrame(columns=['30mer', '30mer_mut', 'Annotation'], index=range(len(wt)))
        df['30mer'] = wt
        df['30mer_mut'] = mut
        annot = []
        for i in range(len(wt)):
            annot.append(elevation.load_data.annot_from_seqs(wt[i], mut[i]))
        df['Annotation'] = annot
        print "Time spent parsing input: ", time.time() - start

        base_model_time = time.time()
        nb_pred, individual_mut_pred = elevation.prediction_pipeline.predict(base_model, df, learn_options)
        print "Time spent in base model predict(): ", time.time() - base_model_time

        stacker_time = time.time()
        pred = elevation.prediction_pipeline.stacked_predictions(df, individual_mut_pred, models=['linear-raw-stacker'],
                                                                 guideseq_data=guideseq_data, preds_guideseq=preds_guideseq,
                                                                 use_mut_distances=False,
                                                                 prob_calibration_model=prob_calibration_model,
                                                                 learn_options=learn_options)['linear-raw-stacker']
        end = time.time()
        print "Time spent in stacker predict(): ", end - stacker_time
        print "Total time: ", end-start

        return {'elevation score': pred.tolist(), 'annotation': annot}

api.add_resource(ElevationPrediction, '/elevation')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
