import numpy as np
import elevation.model_comparison
import os
import pandas
import multiprocessing
cur_dir = os.path.dirname(os.path.abspath(__file__))

class CFDModel(object):
    def __init__(self, cfd_table=None, cfd_table_file=None):
        
        if cfd_table is None:
            #print "Loading CFD table from file"
            self.cfd_table = elevation.model_comparison.get_NBT_cfd(cfd_table_file)
        else:
            self.cfd_table = cfd_table
        self.cfd_table.index = self.cfd_table['Mismatch Type']

    def fit(self):
        pass

    def predict(self, annots_list, num_proc=20):
        if len(annots_list) == 0:
            preds = 1.0

        preds = np.ones(len(annots_list))

        if num_proc > 1:
            pool = multiprocessing.Pool(processes=num_proc)
            jobs = []

            for i, annots in enumerate(annots_list):
                jobs.append(pool.apply_async(predict_annot, (annots, self.cfd_table)))

            pool.close()
            pool.join()

            for i, j in enumerate(jobs):
                pred = j.get()
                preds[i] = pred

            pool.terminate()
        else:

            for i, annots in enumerate(annots_list):
                preds[i] = predict_annot(annots, self.cfd_table)

        return preds

def predict_annot(annots, cfd_table):
    pred_i = 1.0
    for a in annots:
        letters, pos = elevation.model_comparison.parse_mismatch_annot(a)
        if pos=='':
            annot_new = letters # a PAM mutation
        else:
            letters = str(letters)
            annot_new = letters[0] + ":" + letters[1] + "," +  str(pos)
        if a == 'GG':
            tmp_pred = 1.0
        else:
            tmp_pred = cfd_table["Percent-Active"].loc[annot_new]
        # preds[i] = tmp_pred*preds[i]
        pred_i = pred_i * tmp_pred
    return pred_i
