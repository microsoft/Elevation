import os
import elevation
import logging
import socket
# from tempfile import mkdtemp

pj = lambda *paths: os.path.abspath(os.path.join(*paths))

# root project directories
package_root = os.path.abspath(os.path.dirname(elevation.__file__))
repo_root = pj(package_root, "../")

# cache dir used by joblib
default_cachedir = pj(repo_root, "cache")
cachedir = default_cachedir
if not os.path.exists(cachedir):
    logging.info("creating cache dir: %s " % cachedir)
    os.mkdir(cachedir)
else:
    pass

# dependencies for recomputing pickle files
def update_CRISPR(new_dir):
    global CRISPR_dir
    global seq_dir_template
    global offtarget_data_dir
    CRISPR_dir = new_dir
    seq_dir_template = pj(CRISPR_dir, 'gene_sequences/{gene_name}_sequence.txt')
    offtarget_data_dir = pj(CRISPR_dir, 'data/offtarget')

CRISPR_dir = pj(repo_root, "..")
seq_dir_template = pj(CRISPR_dir, 'gene_sequences/{gene_name}_sequence.txt')
offtarget_data_dir = pj(CRISPR_dir, 'data/offtarget')
if not os.path.exists(offtarget_data_dir):
    update_CRISPR(pj(repo_root, "../CRISPR"))
if not os.path.exists(offtarget_data_dir):
    logging.warning("Unable to locate CRISPR repository.")

# temp dir for pickle files
tmpdir = pj(repo_root, "tmp")
base_model_file = pj(tmpdir, 'base_model.pkl')

guideseq_data = pj(tmpdir, 'guideseq_data.pkl')
hauessler_data = pj(tmpdir, 'hauessler_data.pkl')
hmg_data = pj(tmpdir, 'hmg_data.pkl')

gspred_filename = pj(tmpdir, 'gspred.pkl')
hpred_filename = pj(tmpdir, 'hpred.pkl')
hmgpred_filename = pj(tmpdir, 'hmgpred.pkl')

guideseq_all_zeros_pred_file = pj(tmpdir, 'guideseq_all_zeros_pred.pkl')
trainhu_gs_file = pj(tmpdir, 'trainhu_gs.pkl')
calibration_file = pj(tmpdir, 'calibration_models.pkl')
cd33_file = pj(tmpdir, 'cd33.pkl')
base_preds_dir = pj(tmpdir, 'base_preds')

if not os.path.exists(tmpdir):
    os.mkdir(tmpdir)

# aggregation model
agg_model_file = pj(repo_root, 'elevation/saved_models/aggregation_model.pkl')
agg_test_max_batch_size = 2000
agg_default_fixture_file = pj(repo_root, 'tests/fixtures/agg_default.xls')
agg_nicolo_fixture_file = pj(repo_root, 'tests/fixtures/aggregation_test.pkl')

# guide seq model
pred_default_fixture_file = pj(repo_root, 'tests/fixtures/pred_default.xls')
pred_test_num = 1000

default_random_seed = 12345
