import numpy as np
import settings

learn_options = {
    # 'num_proc': 1,
    'num_proc': 28,
    'nuc_features_WT': False, 'include_pi_nuc_feat': False,
    'annotation position one-hot': False,
    'mutation_type' : False,
    'annotation_onehot' : True, # featurize like CFD
    'annotation_decoupled_onehot' : ['pos', 'let', 'transl'], # decouple the CFD features into letters and position
    "include_Tm": False,
    'include_azimuth_score': None, # all of them ["WT","MUT","DELTA"]
    'azimuth_feat' : None, # was: ['WT']
    "include_gene_position": False,
    "cv": "stratified",
    'adaboost_loss' : 'ls',
    'adaboost_CV': False, "algorithm_hyperparam_search" : "grid",
    'n_folds' : 10,
    'allowed_category' : None,#"Mismatch",#"Insertion",
    "include_NGGX_interaction": False,
    'normalize_features' : False, 'class_weight': None,
    "phen_transform": 'kde_cdf', #  'kde_cdf',
    "training_metric": 'spearmanr',
    "fit_intercept" : True,
    "seed": settings.default_random_seed,
    "alpha": np.array([1.0e-3]),
    "V": "CD33",
    "left_right_guide_ind": [4,27,30], # 21-mer    
    "order": 1,
    "testing_non_binary_target_name": 'ranks',
    'models': ["AdaBoost"],
    'post-process Platt': False,
    'azimuth_score_in_stacker': False,
    'guide_seq_full': True,
    'use_mut_distances': False,
    'reload guideseq': False,
    "renormalize_guideseq": True, # 6/30/2017 - per Jennifer, changed back to True after latest experiments
    'haeussler_version': 1,
    'guideseq_version': 2,
    #"kde_normalize_guideseq": False,
}
