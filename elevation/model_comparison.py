import azimuth
import azimuth.predict as pd
import azimuth.model_comparison as mc
import azimuth.local_multiprocessing
import azimuth.load_data
import azimuth.features.featurization as feat
import sklearn
import copy
import os
import numpy as np
import scipy.stats as ss
import azimuth.util as util
import azimuth.features.featurization as ft
import shutil
import pickle
import pylab as plt
import time
import pandas
import Bio.SeqUtils.MeltingTemp as Tm
import Bio.SeqUtils as SeqUtil
import Bio.Seq as Seq
import cStringIO
import sys
import elevation
import elevation.load_data
import itertools
from stacker import Stacker, StackerFeat
import multiprocessing

cur_dir = os.path.dirname(os.path.abspath(__file__))

def set_target_elevation(learn_options, classification):
    # assert 'target_name' not in learn_options.keys() or learn_options['target_name'] is not None, "changed it to be automatically set here"
    if 'target_name' not in learn_options.keys() or learn_options['target_name'] is None:
        learn_options["target_name"] = 'Day21-ETP'
        learn_options['ground_truth_label'] = 'Day21-ETP'

    if classification:
        assert learn_options['phen_transform']=="binarize", "need binarization for classification"

    return learn_options

def setup_elevation(test=False, order=1, learn_options=None, pam_audit=False, length_audit=False, subset_ind=None):
    # need these to be compatible with azimuth's setup
    assert pam_audit==False, "doesn't handle anything else"
    assert length_audit==False, "doesn't handle anything else"

    np.random.seed(learn_options['seed'])
    num_proc = mc.shared_setup(learn_options, order, test)

    if test:
        learn_options["order"] = 1

    # defaults needed for azimuth featurization/calls
    learn_options['ignore_gene_level_for_inner_loop'] = True#False

    if learn_options['V'] == 'CD33':
        data, Y, target_genes = elevation.load_data.load_cd33(learn_options)        
    elif learn_options['V'] == 'guideseq':
        data, Y, target_genes = elevation.load_data.load_guideseq(learn_options)
    elif learn_options['V'] == 'HsuZhang':
        data, Y, target_genes = elevation.load_data.load_HsuZang_data(version="hsu-zhang-single")
    elif learn_options['V'] == 'CD33+HsuZhang':
        data, Y, target_genes = elevation.load_data.load_cd33_plus_hsuzhangsingle()
    else:
        raise Exception("invalid 'V' in learn options, V=%s" % learn_options['V'])

    # subset ind is used in cross validation when we want a subset of the data
    if 'training_indices' in learn_options.keys() and learn_options['training_indices'] is not None:
        training_indices = learn_options['training_indices']
        data = data.iloc[training_indices]
        Y = Y.iloc[training_indices]

    print "featurizing data..."
    feature_sets, _garb = featurize_data_elevation(data, learn_options)
    print "done."

    return Y, feature_sets, target_genes, learn_options, num_proc


def Tm_feature(data):
    raise Exception("need to revert back to Tm of WT and mutated, but this didn't work anyhow, so leaving like this")
    # http://biopython.org/DIST/docs/api/Bio.SeqUtils.MeltingTemp-module.html#Tm_NN
    # as far as I can tell, Tm_NN only accepts a single mismatch and dangling ends, so
    # cannot use for deletions/insertions
    feat = None
    mutseq = data['MutatedSequence'].values
    wtseq = data['WTSequence']
    featarray = np.ones((mutseq.shape[0],1))
    for i, seq in enumerate(mutseq):
        c_seq = Seq.Seq(wtseq[i])
        c_seq_comp = str(c_seq.complement())
        # postive shift moves seq up relative to c_seq
        if len(c_seq)==len(seq):
            newTm = Tm.Tm_NN(seq, c_seq=c_seq_comp, shift=0)
        elif len(c_seq)==len(seq)+1:
            # deletion
            newTm = Tm.Tm_NN(seq, c_seq=c_seq_comp, shift=0)
        elif len(seq)==len(c_seq) + 1:
            # insertion
            newTm = Tm.Tm_NN(seq, c_seq=c_seq_comp, shift=0)
        else:
            raise Exception()
        featarray[i,0] = newTm
    feat = pandas.DataFrame(featarray, index=data.index, columns=["Tm %s" % seqtype])
    return feat

def parse_mismatch_annot(x):
    if ":" in x:
        tmp_pos = int(x.split(',')[1])
        letters = x.split(',')[0].split(':')
        assert len(letters)==2, "expecting two letters here"
        if letters[0]=='I' or letters[0]=='D':
            letters[0] = 'X'
        letters = letters[0] + letters[1]
    else:
        letters = x;
        tmp_pos = ""
    return letters, tmp_pos


def get_unique_annot(decoupled=False):
    # this gets saved in elevation/load_data.py: read_doench()
    # unique_annot = pickle.load(open(r"D:\Source\CRISPR\elevation\elevation\saved_models\unique_annot.p", "rb" ))
    all_2_nucl_sub = list(itertools.permutations(['A', 'T', 'C', 'G'], 2))
    unique_annot = list(itertools.product(all_2_nucl_sub, range(1, 21)))
    unique_annot.extend(list(itertools.product('ATCG', repeat=2)))
    unique_annot = ['%s:%s,%d' % (u[0][0],u[0][1],u[1]) if len(u[0]) ==2 else '%s%s' % (u[0], u[1]) for u in unique_annot]


    if decoupled:
        letters, positions = [], []
        for a in unique_annot:
            if ":" in a:
                let, pos = a.split(",")
                letters.append(let)
                positions.append(int(pos))
            else:
                positions.append(20) # WARNING assuming 20 mer
                letters.append(a)
                unique_annot = (np.unique(letters).tolist(), positions)
        str_to_int =  None
    else:
        str_to_int = {}
        for i, str in enumerate(unique_annot):
            str_to_int[str] = i

    return unique_annot, str_to_int

def one_hot_annotation_position(data, feature_sets):
    """
    seperate out position and letter changes, but don't one-hot encode the position (preserves smoothness)
    """
    unique_annot, str_to_int = get_unique_annot(decoupled=True)
    letters, positions = unique_annot
    pos_matrix = np.zeros((data.shape[0], len(np.unique(positions))+1)) # +1 is because of pam position
    pam_pos = len(np.unique(positions))+1
    for n in range(data.shape[0]):
        annots_tmp = data['Annotation'].values[n]
        if not isinstance(annots_tmp, list):
            #trivially make it a single-item list
            annots_tmp = [annots_tmp]
        for a in annots_tmp:
            if ":" in a:
                let, pos = a.split(",")
            else: #it's a PAM
                pos = pam_pos

            pos_matrix[n, int(pos)-1] = 1.0

    feature_sets['annot_pos'] = pandas.DataFrame(pos_matrix, columns=['pos_%d' % (i+1) for i in range(len(np.unique(positions))+1)], index=data.index)

def one_hot_annotation_letters_decoupled(data, feature_sets):
    """
    get two letters from "A:G" or "CG" features in one-hot-encoding
    """
    alph = feat.get_alphabet(order=1, raw_alphabet = ['A', 'T', 'C', 'G'])
    annot = np.zeros((data.shape[0], len(alph)*2))

    for n in range(data.shape[0]):
        annots_tmp = data['Annotation'].values[n]
        if not isinstance(annots_tmp, list):
            annots_tmp = [annots_tmp]
        for a in annots_tmp:
            if ":" in a: #mismatch
                let, pos = a.split(",")
                let0 = let[0]
                let1 = let[2]
            else: #PAM
                let0 = a[0]
                let1 = a[1]
            annot[n, alph.index(let0)] = 1.0
            annot[n, alph.index(let1) + len(alph)] = 1.0

    col_headers = [let + "0" for let in alph] + [let + "1" for let in alph]
    feature_sets['annot_decoupled'] = pandas.DataFrame(annot, columns=col_headers, index=data.index)


def one_hot_annotation_decoupled(data, feature_sets, option_vals, pam_pos=21):
    """
    seperate out position and letter changes, but don't one-hot encode the position (preserves smoothness)
    """
    unique_annot, str_to_int = get_unique_annot(decoupled=True)
    letters, positions = unique_annot
    annot = np.zeros((data.shape[0], len(letters)+1))
    transl_transv = np.zeros((data.shape[0], 3)) # wobble transition, non-wobble transition, transversion
    for n in range(data.shape[0]):
        annots_tmp = data['Annotation'].values[n]
        if not isinstance(annots_tmp, list):
            #trivially make it a single-item list
            annots_tmp = [annots_tmp]
        for a in annots_tmp:
            if ":" in a:
                let, pos = a.split(",")
                annot[n, letters.index(let)+1] = 1.0
                annot[n, 0] = pos
                let_to, let_from = let.split(":")
                if (let_from == 'G' and let_to == 'A') or (let_from == 'T' and let_to == 'C'):
                    transl_transv[n, 0] = 1.0
                elif (let_from == 'A' and let_to == 'G') or (let_from == 'C' and let_to == 'T'):
                    transl_transv[n, 1] = 1.0
                else:
                    transl_transv[n, 2] = 1.0
            else: #it's a PAM
                annot[n, letters.index(a)+1] = 1.0
                annot[n, 0] = pam_pos

    if "let" in option_vals and "pos" in option_vals:
        feature_sets['annot_decoupled'] = pandas.DataFrame(annot, columns=['pos'] + letters, index=data.index)
    elif "let" in option_vals:
        feature_sets['annot_decoupled'] = pandas.DataFrame(annot[:, 1:annot.shape[1]], columns=letters, index=data.index)
    elif "pos" in option_vals:
        feature_sets['annot_decoupled'] = pandas.DataFrame(annot[:,0], columns=['pos'], index=data.index)

    if 'transl' in option_vals:
        transl = pandas.DataFrame(transl_transv, columns=['wobble_trans', 'nonwobble_trans', 'translation'], index=data.index)
        feature_sets['annot_decoupled'] = pandas.concat((feature_sets['annot_decoupled'], transl), axis=1)

def one_hot_annotation(data, feature_sets):
    """
    use the same features as in CFD, namely one for every mutation type and position
    """
    # this gets saved in elevation/load_data.py: read_doench()
    # don't want to generate it on the fly because test data may not have all the features
    unique_annot, str_to_int = get_unique_annot(decoupled=False)
    annot_onehot = np.zeros((data.shape[0], len(unique_annot)))
    for n in range(data.shape[0]):
        annots_tmp = data['Annotation'].values[n]
        if isinstance(annots_tmp, list):
            for a in annots_tmp:
                annot_onehot[n, str_to_int[a]] = 1.0
        else:
            annot_onehot[n, str_to_int[annots_tmp]] = 1.0

    # recreate CFD scores from here as a sanity check:
    if False:
        M = len(str_to_int)
        counts = np.zeros((M,3)) # 0-CFD freq, 1-total, 2-active
        counts[:,1] = np.sum(annot_onehot, axis=0)
        counts[:,2] = np.sum(annot_onehot[np.where(data['Day21-ETP'])[0]], axis=0)
        counts[:,0] = np.divide(counts[:,2], counts[:,1])
        ordered_col_names = np.empty(M, dtype='S10')
        for key in str_to_int.keys(): ordered_col_names[str_to_int[key]] = key
        df = pandas.DataFrame(counts.T, columns=ordered_col_names).T
        df.columns=['CFD','#total', '#active']
        df.to_csv(r"D:\Source\CRISPR\elevation\elevation\saved_models\our_cfd.csv")
        import ipdb; ipdb.set_trace()

    feature_sets['annot'] = pandas.DataFrame(annot_onehot, columns=unique_annot, index=data.index)

def get_mutation_details(data, feature_sets, skip_pam_feat=False, letpos_inter_ft=False, letpos_indep_ft=True):
    """
        grab the annotation itself, plus components of it.
        The annotations look like one of these:
        G:A, 5 (G on WT and A on MutSeq)
        I:C, 3 (insert C in position 3)
        D:C, 10 (delete C in position 10)
        AGG (the actual PAM)
        """

    N = data.shape[0]

    # note there seem to be about 65 unique PAMs, using X when don't know the N in NGG
    full_annot = data['Annotation'].values

    pam = []
    let = []
    pos = np.zeros_like(full_annot)

    for i, x in enumerate(full_annot):
        # (order here is important!!!)
        x = str(x)
        mut, mut_pos = x.split(",")
        if ':' in mut: # insertion/deletion/change
            letters, pos[i] = parse_mismatch_annot(x)
            let.append(letters)
            pam.append('NGG') # placeholder
        elif len(mut)==3: #"e.g. "CGG,-1"
            pam.append(mut)
            pos[i] = int(mut_pos)
            let.append('XX') # for insertions/deletions, we will just repeat the letter
        else:
            raise Exception("unexpected annotation found: %s" % x)

    # now nucleotide one-hot encode the pam, and the letters, up to max order
    raw_alphabet = ['A', 'T', 'C', 'G', 'X']  # add an X for missing

    if not skip_pam_feat:
        data["PAM"] = pam
        max_index_to_use = 3; order = 3; prefix = "PAM"
        pam_onehot = data["PAM"].apply(feat.nucleotide_features, args=(order, max_index_to_use, prefix, 'pos_dependent', raw_alphabet))
        feature_sets["PAM"] = pam_onehot
        assert feature_sets["PAM"].shape[0]==N

    order = 2

    if letpos_indep_ft:
        # this lists the mutation letters (e.g. AG) and the position, SEPERATELY AS FEATURES (e.g. some are just "AG", and some are "2")
        data["mut_let"] = let
        max_index_to_use = 2; prefix = "mut_let"
        let_df = data["mut_let"].apply(feat.nucleotide_features, args=(order, max_index_to_use, prefix, 'pos_dependent', raw_alphabet))
        let_df["mut_pos"] = np.array(pos, dtype=float)
        feature_sets["mutation_let_pos"] = let_df
        assert feature_sets["mutation_let_pos"].shape[0]==N
        raise Exception("need to add in PAM and maybe indel here")

    # this lists the mutation letters and the position TOGETHER as features as in CFD (e.g. "AG2")
    if letpos_inter_ft:
        alphabet = feat.get_alphabet(order, raw_alphabet = raw_alphabet)
        # this already contains only mutations because we don't have "AA" and "CC" ,etc.
        positions = [str(i+1) for i in range(len(data['30mer'].iloc[0]))]
        letter_pos_pairs = list(itertools.product(alphabet, positions))
        feature_keys = [pr[0] + pr[1] for pr in letter_pos_pairs]
        feature_dict = {}
        for j, val in enumerate(feature_keys):
            feature_dict[val] = j
        let_pos = [let[i] + str(pos[i]) for i in range(N)]
        features_poslet = np.zeros((N, len(feature_keys)))
        for n, ft in enumerate(let_pos):
            features_poslet[n, feature_dict[ft]] = 1
        # remove features that have only zero counts--NO CANNOT DO THIS AS WON'T KNOW WHEN FEATURIZE NEW DATA
        features_poslet_df = pandas.DataFrame(features_poslet, index=data.index, columns=np.array(feature_keys))
        feature_sets["mutletpos"] = features_poslet_df

def compute_mut_azimuth(data, model_file):
    # note some of this is done for only mismatches inside of get_thirtymers_for_CD33
    thirtytwomer_arr = np.array(data["32mer"].apply(str))
    annotations = data["Annotation"]
    azimuth_mut_score = []
    num_mis = 0
    num_del = 0
    num_ins = 0
    num_pam = 0
    for i, seq in enumerate(thirtytwomer_arr):
        mutseq = data["MutatedSequence"][i]
        try:
            pos =  int(annotations[i].split(',')[1]) #irrelevant if a PAM and will fail
            pos = pos-1 #because they were using 1-based system, not 0
            letters = annotations[i].split(',')[0].split(':')
        except:
            assert len(annotations[i])==3, "should be a pam and is not"
            pos = "PAM"

        if pos=="PAM":
            num_pam += 1
            ind = seq.find(mutseq)
            assert ind==5 #because assuming this later on for non-PAM
            assert np.all(seq[ind:(ind+20)]==mutseq)
            mutseq = seq[1:31]
            # stdout stuff is to suppress output from azimuth predict
            save_stdout = sys.stdout; sys.stdout = cStringIO.StringIO()
            tmp_score = azimuth.model_comparison.predict(np.array([mutseq]), None, None, pam_audit=False, model_file=model_file_azimuth)
            sys.stdout = save_stdout
        elif len(mutseq)==20: #and pos!="PAM":  # mismatch
            num_mis += 1
            ind = 5
            assert seq[ind+pos]==letters[0], "expecting a different letter here"
            assert mutseq[pos]==letters[1], "expecting a different letter here"
            newseq = seq[0:ind+pos] + letters[1] + seq[ind+pos+1:]   #seq[ind+pos] = letters[1]
            ind = newseq.find(mutseq)
            assert ind==5
            mutseq = newseq[1:31]
            save_stdout = sys.stdout; sys.stdout = cStringIO.StringIO()
            tmp_score = azimuth.model_comparison.predict(np.array([mutseq]), None, None, model_file=model_file_azimuth)
            sys.stdout = save_stdout
        elif len(mutseq)==19: # deletion
            num_del += 1
            # delete each of two ways, keeping a 30mer, and average
            assert seq[ind+pos]==letters[1]
            newseq = seq[0: ind+pos] + seq[ind+pos+1:]
            assert len(newseq)==31
            mutseq1 = newseq[0:30]
            mutseq2 = newseq[1:31]
            save_stdout = sys.stdout; sys.stdout = cStringIO.StringIO()
            tmp_score1 = azimuth.model_comparison.predict(np.array([mutseq1]), None, None, pam_audit=True, model_file=model_file_azimuth)
            tmp_score2 = azimuth.model_comparison.predict(np.array([mutseq2]), None, None, pam_audit=False, model_file=model_file_azimuth)
            sys.stdout = save_stdout
            tmp_score = [tmp_score1, tmp_score2]
        elif len(mutseq)==21: #insertion
            num_ins += 1
            # fill in the missing item each of two ways to keep a 30mer, and average
            assert mutseq[pos]==letters[1]
            newseq = seq[0:ind+pos] + letters[1] + seq[ind+pos:]
            assert len(newseq)==33
            mutseq1 = newseq[1:31]
            mutseq2 = newseq[2:32]
            save_stdout = sys.stdout; sys.stdout = cStringIO.StringIO()
            tmp_score1 = azimuth.model_comparison.predict(np.array([mutseq1]), None, None, pam_audit=False, model_file=model_file_azimuth)
            tmp_score2 = azimuth.model_comparison.predict(np.array([mutseq2]), None, None, pam_audit=True, model_file=model_file_azimuth)
            sys.stdout = save_stdout
            tmp_score = [tmp_score1, tmp_score2]
        else:
            raise Exception("shoudn't be here")
        azimuth_mut_score.append(np.mean(tmp_score))
        if (i % 100 ==0): print "done %d of %d" % (i, len(thirtytwomer_arr))
    return azimuth_mut_score

def get_on_target_predictions(data, score_list):

    for nm in score_list:
        assert nm in ["MUT","WT","DELTA"], 'allowed types are only: ["MUT","WT","DELTA"]'

    azimuth_score_arr = None
    azimuth_score_mut_arr = None
    azimuth_delta_score_arr = None

    wt_file = cur_dir + "/saved_models/azimuth_score_arr.p"
    mut_file = cur_dir + "/saved_models/azimuth_score_mut_arr.p"
    model_file_azimuth = cur_dir + "/saved_models/Azimuth_23mer.p"

    t0 = time.time()
    # for wild type guides
    if "WT" in score_list or "DELTA" in score_list:
        if False:#os.path.isfile(wt_file):
            #print "loading up azimuth_score_arr from pickle"
            azimuth_score_df = pickle.load(open(wt_file, "rb"))
            azimuth_score_arr = np.array(azimuth_score_df.ix[data.index].values)
        else:
            #print "computing up azimuth_score_arr"
            azimuth_score_arr = azimuth.model_comparison.predict(np.array(data["30mer"].apply(str)), None, None, pam_audit=False, model_file=model_file_azimuth)
            azimuth_score_df = pandas.DataFrame(azimuth_score_arr)
            azimuth_score_df.index = data.index
            azimuth_score_df.columns = ["azimuth_score"]
            pickle.dump(azimuth_score_df,open(wt_file, "wb" ))
        t1 = time.time()
        # print "\t\tElapsed time for perfect match azimuth is %.2f seconds" % (t1-t0)

    # for mutated guides (in ad hoc manner)
    if "MUT" in score_list or "DELTA" in score_list:
        t0 = time.time()
        if os.path.isfile(mut_file):
            #print "loading up azimuth_score_mut_arr from pickle"
            azimuth_score_mut_df = pickle.load(open(mut_file, "rb"));
            azimuth_score_mut_arr = np.array(azimuth_score_mut_df.ix[data.index].values)
        else:
            #print "computing up azimuth_score_mut_arr"
            azimuth_score_mut = compute_mut_azimuth(data, model_file=model_file_azimuth)
            azimuth_score_mut_arr = np.array(azimuth_score_mut)
            azimuth_score_mut_df = pandas.DataFrame(azimuth_score_mut_arr)
            azimuth_score_mut_df.index = data.index
            azimuth_score_mut_df.columns = ["azimuth_score_mut"]
            pickle.dump(azimuth_score_mut_df,open(mut_file, "wb" ))
        t1 = time.time()
        # print "\t\tElapsed time for prediciton features is %.2f seconds" % (t1-t0)

    if "DELTA" in score_list:
        azimuth_delta_score_arr = azimuth_score_arr - azimuth_score_mut_arr
    return azimuth_score_arr, azimuth_score_mut_arr, azimuth_delta_score_arr

def azimuth_featurize(data, learn_options):

    learn_options_to_use = learn_options.copy()

    N, M = data.shape
    # print "found %d rows and %d feature colums to azimut_featurize" % (N, M)
    # ignore gene position
    gene_position = pandas.DataFrame(columns=[u'Percent Peptide', u'Amino Acid Cut position'], data=zip(np.ones(N)*-1, np.ones(N)*-1))

    # load up learn_options needed (grab those we don't yet have)
    model_name = 'V3_model_nopos.pickle'
    azimuth_saved_model_dir = os.path.join(os.path.dirname(azimuth.__file__), 'saved_models')
    model_file = os.path.join(azimuth_saved_model_dir, model_name)
    with open(model_file, 'rb') as f:
        modelgarb, learn_options_az = pickle.load(f)

    for option in learn_options_az.keys():
        if not learn_options_to_use.has_key(option):
            learn_options_to_use[option] = learn_options_az[option]
    learn_options_to_use = mc.fill_learn_options(learn_options_az, learn_options_to_use)

    if learn_options_to_use.has_key("azimuth_gc_features") and learn_options_to_use["azimuth_gc_features"] is not None:
       if learn_options_to_use["azimuth_gc_features"]:
           learn_options_to_use["gc_features"] = True
       else:
            learn_options_to_use["gc_features"] = False

    #if not using long enough sequence, cannot use NGGX interaction feature
    if learn_options_to_use.has_key('include_NGGX_interaction') and learn_options_to_use['include_NGGX_interaction']:
        raise Exception("need to check that this works properly with cd33 and guideseq data")
    if learn_options_to_use.has_key('include_NGGX_interaction') and learn_options_to_use['include_NGGX_interaction'] and learn_options_to_use.has_key('left_right_guide_ind') and learn_options_to_use['left_right_guide_ind'] is not None:
        pos_range = range(learn_options_to_use['left_right_guide_ind'][0],learn_options_to_use['left_right_guide_ind'][1])
        cando_NGGX_features = (27 in pos_range) and (24 in pos_range)
        if learn_options_to_use['include_NGGX_interaction'] and not cando_NGGX_features:
            print "turning off NGGX feature because don't have those nucleotides"
            learn_options_to_use['include_NGGX_interaction'] = False

    check_seq_len(data)

    feature_sets_azimuth = feat.featurize_data(data, learn_options_to_use, pandas.DataFrame(), gene_position, pam_audit=False, length_audit=False, quiet=True)

    return feature_sets_azimuth

def check_seq_len(data, colname='30mer', expected_len=None):
    all_lens = data[colname].apply(len).values
    unique_lengths = np.unique(all_lens)
    num_lengths = len(unique_lengths)
    assert num_lengths == 1, "should only have sequences of a single length, but found %s: %s, probably owing to selecting only some categories of mutations" % (num_lengths, str(unique_lengths))
    if expected_len is not None:
        assert expected_len == unique_lengths[0], "did not find expected length"

def featurize_data_elevation(data, learn_options, verbose=False):
    '''
    assumes that data contains the 30mer
    returns set of features from which one can make a kernel for each one
    '''

    # need in feature_data_elevation and setup
    if "allowed_category" in learn_options and (learn_options["allowed_category"] is not None):
        keep_ind = (data['Category']==learn_options["allowed_category"])
        data = data[keep_ind]

    check_seq_len(data)

    if verbose:
        print "Constructing features for elevation..."
    t0 = time.time()

    feature_sets = {}

    if learn_options.has_key('azimuth_feat') and learn_options['azimuth_feat'] is not None:
        if "WT" in learn_options['azimuth_feat']:
            assert not learn_options["nuc_features_WT"], "should not use both nuc_features_WT and azimuth_feat_WT' as they are redundant"
            feature_sets_azimuth = azimuth_featurize(data, learn_options)
            for set in feature_sets_azimuth.keys():
                feature_sets[set + "_WT"] = feature_sets_azimuth[set]
        if "MUT" in learn_options['azimuth_feat']:
            data_tmp = data.copy()
            data_tmp['30mer'] = data_tmp["30mer_mut"]
            data_tmp['30mer_mut'] = None
            feature_sets_azimuth = azimuth_featurize(data_tmp, learn_options)
            for set in feature_sets_azimuth.keys():
                feature_sets[set + "_MUT"] = feature_sets_azimuth[set]

    if learn_options.has_key('include_TM') and learn_options["include_Tm"]:
        feature_sets["Tm"] = Tm_feature(data)

    if learn_options.has_key('include_azimuth_score') and learn_options['include_azimuth_score'] is not None:
        # this is very expensive to compute from scratch (an hour or more on one CPU)
        azimuth_score, azimuth_mut_score, azimuth_delta_score = get_on_target_predictions(data,  score_list=learn_options['include_azimuth_score'])
        tmp_feature_sets = []
        if "WT" in learn_options['include_azimuth_score']:
            assert azimuth_score is not None
            data['az_score'] = azimuth_score
            tmp_feature_sets.append("az_score")
        if "MUT" in learn_options['include_azimuth_score']:
            assert azimuth_mut_score is not None
            data['az_mut_score'] = azimuth_mut_score
            tmp_feature_sets.append("az_mut_score")
        if "DELTA" in learn_options['include_azimuth_score']:
            assert azimuth_delta_score is not None
            data['az_delta_score'] = azimuth_delta_score
            tmp_feature_sets.append("az_delta_score")
        feature_sets['azimuth'] = data[tmp_feature_sets]

    if learn_options.has_key('annotation_onehot') and learn_options['annotation_onehot']:
        one_hot_annotation(data, feature_sets)

    if learn_options.has_key('annotation_decoupled_onehot') and learn_options['annotation_decoupled_onehot']:
        one_hot_annotation_decoupled(data, feature_sets, learn_options['annotation_decoupled_onehot'])

    if learn_options.has_key('annotation position one-hot') and learn_options['annotation position one-hot']:
        one_hot_annotation_position(data, feature_sets)

    if learn_options.has_key('annotation_letters_decoupled_onehot') and learn_options['annotation_letters_decoupled_onehot']:
        one_hot_annotation_letters_decoupled(data, feature_sets)

    if learn_options.has_key('mutation_details') and learn_options['mutation_details']:
        # TODO: change this to use the actual N in NGG
        get_mutation_details(data, feature_sets, skip_pam_feat=learn_options["skip_pam_feat"], letpos_inter_ft=learn_options["letpos_inter_ft"], letpos_indep_ft=learn_options["letpos_indep_ft"])

    if learn_options.has_key('nuc_features_WT') and learn_options["nuc_features_WT"]:
        # TODO: change this to use the actual 30mer
        # spectrum kernels (position-independent) and weighted degree kernels (position-dependent)
        feat.get_all_order_nuc_features(data['WTSequence'], feature_sets, learn_options, learn_options["order"], max_index_to_use=20)

    if learn_options.has_key('nuc_features_MUT') and learn_options["nuc_features_MUT"]:
        raise Exception("not sure this makes sense--just encode the actual mutation as annotated")
        # spectrum kernels (position-independent) and weighted degree kernels (position-dependent)
        feat.get_all_order_nuc_features(data['MutatedSequence'], feature_sets, learn_options, learn_options["order"], max_index_to_use=20)

    if learn_options.has_key('mutation_type') and learn_options['mutation_type']:
        enc = sklearn.preprocessing.OneHotEncoder()
        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_encoder.fit(np.unique(data['Category'].values)) # [u'Deletion', u'Insertion', u'Mismatch', u'PAM']
        one_hot_cat = np.array(enc.fit_transform(label_encoder.transform(data['Category'].values)[:, None]).todense())
        feature_sets['category'] = pandas.DataFrame(one_hot_cat)

    # not sure what this is, but don't think it's being used
    if learn_options["include_gene_position"]:
        feature_sets["Percent Peptide"] = pandas.DataFrame(data['Protein Annotation'])

    ft.check_feature_set(feature_sets)

    return feature_sets, data


def predict_elevation(data=None, wt_seq30=np.array(['TGTCGTAGTAGGGTATGGGA', 'AAAGCAGCCAGGACAGCAGT','AAAGCAGCCAGGACAGCAGT','AAAGCAGCCAGGACAGCAGT']), mutation_details=np.array(['TGG', 'G:A,7', 'I:C,2', 'D:G,7']), category=None, model=None, model_file=None, pam_audit=False, learn_options_override=None, force_zero_intercept=False, naive_bayes_combine=True, verbose=False, parallel_block_size=10000):
    """
    if pam_audit==False, then it will not check for GG in the expected position
    this is useful if predicting on PAM mismatches, such as with off-target

    Can use 'data' frame containing the columns: 'WTSequence', 'Annotation' 'Category' (pandas object)

    OR all of these

    'wt_seq30' is the 30mer wild type sequence, as in our NBT paper for Azimuth/on-target
    'category' should be one of: Insertion, Deletion, Mismatch, PAM
    'mutation_details' should be:
        Mismatch e.g. 'G:A,7', which means position 7 of our 30mer was changed from a G to an A
        Deletion e.g. 'D:A,3' which means position 3 which had an A was deleted
        Insertion e.g. 'I:T,10' means that a T was insert at position 10
        PAM e.g. 'GGA' means the PAM site is GGA
        NB. position is 1-based
    """
        
    assert naive_bayes_combine==True, "only option right now, and have removed any ifs around it so won't even see it used"
    assert category is None, "haven't yet put in this param functionality"
    #np.array(['PAM','Mismatch', 'Insertion', 'Deletion'])
    
    if model_file is None:
        #azimuth_saved_model_dir = os.path.join(os.path.dirname(elevation.__file__), 'saved_models')
        azimuth_saved_model_dir = r'saved_models'
        model_name = 'final.2016-02-05_17_23_59.p'#'model_12.21.2015.GBRT.elevation.pickle'
        model_file = os.path.join(azimuth_saved_model_dir, model_name)

    if model is None:
        with open(model_file, 'rb') as f:
            model, learn_options, feature_names = pickle.load(f)
        if model_file is None:
            print "No model file specified, using default off-target model: %s", model_name
    else:
        model, learn_options = model

    if force_zero_intercept:
        try:
            model.intercept_=0.0
            print "forced model intercept to 0.0"
        except:
            pass

    learn_options = mc.override_learn_options(learn_options_override, learn_options)

    if data is None:
        data = pandas.DataFrame(columns=[u'WTSequence', u'Annotation', 'Category'], data=zip(wt_seq30, mutation_details, category))

    unique_annot, str_to_int = get_unique_annot(decoupled=True)
    letters, positions = unique_annot
        
    # DEBUG--feel free to delete this block
    #my_range = np.arange(0, 10)
    #pred0, nb_pred0, feature_names0 = score_offtarget(data.iloc[[0]].copy(), model, learn_options, positions, msg="regular_call_debug")
    #pred, nb_pred, feature_names = score_offtarget_many(data.iloc[my_range].copy(), model, learn_options, positions, msg="regular_call_debug")
    #import ipdb; ipdb.set_trace()
    
    print "predict_elevation allocating", learn_options["num_proc"], 'cores'
    N =  data.shape[0]             
    num_blocks = int(np.floor(N/parallel_block_size) + 1*(N % parallel_block_size > 0)) # mod is for left-over if doesn't divide evenly
    
    pool = multiprocessing.Pool(processes=learn_options["num_proc"])
    jobs = []    
    for iter in range(num_blocks):                                                
        jobs.append(pool.apply_async(score_offtarget_many, (data, model, learn_options, positions, iter, parallel_block_size, N)))        
    pool.close()
    pool.join()
    
    all_predictions = None
    all_predictions_ind = None
    feature_names = None
    
    for i, j in enumerate(jobs):                
        pred, nb_pred, feature_names = j.get()        
        if all_predictions is None:
            all_predictions = nb_pred
            all_predictions_ind = pred            
        else:
            all_predictions = np.concatenate((all_predictions, nb_pred), axis=0)
            all_predictions_ind = np.concatenate((all_predictions_ind, pred), axis=0)

    pool.terminate()       

    return all_predictions, model, learn_options, data, feature_names, np.array(all_predictions_ind.tolist())

# need this because if do this increments in the actual loop, the multiprocessing messes it up, and I 
# can't get the Lock() to work. It's not super efficient, but hopefully efficient enough, esp if the paralle_block_size is
# large enough
def iteration_to_range(this_iter, parallel_block_size, N):
    assert N >=0, "N must be >=0"
    assert parallel_block_size >= 1, "parallel_block_size must be >1 ="
    assert this_iter >= 0 and this_iter <=N, "iter must lie between 0 and N and in fact is more constrained than that"
    start_ind = 0
    iter = 0
    while start_ind < N:
        end_ind = np.min((start_ind + parallel_block_size, N))        
        my_range = np.arange(start_ind, end_ind)                        
        if this_iter == iter:
            return my_range, start_ind, end_ind
        start_ind = start_ind + parallel_block_size         
        iter += 1
    return None, None, None

# to do several
def score_offtarget_many(row_data_all, model, learn_options, positions, iter, parallel_block_size, N, verbose=True):
    
    #----------------------------
    # this used to be outside of here, but multiprocessing was messing it up, so seeing if this fixes it-
    my_range, start_ind, end_ind = iteration_to_range(iter, parallel_block_size, N)

    if verbose:
        print "start_range=%d, end_range=%d" % (start_ind, my_range[-1])
    if verbose and (iter % (10000/parallel_block_size)) == 0:
        msg_tmp = "predict_elevation: %0.2f perc. done (%d of %d using block_size=%d) " % (float(start_ind)/N*100, start_ind, N, parallel_block_size)
        print msg_tmp
    else:
        msg_tmp = None
    row_data = row_data_all.iloc[my_range].copy()
    #----------------------------
    
    N = row_data.shape[0]
    all_pred = None
    all_nb_pred = []
    for i in range(N):
        pred, nb_pred, feature_names = score_offtarget(row_data.iloc[[i]], model, learn_options, positions)
        all_nb_pred.append(nb_pred)
        if all_pred is None:
            all_pred = pred.copy()[:,None]            
        else:
            all_pred = np.concatenate((all_pred, pred[:,None]), axis=1)
            
    return all_pred.T, np.array(all_nb_pred, dtype='float64'), feature_names

# This only takes data for a single guide-target pair
def score_offtarget(row_data, model, learn_options, positions, msg=None):
    all_predictions_ind = np.zeros(((len(np.unique(positions))+1),))*np.nan
    annots = row_data['Annotation'].copy()
    row_data = row_data.copy()
    
    if msg is not None:
        print msg

    # note make loop only around the first part
    for a in annots.values[0]:        
        row_data['Annotation'] = [a]

        if ":" in a:
            let, pos = a.split(",")
        else: #it's a PAM
            pos = len(np.unique(positions))+1

        feature_sets, _ = featurize_data_elevation(row_data, learn_options)
        inputs, dim, dimsum, feature_names = azimuth.util.concatenate_feature_sets(feature_sets)
        use_predict_proba = isinstance(model, sklearn.ensemble.gradient_boosting.GradientBoostingClassifier) or isinstance(model, sklearn.linear_model.logistic.LogisticRegression)
        if use_predict_proba:
            pred = model.predict_proba(inputs)[:,1][0]
        else:
            try:
                pred = model.predict(inputs)[0][0]
            except:
                pred = model.predict(inputs)[0]
        assert isinstance(pred, float)
        all_predictions_ind[int(pos)-1] = pred

    if len(annots.values[0])==0:        
        all_predictions_ind = np.ones(((len(np.unique(positions))+1),))                
        feature_names = None #ok for now as being called on cluster, may need to get this out at some point

    return all_predictions_ind, np.nanprod(all_predictions_ind), feature_names


def save_final_model(filename, learn_options, order, shortname):
    '''
    run_models(produce_final_model=True, CV=False) is what saves the model

    '''
    test = False
    assert filename is not None, "need to provide filename to save final model"

    learn_options['cv'] = "gene"
    learn_options["testing_non_binary_target_name"] = 'ranks'
    learn_options_set = {'final': learn_options}
    results, all_learn_options = mc.run_models(models=models, orders=[order], adaboost_learning_rates=[0.1],
                                                adaboost_max_depths=[3], adaboost_num_estimators=[100],
                                                learn_options_set=learn_options_set,
                                                test=False, CV=False, setup_function=setup_elevation,set_target_fn=set_target_elevation, pam_audit=False, length_audit=False)
    model = results.values()[0][3][0]
    feature_names = np.array(results[results.keys()[0]][6], dtype=str)

    with open(filename, 'wb') as f:
        pickle.dump((model, learn_options, feature_names), f, -1)

    if hasattr(model, "coef_"):
        model_description_file = filename[0:-1] + "tostr.csv"
        #param_names = np.concatenate((feature_names, ["bias"]))
        #param_vals = np.concatenate((model.coef_.flatten(), np.array([model.intercept_]).flatten()))
        param_names = feature_names#np.concatenate((feature_names, ["bias"]))
        param_vals = np.concatenate((model.coef_.T, model.intercept_*np.ones(model.coef_.T.shape)), axis=1)

        tmp_one_hot_inputs = np.zeros(model.coef_.shape)
        try:
            pred_bias_only = model.predict_proba(tmp_one_hot_inputs)[0][0]
        except:
            pred_bias_only = model.predict(tmp_one_hot_inputs)[0][0]
        py1x0 = pred_bias_only*np.ones(model.coef_.T.shape)
        py1x1 = np.zeros(model.coef_.T.shape)
        for j in range(tmp_one_hot_inputs.shape[1]):
            tmp_one_hot_inputs = np.zeros(model.coef_.shape)
            tmp_one_hot_inputs[0,j] = 1.0
            try:
                py1x1[j] = model.predict_proba(tmp_one_hot_inputs)[0][0]
            except:
                py1x1[j] = model.predict(tmp_one_hot_inputs)[0][0]

        param_data = np.concatenate((param_names[:,None], param_vals, py1x1,py1x0), axis=1)
        tmp = pandas.DataFrame(param_data, columns=["features", "coef", "bias", "p(y=1|x=1)", "p(y=1|x=0)"])
        tmp = tmp[tmp["coef"] !='0.0']
        tmp.to_csv(model_description_file)

    return model, feature_names

def predict_cfd_and_naivebayes(annots_list, cfd_table_file=None, use_const_px=False):
    """
    Given a list of a set of annotations
    """
    if False:
        "predicting from our own computations of CFD, which should match below"
        cfd_table_loaded_from_file = pandas.read_csv(cfd_table_file, index_col=[0])
    else:
        "predicting from NBT CFD table"
        cfd_table_loaded_from_file = get_NBT_cfd()
        # hack this so things don't crash
        cfd_table_loaded_from_file['p(X=1|Y=1)'] = 1.0
        cfd_table_loaded_from_file['p(Y=1|X=0)'] = 1.0
        cfd_table_loaded_from_file['p(X=1)'] = 1.0
        cfd_table_loaded_from_file['p(Y=1)'] = 1.0

    cfd_table_loaded_from_file.index = cfd_table_loaded_from_file['Mismatch Type']
    unique_annot = cfd_table_loaded_from_file['Mismatch Type'].values

    preds_cfd = np.ones(len(annots_list))
    preds_naive_bayes = np.ones(len(annots_list))
    preds_cfd_corrected = np.ones(len(annots_list))

    py = cfd_table_loaded_from_file['p(Y=1)'][0]
    num_feat = cfd_table_loaded_from_file.shape[0]

    for i, annots in enumerate(annots_list):

        for a in unique_annot:
            letters, pos = parse_mismatch_annot(a)
            if pos=='':
                annot_new = letters # a PAM mutation
            else:
                letters = str(letters)
                annot_new = letters[0] + ":" + letters[1] + "," +  str(pos)

            px_given_y = cfd_table_loaded_from_file['p(X=1|Y=1)'].loc[annot_new]
            if not use_const_px:
                # from CD33 data
                px_annot = cfd_table_loaded_from_file['p(X=1)'].loc[annot_new]
            else:
                # if want uniform, then use this (1/255)
                px_annot = 1.0/num_feat

            if a in annots:
                #preds_cfd[i] *= cfd_table_loaded_from_file["Percent-Active"].loc[annot_new]
                preds_cfd[i] *= cfd_table_loaded_from_file["Percent-Active"].loc[annot_new]
                preds_cfd_corrected[i] *= cfd_table_loaded_from_file["Percent-Active"].loc[annot_new]
                preds_naive_bayes[i] *= (px_given_y/px_annot)
            else:
                preds_naive_bayes[i] *= ((1.0 - px_given_y)/(1.0 - px_annot))
                preds_cfd_corrected[i] *= cfd_table_loaded_from_file["p(Y=1|X=0)"].loc[annot_new]

        preds_naive_bayes[i] *= py

    return preds_cfd, preds_naive_bayes, preds_cfd_corrected

def feature_name_conversion_to_cfd(feature_names, reverse=False):
    """
    'cfd' is of the form 'rA:dU' and seperately the position, say 3
    the reverse is us, and is of the form "T:T,3"
    """

    rna_comp = {'A':'T', 'T':'A', 'G':'C', 'C':'G','X':'X'}
    rna_equiv = {'A':'A', 'G':'G', 'C':'C','T':'U', 'X':'X'}

    if reverse:
        rna_comp = dict((v,k) for k,v in rna_comp.iteritems())
        rna_equiv = dict((v,k) for k,v in rna_equiv.iteritems())
        pos = None

    feature_names_new = np.empty(feature_names.shape[0], dtype='S12')
    pos = feature_names.copy()
    for i, nm in enumerate(feature_names):

        if not reverse:

            if ":" in nm:
                let, postmp = nm.split(",")
                d = rna_comp[let[0]]
                r = rna_equiv[let[2]]
                new_name = 'r%s:d%s' % (r, d)
            else:
                new_name = nm
                postmp = -1
            pos[i] = postmp

        else:

            if nm[1]==-1:
                new_name = nm[0]
            else:
                let0, let1 = nm[0].split(":")
                new_name = rna_comp[let1[1]] + ":" + rna_equiv[let0[1]] + "," + str(int(nm[1]))

        feature_names_new[i] = new_name
    return feature_names_new, pos

def make_cfd_from_data(cfd_file_out=None, learn_options=None):
    if learn_options is None:
        learn_options_tmp = {'num_proc': 10,
                         'nuc_features_WT': False, 'include_pi_nuc_feat': False,
                         'mutation_type' : False,
                         'mutation_details' : False,
                         'annotation_onehot' : True, #featurize like CFD
                         'annotation_decoupled_onehot' : False, #decouple the CFD features into letters and position
                         'annotation_letters_decoupled_onehot': False,
                         "include_Tm": False,
                         'include_azimuth_score': None, # all of them ["WT","MUT","DELTA"]
                         'azimuth_feat' : None,#  ['WT'],#["MUT", "WT"],
                         "include_gene_position": False,
                         "cv": "stratified",
                         'adaboost_loss' : 'ls',
                         'adaboost_CV': False, "algorithm_hyperparam_search" : "grid",
                         'n_folds' : 10,
                         'allowed_category' : None,#"Mismatch",#"Insertion",
                         "include_NGGX_interaction": False,
                         'normalize_features' : False, 'class_weight': None,
                         "phen_transform": 'binarize',
                         "training_metric": 'spearmanr',
                         "skip_pam_feat" : True, "letpos_indep_ft": False, "letpos_inter_ft": True,
                         "fit_intercept" : True,
                         "seed" : 12345,
                         "num_proc": 1,
                         }
        learn_options_tmp["V"] = "CD33"; learn_options_tmp["left_right_guide_ind"] = [4,25,30]
    else:
        learn_options_tmp = learn_options
        learn_options_tmp['annotation_decoupled_onehot'] = False
        learn_options_tmp['annotation_onehot'] = True
        learn_options_tmp["phen_transform"] = 'binarize'
        learn_options_tmp['azimuth_feat'] = None

    [Y, feature_sets, target_genes, learn_options, num_proc] = setup_elevation(test=False, order=1, learn_options=learn_options_tmp)

    inputs, dim, dimsum, feature_names = util.concatenate_feature_sets(feature_sets)
    assert len(np.unique(np.sum(inputs, axis=1)))==1 and np.unique(np.sum(inputs, axis=1))[0]==1
    feature_names = np.array(feature_names, dtype=str)

    not_ggind = np.where(feature_names != "GG")[0]
    feature_names = feature_names[not_ggind]
    inputs = inputs[:, not_ggind]
    # remove individuals with GG feature, since they have "no annotation"
    good_ind = np.where(np.sum(inputs, axis=1)==1)[0]
    inputs = inputs[good_ind, :]

    assert len(np.unique(np.sum(inputs, axis=1)))==1 and np.unique(np.sum(inputs, axis=1))[0]==1

    y = Y['Day21-ETP'].values[good_ind]

    num_guides_per_annot = np.sum(inputs, axis=0)
    num_active_guides_per_annot = np.dot(y, inputs).flatten()
    num_active_guides = len(np.where(y)[0])
    assert np.allclose(num_active_guides, np.sum(num_active_guides_per_annot))
    num_inactive_guides = len(np.where(y==False)[0])
    num_guides = len(y)
    assert num_active_guides + num_inactive_guides == num_guides

    py1_given_x1 = np.divide(num_active_guides_per_annot, num_guides_per_annot)
    perc_activity = py1_given_x1
    px1 = num_guides_per_annot/np.sum(num_guides_per_annot)
    assert np.allclose(np.sum(px1), 1)
    py1 = (y >= 1.0).sum()/float(len(good_ind))*np.ones(inputs.shape[1])
    px1_given_y1 = num_active_guides_per_annot/np.sum(num_active_guides_per_annot)
    assert np.allclose(np.sum(px1_given_y1), 1)

    num_active_other_guides_per_annot = num_active_guides - num_active_guides_per_annot
    num_other_guides_per_annot = num_guides - num_guides_per_annot
    py1_given_x0 = num_active_other_guides_per_annot/num_other_guides_per_annot
    py1_x1 = np.multiply(py1_given_x1, px1)
    py1_x0 = np.multiply(py1_given_x0, 1 - px1)
    assert np.allclose(py1_x0 + py1_x1, py1)

    # convert our feature names to those from NBT Sup T 19
    feature_names_cfd, pos = feature_name_conversion_to_cfd(feature_names)

    # print a table so we can match to CFD SupT19
    tmp_array = np.concatenate((feature_names[:,None], pos[:,None], perc_activity[:,None], num_guides_per_annot[:,None], num_active_guides_per_annot[:,None], px1_given_y1[:,None], px1[:,None], py1[:, None], py1_given_x0[:, None]), axis=1)
    cfd_df = pandas.DataFrame(tmp_array, index=[feature_names_cfd, np.array(pos, dtype=float)], columns=["Mismatch Type", "Position", "Percent-Active","# guides", "# active guides", "p(X=1|Y=1)", "p(X=1)", "p(Y=1)", "p(Y=1|X=0)"])
    cfd_df = cfd_df.sort_index()

    if cfd_file_out is not None:
        cfd_df.to_csv(cfd_file_out)

    return cfd_df

def get_NBT_cfd(cfdtable):
    d_mismatch = pandas.read_excel(cfdtable, index_col=[0, 1], sheetname="Mismatch", parse_cols=[0,1,5])
    d_pam = pandas.read_excel(cfdtable, index_col=[0], sheetname="PAM", parse_cols=[0, 1])
    # remove the "GG columns"
    # d_pam = d_pam[d_pam.index != "GG"]
    d_pam.index = list(zip(d_pam.index, -1*np.ones(len(d_pam.index))))
    d_cfd = pandas.concat((d_mismatch, d_pam))
    feature_names_us, pos = feature_name_conversion_to_cfd(d_cfd.index.values, reverse=True)
    d_cfd["Mismatch Type"] = feature_names_us
    d_cfd = d_cfd.sort_index()

    return d_cfd

if __name__ == '__main__':

    np.random.seed(1234)

    verbose = False
    # run from .\CRISPR using %run Elevation/model_comparison.py

    # classification vs. regression
    # phen_transform = "binarize";      training_metric = "AUC"; models = ["logregL1"]
    # phen_transform = "rank_transform"; training_metric = "spearmanr";
    # phen_transform = "rescale"; training_metric = "spearmanr";
    # phen_transform = "identity"; training_metric = "spearmanr";
    phen_transform = "kde_cdf"; training_metric = "spearmanr";


    models = ['AdaBoost']# ['RandomForest']#["AdaBoost"]#["linreg","L2","L1", "AdaBoost"]

    #models = ["AdaBoostClassifier"]

    learn_options_cv = {'num_proc': 10,
                     'nuc_features_WT': False, 'include_pi_nuc_feat': False,
                     'mutation_type' : False,
                     'mutation_details' : False,
                     'annotation_onehot' : True, #featurize like CFD
                     'annotation_decoupled_onehot' : True, #decouple the CFD features into letters and position
                     'annotation_letters_decoupled_onehot': False,
                     "include_Tm": False,
                     'include_azimuth_score': None, # all of them ["WT","MUT","DELTA"]
                     'azimuth_feat' : None,#  ['WT'],#["MUT", "WT"],
                     "include_gene_position": False,
                     "cv": "stratified",
                     'adaboost_loss' : 'ls',
                     'adaboost_CV': False, "algorithm_hyperparam_search" : "grid",
                     'n_folds' : 10,
                     'allowed_category' : None,#"Mismatch",#"Insertion",
                     "include_NGGX_interaction": False,
                     'normalize_features' : False, 'class_weight': None,
                     "phen_transform": phen_transform,
                     "training_metric": training_metric,
                     "skip_pam_feat" : True, "letpos_indep_ft": False, "letpos_inter_ft": True,
                     "fit_intercept" : True,
                     "seed" : 12345,
                     "num_proc": 1,
                     }

    # if dont't want regularization, put this back in
    #"alpha": np.array([1.0e-3]),

    date_stamp = azimuth.util.datestamp()
    exp_name = date_stamp

    orders = [2] # [1,2]

    # train on single-mutation data CD33
    if False:
        learn_options_cv["V"] = "CD33"; learn_options_cv["left_right_guide_ind"] = [4,25,30]
        #learn_options_cv["V"] = "guideseq"; learn_options_cv["left_right_guide_ind"] = [0,23,23]#[0,21,23]
        shortname = "elevation_" + learn_options_cv["V"] + "." + models[0]
        model_file_elevation = shortname +  "." + date_stamp + ".p"
        savemodel_filename = r"./saved_models/" + model_file_elevation
        model, feature_names = save_final_model(savemodel_filename, learn_options_cv, order=2, shortname=shortname)
        print "done learning and saving final model"


        # also learn cfd model from scratch
        if True:
            print "learning cfd from scratch"
            cfd_file_out = r"saved_models\feature_weights_cfd_from_us.csv"
            cfd_us = make_cfd_from_data(cfd_file_out)

            #compare to NBT cfd
            cfd_them = get_NBT_cfd()
            import ipdb; ipdb.set_trace()

            assert np.all(cfd_them.index==cfd_us.index), "indexes dont have same values"
            if not np.all(cfd_them["Mismatch Type"]==cfd_us["Mismatch Type"]):
                for j in range(len(cfd_them["Mismatch Type"])):
                    print "them=%s, us=%s" % (cfd_them["Mismatch Type"].values[j], cfd_us["Mismatch Type"].values[j])

                raise Exception("annotations don't match")

            them = np.array(cfd_them['Percent-Active'].values, dtype=float)
            us = np.array(cfd_us['Percent-Active'].values, dtype=float)
            atol = 1e-6
            if not np.allclose(them, us, atol=atol):
                print "CFD scores DO NOT match when computing from data vs. table from NBT SI"
                plt.figure(); plt.plot(them, us, '.'); plt.show()
                mydiff = np.abs(them-us)
                badind = np.where(mydiff >= atol)[0]
                for j in badind:
                    print "%s: us[%d]=%f vs them[%d]=%f, US: numer=%s, denom=%s, " % (cfd_them.index[j], j, us[j], j, them[j], cfd_us.iloc[j]["# active guides"], cfd_us.iloc[j]["# guides"])
                import ipdb; ipdb.set_trace()
            else:
                print "CFD scores MATCH when computing from data vs. table from NBT SI"

    # predict on guideseq data
    if True:
        print "predicting on guideseq data"

        #model_file_elevation = "elevation_CD33.AdaBoost.2016-05-03_11_41_08.p"
        #model_file_elevation = "elevation_CD33.logregL1.2016-05-12_16_34_17.p" # with intercept
        #model_file_elevation = "elevation_CD33.logregL1.2016-05-12_16_40_01.p" # without intercept
        #model_file_elevation = "elevation_CD33.AdaBoost.2016-05-18_11_34_49.p" # kde phen
        model_file_elevation = "elevation_CD33.AdaBoost.2016-05-24_18_04_53.p"

        cfd_file_out = r"saved_models/feature_weights_cfd_from_us.csv"

        #learn_options_cv["V"] = "guideseq"; learn_options_cv["left_right_guide_ind"] = [0,21,23]
        learn_options_cv["V"] = "mouse";
        #learn_options_cv["V"] = "hsu-zhang-single";
        #learn_options_cv["V"] = "hsu-zhang-multi";
        #learn_options_cv["V"] = "hsu-zhang-both";

        prediction_file = r".\saved_models\\" + "pred." + learn_options_cv["V"] + "." + model_file_elevation
        model_file = "saved_models" + r"/" + model_file_elevation
        savemodel_filename = r".\saved_models\\" + model_file_elevation

        if not os.path.isfile(prediction_file):
            print "getting predictions"
            force_zero_intercept = False

            if learn_options_cv["V"] == "guideseq":
                data, Y, target_genes = elevation.load_data.load_guideseq(learn_options_cv)
            elif learn_options_cv["V"] == "mouse":
                data, Y, target_genes = elevation.load_data.load_mouse_data()
            elif "hsu-zhang" in learn_options_cv["V"]:
                data, Y, target_genes = elevation.load_data.load_HsuZang_data(learn_options_cv["V"])
            else:
                raise Exception()

            N = data.shape[0]
            learn_options_override = {'left_right_guide_ind' : None}

            preds_cfd, preds_naivebayes, preds_cfd_corrected = predict_cfd_and_naivebayes(data["Annotation"].values, cfd_file_out)

            predictions, model, learn_options, _tmpdata, feature_names, all_predictions_ind = predict_elevation(data=data, model=None, model_file=model_file, pam_audit=False, learn_options_override=learn_options_override, force_zero_intercept=force_zero_intercept)

            guide_seq = data['GUIDE-SEQ Reads'].values

            pickle.dump([predictions, preds_cfd, preds_cfd_corrected, preds_naivebayes, guide_seq, data, feature_names, model, all_predictions_ind], open(prediction_file, "wb" ))
            #plt.figure(); plt.plot(cfd_us, cfd_them, '.')

        else:
            print "loading saved predictions from %s" % prediction_file
            [predictions, preds_cfd, preds_cfd_corrected, preds_naivebayes, guide_seq, data, feature_names, model, all_predictions_ind] = pickle.load(open(prediction_file, "rb"));


        if True:
            plt.figure(); plt.plot(predictions, np.log(guide_seq), '.');
            pearson_us, _garb = ss.pearsonr(predictions, guide_seq)
            spearman_us, _garb = ss.spearmanr(predictions, guide_seq)
            plt.title("guide seq. vs. us (all mismatches)\n%s" % model_file_elevation)
            plt.legend(["pearson=%0.3f, sr=%0.3f" % (pearson_us, spearman_us)])
            plt.ylabel("guide seq")
            plt.xlabel("predictions")

            pearson_cfd, _garb = ss.pearsonr(preds_cfd, guide_seq)
            spearman_cfd, _garb = ss.spearmanr(preds_cfd, guide_seq)
            plt.figure(); plt.plot(preds_cfd, np.log(guide_seq),'.');
            plt.title("preds_cfd vs guide_seq")
            plt.legend(["pearson=%0.3f, sr=%0.3f" % (pearson_cfd, spearman_cfd)])

            plt.show()

        # cross-validate on guide seq, always using CD33 trained model as bases in a stacker
        if False:
            n_folds = 20
            num_seed = 1

            N = data.shape[0]

            num_annot = np.array([len(t) for t in data["Annotation"].values])

            label_encoder = sklearn.preprocessing.LabelEncoder()
            # label_encoder.fit(data['Targetsite'].values)
            label_encoder.fit(num_annot)
            # cv_classes = label_encoder.transform(data['Targetsite'].values)
            cv_classes = label_encoder.transform(num_annot)

            num_cd33_pred = np.array([len(t) for t in all_predictions_ind])
            assert np.all(num_annot==num_cd33_pred), "# annot don't match # predictions---had been changing code and stopped--need to figure out"
            max_num_mut = np.max(num_cd33_pred)
            preds_cd33_model = np.nan*np.zeros((N, max_num_mut))
            for n in range(N):
                for i in range(num_annot[n]):
                    preds_cd33_model[n, i] = all_predictions_ind[n][i]

            y = data['GUIDE-SEQ Reads'].values[:, None]

            results_naive = np.nan*np.zeros((n_folds, num_seed))

            results_stacker = np.nan*np.zeros((n_folds, num_seed))
            results_cfd = np.nan*np.zeros((n_folds, num_seed))
            results_naivebayes = np.nan*np.zeros((n_folds, num_seed))

            results_naive_agg =    np.nan*np.zeros(num_seed)
            results_stacker_agg =    np.nan*np.zeros(num_seed)

            steiger_stack_nb = np.nan*np.zeros(num_seed)
            steiger_stack_cfd = np.nan*np.zeros(num_seed)
            steiger_naivebayes_cfd = np.nan*np.zeros(num_seed)

            for j, seed in enumerate(range(num_seed)):
                np.random.seed(1234*seed)
                cv = sklearn.cross_validation.StratifiedKFold(cv_classes, n_folds=n_folds, shuffle=True)

                contour = None
                preds_per_fold_naive = None
                preds_per_fold_stacker = None
                preds_per_fold_cfd = None
                preds_per_fold_naivebayes = None
                y_per_fold = None

                for fold, [train, test] in enumerate(cv):

                    if True:
                        m_stacker = Stacker(y[train], preds_cd33_model[train], warp_out=False, loss='spearman', opt_method="optimize", combiner="nb")
                        contour, A, B, K = m_stacker.maximize()
                    else:
                        m_stacker = StackerFeat()
                        stack_model = "GP";
                        normalize_feat = False
                        stacker_phen_transform = "identity" # "sqrt"
                        m_stacker.fit(preds_cd33_model[train], y[train], model=stack_model, normalize_feat=normalize_feat, phen_transform=stacker_phen_transform)
                    pred_stacker = m_stacker.predict(preds_cd33_model[test])
                    pred_naive = np.nanprod(preds_cd33_model[test], axis=1)[:,None]

                    if y_per_fold is None:
                        preds_per_fold_naive = pred_naive
                        preds_per_fold_stacker = pred_stacker
                        y_per_fold = y[test].flatten()
                        preds_per_fold_cfd = preds_cfd[test].flatten()
                        preds_per_fold_cfd_corrected = preds_cfd_corrected[test].flatten()
                        preds_per_fold_naivebayes = preds_naivebayes[test].flatten()
                    else:
                        preds_per_fold_naive = np.concatenate((preds_per_fold_naive, pred_naive))
                        preds_per_fold_stacker = np.concatenate((preds_per_fold_stacker, pred_stacker))
                        y_per_fold = np.concatenate((y_per_fold, y[test].flatten()))
                        preds_per_fold_cfd = np.concatenate((preds_per_fold_cfd, preds_cfd[test].flatten()))

                        preds_per_fold_cfd_corrected = np.concatenate((preds_per_fold_cfd_corrected, preds_cfd_corrected[test].flatten()))
                        preds_per_fold_naivebayes = np.concatenate((preds_per_fold_naivebayes, preds_naivebayes[test].flatten()))

                    results_stacker[fold, j] = ss.spearmanr(y[test].flatten(), pred_stacker.flatten())[0]
                    results_naive[fold, j] = ss.spearmanr(y[test].flatten(), pred_naive.flatten())[0]
                    results_cfd[fold, j] = ss.spearmanr(y[test].flatten(),  preds_cfd[test].flatten())[0]
                    results_naivebayes[fold, j] = ss.spearmanr(y[test].flatten(),  preds_naivebayes[test].flatten())[0]

                    print "[%d, %d]: Spearman R for fold: %.3f naive, %.3f stacker, %.3f naive-bayes, %.3f cfd" % (fold, j, results_naive[fold, j],results_stacker[fold, j], results_naivebayes[fold, j], results_cfd[fold, j])

                #print "median Spearman r: %.3f naive, %.3f stacker" % (np.median(results_naive[:, j]),

                #                                                       np.median(results_stacker[:, j]))
                #print "[seed=%d] mean Spearman r: %.3f naive, %.3f stacker" % (j, np.mean(results_naive[:, j]), np.mean(results_stacker[:, j]))

                y_per_fold = np.array(y_per_fold, dtype=float)
                preds_per_fold_naive = np.array(preds_per_fold_naive, dtype=float)
                preds_per_fold_stacker = np.array(preds_per_fold_stacker, dtype=float)
                preds_per_fold_naivebayes = np.array(preds_per_fold_naivebayes, dtype=float)

                results_stacker_agg[j] = ss.spearmanr(y_per_fold,  preds_per_fold_stacker)[0]
                results_naive_agg[j] = ss.spearmanr(y_per_fold,  preds_per_fold_naive)[0]

                # compare stacker to naive bayes
                t2_stack, pv_stack, corr_naive, corr_stacker, corr01_stack = util.get_pval_from_predictions(preds_per_fold_naive, preds_per_fold_stacker,  y_per_fold, twotailed=False, method='steiger')
                print "[seed=%d] Steiger stacker to naive: %1.1e" % (j, pv_stack)

                # compare stacker to cfd
                t2_cfd, pv_cfd, corr_cfd, corr1, corr01_cfd = util.get_pval_from_predictions(preds_per_fold_cfd, preds_per_fold_stacker,  y_per_fold, twotailed=False, method='steiger')
                print "[seed=%d] Steiger stacker to cfd: %1.1e" % (j, pv_cfd)

                # compare naive to cfd
                t2, pv, corr0, corr1, corr01 = util.get_pval_from_predictions(preds_per_fold_cfd, preds_per_fold_naive,  y_per_fold, twotailed=False, method='steiger')
                print "[seed=%d] Steiger naive to cfd: %1.1e" % (j, pv)

                # compare naivebayes co cfd
                t2, pv, corr0, corr_naivebayes, corr01 = util.get_pval_from_predictions(preds_per_fold_cfd, preds_per_fold_naivebayes,  y_per_fold, twotailed=False, method='steiger')
                print "[seed=%d] Steiger naivebayes to cfd: %1.1e" % (j, pv)

                # compare cfd to cfd_corrected
                t2, pv, corr0, corr_cfd_corrected, corr01 = util.get_pval_from_predictions(preds_per_fold_cfd, preds_per_fold_cfd_corrected,  y_per_fold, twotailed=False, method='steiger')
                print "[seed=%d] Steiger cfd to cfd_corrected: %1.1e" % (j, pv)

                print "cfd spearman = %f" % corr_cfd
                print "cfd_corrected spearman = %f" % corr_cfd_corrected
                print "naive spearman = %f" % corr_naive
                print "naivebayes spearman = %f" % corr_naivebayes
                print "Stacker spearman = %f" % corr_stacker
                print "--------------------------------------"
                print "CFD pearson = %f" % ss.pearsonr(preds_per_fold_cfd, y_per_fold)[0]
                print "Stacker pearson = %f" % ss.pearsonr(preds_per_fold_stacker.flatten(), y_per_fold)[0]

                import ipdb; ipdb.set_trace()

                plt.plot(preds_per_fold_cfd_corrected, preds_per_fold_naivebayes, '.');
                plt.show()

                steiger_stack_cfd[j] = pv_stack
                steiger_stack_nb[j] = pv_cfd

                if contour is not None:
                    if m_stacker.combiner=="nb":
                        plt.matshow(contour); plt.colorbar(); plt.show()
                    else:
                        plt.plot(K, contour[0,0,:], 'x'); plt.title('1D contour'); plt.show()

        plt.figure();
        plt.plot(results_naive, results_stacker, 'k.')
        [stat, pval] = ss.mannwhitneyu(results_stacker, results_naive, use_continuity=True, alternative='less')
        plt.title("stacker vs naive over random seeds (per fold and per seed), p=%f" % pval)
        plt.xlabel("results_naive")
        plt.ylabel("results_stacker")
        plt.plot([0,1],[0,1], '--')

        plt.figure();
        d_naive = np.mean(results_naive, axis=1)
        d_stack = np.mean(results_stacker, axis=1)
        plt.plot(d_naive, d_stack, 'k.')
        [stat_agg, pval_agg] = ss.mannwhitneyu(d_naive, d_stack, use_continuity=True, alternative='less')
        plt.title("stacker vs naive over random seeds (per seed), p=%f" % pval_agg)
        plt.xlabel("results_naive")
        plt.ylabel("results_stacker")
        plt.plot([0,1],[0,1], '--')

        plt.figure();
        plt.plot(results_naive_agg,  results_stacker_agg, 'k.')
        [stat_agg2, pval_agg2] = ss.mannwhitneyu(results_naive_agg, results_stacker_agg, use_continuity=True, alternative='less')
        plt.title("stacker vs naive over random seeds (per seed, w one spearman/seed), p=%f" % pval_agg2)
        plt.xlabel("results_naive")
        plt.ylabel("results_stacker")
        plt.plot([0,1],[0,1], '--')


        plt.show()

        import ipdb; ipdb.set_trace()



    # for local runs (otherwise need to use runner which calls this):
    if False:
        learn_options_tmp = learn_options_cv.copy()
        #learn_options_tmp['train_genes'] = azimuth.load_data.get_V3_genes()
        #learn_options_tmp['test_genes'] = azimuth.load_data.get_V3_genes()
        learn_options_tmp['V'] = 'guideseq'

        learn_options_set = {date_stamp:learn_options_tmp}
        results, all_learn_options = mc.run_models(models=models, orders=orders, adaboost_learning_rates=[0.1],
                                                adaboost_max_depths=[3], adaboost_num_estimators=[100],
                                                learn_options_set=learn_options_set,
                                                test=False, CV=True, setup_function=setup_elevation,set_target_fn=set_target_elevation, pam_audit=False, length_audit=False)
        all_metrics, gene_names = azimuth.util.get_all_metrics(results, learn_options_set)
        azimuth.util.plot_all_metrics(all_metrics, gene_names, all_learn_options, save=True)
        # AB_or2_md3_lr0.10_n100_2015-12-21_17_48_36 spearmanr 0.592423433051
        mc.pickle_runner_results(exp_name, results, all_learn_options, relpath=r"Elevation\results")
        import ipdb; ipdb.set_trace()

    # make a call to the final model
    if False:
        predictions = predict_elevation(wt_seq30=np.array(['TGTCGTAGTAGGGTATGGGA', 'AAAGCAGCCAGGACAGCAGT','AAAGCAGCCAGGACAGCAGT','AAAGCAGCCAGGACAGCAGT']),
                          mutation_details=np.array(['TGG', 'G:A,7', 'I:C,2', 'D:G,7']),
                          category=np.array(['PAM','Mismatch', 'Insertion', 'Deletion']),
                          model=None, model_file=None, pam_audit=False)
        # expected output: [ 1.57862003  0.43694794  0.69038882 -0.96302201]
        print predictions

    # predict on the entire training data set, and check correlation with LFC as sanity check
    if False:
        data = elevation.load_data.read_doench(phen_transform=learn_options_cv["phen_transform"])
        model_file_elevation = "saved_models/elevation_missmatch_21mer.2016-03-22_17_40_15.p"
        predictions, model, learn_options, data = predict_elevation(data=data, model=None, model_file=model_file_elevation, pam_audit=False)
        # print these to file for the record, can be used as a regression test
        data['elevation'] = predictions
        data.to_csv(r'../../data/offtarget/SuppTable8_withPredicitions_mismatch.csv')

        plt.figure(); plt.plot(np.sort(predictions),'.'); plt.title('predicted off-target vs. enumeration of training cases')

        spearman = azimuth.util.spearmanr_nonan(predictions, data['Day21-ETP'])[0]
        #  0.60122173263647094, barely higher than in CV (0.59), demonstrating no over-fitting
        # for mismatch only, 0.66443
        plt.figure(); plt.plot(predictions, data['Day21-ETP'], '.', alpha=0.2); plt.title('LFC vs elevation_predictions')
        plt.xlabel('elevation'); plt.ylabel('ranks of LFC')
        plt.legend(['spearman rho=%0.2f' % spearman])
        plt.show()
        import ipdb; ipdb.set_trace()

    # learn Azimuth model for just 21mers:
    if False:
        model_file_azimuth = "saved_models/Azimuth_23mer.p"
        short_name = "azimuth_21mer"
        include_NGGX_interaction = False
        learn_options = {"V": 3,
            'train_genes': azimuth.load_data.get_V3_genes(),
            'test_genes': azimuth.load_data.get_V3_genes(),
            "testing_non_binary_target_name": 'ranks',
            'include_pi_nuc_feat': True,
            "gc_features": True,
            "nuc_features": True,
            "include_gene_position": False,
            "include_NGGX_interaction": include_NGGX_interaction,
            "include_Tm": True,
            "include_strand": False,
            "include_gene_feature": False,
            "include_gene_guide_feature": 0,
            "extra pairs": False,
            "weighted": None,
            "training_metric": 'spearmanr',
            "NDGC_k": 10,
            "cv": "gene",
            "include_gene_effect": False,
            "include_drug": False,
            "include_sgRNAscore": False,
            'adaboost_loss' : 'ls', # main "ls", alternatives: "lad", "huber", "quantile", see scikit docs for details
            'adaboost_alpha': 0.5, # this parameter is only used by the huber and quantile loss functions.
            'adaboost_CV' : False,
            'left_right_guide_ind' : [4,27,30], #inds are relative to 30mer such that [0,30] gives 30mer
            'normalize_features' : False,   ## so far this is only for logistic regression(?)
            }
        mc.save_final_model_V3(filename=model_file_azimuth, learn_options=learn_options, short_name=short_name,  pam_audit=False, length_audit=False)

        import ipdb; ipdb.set_trace()
