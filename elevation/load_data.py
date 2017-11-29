import os
import pickle
import copy
import warnings

import pandas
import numpy as np
import scipy as sp
import sklearn.neighbors.kde as kde
import sklearn.grid_search as gs
import sklearn.linear_model
import Bio.SeqUtils as SeqUtil
import Bio.Seq as Seq

import azimuth
import azimuth.util

import elevation
import elevation.util
from elevation import settings


def load_ben_guideseq(datdir=r'\\nerds5\compbio_storage\CRISPR.offtarget\ben_guideseq', learn_options=None):
    assert learn_options is not None
    learn_options = copy.deepcopy(learn_options)
  
    # file directly from Ben by email on 5/19/2017
    # N.B. the "gecko" in the filenames comes from Ben's file, which he presumably named because the guides were 
    # chosen from the gecko library.
    data_w_readcounts_file = os.path.join(datdir, "17-05-18 GUIDE-seq output.xlsx")
    data_from_dsnickfury_file = os.path.join(datdir, "17-04-01_gecko_GUIDEseq_sites_ordered_MM6_end0_lim500000.hdf5")

    data = pandas.read_excel(data_w_readcounts_file)

    # drop EMX1_1 because these are some sort of control he does everywhere
    data = data[data['sgRNA'] != "EMX1_1"]
    
    data['30mer'] = data['Target Sequence']    
    data['30mer_mut'] = data['Off-Target Sequence']    
    data = data.set_index(['30mer', '30mer_mut'], drop=False, verify_integrity=True)
    
    del data['Target Sequence']
    del data['Off-Target Sequence']

    #guide_seq_full_orig = pandas.read_hdf(data_from_dsnickfury_file, 'allsites')
    guide_seq_full = pandas.read_hdf(data_from_dsnickfury_file, 'allsites')
    #guide_seq_full = guide_seq_full.rename(index=str, columns={'mismatches': 'Num mismatches'})
        
    # we intent this to be similar to the CCTop isnan filter above (in version 1)
    cctop_pams = ["AG", "GG"]
    ag_gg_filter = guide_seq_full["30mer_mut"].apply(lambda x: x[-2:] in cctop_pams)
    guide_seq_full = guide_seq_full[ag_gg_filter]
    # this reduces it from  around 2 million to 722,336

    # Michael by email: There may very well be repeated regions in the genome, and the Y chrm especially so, this would yield repeats in 30mer, 30mer_mut pairs (and potentially quite a few)   
    guide_seq_full = guide_seq_full.drop_duplicates(['30mer_mut','30mer'])
    # this drops it now from 722,336 to 560,562

    # all are AGG because that's what we give Michael's code
    #unique_pams = np.unique(guide_seq_full["30mer"].apply(lambda x: x[-3:] )) 
    # just to make it clearer, make them NGG    
    guide_seq_full["30mer"] = guide_seq_full["30mer"].apply(lambda x: x[:-3] + "NGG")

    guide_seq_full = guide_seq_full.set_index(['30mer', '30mer_mut'], drop=False, verify_integrity=False)

    assert np.unique(data['30mer'].apply(lambda x: x[-3:])) == np.unique(guide_seq_full['30mer'].apply(lambda x: x[-3:]))

    # this is just for debugging and is not needed for final result
    data_left_merge = data.merge(guide_seq_full, how="left", on=["30mer", "30mer_mut"], indicator=True)
    data_left_only = data_left_merge[data_left_merge['_merge'] == "left_only"]
    ommitted_guides = data_left_only[['30mer', '30mer_mut', 'GUIDE-seq read counts', 'Mismatches', 'sgRNA']]
    ommitted_guide_pams = np.unique(ommitted_guides['30mer_mut'].apply(lambda x: x[-2:]))    
    for pam, j in enumerate(ommitted_guide_pams):
        assert pam not in cctop_pams, "offtarget=%s, guide=%s shouldn't appear here" % (ommitted_guides.iloc[j]["30mer_mut"], ommitted_guides.iloc[j]["30mer"])

    print "WARNING: have ommitted %d of guides from the actual guide-seq assay file" % data_left_only.shape[0]

    data_right_merge = data.merge(guide_seq_full, how="right", on=["30mer", "30mer_mut"], indicator=True)
    data = data_right_merge
    assert data.shape[0] == guide_seq_full.shape[0] # only true for right join
        
    # to fill out zeros for all the ones only in the search file, but not found in the lab
    data['GUIDE-seq read counts'] = data['GUIDE-seq read counts'].fillna(value=0)
    
    data['30mer'] = data.apply(lambda x: replace_NGG(x['30mer'], x['30mer_mut']), axis=1)

    # coerce data as needed for off-target prediction, based on Supp. Table 8 data
    categories = ['SomeTypeMutation' for x in range(len(data))]   # in CD33 training would be Mismatch/PAM, etc.
    data['Category'] = categories
        
    data['Annotation'] = data.apply(lambda x : annot_from_seqs(x['30mer'], x['30mer_mut'], x['Mismatches'], warn_not_stop=True), axis=1)

    del data["_merge"]

    assert not np.any(np.isnan(data['GUIDE-seq read counts'].values)), "found nan read counts"
    
    data = data.rename(columns={'GUIDE-seq read counts' : "GUIDE-SEQ Reads"})

    if learn_options["renormalize_guideseq"]:
        #raise Exception("potentially make sure pam filtering happens only after this")                
        data = renormalize_guideseq(data)
    
    # for guides which had no positive read counts, these will get normalized to NaN, so convert them back to zeros:
    data["GUIDE-SEQ Reads"] = data["GUIDE-SEQ Reads"].fillna(value=0)

    assert not np.any(np.isnan(data["GUIDE-SEQ Reads"].values)), "found nan read counts"

    Y = data["GUIDE-SEQ Reads"]
    target_genes = np.unique(data['sgRNA'])

    data['Annotation'] = data.apply(lambda x : annot_from_seqs(x['30mer'], x['30mer_mut']), axis=1)
                    
    return data, Y, target_genes

def load_HF_guideseq(learn_options):

    
    df_gs = pandas.read_excel(settings.pj(settings.offtarget_data_dir, 'HFguideseq/nature16526-s5.xlsx'), sheetname='included sites')    
    df_all = pandas.read_hdf(settings.pj(settings.offtarget_data_dir, 'HFguideseq/nature16526-s5_MM6_end0_lim500000.hdf5'))

    # rename
    df_gs['30mer'] = df_gs['Target Sequence']
    df_gs['30mer_mut'] = df_gs['Off-Target Sequence']
    df_gs['GUIDE-SEQ Reads'] = df_gs['GUIDE-seq Reads']
    df_gs = df_gs.drop(['GUIDE-seq Reads', 'Off-Target Sequence','Target Sequence'], axis=1)

    # filter
    df_gs = df_gs[df_gs["Treatment"]=='wild-type']
    df_gs = df_gs[df_gs["Mismatches (pre-SNP correction)"] > 0]

    # we noticed that two off-targets are length 24, with the NGG at the end, so lets fix those here:
    # waiting to hear back from Ben to validate this assumption (5/31/2017)
    df_gs["30mer_mut"] = df_gs["30mer_mut"].apply(lambda x: x[1:] if len(x) > 23 else x)
    
    assert len(np.unique(df_gs["Cells"])) == 1

    df_gs['20mer'] = df_gs["30mer"].apply(lambda x: x[0:20])
    df_all['20mer'] = df_all["30mer"].apply(lambda x: x[0:20])

    # should not do anything;
    #tmp = df_gs.drop_duplicates(['20mer', "30mer_mut"]) # same as df_gs, namely 251
    #tmp2 = df_gs.drop_duplicates(['30mer', "30mer_mut"])# same as df_gs, namely 251
        
    #Michael's code creates duplicates (we expect it to), but not sure if Haeussler's does.       
    df_gs = df_gs.drop_duplicates(['20mer', '30mer_mut']) #even though we know from above this does nothing...
    df_all = df_all.drop_duplicates(['20mer', '30mer_mut']) # from 105,530 to 88,033  

    elevation.model_comparison.check_seq_len(df_gs, colname='30mer', expected_len=23)
    elevation.model_comparison.check_seq_len(df_gs, colname='30mer_mut', expected_len=23)
    elevation.model_comparison.check_seq_len(df_all, colname='30mer', expected_len=23)
    elevation.model_comparison.check_seq_len(df_all, colname='30mer_mut', expected_len=23)
    
    # just for debugging; which rows are we missing from our search (not needed to return)
    #missing_gs_df = pandas.merge(df_gs, df_all, on=['20mer', '30mer_mut'], how='left', suffixes=['', '_right'], indicator=True)
    #missing_gs_df = missing_gs_df[missing_gs_df["_merge"] == "left_only"]
        
    merge_df = pandas.merge(df_gs, df_all, on=['20mer', '30mer_mut'], how='right', suffixes=['_left', ''], indicator=True)
    # dsNickFury was run only on "wild-type", from "included_sites", EMX1-2, FANCF-2, FANCF-3, RUNX1-1, ZSCAN2, which should contain
    # precisely 53 rows in excel if filtered like this.
    tmp = merge_df[merge_df['GUIDE-SEQ Reads'] > 0]
        
    assert tmp.shape[0] == 53 
    assert merge_df['GUIDE-SEQ Reads'].sum() == 18015

    assert merge_df.shape[0] == df_all.shape[0]
    merge_df['GUIDE-SEQ Reads'] = merge_df['GUIDE-SEQ Reads'].fillna(0)
        
    elevation.model_comparison.check_seq_len(merge_df, colname='30mer', expected_len=23)
    elevation.model_comparison.check_seq_len(merge_df, colname='30mer_mut', expected_len=23)
    
    # PAM FILTER (must come after normalization)

    # see loadGuideseq data--this originally came as a filter via CCTOp who only used NRG PAMs (R=A and G)
    ag_gg_filter = merge_df["30mer_mut"].apply(lambda x: x[-2:] == 'AG' or x[-2:] == 'GG')
    merge_df = merge_df[ag_gg_filter]
        
    if learn_options["renormalize_guideseq"]:
        #raise NotImplementedError("look at other load gs data sets--make sure to do after PAM filterings")
        merge_df = renormalize_guideseq(merge_df)

    merge_df['Annotation'] = merge_df.apply(lambda x : annot_from_seqs(x['30mer'], x['30mer_mut']), axis=1)
    return merge_df


def load_cd33(learn_options={"left_right_guide_ind":[4,25,30]}):

    if not learn_options.has_key("left_right_guide_ind"):
        learn_options["left_right_guide_ind"] = [4,25,30]

    if not learn_options.has_key("pam_pos_filter"):
        learn_options["pam_pos_filter"] = True

    save_cd33_file = "CD33.processed.%s.pamfilt%s.p" % (str(learn_options["left_right_guide_ind"]), str(learn_options["pam_pos_filter"]))
    save_cd33_file = settings.pj(settings.tmpdir, save_cd33_file)

    if False:#os.path.isfile(save_cd33_file):
        print "loading processed data from file: %s..." % save_cd33_file
        [data] = pickle.load(open(save_cd33_file, "rb" ))
        print "done."
    else:
        print "reading and featurizing CD33 data..."

        # this is a filtered version of the one below, but doesn't have the PAM rows we need
        # we only want to keep Mismatch from here
        data_file_filt = settings.pj(settings.offtarget_data_dir, 'CD33_data_postfilter.xlsx')
        data_filt = pandas.read_excel(data_file_filt, index_col=[0], parse_cols=range(1,9))
        # it has some PAM and other rows, but lets remove them so no collisions with below
        data_filt = data_filt[data_filt['Category']=="Mismatch"]

        # this has the PAM info, for which we want to
        data_file_full = settings.pj(settings.offtarget_data_dir, 'STable 18 CD33_OffTargetdata.xlsx')
        data_full = pandas.read_excel(data_file_full, index_col=[0], parse_cols=range(1,9))
        data_full = data_full[data_full['Category']=="PAM"]
        data_full = data_full[data_full['TranscriptID']=="ENST00000262262"]
        if learn_options["pam_pos_filter"]:
            data_full = data_full[(data_full["Protein Annotation"] <= 59.88) & (data_full["Protein Annotation"] >= 7.97)]

        data = pandas.concat((data_filt, data_full), axis=0)

        elevation.model_comparison.check_seq_len(data, colname="WTSequence")

        # for backward compatiblity with previous file format above (Supplementary Table 8.xlsx)
        data['Position'][np.isnan(data['Position'])] = -1
        #data['Annotation'] = [str(data['Annotation'].values[i]) + ',' + str(int(data['Position'].values[i])) for i in range(data.shape[0])]
        for i, categ in enumerate(data["Category"]):
            if categ=="PAM":
                data['Annotation'].iloc[i] = data['Annotation'].iloc[i][1:]
            else:
                data['Annotation'].iloc[i] = data['Annotation'].iloc[i] + "," + str(int(data['Position'].iloc[i]))

        # want index to also be a regular column...
        data[data.index.name] = data.index

        if learn_options.has_key("allowed_category") and learn_options["allowed_category"] is not None:
            # e.g allowed_categories=["Deletion", "Insertion", "Mismatch", "PAM"]
            assert learn_options["allowed_category"] in data["Category"].values, "allowed category not found: %s" % learn_options["allowed_category"]
            data = data[data['Category']==learn_options["allowed_category"]]

        # get sequence context that we will need:
        print "generating 30 and 32 mer sequences for 20mer guides and saving to file"
        thirtymer, thirtytwomer, thirtymer_mut = get_thirtymers_for_geneX(data)
        data['30mer'] = map(str, thirtymer)
        #data['32mer'] = map(str, thirtytwomer)
        data['30mer_mut'] = map(str, thirtymer_mut)
        #data.to_excel(pandas.ExcelWriter(data_file_aug), index=True)

        elevation.model_comparison.check_seq_len(data, colname='30mer')
        elevation.model_comparison.check_seq_len(data, colname="30mer_mut")

        if learn_options.has_key('left_right_guide_ind') and learn_options['left_right_guide_ind'] is not None:
            if len(learn_options['left_right_guide_ind'])==2:
                learn_options['left_right_guide_ind'].append(30) #default is 30mer
            all_len = data['30mer'].apply(len).values
            unique_len = np.unique(all_len)
            assert len(unique_len)==1, "should have only one length for each guide"
            expected_length = learn_options['left_right_guide_ind'][2]
            assert unique_len[0] == expected_length, "these indexes are relative to a particular mer but found %smer" % unique_len[0]
            seq_start, seq_end = learn_options['left_right_guide_ind'][0:2]
            data['30mer'] = data['30mer'].apply(lambda seq: seq[seq_start:seq_end])
            data['30mer_mut'] = data['30mer_mut'].apply(lambda seq: seq[seq_start:seq_end])


        elevation.model_comparison.check_seq_len(data, colname="30mer")
        elevation.model_comparison.check_seq_len(data, colname="30mer_mut")

        if not learn_options["pam_pos_filter"]:
            assert data.shape[0]==5649, "did not find expected # of rows, must be key clash(?)"
        else:
            assert data.shape[0]==4853, "did not find expected # of rows, must be key clash(?)"

        # need in featurize_data_elevation and here
        if "allowed_category" in learn_options and learn_options["allowed_category"] is not None:
            keep_ind = (data['Category']==learn_options["allowed_category"])
            data = data[keep_ind]

        print "Done."

        #print "saving to file: %s" % save_cd33_file
        #pickle.dump([data], open(save_cd33_file, "wb" ))

    unique_annot = np.unique(data['Annotation'].values)
    # pickle.dump(unique_annot, open(r"D:\Source\CRISPR\elevation\elevation\saved_models\unique_annot.p", "wb" ))

    target_genes = get_cd33_genes()

    # Add a binarized column (for prob calibration later on)
    data['Day21-ETP-binarized'] = (data['Day21-ETP'] > 1.0)*1.0

    # this must appear after the other code above otherwise it can get written over from the loading
    if not learn_options.has_key("phen_transform"):
        learn_options["phen_transform"] = "identity"

    if learn_options["phen_transform"]=="rank_transform":
        print "rank transforming target variable"
        data['Day21-ETP'] = azimuth.util.get_ranks(pandas.DataFrame(data['Day21-ETP']))[0]
    elif learn_options["phen_transform"]=="binarize":
        print "binarizing target variable"
        data['Day21-ETP'] = data['Day21-ETP'] > 1.0
    elif learn_options["phen_transform"]=="rescale":
        tmp_data = data['Day21-ETP'].copy().values
        tmp_data = (tmp_data - np.min(tmp_data))/np.max(tmp_data - np.min(tmp_data))
        data['Day21-ETP'] = tmp_data
    elif learn_options["phen_transform"]=="kde_cdf":
        # kernel density estimation cdf
        dat = np.array(data['Day21-ETP'].values, dtype=float)
        pdf, kd = kde_cv_and_fit(dat, bandwidth_range=np.linspace(0.01, 1, 10))
        tmp_cdf = cdf(dat,dat,pdf)
        data['Day21-ETP'] = tmp_cdf
    elif learn_options["phen_transform"]=="identity":
        pass
    elif learn_options["phen_transform"]=='log':
        data['Day21-ETP'] = np.log(np.array(data['Day21-ETP'].values - np.array(data['Day21-ETP'].values.min())+1, dtype=float))
    elif learn_options["phen_transform"]=='Platt':
        print "Platt scaling"
        dat = np.array(data['Day21-ETP'].values, dtype=float)
        dat_bin = (dat >= 1.)*1
        # clf = sklearn.linear_model.LogisticRegression()
        clf = sklearn.linear_model.LogisticRegressionCV(n_jobs=20)
        clf.fit(dat[:, None], dat_bin[:, None])
        Y_prob = clf.predict_proba(dat[:, None])[:, 1]
        data['Day21-ETP'] = Y_prob
    else:
        raise Exception("invalid phen_transform=%s" % learn_options["phen_transform"])



    Y = pandas.DataFrame(data['Day21-ETP'], index=data.index)
    p_Y = (Y['Day21-ETP'] >= 1.0).sum()/float(Y.shape[0])
    Y['pY'] = p_Y

    # to make use of azimuth code, hack like this so we can stratify by category or something else:
    Y['Target gene'] = data['Category']
    if "allowed_categories" not in learn_options:
        learn_options['all_genes'] = np.unique(data['Category'].values)
    else:
        learn_options['all_genes'] = learn_options["allowed_categories"]

    #for evaluation, hack this too, making them all equal to the "Day21-ETP" now
    learn_options['rank-transformed target name'] = 'Day21-ETP'
    learn_options['binary target name'] = 'Day21-ETP'
    learn_options['raw target name'] = 'Day21-ETP'
    learn_options["testing_non_binary_target_name"] = 'raw'
    learn_options['raw_target_name'] = 'Day21-ETP'

    Y.index = Y['Target gene']

    return data, Y, target_genes


def get_cd33_genes():
    """
    for this to work, there needs to be a column in the pandas dataframe called "Target gene"
    """
    return ['CD33']

def get_guideseq_genes():
    """
    for this to work, there needs to be a column in the pandas dataframe called "Target gene"
    N.B. These are nto really all genes, but we're using that to streamline with Azimuth
    """
    return ['RNF2', 'FANCF', 'EMX1', 'VEGFA_site1', 'VEGFA_site3',
            'HEK293_sgRNA4', 'HEK293_sgRNA3', 'HEK293_sgRNA1',
            'VEGFA_site2', 'HEK293_sgRNA2']

# GuideSeq sequences have N in the NGG, but we need the actual letter...
def replace_NGG(thirtymer, thirtymer_mut):
    N_nt = thirtymer_mut[-3]
    seq = thirtymer
    seq = seq[0:-3] + N_nt + seq[-2:]
    return seq

def renormalize_guideseq(data):
    assert "GUIDE-SEQ Reads" in data.columns
    # need to go from 21mer ("30mer") to just 20 by taking off the "N" which is not strictly part of the guide
    # otherwise we end up with 36=9x4 normalization groups instead of 9
    data["20mer"] = data["30mer"].apply(lambda x: x[0:20])
    data_norm = data[['30mer', "GUIDE-SEQ Reads", "20mer"]].groupby('20mer').transform(lambda x: x/np.sum(x))
    data_tmp = data.copy()
    data_tmp.loc[data_norm.index, 'GUIDE-SEQ Reads' ] = data_norm['GUIDE-SEQ Reads']
    data = data_tmp
    print "Done."
    return data

def load_guideseq(learn_options):
    """
    CAVEAT--this data now has only NRG PAMs for the "implicit"
    zero counts, but has all PAMs for the actually observed guide-seq data.

    One can both renormalize_guide_seq, and then subsequently kde_normalize_guideseq"
    """

    #if learn_options is None:
    #    learn_options = {
    #        "left_right_guide_ind": [0, 21, 23],
    #        "guide_seq_full": True,
    #        "reload guideseq": False,
    #        "renormalize_guideseq": False,
    #        "kde_normalize_guideseq": False,
    #        "guideseq_version": 2
    #    }

    #assert learn_options["renormalize_guideseq"], "highly recommend normalize the guideseq data even if doing further renormalization"

    basename = "guideseq_full.v3_%s_%r_%s.p" % (learn_options["guide_seq_full"], learn_options["renormalize_guideseq"], str(learn_options['left_right_guide_ind']))
    pickle_file = os.path.join(settings.tmpdir, basename)

    if False: #os.path.exists(pickle_file) and not learn_options['reload guideseq']:
        # import ipdb; ipdb.set_trace()
        print "loading GuideSeq data from pickle..."
        [data] = pickle.load(open(pickle_file, "rb" ))
        print "done."
    else:
        print "reading GuideSeq data and saving to pickle..."
        guide = settings.pj(settings.offtarget_data_dir, 'Supplementary Table 10.xlsx')
        #guide = '../../data/offtarget/GuideSeqValidationDataDontUseMoreThanOnce/Supplementary Table 10_replace_dup_w_largest_val.xlsx'

        data = pandas.read_excel(guide, sheetname='Supplementary Table 10')

        # filter out those with 0 mismatches
        data = data[data['Num mismatches']>0]

        #filter out the one line with a '-'
        is_ok = data['Offtarget_Sequence'].apply(lambda s: '-' not in s)
        data = data[is_ok]

        N = data.shape[0]
        assert len(data['Offtarget_Sequence'].iloc[0])==23, "expecting length of seq to be 23 so know where PAM is"

        data['30mer'] =  data['Target_Sequence']
        data['WTSequence'] =  data['Target_Sequence']
        data['30mer_mut'] =  data['Offtarget_Sequence']
        #data = data.set_index(['30mer', '30mer_mut'], drop=False, verify_integrity=True)
        # remove replicate keys by taking the one with the higher GUIDE-seq count until we hear back from Mudra/John
        data = data.sort_values('GUIDE-SEQ Reads', ascending=False)
        # this causes us to lose a few points relative to when we were first working on this, in particular, from 402 to 396
        data = data.drop_duplicates(['30mer', '30mer_mut'], keep="first")
        data = data.set_index(['30mer', '30mer_mut'], drop=False, verify_integrity=True)
        # goes from 402 to 395

        del data['Target_Sequence']
        del data['Offtarget_Sequence']

        # even if only want small data set, doing it by first combining is a (perhaps weird way) to achieve the same
        # PAM filtering as Mudra (i.e. NRG), whereas if we only load the small file, we get all PAMs
        if True:#learn_options["guide_seq_full"]:

            print "loading guidseq version", learn_options.get('guideseq_version', 1)
            if learn_options.get('guideseq_version', 1) == 1:
                data_dir = settings.pj(settings.offtarget_data_dir, "guide_seq_comparison_from_Mudra")

                guide_seq_file_base = "score_off_target_9guides_6mm_noperfectmatches"
                guide_seq_file = guide_seq_file_base + ".txt"
                print "reading guideseq_v1: %s" % os.path.join(data_dir, guide_seq_file)
                guide_seq_full = pandas.read_table(os.path.join(data_dir, guide_seq_file), header=1, names=["30mer", "30mer_mut", "Num mismatches", "Mismatch position", "CFD", "CCTop", "HsuZhang"])

                # CCTOP uses NRG PAMS, where R is in A or G
                # we think that this amounts to a filter on the offtargets that don't have a PAM ending in AG or GG
                # the following should output ['AG', 'GG'] only
                # guide_seq_full['30mer_mut'].apply(lambda x: x[-2:]).unique()
                guide_seq_full = guide_seq_full[~np.isnan(guide_seq_full.CCTop)]

                # we haven't yet converted to uppercase, but that shouldn't matter if lowercase
                # just represents mismatch
                guide_seq_full = guide_seq_full.drop_duplicates(['30mer_mut','Num mismatches'])

                #guide_seq_file_base = "result"
                #guide_seq_file = guide_seq_file_base + ".txt"
                #guide_seq_full = pandas.read_table(os.path.join(data_dir, guide_seq_file), header=None, names=["30mer", "chrm", "pos", "30mer_mut", "strand", "Num mismatches"])
                ## we *expect* to see many replicates here, but as Mudra did (see scripts/paper.jenn.py def mudra_get_roc(df, guide_seq), we will keep each possible row
                ##isdup = guide_seq_full.duplicated(['30mer', '30mer_mut']).values

                guide_seq_full["30mer"] = guide_seq_full["30mer"].apply(lambda x: x[:-3] + "NGG")
                guide_seq_full["30mer_mut"] = guide_seq_full["30mer_mut"].apply(lambda x: x.upper())
            elif learn_options.get('guideseq_version') == 2:
                # columns:
                # '30mer', 'mismatches', 'chromosome', 'start', 'end', 'gene', '30mer_mut', 'strand'
                print "reading guideseq_v2: %s" % settings.pj(settings.CRISPR_dir, 'guideseq/guideseq_unique_MM6_end0_lim999999999.hdf5')
                guide_seq_full = pandas.read_hdf(settings.pj(settings.CRISPR_dir, 'guideseq/guideseq_unique_MM6_end0_lim999999999.hdf5'), 'allsites')
                guide_seq_full = guide_seq_full.rename(index=str, columns={'mismatches': 'Num mismatches'})

                guide_seq_full["30mer_mut"] = guide_seq_full["30mer_mut"].apply(lambda x: x.upper())

                # we intent this to be similar to the CCTop isnan filter above (in version 1)
                ag_gg_filter = guide_seq_full["30mer_mut"].apply(lambda x: x[-2:] == 'AG' or x[-2:] == 'GG')
                guide_seq_full = guide_seq_full[ag_gg_filter]

                guide_seq_full = guide_seq_full.drop_duplicates(['30mer_mut','Num mismatches'])
                guide_seq_full["30mer"] = guide_seq_full["30mer"].apply(lambda x: x[:-3] + "NGG")
                # not clear if this is needed.
            else:
                raise Exception("Unknown guideseq_version %s" % learn_options.get('guideseq_version'))

            guide_seq_full = guide_seq_full.set_index(['30mer', '30mer_mut'], drop=False, verify_integrity=False)

            # ---------------------------
            # debugging the drop from 396 to 369 (27 things), which is attributable to some things from the small guide seq
            # table not being in the larger one--Mudra throws them out because they are not NRG PAMs, but we may
            # not want to do that, so not doing that here. CAVEAT--this data now has only NRG PAMs for the "implicit"
            # zero counts, but has all PAMs for the actually observed guide-seq data.
            # ---------------------------
            #tmpright = data.copy()
            #tmpright = tmpright.merge(guide_seq_full, how="right", left_index=True, right_index=True, indicator=True)
            #tmpouter = data.copy()
            #tmpouter = tmpouter.merge(guide_seq_full, how="outer", left_index=True, right_index=True, indicator=True)
            #tmpinner= data.copy()
            #tmpinner = tmpinner.merge(guide_seq_full, how="inner", left_index=True, right_index=True, indicator=True)
            ## which rows are we missing?
            #missingrows = tmpouter[tmpouter['_merge']=='left_only']
            #bothrows = tmpouter[tmpouter['_merge']=='both']
            #import ipdb; ipdb.set_trace()

            # import ipdb; ipdb.set_trace()
            if False:
                # keeps all PAMs (really? don't think so...)
                data = data.merge(guide_seq_full, how="outer", left_index=True, right_index=True, on=["30mer", "30mer_mut", "Num mismatches"], indicator=True)
            else:
                # keep only NRG pams like Mudra did
                # this version yields columns 30mer_mut_x and 30mer_mut_y which causes problems later
                #data = data.merge(guide_seq_full, how="right", left_index=True, right_index=True, indicator=True)
                # this version does not have the problem above

                # ensuring the PAMS are equal to merge on 20nt
                # since guide equivalence is on just 20nts
                assert np.unique(data['30mer'].apply(lambda x: x[-3:])) == np.unique(guide_seq_full['30mer'].apply(lambda x: x[-3:]))
                
                # this is just for debugging and is not needed for final result
                data_left_merge = data.merge(guide_seq_full, how="left", on=["30mer", "30mer_mut", "Num mismatches"], indicator=True)
                data_left_only = data_left_merge[data_left_merge['_merge'] == "left_only"]
                print "have ommitted %d of guides from guide-seq assay file" % data_left_only.shape[0]
                # this appears to be 42, with either version 1 or 2, 5/24/2017

                data_right_merge = data.merge(guide_seq_full, how="right", on=["30mer", "30mer_mut", "Num mismatches"], indicator=True)
                data = data_right_merge
                assert data.shape[0] == guide_seq_full.shape[0] # only true for right join

            # no--Mudrea doesn't do this
            #data_tmp = data.drop_duplicates(['30mer', '30mer_mut', "Num mismatches"])

            #filter out the two rows with "N" in inappropriate places
            #is_ok = data['30mer_mut'].apply(lambda x: 'n' not in x.lower())
            ## why 3 here, when excel file seems to have only 2?
            #data = data[is_ok]

            ## remove perfect matches, and 7-matches

            data['GUIDE-SEQ Reads'] = data['GUIDE-SEQ Reads'].fillna(value=0)
            del data["WTSequence"]

            #del data["pos"]
            #del data["chrm"]
            #del data["strand"]

        data['30mer'] = data.apply(lambda x: replace_NGG(x['30mer'], x['30mer_mut']), axis=1)

        # coerce data as needed for off-target prediction, based on Supp. Table 8 data
        categories = ['SomeTypeMutation' for x in range(len(data))]   # in CD33 training would be Mismatch/PAM, etc.
        data['Category'] = categories

        data['Annotation'] = data.apply(lambda x : annot_from_seqs(x['30mer'], x['30mer_mut'], x['Num mismatches']), axis=1)

        if learn_options.has_key("left_right_guide_ind") and learn_options["left_right_guide_ind"] is not None:
            for colname in ['30mer', '30mer_mut']:
                all_len = data[colname].apply(len).values
                unique_len = np.unique(all_len)
                assert len(unique_len)==1, "should have only one length for each guide"
                expected_length = learn_options["left_right_guide_ind"][2]
                assert unique_len[0] == expected_length, "these indexes are relative to a particular mer but found %smer" % unique_len[0]
                seq_start, seq_end = learn_options["left_right_guide_ind"][0:2]
                data[colname] = data[colname].apply(lambda seq: seq[seq_start:seq_end])

        if not learn_options["guide_seq_full"]:
            keep_ind = data["_merge"]=='both'
            data = data[keep_ind]

        del data["_merge"]

        #pickle.dump([data], open(pickle_file, "wb" ))

    if learn_options["renormalize_guideseq"]:
        #raise Exception("if turn this back on, potentially need/want to do pam filtering only after this, whereas it is now above")
        data = renormalize_guideseq(data)

    if not learn_options.has_key("kde_normalize_guideseq"): learn_options["kde_normalize_guideseq"] = False
    assert not learn_options["kde_normalize_guideseq"]
    # doesn't work (yet) and moreover, doesn't make a lot of sense, so abandoning, but leaving in case change mind
    #if learn_options["kde_normalize_guideseq"]:
    #    import statsmodels.api as sm
    #    print "KDE'ing the GUIDE-SEQ data"
    #    dat = np.array(data['GUIDE-SEQ Reads'].values, dtype=float)

    #    # sub-sample so computationally feasible
    #    ind_nonzero = (dat != 0)
    #    extra_zeros = np.random.permutation(np.where(ind_nonzero == False)[0])[:ind_nonzero.sum()]
    #    ind_keep = ind_nonzero.copy()
    #    ind_keep[extra_zeros] = True
    #    dat = dat[ind_keep]
    #    weights = np.ones(len(dat))
    #    weights[dat==0] = np.sum(data['GUIDE-SEQ Reads'].values == 0) / float(np.sum(dat ==0))

    #    kde = sm.nonparametric.KDEUnivariate(dat)
    #    #kde = sm.nonparametric.KDEUnivariate(dat, bw="cv_ml")
    #    kde.fit(fft=False, weights=weights)
    #    print "found kde bw:%f using %s" % (kde.bw, kde.bw_method)
    #    tmp_cdf = kde.cdf
    #    data['GUIDE-SEQ Reads'] = tmp_cdf
    #    print "done."

    Y = data['GUIDE-SEQ Reads']
    target_genes = get_guideseq_genes()

    #reverse CCTop scores since they are in the wrong direction
    # TODO: Assuming these are not needed - Melih
    # data['CCTop'] = -1.0*data['CCTop']
    return data, Y, target_genes

def get_thirtymers_from_smallermer(smallmers, genes, categories, leftwinsize, rightwinsize):
    """
    left and right win size should be to padd out to 30mer
    """

    # add 1 to window sizes because first getting 32mer
    leftwinsize += 1
    rightwinsize += 1

    num_found = 0
    thirtymer = []
    #thirtytwomer = []
    for i, smallmer in enumerate(smallmers):
        gene_seq = Seq.Seq(elevation.util.get_gene_sequence(genes[i])).reverse_complement()
        guide_seq = Seq.Seq(smallmer)
        categ = categories[i]
        ind = gene_seq.find(guide_seq)
        found = True
        if ind ==-1:
            gene_seq = gene_seq.reverse_complement()
            ind = gene_seq.find(guide_seq)
            if ind == -1:
                print "WARNING: could not find guide in gene %s" % genes[i]
                found = False
            #assert ind != -1, "could not find guide in gene"
        if found:
            assert gene_seq[ind:(ind+len(guide_seq))]==guide_seq, "match not right"
            print "found guide in gene %s" % genes[i]
            num_found += 1
            left_win = gene_seq[(ind - leftwinsize):ind]
            right_win = gene_seq[(ind + len(guide_seq)):(ind + len(guide_seq) + rightwinsize)]
            thirtytwomertmp = left_win + guide_seq + right_win
            assert len(thirtytwomertmp)==32, "thirtytwomer is unexpectedly of length=%i" % len(thirtytwomertmp)
            thirtymertmp =   thirtytwomertmp[1:-1]
            assert len(thirtymertmp)==30, "thirtymer is unexpectedly of length=%i" % len(thirtymertmp)
            if categ!= "PAM":
                assert thirtymertmp[-5:-3] == "GG", "GG not found in either direction"
            thirtymer.append(thirtymertmp)
            #thirtytwomer.append(thirtytwomertmp)
            #if categ=='Mismatch':
            #    # note this is done more generally in compute_mut_azimuth()
            #    mut_20mer = data['MutatedSequence'][i]
            #    find_ind = thirtymertmp.find(twentymer)
            #    assert twentymer == thirtymertmp[find_ind:(find_ind + 20)], "indexing not what expected"
            #    # now know where to replace the mutated sequence to get the right one
            #    thirtymer_mut_tmp = str(thirtymertmp).replace(str(twentymer), mut_20mer)
            #    thirtymer_mut.append(thirtymer_mut_tmp)
            #else:
            #    thirtymer_mut.append(None)
            #if (i % 50 ==0): print "done %d of %d" % (i, len(data))

    import ipdb; ipdb.set_trace()
    return thirtymer#, thirtytwomer#, thirtymer_mut



def get_thirtymers_for_geneX(data, gene='CD33'):
    """
    Need to have column data['WTSequence']
    """
    thirtymer = []
    thirtytwomer = []
    thirtymer_mut = []
    gene_seq = Seq.Seq(elevation.util.get_gene_sequence(gene)).reverse_complement()
    categories = data['Category']
    for i, twentymer in enumerate(data['WTSequence'].values):
        guide_seq = Seq.Seq(twentymer)
        categ = categories[i]
        annot = data['Annotation'].iloc[i]
        ind = gene_seq.find(guide_seq)
        if ind ==-1:
            gene_seq = gene_seq.reverse_complement()
            ind = gene_seq.find(guide_seq)
            assert ind != -1, "could not find guide in gene"
        assert gene_seq[ind:(ind+len(guide_seq))]==guide_seq, "match not right"
        # relative to getting 32mer
        leftwinsize = 5
        rightwinsize = 7
        left_win = gene_seq[(ind - leftwinsize):ind]
        right_win = gene_seq[(ind + len(guide_seq)):(ind + len(guide_seq) + rightwinsize)]
        thirtytwomertmp = left_win + guide_seq + right_win
        thirtymertmp =   thirtytwomertmp[1:-1]
        assert len(thirtymertmp)==30, "thirtymer is unexpectedly of length=%i" % len(thirtymertmp)

        #thirtytwomer.append(thirtytwomertmp)
        thirtytwomer.append("PLACEHOLDER")

        if categ=='Mismatch':
            # note this is done more generally in compute_mut_azimuth()
            find_ind = thirtymertmp.find(twentymer)
            assert twentymer == thirtymertmp[find_ind:(find_ind + 20)], "indexing not what expected"
            mut_20mer = data['MutatedSequence'][i]
            # now know where to replace the mutated sequence to get the right one
            thirtymer_mut_tmp = str(thirtymertmp).replace(str(twentymer), mut_20mer)
            thirtymer_mut.append(thirtymer_mut_tmp)
            thirtymer.append(thirtymertmp)
        elif categ=='PAM':
            ## there is no mismatch here at all, it's just the PAM is wrong
            thirtymer_mut_tmp = thirtymertmp
            thirtymertmp = thirtymertmp[0:25] + "GG" + thirtymertmp[27:]
            assert thirtymer_mut_tmp[25:27]==annot
            thirtymer_mut.append(thirtymer_mut_tmp)
            thirtymer.append(thirtymertmp)
        else:
            thirtymer_mut.append(None)

        assert thirtymertmp[-5:-3] == "GG", "should be GG"

        if (i % 100 ==0): print "done %d of %d" % (i, len(categories))
    return thirtymer, thirtytwomer, thirtymer_mut

def annot_from_seqs(guide, target, expectedNumMismatches=None, warn_not_stop=False):
    """
    Assumes that guide and target sequences start at 20 nucleotides away from the PAM,
    i.e. p1, ... p20, N, G, G.  If there are no non-NGG PAM, one can give only a 20-mer,
    otherwise one shoudl give a 23mer, possibly with an "N" in position 21 which woul be ignored

    Note that by python idexing, this means index 0-19 inclusive contain non-PAM sequence, and
    index 20-22 inclusive are the NGG

    Also note that we are using these terms interchangeably:
    "WT Sequence" ~ "guide" ~ "30mer"
    "Mutated Sequence" ~ "off-target sequence" ~ "30mer_mut"

    Returns list of annotations of the form "G:A,3" (mismatch) and "CG" (PAM)
    from two sequence lengths, with a 1-based position relative to start of wt and mut

    N.B. expectedNumMismatches does not include non NGG PAM in the target

    e.g. PAM region 20-22 (python idexing)
    target:  AGG
    guide:   AAA (there actually is no notion of a PAM in the guide, but some
                  data given to us includes these adjacent positions for context,
                  but they should be ignored)
    """
    M = len(guide)
    assert M==len(target), "wt and mut sequences must be the same length"
    assert M==23, "Elevation only works with 23 mer (where the last 3mer is the PAM)"

    annot_list = []
    num_non_pam = 0

    # first check for non-NGG PAM in target sequence
    # in reality, should always have a PAM annotation, but in the end, with a one-hot encoding as we use, there is
    # a redundant bit, and so this if clause, not putting in 'GG', is still fine (though by accident as that was not the intent)
    if target[21:] != "GG":
        annot_list.append(target[21:])

    # now also check the 20mer adjacent to the PAM
    for pos in range(M):
        if pos < 20 and (guide[pos] != target[pos]) and (guide[pos] != "N" and target[pos] != "N"):
            num_non_pam += 1
            annot_list.append(target[pos] + ":" + guide[pos] + "," + str(pos + 1))

    if expectedNumMismatches is not None and not np.isnan(expectedNumMismatches):
        if warn_not_stop:
            if num_non_pam != expectedNumMismatches:
                print "warning, wrong # for guide=%s, ot=%s, expected=%d, actual=%d" % (guide, target, expectedNumMismatches, num_non_pam)
        else:
            assert num_non_pam == expectedNumMismatches
    return annot_list

def load_HsuZang_data(version="hsu-zhang-both"):
    """
    version= "hsu-zhang-single", "hsu-zhang-multi", "hsu-zhang-both"

    Looking in their paper, Figure 2A, we can see that the 20 nucleotides in their
    Supp. Table 5 and 6 must be the 20mers adjacent to the NGG PAM (and so don't include the N)
    """

    file_list = []
    sheetnames = []
    ynames = []
    cols = []
    shortname = []

    if version=="hsu-zhang-single" or version=="hsu-zhang-both":
        file_list.append(settings.pj(settings.offtarget_data_dir, "2013.HsuZhang.DNAtargetingSpec.NBT.SITab5"))
        sheetnames.append("Data (Single Mismatch)")
        ynames.append("MLE (cutting frequency)")
        cols.append([0,4,6,7,11])
        shortname.append("single")

    if version=="hsu-zhang-multi" or version=="hsu-zhang-both":
        raise NotImplementedError("waiting for info from John to parse")
        file_list.append(settings.pj(settings.offtarget_data_dir, "2013.HsuZhang.DNAtargetingSpec.NBT.SITab6"))
        sheetnames.append("RawData")
        ynames.append("MLE Cleavage")
        #cols.append([])
        shortname.append("multi")

    all_data = []
    for j, file_base in enumerate(file_list):

        if False:#os.path.isfile(file_base + ".p"):
            print "loading processed data from file: %s..." % file_base + ".p"
            data = pickle.load(open(file_base + ".p", "rb" ))
            print "done."
        else:
            print "reading and featurizing Hsu-Zhang data..."

            data = pandas.read_excel(file_base + ".xlsx", index_col=[0], skiprows=3, sheetname=sheetnames[j], parse_cols=cols[j])

            # filter out rows that have weird annotations and have no Target value
            keepind = data["Target"].values != "[]"
            data = data.iloc[keepind]
            keepind = [data.iloc[i]["Target"] != data.iloc[i]["Guide RNA"] for i in range(data.shape[0])]
            data = data.iloc[keepind]
            data[ynames] = data[ynames].apply(lambda x: pandas.to_numeric(x, errors='coerce'))
            data = data.dropna()

            # Make both the target and guide 30mers
            data['Target'] = data['Target'].apply(lambda x: x+'NGG')
            data['Guide RNA'] = data['Guide RNA'].apply(lambda x: x+'NGG')


            data["30mer"] = data["Target"]
            data["30mer_mut"] = data["Guide RNA"]

            # redundant, but later code seems to depend on this.. (ick, I know)
            data["WTSequence"] = data["30mer"]
            data["MutatedSequence"] = data["30mer_mut"]

            data["GUIDE-SEQ Reads"] = data[ynames[j]]
            data['Target gene'] = data[ynames[j]]

            elevation.model_comparison.check_seq_len(data, colname="WTSequence")
            elevation.model_comparison.check_seq_len(data, colname="MutatedSequence")

            N = data.shape[0] # number of rows
            M = len(data["WTSequence"][0]) # length of sequences
            annot = [] # list of lists of annotations e.g. [G:A,5]
            num_annot = np.zeros(N)
            for i in range(N):
                guide = data["30mer"][i]
                off_target = data["30mer_mut"][i]
                tmp_annot = annot_from_seqs(guide, off_target)
                annot.append(tmp_annot)
                num_annot[i] = len(tmp_annot)
                if shortname[j] == "single":
                    assert num_annot[i] == 1
                else:
                    raise NotImplementedError()
            data["Annotation"] = annot
            data["Target gene"] = num_annot

            print "Done. Now saving to file: %s" % file_base + ".p"
            pickle.dump(data, open(file_base + ".p", "wb" ))

            all_data.append(data)

    ext_data = pandas.read_csv(settings.pj(settings.offtarget_data_dir, 'extendedfiles/HsuZangSingleData.csv'), index_col=0)
    data['Extended 30mer'] = np.nan

    for i in range(data.shape[0]):
        if data.iloc[i]['30mer'][0:20] == ext_data.iloc[i]['Extended 30mer'][20:40]:
            data.loc[data.index[i], 'Extended 30mer'] =  ext_data.iloc[i]['Extended 30mer'][20:43]
        else:
            extended_reverse = str(Seq.Seq(ext_data.iloc[i]['Extended 30mer'][17:40]).reverse_complement())
            if data.iloc[i]['30mer'][0:20] in extended_reverse:
                data.loc[data.index[i], 'Extended 30mer'] =  extended_reverse
            else:
                raise Exception("can't find seq in extended seq")

    data['30mer'] = data['Extended 30mer']
    data['WTSequence'] = data['Extended 30mer']

    if version=="hsu-zhang-both":
        raise NotImplementedError()

    Y = pandas.DataFrame(data['GUIDE-SEQ Reads'], index=data.index)
    target_genes = np.unique(data["Target gene"].values)
    return data, Y, target_genes

def load_HauesslerFig2(version):
    
    def merge_hauessler():
        hdf5_data = pandas.read_hdf(settings.pj(settings.offtarget_data_dir, 'Haeussler/fig2-crisporData_withReadFraction_MM6_end0_lim500000.hdf5'), 'allsites')
        new_data = {
            'otSeq': map(lambda x: x[0] + ',' + x[1], hdf5_data[['30mer', '30mer_mut']].values),
            'gene': hdf5_data['gene'],
            'chromosome': hdf5_data['chromosome'],
            'start': hdf5_data['start'],
            'end': hdf5_data['end'],
        }
        new_df = pandas.DataFrame(new_data, columns=['otSeq', 'gene', 'chromosome', 'start', 'end'])                
        org_df = pandas.read_csv(settings.pj(settings.offtarget_data_dir, 'Haeussler/fig2-crisporData_withReadFraction.tab'), delimiter='\t')
         
        #Michael's code creates duplicates (we expect it to), but not sure if Haeussler's does.       
        new_df['otSeq_tmp'] = new_df['otSeq'].apply(lambda x: '%s,%s' % (x[0:20], x[24:]))
        org_df['otSeq_tmp'] = org_df['otSeq'].apply(lambda x: '%s,%s' % (x[0:20], x[24:]))
        new_df = new_df.drop_duplicates(['otSeq_tmp'])
        org_df = org_df.drop_duplicates(['otSeq_tmp', 'readFraction'])

        # this effectively assigns, if possible, a readFraction to everything in Michael's search (after dropping duplicates)
        # for those that don't have a value, next they are filled with 0.
        # in principle, this means if michael's search doesn't find all the non-zeros in Haeussler's original file,
        # we could lose some guides, but we know that the total readFraction stays the same, so it seems to be fine
        final_df = pandas.merge(new_df, org_df, on='otSeq_tmp', how='left', suffixes=['', '_right'])

        del final_df['otSeq_right']
        del final_df['otSeq_tmp']
        final_df = final_df.fillna(0)
        assert not final_df.isnull().any().any()
        print org_df['readFraction'].sum(), final_df['readFraction'].sum()
        assert np.allclose(org_df['readFraction'].sum(), final_df['readFraction'].sum())
        return final_df

    print "loading hauessler version", version
    if version == 1:
        data = pandas.read_csv(settings.pj(settings.offtarget_data_dir, 'Haeussler/fig2-crisporData_withReadFraction.tab'), delimiter='\t')
    elif version == 2:
        data = merge_hauessler()

    data['30mer'] = data['otSeq'].apply(lambda x: x.split(',')[0])
    data['30mer_mut'] = data['otSeq'].apply(lambda x: x.split(',')[1])

    elevation.model_comparison.check_seq_len(data, colname='30mer', expected_len=23)
    elevation.model_comparison.check_seq_len(data, colname='30mer_mut', expected_len=23)

    # assert data.shape[0] == 26052
    #assert data['wasValidated'].sum() == 154

    assert np.allclose(data['readFraction'].sum(), 6.74119043336)
    # not sure where this came from
    # assert np.allclose(data['readFraction'].sum(), 3.3623554844970402)

    # see loadGuideseq data--this originally came as a filter via CCTOp who only used NRG PAMs (R=A and G)
    ag_gg_filter = data["30mer_mut"].apply(lambda x: x[-2:] == 'AG' or x[-2:] == 'GG')
    data = data[ag_gg_filter]
    assert np.allclose(data['readFraction'].sum(), 6.6659440001416188)
    

    target_genes = annotate_etc(data)
    return data, data['wasValidated'], data['readFraction']


def annotate_df(df, colguide="30mer", coltarg="30mer_mut", check_for_empty_annot=False):
    N = df.shape[0] # number of rows
    all_annotations = []#np.array(N)
    for i in range(N):
        annotations = []
        guide_seq = df[colguide].iloc[i]
        offtarg_seq = df[coltarg].iloc[i]
        assert len(guide_seq)==len(offtarg_seq), "should be same size"

        annotations = annot_from_seqs(guide_seq, offtarg_seq)
        if check_for_empty_annot and not annotations:
            raise Exception("found empty annotation")
        all_annotations.append(annotations)

    return all_annotations

def annotate_etc(data, colguide="30mer", coltarg="30mer_mut"):
    N = data.shape[0] # number of rows
    # length of sequences
    M = len(data[colguide].values[0]) 
    all_annotations = annotate_df(data, colguide, coltarg)

    data["Annotation"] = all_annotations
    data["Target gene"] = data["Annotation"].apply(len)

    target_genes = np.unique(data["Target gene"].values)
    return target_genes

def load_Frock_et_al(which_table=3):
    if which_table == 3:
        data = pandas.read_csv(settings.pj(settings.offtarget_data_dir, 'frocketal_Stable2.csv'))
    elif which_table == 7:
        data = pandas.read_csv(settings.pj(settings.offtarget_data_dir, 'frocketal_STable7.csv'))
    else:
        raise Exception("Can't find this table in the Frock et al data")

    data['30mer'] = data['30mer'].apply(lambda x: x[0:23])
    data['30mer_mut'] = data['30mer_mut'].apply(lambda x: x[0:23])

    data["WTSequence"] = data["30mer"]
    data["MutatedSequence"] = data["30mer_mut"]

    data["GUIDE-SEQ Reads"] = data["score"]
    elevation.model_comparison.check_seq_len(data, colname="WTSequence")
    elevation.model_comparison.check_seq_len(data, colname="MutatedSequence")
    target_genes = annotate_etc(data)
    Y = pandas.DataFrame(data['score'], index=data.index)
    return data, Y, target_genes

def load_mouse_data():

    file_base = settings.pj(settings.offtarget_data_dir, "MouseValidation/STable 20 H2D_H2K")

    if False: #os.path.isfile(file_base + ".p"):
        print "loading processed data from file: %s..." % file_base + ".p"
        [data] = pickle.load(open(file_base + ".p", "rb" ))
        print "done."
    else:
        print "reading mouse data..."

        data = pandas.read_excel(file_base + ".xlsx", index_col=[5,6], parse_cols=range(0,11))
        data["30mer"] = [ind[0] for ind in data.index]
        data["30mer_mut"] = [ind[1] for ind in data.index]

        # redundant, but later code seems to depend on this.. (ick, I know)
        data["WTSequence"] = data["30mer"]
        data["MutatedSequence"] = data["30mer_mut"]
        data["GUIDE-SEQ Reads"] = data["H2-K LFC"]

        elevation.model_comparison.check_seq_len(data, colname="WTSequence")
        elevation.model_comparison.check_seq_len(data, colname="MutatedSequence")

        N = data.shape[0] # number of rows
        M = len(data["WTSequence"][0]) # length of sequences
        annot = [] # list of lists of annotations e.g. [G:A,5]
        num_annot = np.zeros(N)
        for i in range(N):
            wt = data["30mer"][i]
            mut = data["30mer_mut"][i]
            tmp_annot = annot_from_seqs(wt, mut)
            annot.append(tmp_annot)
            num_annot[i] = len(tmp_annot)
            assert num_annot[i] == len(data['Position of mismatches'].iloc[i].split(","))
        data["Annotation"] = annot
        data["Target gene"] = num_annot

        print "Done. Now saving to file: %s" % file_base + ".p"
        pickle.dump([data], open(file_base + ".p", "wb" ))

    Y = pandas.DataFrame(data['H2-K LFC'], index=data.index)
    target_genes = np.unique(data["Target gene"].values)
    return data, Y, target_genes


def cdf(x, data, mypdf):
    """
    returns the cdf at point x for the pdf defined at points data
    x can be an array
    mypdf and data need to be arrays of the same length
    """
    res = np.zeros_like(x)
    norm = np.sum(mypdf)
    for i in range(x.shape[0]):
        res[i] = np.sum(mypdf[data<=x[i]])/norm
    return res

def kde_cv_and_fit(data, bandwidth_range=np.linspace(0.01, 1, 100)):
    grid = gs.GridSearchCV(kde.KernelDensity(kernel='gaussian', rtol=1e-7),{'bandwidth': bandwidth_range}, cv=10, refit=True)
    grid.fit(data[:, None])
    print grid.best_params_
    kd = grid.best_estimator_
    log_pdf = kd.score_samples(data[:, None])
    return np.exp(log_pdf), kd


def checkfor_pam_mismatch(pos=20):
    for j in range(data.shape[0]):
        if data.iloc[j]["30mer"][pos] != data.iloc[j]["30mer_mut"][pos]:
            raise Exception("found PAM mismatch: wt=%s, mut=%s" % (data.iloc[j]["30mer"], data.iloc[j]["30mer_mut"]))

    return


def load_cd33_plus_hsuzhangsingle():
    """
    to combine data sets, need to do a rank transformation, the result of which is in column 'Yranks'
    """
    cols_to_keep = ["30mer", "30mer_mut", "Annotation", "Target gene", "Yranks", "Data set"]

    data_h, Y_h, target_genes_h = load_HsuZang_data(version="hsu-zhang-single")
    data_h["Data set"] = "hsuzhang"
    data_h["cutting frequency"] = Y_h # this is the cutting frequency, judging from the table headings in the excel sheet
    yranks =  sp.stats.mstats.rankdata (Y_h)
    yranks /= np.max(yranks)
    data_h["Yranks"] = yranks
    data_h["Target"] = data_h.index
    data_h = data_h[cols_to_keep]

    data_c, Y_c, target_genes_c = load_cd33(learn_options = {"left_right_guide_ind":[4,27,30]})
    data_c["Data set"] = "cd33"
    yranks =  sp.stats.mstats.rankdata (Y_c["Day21-ETP"])
    yranks /= np.max(yranks)
    data_c["Yranks"] = yranks
    data_c.rename(columns = {"TranscriptID" : "Target gene"}, inplace=True)
    data_c = data_c[cols_to_keep]

    data_h_plus_c = pandas.concat([data_h, data_c])
    assert data_c.shape[0] + data_h.shape[0] == data_h_plus_c.shape[0], "#s don't add up"
    elevation.model_comparison.check_seq_len(data_h_plus_c, colname="30mer", expected_len=23)
    target_genes = data_h_plus_c['Target gene']
    Y_h_c = pandas.DataFrame(data_h_plus_c['Yranks'])
    Y_h_c.index = target_genes
    return data_h_plus_c, Y_h_c, target_genes


def load_hauessler_plus_guideseq():
    cols_to_keep = ["30mer", "30mer_mut", "Annotation", "Mismatch position", "Targetsite","readFraction", "Num mismatches"]

    data_h = load_hauessler_minus_guideseq()
    data_h.rename(columns = {'Target gene' : 'Num mismatches'}, inplace=True)
    data_h = data_h[cols_to_keep]

    data_g, Y, target_genes = load_guideseq(learn_options={"left_right_guide_ind":[0,23,23], "guide_seq_full":True, "reload guideseq":False, "renormalize_guideseq":True})
    data_g.rename(columns = {"GUIDE-SEQ Reads" : "readFraction"}, inplace=True)
    data_g = data_g[cols_to_keep]

    data_h_plus_g = pandas.concat([data_h, data_g])#, how="outer", on=cols_to_join)
    assert data_g.shape[0] + data_h.shape[0] == data_h_plus_g.shape[0], "#s don't add up"

    return data_h_plus_g

def load_hauessler_minus_guideseq(learn_options=None):
    if learn_options is None:
        learn_options = {}
    else:
        learn_options = copy.deepcopy(learn_options)

    learn_options.update({
        "left_right_guide_ind": [0, 23, 23],
        "guide_seq_full": True,
        "reload guideseq": False,
        "renormalize_guideseq": False
    })
    
    data_h, Y1, Y2 = load_HauesslerFig2(learn_options['haeussler_version'])
    data_g, Y, target_genes = load_guideseq(learn_options)

    # note cannot merge on Annotation because each is a list
    cols_to_keep = ["30mer", "30mer_mut", "Annotation", 'Target gene', 'wasValidated', 'GUIDE-SEQ Reads', "Targetsite",
                    "readFraction", "20mer"]

    # right now only v2 has DNAse and only DNAse needs chr start and end so we are enforcing
    # this               
    if learn_options['haeussler_version'] == 2 and learn_options['guideseq_version'] == 2:
        cols_to_keep.extend(['chromosome', 'start', 'end'])

    data_h["20mer"] = data_h["30mer"].apply(lambda x: x[0:20])
    data_g["20mer"] = data_g["30mer"].apply(lambda x: x[0:20])

    cols_to_join = ["20mer", "30mer_mut"]  # what we now believe to be correct (4/5/2017)
    data_h_left_g = data_h.merge(data_g, how="left", on=cols_to_join, indicator=True)
    # '_merge' is generate by using indicator=True in the join
    data_h_left_g = data_h_left_g[data_h_left_g['_merge'] == 'left_only']
    rename_dict = {
        "Annotation_x": "Annotation",
        "20mer_x": "20mer",
        "30mer_x": "30mer",
    }

    if learn_options['haeussler_version'] == 2 and learn_options['guideseq_version'] == 2:
        rename_dict['chromosome_x'] =  'chromosome'
        rename_dict['start_x'] = 'start'
        rename_dict['end_x'] = 'end' 

    data_h_left_g.rename(columns=rename_dict, inplace=True)
    data_h_left_g = data_h_left_g[cols_to_keep]

    return data_h_left_g

def load_guideseq_inner_hauessler():
    data_h, Y_wasValidated, Y_readFraction = load_HauesslerFig2();
    data_g, Y, target_genes = load_guideseq(learn_options={"left_right_guide_ind":[0,21,23], "guide_seq_full":True, "reload guideseq":False, "renormalize_guideseq":True})

    # in retrospect, don't need to do this if only oaded guideseq with [0,23,23]
    data_h['30mer'] = data_h['30mer'].apply(lambda x: x[0:21])
    data_h['30mer_mut'] = data_h['30mer_mut'].apply(lambda x: x[0:21])

    # note cannot merge on Annotation because each is a list
    cols_to_keep = ["30mer", "30mer_mut", "Annotation_x", 'Target gene', 'wasValidated', 'GUIDE-SEQ Reads',"Mismatch position", "CFD", "CCTop", "HsuZhang", "Targetsite"]
    cols_to_join = ["30mer", "30mer_mut"]
    data_g_inner_h = data_g.merge(data_h, how="inner", on=cols_to_join, indicator=True)
    data_g_inner_h = data_g_inner_h[cols_to_keep]
    data_g_inner_h.rename(columns = {"Annotation_x" : "Annotation"}, inplace=True)


    if False:
        data_g_inner_h.to_csv(settings.pj(settings.offtarget_data_dir, 'Haeussler/HaeusslerGuideSeqIntersection.Norm.csv'))
        bad_gene_names = ['VEGFA_site2', 'HEK293_sgRNA4']
        bad_rows = data_g[data_g['Targetsite'].isin(bad_gene_names)]
        good_rows = data_g[~data_g['Targetsite'].isin(bad_gene_names)]

        np.sum(bad_rows["GUIDE-SEQ Reads"].values > 0)
        np.sum(good_rows["GUIDE-SEQ Reads"].values > 0)
        np.sum(data_g["GUIDE-SEQ Reads"].values > 0)

        data_h.shape
        data_g.shape
        data_g_inner_h.shape
        np.sum(data_g_inner_h["GUIDE-SEQ Reads"].values > 0)
        np.sum(data_g_inner_h["wasValidated"])

        my_ind_1 = np.where(data_g_inner_h["GUIDE-SEQ Reads"].values > 0)[0]
        my_ind_2 = np.where(data_g_inner_h["wasValidated"].values)[0]
        my_ind = np.unique(np.append( np.array(my_ind_1), np.array(my_ind_2)))

        val1 = data_g_inner_h["GUIDE-SEQ Reads"].iloc[my_ind].values
        val2 = data_g_inner_h["wasValidated"].iloc[my_ind].values
        plt.figure(); plt.plot(val1, val2, '.')
        import ipdb; ipdb.set_trace()

    return data_g_inner_h

def load_gecko():
    """
    target variable is column "A375 Percent rank"
    """
    data_nonessential = pandas.read_excel(settings.pj(settings.offtarget_data_dir, 'GeCKOv2_Non_essentials_Achilles_A375_complete.xls')) #(4697, 31)
    data_all_A375 = pandas.read_csv(settings.pj(settings.offtarget_data_dir, 'GeckoAvanaSameUnits/GeCKOv2_DMSO_lentiGuide_A375.txt', sep="\t")) # (121964, 25)

    guides = data_nonessential['sgRNA Sequence'].values
    data = data_nonessential[data_all_A375["sgRNA Sequence"].isin(guides)]

    #missing_guides = set(guides).difference(set(data_all_A375["sgRNA Sequence"].values))
    #tmp = set(data_all_A375["sgRNA Sequence"].values).difference(set(guides))
    return data


def load_avana():
    """
    target variable is column "A375 Percent rank"
    """
    data_nonessential = pandas.read_excel(settings.pj(settings.offtarget_data_dir, 'Non-Essentials_Avana.xlsx'))
    data_all_A375 = pandas.read_csv(settings.pj(settings.offtarget_data_dir, 'GeckoAvanaSameUnits/Avana_DMSO_lentiGuide_A375.txt', sep="\t"))
    guides = data_nonessential['sgRNA Target Sequence'].values
    data = data_all_A375[data_all_A375["sgRNA Sequence"].isin(guides)]

    return data

if __name__ == '__main__':

    ben = load_ben_guideseq(datdir=r'\\nerds5\compbio_storage\CRISPR.offtarget\ben_guideseq', learn_options=None)

    hf = load_HF_guideseq({"renormalize_guideseq" : False})
    import ipdb; ipdb.set_trace()

    data, garb1, garb2 = load_HauesslerFig2(1)
    import ipdb; ipdb.set_trace()
    data, garb1, garb2 = load_HauesslerFig2(2)
    import ipdb; ipdb.set_trace()

    data, Y, target_genes = load_ben_guideseq(datdir=r'\\nerds5\compbio_storage\CRISPR.offtarget\ben_guideseq')
    import ipdb; ipdb.set_trace()

    # gs = load_guideseq({'left_right_guide_ind':[0,21,23], "guide_seq_full":True, "reload guideseq":True, "renormalize_guideseq":True, "kde_normalize_guideseq" : False})

    import pylab as plt

    data, Y, target_genes = load_cd33_plus_hsuzhangsingle()
    import ipdb; ipdb.set_trace()

    data_g_inner_h = load_guideseq_inner_hauessler()

    data = load_hauessler_minus_guideseq()



    data = load_avana()





    data = load_hauessler_plus_guideseq()



    data_g = load_guideseq(learn_options={"left_right_guide_ind":[0,21,23], "guide_seq_full":True, "reload guideseq":True, "renormalize_guideseq":True})

    data_g_inner_h = load_guideseq_inner_hauessler()



    learn_options = {}

    learn_options["phen_transform"] = "rescale"
    data, Y_rescale, target_genes = load_cd33(learn_options)

    learn_options["phen_transform"] = "kde_cdf"
    data, Y_kde, target_genes = load_cd33(learn_options)

    learn_options["phen_transform"] = "identity"
    data, Y_id, target_genes = load_cd33(learn_options)


    checkfor_pam_mismatch(20)

    data, Y, target_genes = load_cd33(learn_options={"left_right_guide_ind":[4,27,30]})
    data, Y, target_genes = load_cd33()
    data, Y, target_genes = load_guideseq()

    data, Y, target_genes = load_HsuZang_data("hsu-zhang-single")
    data, Y, target_genes = load_mouse_data()
    data, Y, target_genes = load_Frock_et_al(3)
    data, Y, target_genes = load_Frock_et_al(7)

    save_cd33_file = settings.pj(settings.offtarget_data_dir, "CD33.processed.%s.p") % str([4,26,30])
    [data, Y, target_genes] = pickle.load(open(save_cd33_file, "rb" ))

    dat = np.array(data['Day21-ETP'].values, dtype=float)

    pdf, kd = kde_cv_and_fit(dat, bandwidth_range=np.linspace(0.01, 1, 10))
    cdf = cdf(dat,dat,pdf)

    plt.figure(); plt.plot(np.sort(dat),'r.'); plt.title("original data")
    plt.figure(); plt.plot(dat, pdf, '.'); plt.title("pdf with best bandwidth=%s" % kd.bandwidth)
    plt.show()


    import ipdb; ipdb.set_trace()
