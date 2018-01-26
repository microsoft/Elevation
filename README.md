# Elevation

Off-target effects of the CRISPR-Cas9 system can lead to suboptimal gene
editing outcomes and are a bottleneck in its development. Here, we introduce
two interdependent machine learning models for the prediction of off-target
effects of CRISPR-Cas9. The approach, which we named Elevation, scores
individual guide–target pairs, and aggregates such scores into a single,
overall summary guide score.

See our [**official project page**](https://www.microsoft.com/en-us/research/project/crispr/) for more detail.

## Publications

Please cite this paper if using our predictive model:

Jennifer Listgarten\*, Michael Weinstein\*, Benjamin P. Kleinstiver,
Alexander A. Sousa, J. Keith Joung, Jake Crawford, Kevin Gao, Luong Hoang,
Melih Elibol, John G. Doench\*, Nicolo Fusi\*. [**Prediction of off-target
activities for the end-to-end design of CRISPR guide RNAs.**](https://doi.org/10.1038/s41551-017-0178-6)
*Nature Biomedical Engineering*, 2018.

(\* = equal contributions/corresponding authors)

## Dependencies

### Software dependencies

1. Install Anaconda >= 4.1.1: `https://www.continuum.io/downloads`

### Download and process data dependencies

1. First, download and process necessary public data files using the
   `CRISPR/download_data.sh` script (on Windows, you can run similar
   commands by hand to the wget commands in the script, or run the
   script using Bash on Windows/Cygwin/etc).

   After running the script, your directory structure should look like:

```
elevation/
    CRISPR/
        data/
            offtarget/
                Haeussler/
                CD33_data_postfilter.xlsx
                nbt.3117-S2.xlsx
                STable 18 CD33_OffTargetdata.xlsx*
                STable 19 FractionActive_dlfc_lookup.xlsx*
                Supplementary Table 10.xlsx
        gene_sequences/
            CD33_sequence.txt
        guideseq/
            guideseq.py
            guideseq_unique.txt
    cache/
    CHANGELOG.md
    elevation/
    ...
```

2. One additional data file must be generated via a genome search,
   using our Elevation-search (aka "dsNickFury") software, which
   must be installed separately. The repository is located at
   `https://github.com/michael-weinstein/dsNickFury3PlusOrchid`.

   (If you are not planning to run a genome search for off-targets,
    you do not need to follow the instructions in the dsNickFury
    documentation for data dependencies, just cloning the repository
    is sufficient. You will need to download the indexed genome data.)

   After installing dsNickFury, edit the `CRISPR/guideseq/guideseq.py`
   script to point to the directory where it is installed, then run
   the script. (This will take some time to run; ~8 hours on a desktop)

   Once the script finishes, there should be a file called
   `guideseq_unique_MM6_end0_lim999999999.hdf5` in the `CRISPR/guideseq`
   directory.

3. Your directory structure should now look something like this:

```
elevation/
    CRISPR/
        data/
            offtarget/
                Haeussler/
                CD33_data_postfilter.xlsx
                nbt.3117-S2.xlsx
                STable 18 CD33_OffTargetdata.xlsx*
                STable 19 FractionActive_dlfc_lookup.xlsx*
                Supplementary Table 10.xlsx
        gene_sequences/
            CD33_sequence.txt
        guideseq/
            guideseq.py
            guideseq_unique.txt
            guideseq_unique_MM6_end0_lim999999999.hdf5
            ...
    cache/
    CHANGELOG.md
    elevation/
    ...
```

   You can now install the elevation dependencies and run the software.

### Install / Develop

1. Create conda env for elevation: `conda create -n elevation python=2.7`
2. Activate conda env:
    * (windows) `activate elevation`
    * (linux) `source activate elevation`
3. Install Azimuth version 2.0.0: `pip install git+https://github.com/MicrosoftResearch/Azimuth.git`
4. Overwrite some of the Azimuth dependencies, since Elevation uses different versions:
    * `conda install pytables`
    * `conda install scikit-learn==0.18.1`
    * `pip install pandas==0.19.1`
   (installing these packages via conda/pip avoids recompiling them from source)
4. Install/Develop elevation:
    * To install, `python setup.py install`
    * To develop, `python setup.py develop`

### Test installation

Make sure everything is set up properly by running the following command from
the root directory of the repository.

`python -m pytest tests` or `nosetests tests`

## Use

### Guide Sequence Prediction

```python
import elevation.load_data
from elevation.cmds.predict import Predict

# load data
num_x = 100
roc_data, roc_Y_bin, roc_Y_vals = elevation.load_data.load_HauesslerFig2(1)
wildtype = list(roc_data['30mer'])[:num_x]
offtarget = list(roc_data['30mer_mut'])[:num_x]

# initialize predictor
p = Predict()

# run prediction
preds = p.execute(wildtype, offtarget)

# preds is a dictionary of the form {'linear-raw-stacker': [...], 'CFD': [...]}
for i in range(num_x):
    print(wildtype[i], offtarget[i], map(lambda kv: kv[0] + "=" + str(kv[1][i]), preds.iteritems()))
```

### Aggregation Prediction

```python
import numpy as np
import pickle
import elevation.load_data
from elevation.cmds.predict import Predict
from elevation import settings
from elevation import aggregation

# load data
num_x = 100
roc_data, roc_Y_bin, roc_Y_vals = elevation.load_data.load_HauesslerFig2()
wildtype = list(roc_data['30mer'])[:num_x]
offtarget = list(roc_data['30mer_mut'])[:num_x]

# initialize guide seq predictor
p = Predict()

# run prediction
preds = p.execute(wildtype, offtarget)

# load aggregation model
with open(settings.agg_model_file) as fh:
    final_model, other = pickle.load(fh)

# compute aggregated score
isgenic = np.zeros(num_x, dtype=np.bool)
result = aggregation.get_aggregated_score(
         preds['linear-raw-stacker'],
         preds['CFD'],
         isgenic,
         final_model)
print result
```

### Recomputing Models

Models are persisted as pickle files and, under certain circumstances,
may need to be recomputed. Elevation models depend on the CRISPR repository.
To recompute models, run the following command.

```shell
elevation-fit --crispr_repo_dir /home/melih/dev/CRISPR
```

where `/home/melih/dev/CRISPR` corresponds to the directory that contains the
CRISPR repository you'd like to use to recompute the models.

### New Fixtures

After making changes to the models, to generate new fixtures (data used to test
prediction consistency), run `elevation-fixtures`.

Run `python -m pytest tests` to make sure tests are still passing.

### Settings

If you'd like to reconfigure the default location of CRISPR, the temp dir in
which pickles are stored, etc., copy `elevation/settings_template.py` to
`elevation/settings.py` and edit `elevation/settings.py` before installation.
If `elevation/settings.py` does not exist at install time, then
`elevation/settings_template.py` is used to create `elevation/settings.py`.

## Contacting us

You can submit bug reports using the GitHub issue tracker. If you have any
other questions, please contact us at crispr@lists.research.microsoft.com.
