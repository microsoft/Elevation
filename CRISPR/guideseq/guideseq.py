import os
import sys
import pandas as pd

# replace this with the location of your dsNickFury install
DSNF_DIRECTORY = "/home/jake/repos/dsNickFury/dsNickFury3PlusOrchid"

pj = os.path.join
workdir = os.path.dirname(os.path.realpath(__file__))
os.chdir(DSNF_DIRECTORY)
sys.path.append("./")

from offtarget_manager import OfftargetManager


def get_sequences(fname):
    df = pd.read_csv(fname)
    targets = df['seq'].apply(lambda x: x.split(",")[0][:20]).unique()
    targets = map(lambda x: ("NONE", x.upper(), 'AGG'), targets)
    return targets

if __name__ == "__main__":
    filename = pj(workdir, "guideseq_unique.txt")
    om = OfftargetManager(filename, mismatchTolerance=6, endClip=0, mismatchLimit=999999999)

    targets = get_sequences(filename)
    om.set_targets(targets)
    om.generate_command_list(output_to_file=True)

    om.write_commands()
    om.execute_commands()

    om.merge_output_files()
    om.write_hdf5()

    # print om.get_hdf5().columns
