import pandas as pd
import numpy as np


class DnaseRegions(object):

    def __init__(self, filename, min_value=0.0, check_data=False, has_headers=False):
        self.filename = filename
        # the smallest signal value will be 1.0/95
        # initialize this to something significantly smaller
        self.min_value = min_value
        if not has_headers:
            self.all_columns = ["#bin", "chrom", "chromStart", "chromEnd", "name", "score", "strand", "signalValue", "pValue", "qValue", "peak"]
            self.df = pd.read_csv(filename, delimiter="\t", header=None, names=self.all_columns)
            self.df.drop(["#bin", "name", "score", "strand", "pValue", "qValue", "peak"], inplace=True, axis=1)
        else:
            self.df = pd.read_csv(filename, delimiter="\t")
        self.chromosomes = self.df['chrom'].unique()
        self.data = self.build_data()
        if check_data:
            self.do_check_data()

    def build_data(self):
        data = {chrom: self.df[self.df['chrom'] == chrom][['chromStart', 'chromEnd', 'signalValue']].sort_values(by='chromStart').values.T for chrom in self.chromosomes}
        return data

    def do_check_data(self):
        for chrom in self.data:
            values = self.data[chrom]
            prev = None
            for start, end, signalValue in values.T:
                if prev is None:
                    prev = start, end
                    continue
                assert prev[1] <= start
                prev = start, end

    def find(self, chrom_num, qstart, qend):

        def b(t, a, p=0):
            if len(a) == 0:
                return p
            i = len(a)//2
            if t < a[i]:
                return b(t, a[:i], p)
            if t > a[i]:
                return b(t, a[i+1:], p + i + 1)
            return p + i

        chrom = "chr%s" % chrom_num
        if chrom not in self.data:
            return self.min_value

        start_values, end_values, signals = self.data[chrom]
        pos = b(qstart, start_values)

        if pos == start_values.shape[0] or (pos == 0 and qstart < start_values[pos]):
            return self.min_value

        # b returns insertion position s.t. t <= a[p]
        # we want a[p] <= t
        if start_values[pos] != qstart and start_values[pos-1] <= qstart:
            pos -= 1

        assert start_values[pos] <= qstart, "%s <= %s" % (start_values[pos], qstart)
        if qend <= end_values[pos]:
            return signals[pos]
        else:
            return self.min_value

    def compute_from_dataframe(self, df, chrom_col='chromosome', start_col='start', end_col='end'):
        return np.array(map(lambda x: self.find(*x), df[[chrom_col, start_col, end_col]].values))

if __name__ == "__main__":
    dnase_raw = DnaseRegions(r"\\nerds5\compbio_storage\CRISPR.offtarget\dnase\individual\wgEncodeRegDnaseUwA549Peak.txt", no_headers=True)
    print dnase_raw.df.columns
    dnase_avg = DnaseRegions(r"\\nerds5\compbio_storage\CRISPR.offtarget\dnase\average_pk.txt")
    print dnase_avg.df.columns

    for i in range(dnase_raw.df.columns.shape[0]):
        assert dnase_raw.df.columns[i] == dnase_avg.df.columns[i]
