from __future__ import print_function
import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data_file',
                   default='./nbt.3117-S2.xlsx')
    p.add_argument('-o', '--output_file',
                   default='./tsai_s2_filtered.xlsx')
    args = p.parse_args()

    df = pd.read_excel(args.data_file)
    print(len(df))

    df = df.filter(items=['Targetsite', 'Target_Sequence', 'Offtarget_Sequence',
                          '20 bp protospacer # mismatches', 'GUIDE-Seq Reads'])
    df = df.rename(columns={'20 bp protospacer # mismatches': 'Num mismatches',
                            'GUIDE-Seq Reads': 'GUIDE-SEQ Reads'})
    df = df.loc[~df['Targetsite'].str.startswith('tru')]
    print(len(df))

    df.to_excel(args.output_file, index=False,
                sheet_name='Supplementary Table 10')

if __name__ == '__main__':
    main()

