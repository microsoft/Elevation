from __future__ import print_function
import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data_file',
                   default='./STable 18 CD33_OffTargetdata.xlsx')
    p.add_argument('-o', '--output_file',
                   default='./CD33_data_postfilter.xlsx')
    args = p.parse_args()

    df = pd.read_excel(args.data_file)
    print('Original length: {}'.format(len(df)))

    # filter out Thy1 guides
    df = df[~df['TranscriptID'].isin(['ENSMUST00000114840'])]
    print('Length after filtering Thy1 guides: {}'.format(len(df)))

    # filter out rows that are in category PAM, and are not NGG
    df = df[~(df['Category'].isin(['PAM']) & (~df['Annotation'].str.endswith('GG')))]
    print('Length after filtering non-NGG PAMs: {}'.format(len(df)))

    # filter out rows with protein annotation < 7.97 or > 59.34
    df = df[df['Protein Annotation'] >= 7.97]
    df = df[df['Protein Annotation'] <= 59.34]
    print('Length after filtering for protein annotation between 7.97 and 59.34: {}'.format(len(df)))

    # filter out rows having perfect-match sgRNAs with Day21-ETP < 2.25
    perfect_matches = df[(df['Category'].isin(['PAM'])) & (df['Day21-ETP'] > 2.25)]
    match_sgRNAs = perfect_matches['WTSequence'].tolist()
    df = df[df['WTSequence'].isin(match_sgRNAs)]
    print('Length after filtering for Day21-ETP > 2.25: {}'.format(len(df)))

    # split annotation column into annotation and position
    f = lambda x: pd.Series([i for i in x.split(',')])
    col = df['Annotation'].apply(f)
    df['Annotation'] = col[0]
    df['Position'] = pd.to_numeric(col[1])

    df.to_excel(args.output_file, index=False, sheet_name='CD33_data_postfilter')

if __name__ == '__main__':
    main()

