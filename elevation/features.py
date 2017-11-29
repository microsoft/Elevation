import numpy as np
import pandas
import itertools

def extract_mut_positions_stats(df, annotation_column_name='Annotation'):
    df = df.copy()
    df['mut positions'] = df[annotation_column_name].apply(parse_positions_from_mut_list)
    df['mut abs distances'] = df['mut positions'].apply(absolute_mutation_distances)
    df['mut abs distances'] = df['mut abs distances'].apply(fill_in_single_mismatch_distances)
    df['mut mean abs distance'] = df['mut abs distances'].apply(np.mean)
    df['mut median abs distance'] = df['mut abs distances'].apply(np.median)
    df['mut min abs distance'] = df['mut abs distances'].apply(np.min)
    df['mut max abs distance'] = df['mut abs distances'].apply(np.max)
    df['mut sum abs distance'] = df['mut abs distances'].apply(np.sum)

    df['consecutive mut distance'] = df['mut positions'].apply(consecutive_mutation_distance)
    df['consecutive mut distance'] = df['consecutive mut distance'].apply(fill_in_single_mismatch_distances)
    df['mean consecutive mut distance'] = df['consecutive mut distance'].apply(np.mean)
    df['median consecutive mut distance'] = df['consecutive mut distance'].apply(np.median)
    df['min consecutive mut distance'] = df['consecutive mut distance'].apply(np.min)
    df['max consecutive mut distance'] = df['consecutive mut distance'].apply(np.max)
    df['sum consecutive mut distance'] = df['consecutive mut distance'].apply(np.sum)

    return df

def parse_positions_from_mut_list(mut_list):
    positions = []
    for m in mut_list:
        if ":" in m:
            change, pos = m.split(',')
        else: # it's a PAM
            pos = 22

        positions.append(int(pos))

    return positions

def absolute_mutation_distances(pos_list):
    return [abs(pair[1] - pair[0]) for pair in itertools.combinations(pos_list,2)]

def fill_in_single_mismatch_distances(distances, missing_distance_value=18):
    if distances == []:
        return [missing_distance_value]
    else:
        return distances

def consecutive_mutation_distance(pos_list):
    return np.diff(np.sort(pos_list)).tolist()
