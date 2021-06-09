import pandas as pd
import numpy as np
from necsuf_tabular_text import \
    suff_nec_pipeline, deg_nec_suff
from IPython.display import display


def model_gaming(clf, refs, dataset, feature_num, num_inp=20, df_flattened=None, df_raw=None):
    smallest_cardinality_suff = pd.DataFrame([])
    smallest_card = len(refs.columns)
    f_ref = refs.iloc[0][-1]
    relevant_dataset = dataset[dataset['Model_pred'] != f_ref]
    for i in range(num_inp):
        # seed fixed for reproducability of demonstration, change to get random examples
        inp_i = relevant_dataset.sample(n=1, random_state=42+i)
        f_inp_i = relevant_dataset.loc[inp_i.index]['Model_pred'].values
        # for this experiment, refs are given, don't need to sample them.
        # therefore, can have an empty string for the sampling condition for CF.
        CF_condition = ""
        # our method
        if df_flattened is None:
            CF_r2i, CF_i2r, refs_i = suff_nec_pipeline(CF_condition,
                                                       inp_i, clf, dataset,
                                                       feature_num, refs=refs)
        else:
            CF_r2i, CF_i2r, _, refs_i_flat = \
                suff_nec_pipeline(CF_condition,
                                  inp_i, clf, dataset,
                                  feature_num, datatype='Text',
                                  dataset_flattened=df_flattened,
                                  refs=refs)

        CF_df_deg_suff = deg_nec_suff(CF_r2i, inp_i, f_inp_i, clf, feature_num, r2i=True,
                                      deg_thresh=0.65, filter_supersets=True, datatype='Text')

        card = CF_df_deg_suff.iloc[0]['cardinality']
        if card <= smallest_card:
            smallest_card = card
            sub_df_filtered_suff_smallest_card = \
                CF_df_deg_suff[CF_df_deg_suff['cardinality'] == card].copy()
            sub_df_filtered_suff_smallest_card['inp_ind'] = \
                [inp_i.index]*len(sub_df_filtered_suff_smallest_card)
            smallest_cardinality_suff = pd.concat((smallest_cardinality_suff, sub_df_filtered_suff_smallest_card))
    smallest_cardinality_suff = smallest_cardinality_suff[smallest_cardinality_suff['cardinality'] == smallest_card]
    max_suff = np.max(smallest_cardinality_suff['degree'])
    smallest_card_biggest_suff = smallest_cardinality_suff[smallest_cardinality_suff['degree'] == max_suff]

    def display_gaming_opts(smallest_card_biggest_suff, df):
        inds_inps_gaming = smallest_card_biggest_suff['inp_ind'].values
        inds_inps_gaming = [ind.values[0] for ind in inds_inps_gaming]
        smallest_changes = list(smallest_card_biggest_suff['index'].values)

        for i, (game_option, game_cols) in enumerate(list(zip(inds_inps_gaming, smallest_changes))):
            print("----------------------")
            print("gaming option no.: ", i)
            print("----------------------")
            pd.set_option('max_colwidth', 800)
            display(pd.DataFrame(df.loc[game_option][df.columns[eval(game_cols)]]).T)

    if df_raw is None:
        display_gaming_opts(smallest_card_biggest_suff, dataset)
    else:
        display_gaming_opts(smallest_card_biggest_suff, df_raw)

    # return smallest_card_biggest_suff, smallest_cardinality_suff
