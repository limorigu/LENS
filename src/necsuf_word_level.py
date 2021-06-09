import pandas as pd
import numpy as np
import itertools
from necsuf_tabular_text import intervention_order, all_choices


def create_CF_unk_text(inp, refs, predict_lr, r2i=True, datatype='Text', raw_text=False):
    CF = {'cfs': [], 'intervention_sets': [],
               'original': [], 'cost': [], 'model_pred': []}
    num_features = len(inp.iloc[:-1])
    intervention_order_ids = intervention_order(num_features)
    # in this particular setting, just a single ref
    ref = refs
    if r2i:
        cfs_ref = list(itertools.product(*zip(np.array(inp)[:-1],
                                              np.array(ref)[:-1])))
        CF['original'] += [str(ref.values[:-1])] * (2 ** num_features)
    else:
        cfs_ref = list(itertools.product(*zip(np.array(ref)[:-1],
                                              np.array(inp)[:-1])))
        CF['original'] += [str(inp.values[:-1])] * (2 ** num_features)
    CF['cfs'] += cfs_ref
    if not raw_text:
        CF['intervention_sets'] += intervention_order_ids
    CF['model_pred'].extend([predict_lr([" ".join(x)]) for x in cfs_ref])
    # In this example, no point in considering cost
    CF['cost'] += [None] * (2 ** num_features)
    # unlike in other examples, columns don't mean much here
    CF_df = pd.DataFrame(CF['cfs'], columns=inp.index[:-1])
    if not raw_text:
        CF_df['Original'] = CF['original']
        CF_df['Intervention_index'] = CF['intervention_sets']
        CF_df['Model_pred'] = CF['model_pred']
        CF_df['Cost'] = CF['cost']
    return CF_df


def deg_nec_suff(CF_df, inp, f_inp, r2i=True, CF_i2r_raw_text=None):
    # degrees computation
    subsets = all_choices(list(range(len(inp.iloc[:-1]))))
    deg_dict = {}
    if r2i:
        x_f_inp_f = CF_df['Model_pred'] == f_inp[0]
    else:
        x_f_ref_f = CF_df['Model_pred'] != f_inp[0]

    for subset in subsets:  # for each Subset S s.t. X_S = inp_S
        if r2i:
            x_s_inp_s = \
                CF_df['Intervention_index'] == str(subset)
            s_count = sum(x_s_inp_s)  # compute empirical marginal (maximal) P_{CF}(S=s)
            if s_count > 0:
                s_count_o_count = sum(x_s_inp_s & x_f_inp_f)  # compute empirical joint P_{CF}(X_s=inp_s, F(x)=F(inp))
                degree_of_r2i_sub = \
                    s_count_o_count / s_count  # deg of suff = P_{CF}(F(x)=F(inp)|X_s=inp_s)
            else:
                degree_of_r2i_sub = 0
            deg_dict[str(subset)] = \
                (degree_of_r2i_sub, subs_to_str(subset, inp),
                 len(subset), np.mean(CF_df[x_s_inp_s]['Cost']))
        # i2r, necessary
        else:
            degree_of_i2r_sub = 0
            x_s_ref_s = \
                CF_df['Intervention_index'] == str(subset)
            s_count = sum(x_s_ref_s)
            if s_count > 0:
                s_count_o_count = sum(x_s_ref_s & x_f_ref_f)  # compute empirical joint P_{CF}(X_s=ref_s, F(x)=F(ref))
                degree_of_i2r_sub = \
                    s_count_o_count / s_count  # deg of suff = P_{CF}(F(x)=F(ref)|X_s=ref_s)
            deg_dict[str(subset)] = \
                (degree_of_i2r_sub, subs_to_str(subset, inp), len(subset),
                 np.min(CF_df[x_s_ref_s & x_f_ref_f]['Cost']))

    sub_df = pd.DataFrame.from_dict(deg_dict, orient='index',
                                    columns=["degree", "string", "cardinality", "cost"]).reset_index()
    return sub_df


def subs_to_str(sub, inp, r2i=True):
    # pretty print subsets to feature names, helper functions
    if isinstance(sub, str):
        sub = sub.replace("[", " ").replace("]", " ")
        sub_ind = np.fromstring(sub, sep=' ')
    else:
        sub_ind = np.array(sub)

    return inp[sub_ind.astype(int)].to_string().replace("\n", ", ").replace("    ", " ")