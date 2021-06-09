from scipy import stats
from necsuf_tabular_text import \
    intervention_order, cost, subs_to_str, \
    all_choices
import itertools
from operator import itemgetter
from pathlib import Path
import dice_ml
from dice_ml.utils import helpers  # helper functions
import tensorflow as tf

# supress deprecation warnings from TF
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import sys;

sys.path.insert(0, '../src/')
import numpy as np
import seaborn as sns


def recourse_experiment_main(CF_condition, clf, dataset, dataset_raw,
                             num_features, datatype,
                             d, exp, num_inp=5):
    results_dict = {'our_mean_cost': [], 'our_min_cost': [], 'our_max_cost': [],
                    'dice_mean_cost': [], 'dice_min_cost': [], 'dice_max_cost': [],
                    'our_val_cands': [], 'dice_val_cands': [],
                    'num_our_val_cands': [], 'num_dice_val_cands': [],
                    'our_dfs': [], 'dice_dfs': [], 'recall_score': [],
                    'inps': [], 'inps_raw': []}

    col_con = [i for i in range(len(dataset.columns[:num_features]))
               if (isinstance(dataset.iloc[0, i], int) | isinstance(dataset.iloc[0, i], float))]
    col_cat = list(set(range(len(dataset.columns[:num_features]))) - set(col_con))
    categorical_cols_names = dataset.columns[col_cat]
    cont_cols_names = dataset.columns[col_con]

    types = dataset.dtypes
    MAD_features_cost = [stats.median_abs_deviation(dataset[col].values)
                         if dtype != 'object' else 1 for
                         (col, dtype) in types.iteritems()]
    for i in range(num_inp):
        inp_i = dataset.sample(n=1, random_state=i)
        inp_i_raw = dataset_raw.loc[inp_i.index]
        f_inp_i = dataset.loc[inp_i.index]['Model_pred'].values
        CF_i2r, CF_i2r_raw_text, refs = \
            recourse_suff_nec_pipeline(CF_condition=eval(CF_condition), inp=inp_i,
                                       f_inp=f_inp_i, clf=clf, dataset=dataset,
                                       num_features=num_features, datatype=datatype,
                                       inp_raw=inp_i_raw, dataset_raw=dataset_raw,
                                       MAD_features_cost=MAD_features_cost)

        for cat in categorical_cols_names:
            CF_i2r[cat] = CF_i2r[cat].apply(lambda x: np.array(eval(x)))

        CF_i2r = CF_i2r[CF_i2r.Intervention_index != '[]']

        X = CF_i2r.iloc[:, :num_features][list(cont_cols_names) +
                                          list(categorical_cols_names)].values
        preds = get_pred(clf, X, datatype='Dice')
        CF_i2r['Model_pred'] = preds

        filtered_df_i2r = deg_nec_suff(CF_i2r, inp_i, f_inp_i, clf, num_features,
                                       r2i=False, deg_thresh=0.00, datatype='Dice',
                                       filter_supersets=True, filter_cost=True)
        # based on which string rep is wanted
        # can use CF_i2r_raw_text = CF_i2r_raw_text

        recall_score = \
            recall_nec_score(CF_i2r, filtered_df_i2r, f_inp_i, r2i=False)

        # compute cost comparison
        our_mean_cost = np.mean(filtered_df_i2r['cost'])
        our_min_cost = np.min(filtered_df_i2r['cost'])
        our_max_cost = np.max(filtered_df_i2r['cost'])

        dice_mean_cost, dice_min_cost, dice_max_cost, \
        dice_df = get_Dice_cands_cost(d=d, exp=exp,
                                      inp_raw=inp_i_raw.iloc[:, :num_features].to_dict('records')[0],
                                      inp_listed=inp_i, num_features=num_features,
                                      categorical_cols=categorical_cols_names,
                                      cont_cols=cont_cols_names,
                                      total_CFs=len(filtered_df_i2r),
                                      col_con_inds=col_con, col_cat_inds=col_cat,
                                      MAD_features_cost=MAD_features_cost)

        our_cands, dice_cands = valid_cands_count(filtered_df_i2r, dice_df, inp_i, num_features)

        results_dict['our_mean_cost'].append(our_mean_cost)
        results_dict['our_min_cost'].append(our_min_cost)
        results_dict['our_max_cost'].append(our_max_cost)
        results_dict['dice_mean_cost'].append(dice_mean_cost)
        results_dict['dice_min_cost'].append(dice_min_cost)
        results_dict['dice_max_cost'].append(dice_max_cost)
        results_dict['recall_score'].append(recall_score)
        results_dict['our_dfs'].append(filtered_df_i2r)
        results_dict['dice_dfs'].append(dice_df)
        results_dict['our_val_cands'].append(our_cands)
        results_dict['dice_val_cands'].append(dice_cands)
        results_dict['num_our_val_cands'].append(len(our_cands))
        results_dict['num_dice_val_cands'].append(len(dice_cands))
        results_dict['inps'].append(inp_i)
        results_dict['inps_raw'].append(inp_i_raw)
    return results_dict


def check_superset_sub(sub, compare, superset_check=True):
    if superset_check:
        return len(set(sub).intersection(set(compare))) == len(compare)
    else:
        return len(set(compare).intersection(set(sub))) == len(sub)


def find_cands(df, compare_df):
    cands = []
    for ind, row in df.iterrows():
        found_subsets_costs = []
        is_subset = False
        for ind2, row2 in compare_df.iterrows():
            # check if row superset of row2
            if check_superset_sub(row['subset'], row2['subset']):
                found_subsets_costs.append(row2['cost'])
        # if not a superset and not a subset, add to cands
        if len(found_subsets_costs) == 0:
            cands.append(row)
        # if superset of some subsets, check if better cost
        elif len(found_subsets_costs) != 0:
            if row['cost'] < np.min(found_subsets_costs):
                cands.append(row)
    return cands


def valid_cands_count(filtered_df_i2r, dice_df, inp, num_features):
    subsets = []
    indecies = np.transpose((inp.values[:, :num_features] !=
                             dice_df.values[:, :num_features]).nonzero())
    indecies[indecies[:, 0] == 0][:, 1]
    for i in range(len(dice_df)):
        subsets.append(indecies[indecies[:, 0] == i][:, 1])

    dice_df['subset'] = subsets

    dice_cands = find_cands(dice_df, filtered_df_i2r)
    our_cands = find_cands(filtered_df_i2r, dice_df)
    return our_cands, dice_cands


def get_Dice_cands_cost(d, exp, inp_raw, inp_listed, num_features,
                        categorical_cols, cont_cols, total_CFs,
                        col_con_inds, col_cat_inds, MAD_features_cost):
    dice_exp = \
        exp.generate_counterfactuals(inp_raw, total_CFs=total_CFs,
                                     desired_class="opposite")
    exp_df = dice_exp.final_cfs_df
    exp_df_list_of_dict = exp_df.to_dict('records')

    prepared_df_exp = pd.DataFrame([])

    for row in exp_df_list_of_dict:
        prepared_df_exp = pd.concat((prepared_df_exp,
                                     d.prepare_query_instance(row, encode=True)))

    prepared_df_listed = pd.DataFrame([])

    for cat in categorical_cols:
        prepared_df_listed[cat] = list(prepared_df_exp.filter(regex=cat).values)
        prepared_df_listed[cat] = prepared_df_listed[cat].apply(lambda x: str(list(x)))

    for cont in cont_cols:
        prepared_df_listed[cont] = prepared_df_exp[cont].values

    prepared_df_listed = prepared_df_listed[inp_raw.keys()]
    prepared_df_listed['cost'] = \
        prepared_df_listed.apply(lambda x: cost(x.values, inp_listed.values[0][:num_features], MAD_features_cost,
                                                col_con_inds, col_cat_inds), axis=1)
    return np.mean(prepared_df_listed['cost']), \
           np.min(prepared_df_listed['cost']), \
           np.max(prepared_df_listed['cost']), \
           prepared_df_listed


def get_pred(clf, X, datatype='Tabular'):
    if datatype == 'Text':
        hstacked = np.hstack(np.hstack(X)).reshape(len(X), -1)
        preds = clf.predict(hstacked)

    elif datatype == 'Dice':
        hstacked = np.hstack(np.hstack(X)).reshape(len(X), -1)
        preds = (clf.predict(hstacked) >= 0.5) * 1.

    else:
        preds = clf.predict(X)

    return preds


def recourse_suff_nec_pipeline(CF_condition, inp, f_inp,
                               clf, dataset, num_features,
                               datatype='Tabular', inp_raw=None,
                               dataset_raw=None, dataset_flattened=None,
                               refs=None, n_sample=None,
                               MAD_features_cost=None):
    if refs is None:
        dataset_rel_subset = \
            dataset[CF_condition]
        n = len(dataset_rel_subset)
        if n_sample is None:
            refs = dataset_rel_subset.sample(n=n, random_state=42)
        else:
            refs = dataset_rel_subset.sample(n=n_sample, random_state=42)

    CF_i2r = create_CF(inp, refs, clf, num_features, r2i=False, datatype=datatype,
                       MAD_features_cost=MAD_features_cost)
    if dataset_raw is not None:
        refs_raw_text = dataset_raw.loc[refs.index]

        CF_i2r_raw_text = create_CF(inp_raw, refs_raw_text, clf, num_features,
                                    r2i=False, datatype=datatype, raw_text=True)

        return CF_i2r, CF_i2r_raw_text, refs
    if dataset_flattened is not None:
        refs_flattened = dataset_flattened.loc[refs.index]
        return CF_i2r, refs, refs_flattened
    return CF_i2r, refs


def create_CF(inp, refs, clf, num_features, r2i=True,
              datatype='Tabular', raw_text=False,
              MAD_features_cost=None):
    CF = {'cfs': [], 'intervention_sets': [], 'cardinality': [],
          'original': [], 'cost': [], 'model_pred': []}
    intervention_order_ids = intervention_order(num_features)
    cardinalities = [len(eval(x)) for x in intervention_order_ids]

    # Keep track of col types for cost computation
    col_con = [i for i in range(len(inp.columns[:num_features]))
               if (isinstance(inp.iloc[0, i], int) | isinstance(inp.iloc[0, i], float))]
    col_cat = list(set(range(len(inp.columns[:num_features]))) - set(col_con))

    # construction of interventions
    for ind, ref in refs.iterrows():
        if r2i:
            cfs_ref = list(itertools.product(*zip(np.array(inp)[0][:num_features],
                                                  np.array(ref)[:num_features])))
            CF['original'] += [str(ref.values[:num_features])] * (2 ** num_features)
        else:
            cfs_ref = list(itertools.product(*zip(np.array(ref)[:num_features],
                                                  np.array(inp)[0][:num_features])))
            CF['original'] += [str(inp.values[0][:num_features])] * (2 ** num_features)
        CF['cfs'] += cfs_ref

        # for raw text, just interested in text rep of possible interventions
        if not raw_text:
            # otherwise, compute model preds, cost, etc.
            # cost computation
            if r2i:
                costs = cost(cfs_ref, ref.values[:num_features],
                             MAD_features_cost, col_con, col_cat,
                             l1_MAD=True, datatype=datatype)
            else:
                costs = cost(cfs_ref, inp.values[0][:num_features],
                             MAD_features_cost, col_con, col_cat,
                             l1_MAD=True, datatype=datatype)
            CF['cost'] += list(costs)

            # mark intervention targets
            CF['intervention_sets'] += intervention_order_ids
            CF['cardinality'] += cardinalities
            # obtain model prediction for CFs

    CF_df = pd.DataFrame(CF['cfs'], columns=inp.columns[:num_features])
    if not raw_text:
        CF_df['Original'] = CF['original']
        CF_df['Intervention_index'] = CF['intervention_sets']
        CF_df['Cost'] = CF['cost']
        CF_df['Cardinality'] = CF['cardinality']

    return CF_df


def deg_nec_suff(CF_df, inp, f_inp, clf, num_features,
                 r2i=True, CF_i2r_raw_text=None,
                 deg_thresh=0.7, datatype='Tabular',
                 filter_supersets=False, filter_cost=False,
                 pred_on_fly=False, max_output=-1):
    # degrees computation
    subsets = all_choices(list(range(num_features)))[1:]
    deg_dict = {}
    saved_subsets = 0
    if not pred_on_fly:
        if r2i:
            x_f_inp_f = CF_df['Model_pred'] == f_inp[0]
        else:
            x_f_ref_f = CF_df['Model_pred'] != f_inp[0]

    for subset in subsets:  # for each Subset S s.t. X_S = inp_S
        if saved_subsets == max_output:
            break
        else:
            subset_interventions_id = CF_df.Intervention_index == str(subset)
            min_cost_subsets = np.inf
            if filter_supersets:
                keys = list(map(lambda x: np.array(eval(x)), deg_dict.keys()))
                # helper lambda function for filtering
                is_superset = lambda x: (len(set(x).intersection(set(subset))) == len(x))
                if np.array(list(map(is_superset, keys))).any():
                    if filter_cost:
                        subsets_inds = list(np.array(list(map(is_superset, keys))).nonzero()[0])
                        relevant_keys = list(np.take(np.array(list(deg_dict.keys())), subsets_inds))
                        subset_cost = np.min(CF_df[subset_interventions_id]['Cost'])
                        if len(relevant_keys) > 1:
                            min_cost_subsets = np.min(
                                np.array(itemgetter(*relevant_keys)(deg_dict), dtype=object)[:, -1])
                        else:
                            min_cost_subsets = np.min(np.array(itemgetter(*relevant_keys)(deg_dict), dtype=object)[-1])
                        if subset_cost > min_cost_subsets:
                            continue
                    else:
                        continue
            if not pred_on_fly:
                if r2i:
                    x_s_inp_s = \
                        CF_df['Intervention_index'] == str(subset)
                    s_count = sum(x_s_inp_s)  # compute empirical marginal (maximal) P_{CF}(S=s)
                    if s_count > 0:
                        s_count_o_count = sum(
                            x_s_inp_s & x_f_inp_f)  # compute empirical joint P_{CF}(X_s=inp_s, F(x)=F(inp))
                        degree_of_suff_sub = \
                            s_count_o_count / s_count  # deg of suff = P_{CF}(F(x)=F(inp)|X_s=inp_s)
                    else:
                        degree_of_suff_sub = 0

                    subset_cost = np.mean(CF_df[x_s_inp_s]['Cost'])
                    # condition block meant to ensure minimality
                    if filter_cost:
                        if subset_cost < min_cost_subsets:
                            if degree_of_suff_sub >= deg_thresh:
                                deg_dict[str(subset)] = \
                                    (float(degree_of_suff_sub), subs_to_str(subset, inp),
                                     CF_df[x_s_inp_s]['Cardinality'][0], subset_cost)
                                saved_subsets += 1
                        else:
                            continue
                    else:
                        if degree_of_suff_sub >= deg_thresh:
                            deg_dict[str(subset)] = \
                                (float(degree_of_suff_sub), subs_to_str(subset, inp),
                                 CF_df[x_s_inp_s]['Cardinality'][0], subset_cost)
                            saved_subsets += 1
                # i2r
                else:
                    degree_i2r_sub = 0
                    x_s_ref_s = \
                        CF_df['Intervention_index'] == str(subset)
                    s_count = sum(x_s_ref_s)
                    if s_count > 0:
                        s_count_o_count = sum(
                            x_s_ref_s & x_f_ref_f)  # compute empirical joint P_{CF}(X_s=ref_s, F(x)=F(ref))
                        degree_i2r_sub = \
                            s_count_o_count / s_count  # deg of suff = P_{CF}(F(x)=F(ref)|X_s=ref_s)
                    # this is just for grabbing the string rep. of the best cost ref
                    # with subset intervention that also lead to a win.
                    subset_applied_and_won = CF_df[x_s_ref_s & x_f_ref_f].copy()
                    if degree_i2r_sub != 0:
                        min_cost_ind_subset_and_win = \
                            subset_applied_and_won.Cost.idxmin()
                        if CF_i2r_raw_text is not None:
                            ref_values = \
                                CF_i2r_raw_text.loc[min_cost_ind_subset_and_win][inp.columns[:num_features]]
                        else:
                            ref_values = \
                                subset_applied_and_won.loc[min_cost_ind_subset_and_win][inp.columns[:num_features]]
                        string_rep = subs_to_str(subset, ref_values, r2i=False)
                    # The excpet handles cases where subset never lead to a win,
                    # and thus subset_applied_won is empty
                    else:
                        string_rep = ""
                    subset_cost = subset_applied_and_won.Cost.min()

                    # condition block meant to ensure minimality
                    if filter_cost:
                        if subset_cost < min_cost_subsets:
                            if degree_i2r_sub >= deg_thresh:
                                deg_dict[str(subset)] = \
                                    (float(degree_i2r_sub), string_rep,
                                     CF_df[x_s_ref_s]['Cardinality'].values[0],
                                     subset_cost)
                                saved_subsets += 1
                        else:
                            continue
                    else:
                        if degree_i2r_sub >= deg_thresh:
                            deg_dict[str(subset)] = \
                                (float(degree_i2r_sub), string_rep,
                                 CF_df[x_s_ref_s]['Cardinality'].values[0],
                                 subset_cost)
                            saved_subsets += 1

            else:
                X = CF_df[subset_interventions_id].iloc[:, :num_features].values
                if datatype == 'Text':
                    hstacked = np.hstack(np.hstack(X)).reshape(len(X), -1)
                    preds = clf.predict(hstacked)

                elif datatype == 'Dice':
                    hstacked = np.hstack(np.hstack(X)).reshape(len(X), -1)
                    preds = ((clf.predict(hstacked) >= 0.5) * 1.)

                else:
                    preds = clf.predict(X)

                if r2i:
                    # compute empirical joint P_{CF}(X_s=inp_s, F(x)=F(inp))
                    if isinstance(f_inp, (int, float)):
                        s_count_o_count = sum(preds == f_inp)
                    else:
                        s_count_o_count = sum(preds == f_inp[0])

                    s_count = len(X)  # compute empirical marginal (maximal) P_{CF}(S=s)

                    if s_count > 0:
                        degree_of_suff_sub = \
                            s_count_o_count / s_count  # deg of suff = P_{CF}(F(x)=F(inp)|X_s=inp_s)
                    else:
                        degree_of_suff_sub = 0

                    subset_cost = np.mean(CF_df[subset_interventions_id]['Cost'])
                    if filter_supersets:
                        if degree_of_suff_sub >= deg_thresh:
                            if filter_cost:
                                if subset_cost < min_cost_subsets:
                                    deg_dict[str(subset)] = \
                                        (float(degree_of_suff_sub), subs_to_str(subset, inp),
                                         len(subset), subset_cost)
                                    saved_subsets += 1
                                else:
                                    continue
                            else:
                                deg_dict[str(subset)] = \
                                    (float(degree_of_suff_sub), subs_to_str(subset, inp),
                                     len(subset), subset_cost)
                                saved_subsets += 1

                # i2r
                else:
                    degree_i2r_sub = 0
                    s_count = len(X)
                    if s_count > 0:
                        if isinstance(f_inp, (int, float)):
                            x_f_ref_f = preds != f_inp
                        else:
                            x_f_ref_f = preds != f_inp[0]
                        s_count_o_count = sum(x_f_ref_f)  # compute empirical joint P_{CF}(X_s=ref_s, F(x)=F(ref))
                        degree_i2r_sub = \
                            s_count_o_count / s_count  # deg of suff = P_{CF}(F(x)=F(ref)|X_s=ref_s)
                    # this is just for grabbing the string rep. of the best cost ref
                    # with subset intervention that also lead to a win.
                    subset_applied_and_won = CF_df[subset_interventions_id][x_f_ref_f].copy()
                    if degree_i2r_sub != 0:
                        min_cost_ind_subset_and_win = \
                            subset_applied_and_won.Cost == subset_applied_and_won.Cost.min()
                        if CF_i2r_raw_text is not None:
                            ref_values = \
                                CF_i2r_raw_text.loc[min_cost_ind_subset_and_win.index]. \
                                    iloc[0][inp.columns[:num_features]]
                        else:
                            ref_values = \
                                subset_applied_and_won[min_cost_ind_subset_and_win]. \
                                    iloc[0][inp.columns[:num_features]]
                        string_rep = subs_to_str(subset, ref_values, r2i=False)
                    # The excpet handles cases where subset never lead to a win,
                    # and thus subset_applied_won is empty
                    else:
                        string_rep = ""

                    subset_cost = np.min(CF_df[subset_interventions_id][x_f_ref_f]['Cost'])
                    if degree_i2r_sub >= deg_thresh:
                        if filter_cost:
                            if subset_cost < min_cost_subsets:
                                deg_dict[str(subset)] = \
                                    (float(degree_i2r_sub), string_rep,
                                     CF_df[CF_df['Intervention_index'] ==
                                           str(subset)]['Cardinality'].values[0],
                                     subset_cost)
                                saved_subsets += 1
                            else:
                                continue
                        else:
                            deg_dict[str(subset)] = \
                                (float(degree_i2r_sub), string_rep,
                                 CF_df[CF_df['Intervention_index'] ==
                                       str(subset)]['Cardinality'].values[0], subset_cost)
                            saved_subsets += 1
    sub_df = pd.DataFrame.from_dict(deg_dict, orient='index',
                                    columns=["degree", "string",
                                             "cardinality", "cost"]).reset_index()

    return sub_df


def recall_nec_score(CF_input, sub_df_filtered, f_inp, r2i=True):
    CF = CF_input.copy()
    sub_df_filtered.rename(columns={"index": "subset"}, inplace=True)
    sub_df_filtered["subset"] = sub_df_filtered["subset"].apply(lambda x: np.array(eval(x)))
    CF['Intervention_index'] = CF['Intervention_index'].apply(lambda x: np.array(eval(x)))

    if r2i:
        win_inds = CF['Model_pred'] == f_inp[0]
    else:
        win_inds = CF['Model_pred'] != f_inp[0]
    if len(sub_df_filtered) > 0:
        for i, subset in enumerate(sub_df_filtered['subset']):
            current_supersets = CF.Intervention_index.apply(lambda x:
                                                            len(set(x).intersection(set(subset)))
                                                            == len(set(subset)))
            if i == 0:
                all_supsersets = current_supersets
            else:
                all_supsersets = all_supsersets | current_supersets
        deg_cum_nec = sum(all_supsersets & win_inds) / sum(win_inds)
    else:
        deg_cum_nec = 0

    return deg_cum_nec


def plt_cost_comp(results_comparison, type='mean'):
    sns.set_style("whitegrid")
    BIGGER_SIZE = 16
    plt.rc('font', size=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE)
    plt.rc('ytick', labelsize=BIGGER_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
    fig, ax = plt.subplots()
    if type == 'mean':
        key_cost = 'our_mean_cost'
        dice_key_cost = 'dice_mean_cost'
    elif type == 'min':
        key_cost = 'our_min_cost'
        dice_key_cost = 'dice_min_cost'
    else:
        key_cost = 'our_max_cost'
        dice_key_cost = 'dice_max_cost'
    plt.scatter(results_comparison[key_cost], results_comparison[dice_key_cost], s=90, color='darkorange',
                edgecolors='slateblue', linewidth=2, alpha=0.7)
    max_val = max(np.max(results_comparison[key_cost]), np.max(results_comparison[dice_key_cost])) + 1
    plt.xlim(-0.5, max_val)
    plt.ylim(-0.5, max_val)
    ax.axline([0, 0], [max_val, max_val], linestyle='--', color='black')
    plt.xlabel('LENS cost')
    plt.ylabel('DiCE cost')
    Path("../notebooks/out/").mkdir(parents=True, exist_ok=True)
    plt.savefig("../notebooks/out/recourse_{}_cost_comp.png".format(type), dpi=200, bbox_inches='tight')


def plt_valid_cand(results_comparison):
    sns.set_style("whitegrid")
    BIGGER_SIZE = 16
    plt.rc('font', size=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE)
    plt.rc('ytick', labelsize=BIGGER_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
    jitter_divide = 3
    fig, ax = plt.subplots()
    plt.scatter(np.array(results_comparison['num_our_val_cands']) + (
            np.random.rand(len(results_comparison['num_our_val_cands'])) / jitter_divide),
                np.array(results_comparison['num_dice_val_cands']) + (
                        np.random.rand(len(results_comparison['num_dice_val_cands'])) / jitter_divide), s=100,
                color='darkorange', edgecolors='slateblue', linewidth=2, alpha=0.5)
    max_val = max(np.max(results_comparison['num_our_val_cands']), np.max(results_comparison['num_dice_val_cands'])) + 1
    plt.xlim(-0.5, max_val)
    plt.ylim(-0.5, max_val)
    ax.axline([0, 0], [max_val, max_val], linestyle='--', color='black')
    plt.xlabel('# our valid cands')
    plt.ylabel('# dice val cands')
    Path("../notebooks/out/").mkdir(parents=True, exist_ok=True)
    plt.savefig("../notebooks/out/recourse_compare_valid_cands.png", dpi=200, bbox_inches='tight')
