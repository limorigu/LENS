import numpy as np
import pandas as pd
from scipy import stats
import itertools
import time
import pdb
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns


def intervention_order(num_features, text_i2r=False, numeric=False):
    if text_i2r or numeric:
        intervention_inds_numeric = []
    intervention_inds = []
    for f_i in range(2 ** num_features):
        bitstr = to_bitstring(f_i, num_features)
        # Indices where where we have 0 in this bitstring correspond
        # to features where an intervention has been performed (i.e., feature is
        # assigned to input value).
        intervention_inds.append(str([i for i, b in enumerate(bitstr) if b == '0']))
        if text_i2r or numeric:
            intervention_inds_numeric.append([i for i, b in enumerate(bitstr) if b == '0'])
    if text_i2r:
        return intervention_inds, intervention_inds_numeric
    if numeric:
        return intervention_inds_numeric
    return intervention_inds


def to_bitstring(num, num_bits):
    # Returns the bitstring corresponding to (base 10) number 'num'
    bitstr = bin(num)[2:]  # 2: to remove the initial '0b'
    # Append '0's at the beginning to make it of length 'num_bits'
    return ''.join(['0' for _ in range(num_bits - len(bitstr))]) + bitstr


def close_or_distant_neighbours(df, inp, col_name, like=True, perc=0.1):
    similars = df[col_name].apply(lambda x: cosine_similarity(x.reshape(1, -1), inp[col_name].item().reshape(1, -1)))

    if like:
        chosen = similars.sort_values(ascending=False)[:int(len(similars) * perc)]
    else:
        chosen = similars.sort_values(ascending=True)[:int(len(similars) * perc)]

    return chosen.index


def create_CF(inp, refs, clf, num_features, MAD_features_cost,
              r2i=True, datatype='Tabular', raw_text=False, causal_SCM=None,
              col_con=None, col_cat=None, predict=False):
    np.random.seed(42)
    CF = {'cfs': [], 'intervention_sets': [], 'cardinality': [],
          'original': [], 'cost': [], 'model_pred': []}
    intervention_order_ids = intervention_order(num_features)
    cardinalities = [len(eval(x)) for x in intervention_order_ids]

    # Keep track of col types for cost computation
    if not col_con:
        col_con = [i for i in range(num_features)
                   if (isinstance(inp.iloc[0, i], int) | isinstance(inp.iloc[0, i], float))]
        col_cat = list(set(range(num_features)) - set(col_con))

    # construction of interventions
    for ind, ref in refs.iterrows():
        cfs_ref = []
        if r2i:
            if not causal_SCM:
                cfs_ref = list(itertools.product(*zip(np.array(inp)[0][:num_features],
                                                      np.array(ref)[:num_features])))
            else:
                for interv_ind in intervention_order_ids:
                    original_values = pd.DataFrame(ref).T
                    # This line is to prevent from applying intervention that are the same as origin. values
                    # Note that order of columns in diff_values_intervention are different than original
                    # b/c sets do not respect order, but all access in sample_from_SCM is invariant to column order
                    if interv_ind != '[]':
                        intervention_values = inp.iloc[:, eval(interv_ind)].to_dict('records')[0].items()
                        diff_values_intervention = \
                            pd.DataFrame.from_dict(dict(intervention_values -
                                                        (original_values.to_dict('records')[0].items() &
                                                         intervention_values)), orient='index').T
                    else:
                        diff_values_intervention = pd.DataFrame([])
                    cfs_ref.append(sample_from_SCM(diff_values_intervention, original_values, causal_SCM))
            CF['original'] += [str(ref.values[:num_features])] * (2 ** num_features)
        else:
            if not causal_SCM:
                cfs_ref = list(itertools.product(*zip(np.array(ref)[:num_features],
                                                      np.array(inp)[0][:num_features])))
            else:
                for interv_ind in intervention_order_ids:
                    original_values = inp
                    # This block is to prevent from applying intervention that are the same as origin. values
                    # Note that order of columns in diff_values_intervention are different than original
                    # b/c sets do not respect order, but all access in sample_from_SCM is invariant to column order
                    if interv_ind != '[]':
                        intervention_values = pd.DataFrame(ref[eval(interv_ind)]).T.to_dict('records')[0].items()
                        diff_values_intervention = \
                            pd.DataFrame.from_dict(dict(intervention_values -
                                                        (original_values.to_dict('records')[0].items() &
                                                         intervention_values)), orient='index').T
                    else:
                        diff_values_intervention = pd.DataFrame([])
                    cfs_ref.append(sample_from_SCM(diff_values_intervention, original_values, causal_SCM))
            CF['original'] += [str(inp.values[0][:num_features])] * (2 ** num_features)
        CF['cfs'] += cfs_ref

        # for raw text, just interested in text rep of possible interventions
        if not raw_text:
            # otherwise, compute model preds, cost, etc.
            # mark intervention targets
            CF['intervention_sets'] += intervention_order_ids
            CF['cardinality'] += cardinalities
            # obtain model prediction for CFs
            if predict:
                if datatype == 'Text':
                    CF['model_pred'].extend(clf.predict(
                        np.array(cfs_ref).reshape(len(cfs_ref), -1)))

                elif datatype == 'Dice':
                    hstacked = np.hstack(np.hstack(
                        np.array(cfs_ref, dtype=object))).reshape(len(cfs_ref), -1)
                    CF['model_pred'].extend((clf.predict(hstacked) >= 0.5) * 1.)

                else:
                    CF['model_pred'].extend(clf.predict(cfs_ref))

            # cost computation
            if r2i:
                if not causal_SCM:
                    costs = cost(cfs_ref, ref.values[:num_features],
                                 MAD_features_cost, col_con, col_cat,
                                 l1_MAD=True, datatype=datatype)
                else:
                    # This block is to prevent assigning cost to downstream effects of interventions
                    intervention_inds = intervention_order(num_features, numeric=True)
                    intervention_inds_len = len(intervention_inds)
                    cost_mask = np.zeros((intervention_inds_len, num_features))
                    for i, intervention in enumerate(intervention_inds):
                        cost_mask[i, intervention] = 1.
                    ref_tiled = np.tile(ref.values[:num_features], (intervention_inds_len, 1))
                    cfs_ref_masked_w_ref = np.where(cost_mask == 0, ref_tiled, np.array(cfs_ref))
                    costs = cost(cfs_ref_masked_w_ref, ref.values[:num_features],
                                 MAD_features_cost, col_con, col_cat,
                                 l1_MAD=True, datatype=datatype)
            else:
                if not causal_SCM:
                    costs = cost(cfs_ref, inp.values[0][:num_features],
                                 MAD_features_cost, col_con, col_cat,
                                 l1_MAD=True, datatype=datatype)
                else:
                    # This block is to prevent assigning cost to downstream effects of interventions
                    intervention_inds = intervention_order(num_features, numeric=True)
                    intervention_inds_len = len(intervention_inds)
                    cost_mask = np.zeros((intervention_inds_len, num_features))
                    for i, intervention in enumerate(intervention_inds):
                        cost_mask[i, intervention] = 1.
                    inp_tiled = np.tile(inp.values[0][:num_features], (intervention_inds_len, 1))
                    cfs_ref_masked_w_inp = np.where(cost_mask == 0, inp_tiled, np.array(cfs_ref))
                    costs = cost(cfs_ref_masked_w_inp, inp.values[0][:num_features],
                                 MAD_features_cost, col_con, col_cat,
                                 l1_MAD=True, datatype=datatype)

            CF['cost'] += list(costs)

    CF_df = pd.DataFrame(CF['cfs'], columns=inp.columns[:num_features])
    if not raw_text:
        CF_df['Original'] = CF['original']
        CF_df['Intervention_index'] = CF['intervention_sets']
        if predict:
            CF_df['Model_pred'] = CF['model_pred']
        CF_df['Cost'] = CF['cost']
        CF_df['Cardinality'] = CF['cardinality']

    return CF_df


# Causal model fitting and predicting
def fit_scm(dataset):
    np.random.seed(42)
    # Age and Sex are root nodes and don't need fitting

    # Job
    job_fn = RandomForestClassifier()
    job_fn.fit(np.vstack((dataset['Age'].values,
                          dataset['Sex'].values)).reshape(-1, 2),
               dataset['Job'].values)

    # Savings
    savings_fn = smf.ols(formula="Savings ~ Age + Sex + Job", data=dataset).fit()
    savings_rmse = np.sqrt(np.mean(savings_fn.resid ** 2))

    # Housing
    housing_fn = RandomForestClassifier()
    housing_fn.fit(np.vstack((dataset['Job'].values,
                              dataset['Savings'].values)).reshape(-1, 2),
                   dataset['Housing'].values)

    # Checking
    checking_fn = smf.ols(formula="Checking ~ Job + Savings", data=dataset).fit()
    checking_rmse = np.sqrt(np.mean(checking_fn.resid ** 2))

    # Credit
    credit_fn = smf.ols(formula="Credit ~ Age + Job + Housing", data=dataset).fit()
    credit_rmse = np.sqrt(np.mean(credit_fn.resid ** 2))

    # Duration
    duration_fn = smf.ols(formula="Duration ~ Credit + Savings", data=dataset).fit()
    duration_rmse = np.sqrt(np.mean(duration_fn.resid ** 2))

    # Purpose
    purpose_fn = RandomForestClassifier()
    purpose_fn.fit(np.vstack((dataset['Age'].values, dataset['Housing'].values,
                              dataset['Credit'].values, dataset['Duration'].values)).reshape(-1, 4),
                   dataset['Purpose'].values)

    return {'job_fn': job_fn, 'savings_fn': savings_fn, 'savings_rmse': savings_rmse,
            'housing_fn': housing_fn, 'checking_fn': checking_fn, 'checking_rmse': checking_rmse,
            'credit_fn': credit_fn, 'credit_rmse': credit_rmse, 'duration_fn': duration_fn,
            'duration_rmse': duration_rmse, 'purpose_fn': purpose_fn}


def sample_from_SCM(intervention_values, original_values, SCM_model, n=1):
    intervened = 0
    # Age
    if 'Age' in intervention_values.columns:
        age_SCM = intervention_values['Age'].item()
        intervened = 1
    else:
        age_SCM = original_values['Age'].item()

    # Sex
    if 'Sex' in intervention_values.columns:
        sex_SCM = intervention_values['Sex'].item()
        intervened = 1
    else:
        sex_SCM = original_values['Sex'].item()

    # Job
    if 'Job' in intervention_values.columns:
        job_SCM = intervention_values['Job'].item()
        intervened = 1
    else:
        if intervened == 0:
            job_SCM = original_values['Job'].item()
        else:
            predict_proba_job = SCM_model['job_fn'].predict_proba(
                np.vstack((age_SCM, sex_SCM)).reshape(-1, 2))
            job_SCM = np.random.choice(len(predict_proba_job.squeeze(0)),
                                       1, p=predict_proba_job.squeeze(0)).item()

    # Savings
    if 'Savings' in intervention_values.columns:
        savings_SCM = intervention_values['Savings'].item()
        intervened = 1
    else:
        if intervened == 0:
            savings_SCM = original_values['Savings'].item()
        else:
            savings_SCM = (SCM_model['savings_fn'].predict(
                exog=dict(Age=age_SCM, Sex=sex_SCM, Job=job_SCM)).item() +
                           np.random.normal(scale=SCM_model['savings_rmse'], size=n))[0]
            if savings_SCM < 0:
                savings_SCM = 0.

    # Housing
    if 'Housing' in intervention_values.columns:
        housing_SCM = intervention_values['Housing'].item()
        intervened = 1
    else:
        if intervened == 0:
            housing_SCM = original_values['Housing'].item()
        else:
            predict_proba_housing = SCM_model['housing_fn'].predict_proba(
                np.vstack((job_SCM, savings_SCM)).reshape(-1, 2))
            housing_SCM = np.random.choice(len(predict_proba_housing.squeeze(0)),
                                           1, p=predict_proba_housing.squeeze(0)).item()

    # Checking
    if 'Checking' in intervention_values.columns:
        checking_SCM = intervention_values['Checking'].item()
        intervened = 1
    else:
        if intervened == 0:
            checking_SCM = original_values['Checking'].item()
        else:
            checking_SCM = (SCM_model['checking_fn'].predict(
                exog=dict(Job=job_SCM, Savings=savings_SCM)).item() +
                            np.random.normal(scale=SCM_model['checking_rmse'], size=n))[0]
            if checking_SCM < 0:
                checking_SCM = 0

    # Credit
    if 'Credit' in intervention_values.columns:
        credit_SCM = intervention_values['Credit'].item()
        intervened = 1
    else:
        if intervened == 0:
            credit_SCM = original_values['Credit'].item()
        else:
            credit_SCM = (SCM_model['credit_fn'].predict(
                exog=dict(Age=age_SCM, Job=job_SCM, Housing=housing_SCM)).item() +
                          np.random.normal(scale=SCM_model['credit_rmse'], size=n))[0]
            if credit_SCM < 0:
                credit_SCM = 1.

    # Duration
    if 'Duration' in intervention_values.columns:
        duration_SCM = intervention_values['Duration'].item()
        intervened = 1
    else:
        if intervened == 0:
            duration_SCM = original_values['Duration'].item()
        else:
            #         x = np.vstack((credit_SCM, savings_SCM))
            duration_SCM = (SCM_model['duration_fn'].predict(
                exog=dict(Credit=credit_SCM, Savings=savings_SCM)).item() +
                            np.random.normal(scale=SCM_model['duration_rmse'], size=n))[0]
            if duration_SCM < 0:
                duration_SCM = 1

    # Purpose
    if 'Purpose' in intervention_values.columns:
        purpose_SCM = intervention_values['Purpose'].item()
    else:
        if intervened == 0:
            purpose_SCM = original_values['Purpose'].item()
        else:
            predict_proba_purpose = SCM_model['purpose_fn'].predict_proba(
                np.vstack((age_SCM, housing_SCM, credit_SCM, duration_SCM)).reshape(-1, 4))
            purpose_SCM = np.random.choice(len(predict_proba_purpose.squeeze(0)),
                                           1, p=predict_proba_purpose.squeeze(0)).item()

    SCM_list = np.array([age_SCM, sex_SCM, job_SCM, housing_SCM, savings_SCM,
                         checking_SCM, credit_SCM, duration_SCM, purpose_SCM])
    return SCM_list


def cost(intervened, original,
         MAD_features_cost, col_con, col_cat,
         l1_MAD=True, datatype='Tabular'):
    MAD_features_cost = np.array(MAD_features_cost)
    MAD_features_cost = np.where(MAD_features_cost == 0, 1, MAD_features_cost)
    num_features = len(original)
    intervened = np.array(intervened, dtype=object)
    if l1_MAD:  # from Counterfactual Explanations, Wachter et al., P.16/7

        if datatype == 'Tabular' or datatype == 'Dice':
            cost_result = 0
            if len(intervened.shape) == 1:
                axis = 0
            else:
                axis = 1
            try:
                if len(col_con) > 0:
                    con_cost_result = np.sum(np.abs(original[col_con] - intervened[:, col_con]) *
                                             (1 / MAD_features_cost[col_con]), axis=axis)
                    cost_result += con_cost_result
                if len(col_cat) > 0:
                    cat_cost_result = np.sum(original[col_cat] != intervened[:, col_cat], axis=axis)
                    cost_result += cat_cost_result
            except IndexError:
                if len(col_con) > 0:
                    con_cost_result = np.sum(np.abs(original[col_con] - intervened[col_con]) *
                                             (1 / MAD_features_cost[col_con]), axis=axis)
                    cost_result += con_cost_result
                if len(col_cat) > 0:
                    cat_cost_result = np.sum(original[col_cat] != intervened[col_cat], axis=axis)
                    cost_result += cat_cost_result
        else:
            cost_result = 0
            for feature_i in range(num_features):
                try:
                    cost_f_i = np.abs(original[feature_i] -
                                      np.vstack(intervened[:, feature_i]))

                except IndexError:
                    cost_f_i = np.abs(original[feature_i] -
                                      intervened[feature_i])
                MAD_f_i = MAD_features_cost[feature_i]
                if MAD_f_i != 0:
                    try:
                        cost_result += np.mean((cost_f_i / MAD_f_i), axis=1)
                    except np.AxisError:
                        cost_result += np.mean((cost_f_i / MAD_f_i))
                # if MAD for feature i is 0, apply no weight
                else:
                    cost_result += cost_f_i
    return cost_result


def deg_nec_suff(CF_df, inp, f_inp, clf, num_features,
                 r2i=True, CF_i2r_raw_text=None,
                 deg_thresh=0, datatype='Tabular',
                 filter_supersets=False, pred_on_fly=True):
    # degrees computation
    subsets = all_choices(list(range(num_features)))
    deg_dict = {}

    for subset in subsets:  # for each Subset S s.t. X_S = inp_S
        if filter_supersets:
            keys = list(map(lambda x: np.array(eval(x)), deg_dict.keys()))
            # helper lambda function for filtering
            is_superset = lambda x: (len(set(x).intersection(set(subset))) == len(x))
            if np.array(list(map(is_superset, keys))).any():
                continue

        subset_interventions_id = CF_df.Intervention_index == str(subset)
        X = CF_df[subset_interventions_id].iloc[:, :num_features].values
        if pred_on_fly:
            if datatype == 'Text':
                hstacked = np.hstack(np.hstack(X)).reshape(len(X), -1)
                preds = clf.predict(hstacked)

            elif datatype == 'Dice':
                hstacked = np.hstack(np.hstack(X)).reshape(len(X), -1)
                preds = (clf.predict(hstacked) >= 0.5) * 1.

            else:
                preds = clf.predict(X)

        if r2i:
            if pred_on_fly:
                if isinstance(f_inp, (int, float)):
                    x_s_inp_s = preds == f_inp
                else:
                    x_s_inp_s = preds == f_inp[0]
            else:
                if isinstance(f_inp, (int, float)):
                    x_s_inp_s = CF_df[subset_interventions_id]['Model_pred'] == f_inp
                else:
                    x_s_inp_s = CF_df[subset_interventions_id]['Model_pred'] == f_inp[0]
            s_count_o_count = sum(x_s_inp_s)  # compute empirical joint P_{CF}(X_s=inp_s, F(x)=F(inp))
            s_count = len(X)

            if s_count > 0:
                degree_of_suff_sub = \
                    s_count_o_count / s_count  # deg of suff = P_{CF}(F(x)=F(inp)|X_s=inp_s)
            else:
                degree_of_suff_sub = 0

            if degree_of_suff_sub >= deg_thresh:
                deg_dict[str(subset)] = \
                    (float(degree_of_suff_sub), subs_to_str(subset, inp),
                     len(subset), np.mean(CF_df[subset_interventions_id][x_s_inp_s]['Cost']))


        # i2r
        else:
            degree_i2r_sub = 0
            s_count = len(X)
            if s_count > 0:
                if pred_on_fly:
                    if isinstance(f_inp, (int, float)):
                        x_f_ref_f = preds != f_inp
                    else:
                        x_f_ref_f = preds != f_inp[0]
                else:
                    if isinstance(f_inp, (int, float)):
                        x_f_ref_f = CF_df[subset_interventions_id]['Model_pred'] != f_inp
                    else:
                        x_f_ref_f = CF_df[subset_interventions_id]['Model_pred'] != f_inp[0]
                s_count_o_count = sum(x_f_ref_f)  # compute empirical joint P_{CF}(X_s=ref_s, F(x)=F(ref))
                degree_i2r_sub = \
                    s_count_o_count / s_count  # deg of suff = P_{CF}(F(x)=F(ref)|X_s=ref_s)
            # this is just for grabbing the string rep. of the best cost ref
            # with subset intervention that also lead to a win.
            subset_applied_and_won = CF_df[subset_interventions_id][x_f_ref_f].copy()
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
            if degree_i2r_sub >= deg_thresh:
                deg_dict[str(subset)] = \
                    (float(degree_i2r_sub), string_rep, len(subset), subset_cost)

    sub_df = pd.DataFrame.from_dict(deg_dict, orient='index',
                                    columns=["degree", "string",
                                             "cardinality", "cost"]).reset_index()
    return sub_df


def all_choices(ind_array):
    choices = []
    for i in range(len(ind_array) + 1):
        choices.extend([list(x) for x in itertools.combinations(ind_array, i)])

    return choices


def subs_to_str(sub, inp, r2i=True):
    # pretty print subsets to feature names, helper functions
    if isinstance(sub, str):
        sub = sub.replace("[", " ").replace("]", " ")
        sub_ind = np.fromstring(sub, sep=' ')
    else:
        sub_ind = np.array(sub)
    if r2i:
        try:
            return inp.iloc[:, sub_ind.astype(int)].T.squeeze(). \
                to_string().replace("\n", ", ").replace("    ", " ")
        except AttributeError:
            return inp.iloc[:, sub_ind.astype(int)].T.squeeze('columns'). \
                to_string().replace("\n", ", ").replace("    ", " ")
    else:
        return inp.iloc[sub_ind.astype(int)]. \
            to_string().replace("\n", ", ").replace("    ", " ")


def filter_by_degree_and_overalp(degree_df, degree_thresh=0.9, subset_max_num=10):
    sub_df = degree_df.copy()
    sub_df.rename(columns={"index": "subset"}, inplace=True)
    sub_df["subset"] = sub_df["subset"].apply(lambda x: np.array(eval(x)))
    sub_df = sub_df[sub_df['degree'] > degree_thresh]
    filtering_subsets = sub_df.sort_values(by='cardinality', ascending=True)
    for f_subset in filtering_subsets['subset']:
        sub_df = sub_df[sub_df.subset.apply(lambda x: \
                                                (len(set(x).intersection(
                                                    set(f_subset))) != len(f_subset)) |
                                                (np.array_equal(f_subset, x)))]
    sub_df = sub_df.sort_values(by='cost', ascending=True)
    if len(sub_df) >= subset_max_num:
        sub_df = sub_df[:subset_max_num]
    return sub_df


def filter_by_overalp(degree_df, subset_max_num=10):
    sub_df = degree_df.copy()
    sub_df["Intervention_index"] = sub_df["Intervention_index"].apply(lambda x: np.array(eval(x)))
    filtering_subsets = sub_df.sort_values(by='Cardinality', ascending=True)
    for f_subset in filtering_subsets['Intervention_index']:
        sub_df = sub_df[sub_df.Intervention_index.apply(lambda x: \
                                                            (len(set(x).intersection(
                                                                set(f_subset))) != len(f_subset)) |
                                                            (np.array_equal(f_subset, x)))]
    sub_df = sub_df.sort_values(by='Cost', ascending=True)
    if len(sub_df) > subset_max_num:
        sub_df = sub_df[:subset_max_num]
    return sub_df


def recall_nec_score(CF_input, sub_df_filtered, f_inp, r2i=True):
    CF = CF_input.copy()
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


def suff_nec_pipeline(CF_condition, inp,
                      clf, dataset, num_features,
                      datatype='Tabular', inp_raw=None,
                      dataset_raw=None, dataset_flattened=None,
                      refs=None, n_sample=None,
                      causal_SCM=None, col_con=None,
                      col_cat=None, predict=False,
                      r2i_i2r='both'):
    if refs is None:
        dataset_rel_subset = \
            dataset[CF_condition]
        n = len(dataset_rel_subset)
        if n_sample is None:
            refs = dataset_rel_subset.sample(n=n, random_state=42)
        else:
            refs = dataset_rel_subset.sample(n=n_sample, random_state=42)
    if datatype == 'Text':
        MAD_features_cost = [stats.median_abs_deviation(
            np.mean(dataset.iloc[:, f_i]))
            for f_i in range(num_features)]
    elif datatype == 'Dice':
        types = refs.dtypes[:num_features]
        MAD_features_cost = [stats.median_abs_deviation(dataset[col].values)
                             if dtype != 'object' else 1 for
                             (col, dtype) in types.iteritems()]
    else:
        MAD_features_cost = [stats.median_abs_deviation(np.unique(dataset.iloc[:, f_i]))
                             for f_i in range(num_features)]

    if r2i_i2r == 'both':
        CF_r2i = create_CF(inp, refs, clf, num_features, MAD_features_cost,
                           r2i=True, datatype=datatype, causal_SCM=causal_SCM,
                           col_con=col_con, col_cat=col_cat, predict=predict)
        CF_i2r = create_CF(inp, refs, clf, num_features, MAD_features_cost,
                           r2i=False, datatype=datatype, causal_SCM=causal_SCM,
                           col_con=col_con, col_cat=col_cat, predict=predict)
    elif r2i_i2r == 'r2i':
        CF_r2i = create_CF(inp, refs, clf, num_features, MAD_features_cost,
                           r2i=True, datatype=datatype, causal_SCM=causal_SCM,
                           col_con=col_con, col_cat=col_cat, predict=predict)
        CF_i2r = None
    elif r2i_i2r == 'i2r':
        CF_r2i = None
        CF_i2r = create_CF(inp, refs, clf, num_features, MAD_features_cost,
                           r2i=False, datatype=datatype, causal_SCM=causal_SCM,
                           col_con=col_con, col_cat=col_cat, predict=predict)
    if dataset_raw is not None:
        refs_raw_text = dataset_raw.loc[refs.index]

        CF_i2r_raw_text = create_CF(inp_raw, refs_raw_text, clf, num_features, MAD_features_cost,
                                    r2i=False, datatype=datatype, raw_text=True)
        return CF_r2i, CF_i2r, CF_i2r_raw_text, refs
    if dataset_flattened is not None:
        refs_flattened = dataset_flattened.loc[refs.index]
        return CF_r2i, CF_i2r, refs, refs_flattened
    return CF_r2i, CF_i2r, refs


def viz_df(sub_df_filtered, inp, num_features):
    restruct_dict = {colname: [] for colname in inp.columns[:num_features]}

    for i, row in sub_df_filtered.iterrows():
        subs_ind = inp.columns[row['subset']]
        for sub in subs_ind:
            restruct_dict[sub].append('X')
        N_m_sub = list(set(range(len(inp.columns[:num_features]))) - set(row['subset']))
        for complement in inp.columns[N_m_sub]:
            restruct_dict[complement].append(' ')

    return pd.DataFrame(restruct_dict)


def recourse_deg_nec_suff(CF_df, inp, f_inp, clf, num_features,
                          r2i=True, CF_i2r_raw_text=None,
                          deg_thresh=0.7, datatype='Tabular',
                          filter_supersets=False, max_output=20):
    # degrees computation
    subsets = all_choices(list(range(num_features)))
    deg_dict = {}
    saved_subsets = 0
    CF_df = CF_df.sort_values(by='Cost', ascending=True)
    # for subset in subsets:  # for each Subset S s.t. X_S = inp_S
    for subset in subsets:
        if saved_subsets == max_output:
            break
        else:
            subset_interventions_id = CF_df.Intervention_index == str(subset)
            if filter_supersets:
                keys = list(map(lambda x: np.array(eval(x)), deg_dict.keys()))
                # helper lambda function for filtering
                is_superset = lambda x: (len(set(x).intersection(set(subset))) == len(x))
                if np.array(list(map(is_superset, keys))).any():
                    continue
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
                if isinstance(f_inp, (int, float)):
                    s_count_o_count = sum(preds == f_inp)
                else:
                    s_count_o_count = sum(preds == f_inp[0])
                s_count = len(X)

                if s_count > 0:
                    degree_of_suff_sub = \
                        s_count_o_count / s_count  # deg of suff = P_{CF}(F(x)=F(inp)|X_s=inp_s)
                else:
                    degree_of_suff_sub = 0

                if degree_of_suff_sub >= deg_thresh:
                    deg_dict[str(subset)] = \
                        (float(degree_of_suff_sub), subs_to_str(subset, inp),
                         len(subset), np.mean(CF_df[subset_interventions_id]['Cost']))
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
                if degree_i2r_sub > deg_thresh:
                    deg_dict[str(subset)] = \
                        (float(degree_i2r_sub), string_rep, len(subset), subset_cost)
                    saved_subsets += 1

    sub_df = pd.DataFrame.from_dict(deg_dict, orient='index',
                                    columns=["degree", "string",
                                             "cardinality", "cost"]).reset_index()
    return sub_df


def deg_nec_suff_spec_subsets(CF_df, inp, f_inp, clf,
                              num_features, subsets,
                              r2i=True, CF_i2r_raw_text=None,
                              deg_thresh=0.7, datatype='Tabular',
                              filter_supersets=False):
    # degrees computation
    deg_dict = {}

    for subset in subsets:  # for each Subset S s.t. X_S = inp_S
        if filter_supersets:
            keys = list(map(lambda x: np.array(eval(x)), deg_dict.keys()))
            # helper lambda function for filtering
            is_superset = lambda x: (len(set(x).intersection(set(subset))) == len(x))
            if np.array(list(map(is_superset, keys))).any():
                continue

        subset_interventions_id = CF_df.Intervention_index == str(subset)
        X = CF_df[subset_interventions_id].iloc[:, :num_features]
        if datatype == 'Text':
            hstacked = np.hstack(np.hstack(X)).reshape(len(X), -1)
            preds = clf.predict(hstacked)

        elif datatype == 'Dice':
            hstacked = np.hstack(np.hstack(X)).reshape(len(X), -1)
            preds = (clf.predict(hstacked) >= 0.5) * 1.

        else:
            preds = clf.predict(X)

        if r2i:
            if isinstance(f_inp, (int, float)):
                s_count_o_count = sum(preds == f_inp)
            else:
                s_count_o_count = sum(preds == f_inp[0])
            s_count = len(X)

            if s_count > 0:
                degree_of_suff_sub = \
                    s_count_o_count / s_count  # deg of suff = P_{CF}(F(x)=F(inp)|X_s=inp_s)
            else:
                degree_of_suff_sub = 0

            if degree_of_suff_sub >= deg_thresh:
                deg_dict[str(subset)] = \
                    (float(degree_of_suff_sub), subs_to_str(subset, inp), len(subset))


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
            if degree_i2r_sub >= deg_thresh:
                deg_dict[str(subset)] = \
                    (float(degree_i2r_sub), string_rep, len(subset))

    sub_df = pd.DataFrame.from_dict(deg_dict, orient='index',
                                    columns=["degree", "string",
                                             "cardinality"]).reset_index()
    return sub_df


def prec_recall_curve(CF_df, CF_df_deg, f_inp):
    recalls = []
    threshs = [0.1 * (i) for i in range(10)]
    for thresh in threshs:
        sub_df_filtered = filter_by_degree_and_overalp(CF_df_deg, degree_thresh=thresh, subset_max_num=10)
        recalls.append(recall_nec_score(CF_df, sub_df_filtered, f_inp))

    return threshs, recalls


def plot_prec_recall(threshs, recalls, r2i=True):
    sns.set_style("whitegrid")
    BIGGER_SIZE = 16
    plt.rc('font', size=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE)
    plt.rc('ytick', labelsize=BIGGER_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
    plt.plot(threshs, recalls, color='slateblue', linewidth=3)
    if r2i:
        title_name = 'sufficiency r2i'
    else:
        title_name = 'sufficiency i2r'
    plt.xlabel("precision (sufficiency)")
    plt.ylabel("recall (necessity)")
    savename = "../notebooks/out/{}_precision_recall_curve.png".format(title_name)
    plt.savefig(savename, dpi=200, bbox_inches='tight')
    plt.show()
