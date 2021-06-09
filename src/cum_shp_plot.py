import pandas as pd
import shap
import numpy as np
from necsuf_tabular_text import \
    suff_nec_pipeline, deg_nec_suff
import matplotlib.pyplot as plt


def cumul_shap_vs_us_multi_input(dataset, clf, CF_condition, feature_num,
                                 num_inp=4, datatype=None, df_flattened=None):
    all_inp_deg_r2i = pd.DataFrame([])
    all_inp_deg_i2r = pd.DataFrame([])
    all_shaps = np.array([])
    for i in range(num_inp):
        inp_i = dataset.sample(n=1, random_state=i)
        f_inp_i = dataset.loc[inp_i.index]['Model_pred'].values

        # our method
        if df_flattened is None:
            CF_r2i, CF_i2r, refs_i = suff_nec_pipeline(eval(CF_condition),
                                                       inp_i, clf, dataset,
                                                       feature_num)
        else:
            CF_r2i, CF_i2r, refs_i, refs_i_flat = \
                suff_nec_pipeline(eval(CF_condition),
                                  inp_i, clf, dataset,
                                  feature_num, datatype=datatype,
                                  dataset_flattened=df_flattened)

        CF_df_deg_suff_r2i = deg_nec_suff(CF_r2i, inp_i, f_inp_i, clf, feature_num,
                                          r2i=True, deg_thresh=0, filter_supersets=False,
                                          datatype=datatype)
        CF_df_deg_suff_i2r = deg_nec_suff(CF_i2r, inp_i, f_inp_i, clf, feature_num,
                                          r2i=False, deg_thresh=0,
                                          filter_supersets=False, datatype=datatype)
        all_inp_deg_r2i = pd.concat((all_inp_deg_r2i, CF_df_deg_suff_r2i))
        all_inp_deg_i2r = pd.concat((all_inp_deg_i2r, CF_df_deg_suff_i2r))

        # shap
        if df_flattened is None:
            model_fn = lambda x: clf.predict(np.array(x).reshape(len(x), -1))
            explainer = shap.KernelExplainer(model_fn,
                                             shap.sample(refs_i.iloc[:, :feature_num], 50))
            shap_values_single = explainer.shap_values(
                np.array(inp_i.iloc[0, :feature_num]), nsamples=1000)
        else:
            model_fn = lambda x: clf.predict(x)
            explainer = shap.KernelExplainer(model_fn,
                                             shap.sample(refs_i_flat.iloc[:, :feature_num * 50], 50))

            shap_values_single = explainer.shap_values(
                np.hstack(np.hstack(inp_i.iloc[:, :feature_num].values)), nsamples=1000).squeeze()
            # group shapley values
            rep_dim = int(len(shap_values_single) / feature_num)
            shap_values_single = [np.sum(shap_values_single
                                         [(i * rep_dim):((i + 1) * rep_dim)])
                                  for i in range(feature_num)]
        if i == 0:
            all_shaps = np.concatenate((all_shaps, shap_values_single))
        else:
            all_shaps = np.vstack((all_shaps, shap_values_single))
    return all_inp_deg_r2i, all_inp_deg_i2r, all_shaps


def mean_and_plot(all_inp_deg, all_shaps,
                  title_name='sufficiency', dataset_name='credit'):
    BIGGER_SIZE = 16
    plt.rc('font', size=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE)
    plt.rc('ytick', labelsize=BIGGER_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
    all_inp_deg.rename(columns={"index": "subset"}, inplace=True)
    all_inp_deg['subset'] = all_inp_deg['subset'].apply(lambda x: str(x))
    stats_df = all_inp_deg[['subset', 'degree', 'cardinality']].groupby(by=['subset']).agg(['mean', 'count', 'std'])
    stats_df.reset_index(inplace=True)
    stats_df['subset'] = stats_df['subset'].apply(lambda x: eval(x))

    topk_shap_order = np.array([np.argsort(a_shap) for a_shap in all_shaps])
    assert topk_shap_order.shape == (all_shaps.shape[0], all_shaps.shape[1])

    shaps_inputs_mean = []
    shaps_inputs_ci = []
    our_k_inputs_mean = []
    our_k_inputs_ci = []
    # iterate over k shap values by ranking
    for i, _ in enumerate(topk_shap_order[0]):
        shaps_inps = []
        for inp_shap in range(len(topk_shap_order)):
            cum_shap = sorted(topk_shap_order[inp_shap][:i + 1])
            # obtain degree for topk shap
            degree_shap_mean = stats_df[stats_df.subset.apply(lambda x:
                                                              np.array_equal(cum_shap, x))]['degree']['mean'].values

            shaps_inps.append(degree_shap_mean[0])
        shaps_inputs_mean.append(shaps_inps)
        # get best subset for same degree from our approach
        # note: cardinality will be the same for all inps.
        # Take the mean for convenience with the rest of df structure in this case
        relevant_subsets = stats_df[stats_df.cardinality['mean'] == (i + 1)]
        best_degree_k_cardinality = relevant_subsets['degree']['mean'].max()
        best_subest_ind = relevant_subsets['degree']['mean'].idxmax()
        best_degree_count = relevant_subsets.loc[best_subest_ind]['degree']['count']
        best_degree_std = relevant_subsets.loc[best_subest_ind]['degree']['std']
        best_degree_ci = 1.96 * best_degree_std / np.sqrt(best_degree_count)
        our_k_inputs_mean.append(best_degree_k_cardinality)
        our_k_inputs_ci.append(best_degree_ci)

    final_shaps_mean = np.mean(shaps_inputs_mean, axis=1)
    final_shaps_std = np.std(shaps_inputs_mean, axis=1)
    final_shaps_count = len(shaps_inputs_mean)
    final_shaps_ci = 1.96 * final_shaps_std / np.sqrt(final_shaps_count)

    fig, ax = plt.subplots()
    ax.plot(list(range(len(topk_shap_order[0]))), final_shaps_mean, linewidth=3, label='shaps', color='slateblue')
    ax.fill_between(list(range(len(topk_shap_order[0]))),
                    (np.array(final_shaps_mean) - np.array(final_shaps_ci)),
                    (np.array(final_shaps_mean) + np.array(final_shaps_ci)), alpha=.1, color='slateblue')
    ax.plot(list(range(len(topk_shap_order[0]))), our_k_inputs_mean, linewidth=3, label='our k', color='orange')
    ax.fill_between(list(range(len(topk_shap_order[0]))),
                    (np.array(our_k_inputs_mean) - np.array(our_k_inputs_ci)),
                    (np.array(our_k_inputs_mean) + np.array(our_k_inputs_ci)), alpha=.1, color='orange')

    # Save results to file
    dict_results = {'final_shaps_mean': final_shaps_mean, 'our_k_inputs_mean': our_k_inputs_mean,
                    'final_shaps_ci': final_shaps_ci, 'our_k_inputs_ci': our_k_inputs_ci}
    df_results = pd.DataFrame(dict_results)
    df_results.to_csv("out/{}_{}_k_cum_shap_vs_us.csv".format(dataset_name, title_name))

    plt.title("{} {} k cum shap vs. us".format(dataset_name, title_name))
    plt.xlabel("cardinality")
    plt.ylabel("degree")
    savename = "out/{}_{}_k_cum_shap_vs_us.png".format(dataset_name, title_name)
    plt.legend()
    plt.savefig(savename, dpi=200, bbox_inches='tight')
    plt.show()
