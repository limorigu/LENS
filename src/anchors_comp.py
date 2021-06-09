from anchor import anchor_tabular
from necsuf_tabular_text import deg_nec_suff, suff_nec_pipeline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def anchors_card_prec_comp(dataset, clf, num_features, CF_condition, num_inp=5, deg_thresh=0.9):
    all_inp_deg = {}
    anchors_df = pd.DataFrame({'Achor': [], 'Cardinality': [], 'Precision': []})
    # To not get empty factors ranking high on sufficiency
    relevant_dataet = dataset[dataset.outcome == dataset.Model_pred]
    for i in range(num_inp):
        # seed fixed for reproducability of demonstration, change to get random examples
        inp_i = relevant_dataet.sample(n=1, random_state=42 + i)
        f_inp_i = clf.predict(np.array(inp_i.iloc[:, :num_features]))

        # our approach
        CF_r2i, _, refs_i = suff_nec_pipeline(eval(CF_condition),
                                              inp_i, clf, dataset,
                                              num_features, datatype='Tabular',
                                              r2i_i2r='r2i')
        CF_df_deg = deg_nec_suff(CF_r2i, inp_i, f_inp_i, clf, num_features, r2i=True,
                                 deg_thresh=deg_thresh, filter_supersets=True)
        all_inp_deg[i] = CF_df_deg

        # anchors
        explainer = anchor_tabular.AnchorTabularExplainer(
            {}, dataset.columns[:num_features],
            dataset.values[:, :num_features],
            {})

        exp = explainer.explain_instance(inp_i.values[:, :num_features], clf.predict, threshold=deg_thresh)
        anchors_new_row = pd.Series(
            {"Achor": exp.names(), "Cardinality": len(exp.names()), "Precision": exp.precision()})
        anchors_df = anchors_df.append(anchors_new_row, ignore_index=True)

    return all_inp_deg, anchors_df


def plt_cardinality_anchors_comp(all_inp_deg, anchors_df, num_inp, tau=0.9, type='min', shape='scatter'):
    cardinalities_anchors = list(anchors_df['Cardinality'])
    if type == 'min':
        cardinalities_ours = [np.min(all_inp_deg[i]['cardinality']) for i in range(num_inp)]
    elif type == 'mean':
        cardinalities_ours = [np.mean(all_inp_deg[i]['cardinality']) for i in range(num_inp)]
    elif type == 'max':
        cardinalities_ours = [np.max(all_inp_deg[i]['cardinality']) for i in range(num_inp)]

    sns.set_style("whitegrid")
    BIGGER_SIZE = 16
    plt.rc('font', size=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE)
    plt.rc('ytick', labelsize=BIGGER_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
    if shape == 'scatter':
        fig, ax = plt.subplots()
        plt.scatter(cardinalities_ours, cardinalities_anchors, s=90, color='darkorange',
                    edgecolors='slateblue', linewidth=2, alpha=0.7)

        max_val = max(np.max(cardinalities_ours), np.max(cardinalities_anchors)) + 1
        plt.xlim(-0.5, max_val)
        plt.ylim(-0.5, max_val)
        ax.axline([0, 0], [max_val, max_val], linestyle='--', color='black')
        plt.xlabel('LENS {} cardinalities'.format(type))
        plt.ylabel('Anchors cardinalities')
    else:
        df = pd.DataFrame({'Anchors': cardinalities_anchors, r'$\bf{LENS}$': cardinalities_ours})
        df = pd.melt(df)
        my_pal = {"Anchors": "slateblue", r'$\bf{LENS}$': "orange"}
        df.rename(columns={'variable': 'Method', 'value': 'Cardinality'}, inplace=True)
        sns.boxplot(x="Method", y="Cardinality", data=df, showfliers=False, palette=my_pal)

    Path("../notebooks/out/").mkdir(parents=True, exist_ok=True)
    plt.savefig("../notebooks/out/anchors_{}_cardinality_comp.png".format(type), dpi=200, bbox_inches='tight')


def plt_degree_anchors_comp(all_inp_deg, anchors_df, num_inp, tau=0.9, type='min', shape='scatter'):
    precision_anchors = list(anchors_df['Precision'])
    if type == 'min':
        degree_ours = [np.min(all_inp_deg[i]['degree']) for i in range(num_inp)]
    elif type == 'mean':
        degree_ours = [np.mean(all_inp_deg[i]['degree']) for i in range(num_inp)]
    elif type == 'max':
        degree_ours = [np.max(all_inp_deg[i]['degree']) for i in range(num_inp)]

    sns.set_style("whitegrid")
    BIGGER_SIZE = 16
    plt.rc('font', size=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE)
    plt.rc('ytick', labelsize=BIGGER_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
    if shape == 'scatter':
        fig, ax = plt.subplots()
        plt.scatter(degree_ours, precision_anchors, s=90, color='darkorange',
                    edgecolors='slateblue', linewidth=2, alpha=0.7)

        max_val = max(np.max(degree_ours), np.max(precision_anchors) + 0.1)
        plt.xlim(-0.1, max_val)
        plt.ylim(-0.1, max_val)
        ax.axline([0, 0], [max_val, max_val], linestyle='--', color='black')
        plt.xlabel('LENS {} sufficiency'.format(type))
        plt.ylabel('Anchors precision')
    else:
        # fig, ax = plt.subplots()
        df = pd.DataFrame({'Anchors': precision_anchors, r'$\bf{LENS}$': degree_ours})
        df = pd.melt(df)
        my_pal = {"Anchors": "slateblue", r'$\bf{LENS}$': "darkorange"}
        df.rename(columns={'variable': 'Method', 'value': 'Sufficiency/Precision'}, inplace=True)
        ax = sns.boxplot(x="Method", y="Sufficiency/Precision", data=df, showfliers=False, palette=my_pal)
        ax.axhline(tau, linestyle='--', color='black')
        # Add transparency to colors
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .7))

    Path("../notebooks/out/").mkdir(parents=True, exist_ok=True)
    plt.savefig("../notebooks/out/anchors_{}_degree_suff_comp.png".format(type), dpi=200, bbox_inches='tight')
