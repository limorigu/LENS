import pandas as pd
#### Preprcoessing of adult dataset per DiCE, to be compatible with us. Kept here for repoducibility

def preproc_adult(d, datasetlist_dicts, dice_model, query_instance):
    prepared_df = pd.DataFrame([])
    for row in datasetlist_dicts:
        prepared_df = pd.concat((prepared_df, d.prepare_query_instance(row, encode=True)))

    modelPreds = (dice_model.predict(prepared_df.values) >= 0.5)*1.
    prepared_df['modelPred'] = modelPreds
    prepared_df.to_csv("../datasets/adult_clean.csv")


    prepared_df_listed = pd.DataFrame([])
    categorical_cols = ['workclass', 'education', 'marital_status', 'occupation', 'race', 'gender']
    cont_cols = ['age', 'hours_per_week']

    for cat in categorical_cols:
        prepared_df_listed[cat] = list(prepared_df.filter(regex=cat).values)
        prepared_df_listed[cat] = prepared_df_listed[cat].apply(lambda x: list(x))

    for cont in cont_cols:
        prepared_df_listed[cont] = prepared_df[cont].values

    prepared_df_listed = prepared_df_listed[query_instance.keys()]
    prepared_df_listed['modelPred'] = prepared_df['modelPred']
    prepared_df_listed.to_csv("../datasets/adult_clean_listed.csv")