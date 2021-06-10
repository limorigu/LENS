Local Explanations via Necessity and Sufficiency: Unifying Theory and Practice
==============================================================================

Code to reproduce results from the [UAI 2021 paper](https://arxiv.org/abs/2103.14651). It has three subdirectories:

1. notebooks
2. src
3. datasets

All code with explanation of how to run experiments and obtain results can be found in the relevant notebooks. Names should be self-explanatory:

1. German_credit_experiment - includes all experiments on the German Credit dataset, including Precision/Recall curve (Appendix B), Feature Attribution SHAP comparison (5.1), and Anchors comparison (5.2).
2. Spam_expriment - all expriments involving the SpamAssassins dataset, namely Feature Attribution SHAP comparison (5.1), Counterfactual Adverserial example (5.3).
3. Sentiment_experiment - the brittle predictions anchors comparison (5.2).
4. Recourse_adult_experiment - includes all results from Counterfactual recourse DiCE comparison (5.3).
5. German_credit_causal_experiment - includes the causal vs. non-causal recourse comparison (5.3).

Each notebook refers to source code found in the src/ folder. See notebook for details. Most datasets and saved models are included in the datasets folder. Note that the IMDB dataset was too voluminous to be included in this folder, 
please see downloading instructions at the top of Sentiment_experiment.ipynb to rerun the results included therein.

Preprocessing steps of the SpamAssassins and Adult Income datasets are included in src/data/. 
The pip state corresponding to the setup used to run this code is in requirements.txt. 
Alternatively, see environment.yml, to be used with conda virtual environment by activating virtualenv 
```
conda env create -f environment.yml
conda activate LENS
python <script_to_run>.py
```