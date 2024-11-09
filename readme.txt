Subfolder for the reconstruction of missing SST from measurements using machine learning.
Data can be found in /datasets, and weights in /weights.

baselineCreation contains snippets of code used for the creation of the baseline (daily climatology).

testUtils contains custom callbacks created by Alessandro Testa, used for evaluating the model during training.

performanceEvaluation can be used to get the final data of a model, averaging multiple validations. It also tests dincae data for comparison.
