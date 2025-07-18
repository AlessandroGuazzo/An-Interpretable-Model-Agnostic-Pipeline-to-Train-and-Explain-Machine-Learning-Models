# An-Interpretable-Model-Agnostic-Framework-to-Fairly-Train-and-Evaluate-Artificial-Intelligence-Models
Python code for the model development framework described in the paper "An Interpretable Model-Agnostic Framework to Fairly Train and Evaluate Artificial Intelligence Models"

## Instructions:

The main Python code to run the pipeline is the file `pipelineBackward.py`.

The code requires as input a dataset compatible with a Survival Analysis Framework (i.e. it must have a column with the event occurrence and one with the outcome time (or censoring).
The dataset must also have been previously processed in order to obtain several bootstrap sets. All bootstrap sets should be placed together in the same dataset to be provided as input. Specifically, a column named "boot" must be used to identify each subject (row of the dataset) with the corresponding bootstrap set. Bootstrap numbering must start from 1. Odd numbers (1, 3, 5, ...) must be associated with subjects belonging to the training portion of a given bootstrap set while even numbers (2, 4, 6, ...) must be associated with subjects belonging to the corresponding out-of-bag.

When running the code the user must add two arguments in the terminal after the name of the code.
Specifically:
- the model name (either Cox or SSVM).
- the dataset path or name.
Other arguments may be passed as named arguments. Check the beginning of `pipelineBackward.py` or run `python pipelineBackward.py --help` to see the full list.


We also provide the Jupyter file BackwardResultsAnalysis.ipynb to automatically analyze results obtained through the previous code.
Here the User attention is required in the first box to select:
- the considered model (Cox or SSVM).
- the dataset path or name.
- the name of the column in the dataset associated with the outcome occurrence.
- the name of the column in the dataset associated with the outcome time.

After running it the code shows:
- the overall bootstrap performance in the first box.
- a figure representing the performance as a function of the number of features in the third box.
- a figure representing the feature ranking based on the BRFS (features at the top are removed later and thus more informative) in the fourth box.

Please note that the user may need to change the axis limit of figures according to their results.

## Authors

- **Alessandro Guazzo** - Main author, lead developer of the codebase.
- **Isotta Trescato, Enrico Longato, Erica Tavazzi, Barbara di Camillo** - Project collaborators who provided input during the conceptualization phase.
