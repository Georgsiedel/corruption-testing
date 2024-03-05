This is the code built further from the paper:
**"Investigating the Corruption Robustness of Image Classifiers with Random Lp-norm Corruptions"** ([ArXiv](https://arxiv.org/abs/2305.05400))

Here, we train and evaluate (corruption) robust image classification models with various well known data augmentation and related methods. One core concept of this repo is the use of p-norm noise injections, especially combining different p-norm noise for the training process. The noise sampling can be found in experiments/noise.py.

run_exp.py is the test execution that calls train.py and eval.py modules from the experiments folder and allows to execute multiple experiments and runs of the same experiments successively.

run_exp.py uses parameters for every experiment that are defined in a config file, stored at experiments/configs. Name every config file with a number, e.g. config0.py and place the number "0" in the list the first for-loop in run_exp.py iteraters through.

We use a sub-folder structure for clarity: /results/['datasetname']/['modelname']. The same modelstructure needs to be created for: /experiments/trained_models/['datasetname']/['modelname'], as empty folders are not allowed on Github. Please notice the Readme in said folder.

The model-architectures in /experiments/models (e.g. wideresnet.py) are reworked and inherit the parent class from ct_model.py to allow noise injection and mixup in the forward pass. Some model architectures may need to be adjusted to inherit from ct_model.py.

Visualizations (e.g. images with imperceptible p-norm corruptions) can be found in notebooks.
