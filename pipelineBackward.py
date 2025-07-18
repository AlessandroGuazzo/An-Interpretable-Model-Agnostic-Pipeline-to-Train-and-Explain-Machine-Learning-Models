import pandas as pd
import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import GridSearchCV
from scipy.stats import loguniform
import pickle
import warnings
import os
import argparse
import inspect


def run_pipeline_backward(
        model_name,  # name of the model to be used (either Cox or SVM)
        data_name,   # dataset path or name

        outcome_name = "outcome",   # name of the column in the dataset associated with the outcome occurrence
        time_name = "outcome_time", # name of the column in the dataset associated with the outcome time
        jobs = 16,                  # number of jobs to parallelize the code
        nrnd = 500,                 # number of random search samples to be used in the hyperparameter optimization
    ):
        

    # Load data 
    data = pd.read_csv(data_name)
    roN_boot = data['boot'].max()/2

    # String with modelName_dataType (progression or baseline)
    savestr = model_name

    # load old result if present to continue optimisation or start a new one
    if os.path.isfile("backward_optimisation_results_" + savestr + ".p"):
        final_results = pickle.load(open("backward_optimisation_results_" + savestr + ".p", "rb"))
        final_covariates = pickle.load(open("backward_worst_covariates_" + savestr + ".p", "rb"))
        final_val_predictions = pickle.load(open("backward_OOB_predictions_" + savestr + ".p", "rb"))
        n = len(final_results) + 1
        k = n*2 - 1
        print("\nSuccessfully loaded old results!!\n")

    else:
        final_results = []
        final_covariates = []
        final_val_predictions = []
        n = 1
        k = 1
        print("\nNew optimisation started\n")

    # Define hyperparameters space for the model
    if (model_name == "Cox"):
        al = loguniform.rvs(a=1e-6,b=1e3,size = nrnd)
        space = dict()
        space['alphas'] = [[v] for v in al]
    else:
        # ... Insert here your space
        space = dict()
        space['alpha'] = loguniform.rvs(a=1e-5,b=1e5,size = nrnd)

    while k <= roN_boot*2: # reduce to try if everything works if needed
        
        print("\nBoot iteration: " + str(n) + "\n")

        train_data = data.iloc[np.where(data["boot"]==k)]
        val_data = data.iloc[np.where(data["boot"]==k+1)]

        del train_data["boot"]
        del val_data["boot"]

        train_labels = train_data[[outcome_name,time_name]]
        del train_data[outcome_name]
        del train_data[time_name]

        val_labels = val_data[[outcome_name,time_name]]
        del val_data[outcome_name]
        del val_data[time_name]

        x_train = np.asarray(train_data)
        x_val = np.asarray(val_data)

        # Model specific outcome/data manipulation
        if (model_name == "Cox"):
            
            y_train = np.zeros(train_labels.shape[0], dtype={'names':('event', 'T'), 'formats':('?', 'f8')})
            y_train["event"] = train_labels[outcome_name]
            y_train["T"] = train_labels[time_name]

            y_val = np.zeros(val_labels.shape[0], dtype={'names':('event', 'T'), 'formats':('?', 'f8')})
            y_val["event"] = val_labels[outcome_name]
            y_val["T"] = val_labels[time_name]
        else:
            # add code if needed
            y_train = np.zeros(train_labels.shape[0], dtype={'names':('event', 'T'), 'formats':('?', 'f8')})
            y_train["event"] = train_labels[outcome_name]
            y_train["T"] = train_labels[time_name]

            y_val = np.zeros(val_labels.shape[0], dtype={'names':('event', 'T'), 'formats':('?', 'f8')})
            y_val["event"] = val_labels[outcome_name]
            y_val["T"] = val_labels[time_name]

        boot_result = []
        allCovariates = np.asarray(range(0,x_train.shape[1]))
        worstCovariates = []
        val_predictions = []

        # Full model

        # Change here model to be used
        if (model_name == "Cox"):
            model = CoxnetSurvivalAnalysis(max_iter=1000, tol=1e-4, l1_ratio=1e-16)
            search = GridSearchCV(model, space, n_jobs=jobs, cv = 5)
            search_result = search.fit(x_train[:,allCovariates].reshape(-1,len(allCovariates)), y_train)
        else:
            #insert here your model
            model = FastSurvivalSVM(max_iter=100, tol=1e-4, optimizer="avltree", rank_ratio = 0, random_state = 117)
            search = GridSearchCV(model, space, n_jobs=jobs, cv = 5)
            search_result = search.fit(x_train[:,allCovariates].reshape(-1,len(allCovariates)), y_train)

        # Change here for model predictions and output savefile 
        
        if (model_name == "Cox"):
            val_prediction = search_result.best_estimator_.predict(x_val[:,allCovariates].reshape(-1,len(allCovariates)),search_result.best_params_["alphas"][0])
            val_ci = concordance_index_censored(y_val["event"],y_val["T"],val_prediction)
            boot_result.append([np.sum(search_result.best_estimator_.coef_!=0),search_result.best_score_,val_ci[0],search_result.best_params_["alphas"][0]])    
        else:
            # insert here your model predictions on x_val[:,bestCovariates].reshape(-1,len(bestCovariates))
            # C-Index on y_val
            # savefile with format [int counting number of considered features (1,2,3...), medium cross validation C-Index as returned by gridsearch, C-Index computed on the OOB, best hyperparameters as returned by gridsearch]
            val_prediction = search_result.best_estimator_.predict(x_val[:,allCovariates].reshape(-1,len(allCovariates)))
            val_ci = 1- concordance_index_censored(y_val["event"],y_val["T"],val_prediction)[0]
            boot_result.append([np.sum(search_result.best_estimator_.coef_!=0),search_result.best_score_,val_ci,search_result.best_params_["alpha"]])

        val_predictions.append(val_prediction)

        # Backward elimination
        while len(allCovariates) > 1:
            
            metrics = []
            fs_result = []
            currentCovariates = allCovariates.copy()
            
            for i in range(0,len(allCovariates)):
                
                currentCovariates = np.delete(allCovariates,i)
                
                # Change here model to be used
                if (model_name == "Cox"):
                    model = CoxnetSurvivalAnalysis(max_iter=1000, tol=1e-4, l1_ratio=1e-16)
                    search = GridSearchCV(model, space, n_jobs=jobs, cv = 5)
                    search_result = search.fit(x_train[:,currentCovariates].reshape(-1,len(currentCovariates)), y_train)
                else:
                    #insert here your model
                    model = FastSurvivalSVM(max_iter=100, tol=1e-4, optimizer="avltree", rank_ratio = 0, random_state = 117)
                    search = GridSearchCV(model, space, n_jobs=jobs, cv = 5)
                    search_result = search.fit(x_train[:,currentCovariates].reshape(-1,len(currentCovariates)), y_train)
                
                metrics.append(search_result.best_score_)
                fs_result.append(search_result)

            worstCovariates.append(allCovariates[metrics.index(max(metrics))])
            allCovariates = np.delete(allCovariates,metrics.index(max(metrics)))
            fs_result = fs_result[metrics.index(max(metrics))]

            # Change here for model predictions and output savefile 
            if (model_name == "Cox"):
                val_prediction = fs_result.best_estimator_.predict(x_val[:,allCovariates].reshape(-1,len(allCovariates)),fs_result.best_params_["alphas"][0])
                val_ci = concordance_index_censored(y_val["event"],y_val["T"],val_prediction)
                boot_result.append([np.sum(fs_result.best_estimator_.coef_!=0),fs_result.best_score_,val_ci[0],fs_result.best_params_["alphas"][0]])
                
            else:
                # insert here your model predictions on x_val[:,bestCovariates].reshape(-1,len(bestCovariates))
                # C-Index on y_val
                # savefile with format [int counting number of considered features (1,2,3...), medium cross validation C-Index as returned by gridsearch, C-Index computed on the OOB, best hyperparameters as returned by gridsearch]
                val_prediction = fs_result.best_estimator_.predict(x_val[:,allCovariates].reshape(-1,len(allCovariates)))
                val_ci = 1- concordance_index_censored(y_val["event"],y_val["T"],val_prediction)[0]
                boot_result.append([np.sum(fs_result.best_estimator_.coef_!=0),fs_result.best_score_,val_ci,fs_result.best_params_["alpha"]])

            val_predictions.append(val_prediction)

        final_results.append(boot_result)
        final_val_predictions.append(val_predictions)
        worstCovariates.append(allCovariates[0])
        final_covariates.append(worstCovariates)
                    
        pickle.dump(final_results, open("backward_optimisation_results_" + savestr + ".p", "wb"))
        pickle.dump(final_covariates, open("backward_worst_covariates_" + savestr + ".p", "wb"))
        pickle.dump(final_val_predictions, open("backward_OOB_predictions_" + savestr + ".p", "wb"))

        k = k + 2
        n = n + 1
        



# Utility function to generate a command line interface from a function signature
def generate_cli_from_function(func):
    parser = argparse.ArgumentParser(description=f"CLI for {func.__name__}")

    # Extract parameters comments from function signature
    signature_with_comments = inspect.getsource(func).split("):", 1)[0].split("(", 1)[1]
    comments_dict = {}
    for line in signature_with_comments.strip().split('\n'):
        if not "#" in line: continue
        line, comment = line.split("#", 1)
        name = line.split("=")[0].strip().replace(",", "")
        comment = comment.strip()
        comments_dict[name] = comment

    # Inspect function signature
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
        default_value = param.default if param.default != inspect.Parameter.empty else None
        comment = "" if name not in comments_dict else comments_dict[name]        
        if param_type == bool:
            parser.add_argument(f"--{name}", action="store_true", help=f"{comment} (flag)")
        else:
            if default_value is None:
                parser.add_argument(name, type=param_type, help=comment)
            else:
                parser.add_argument(f"--{name}", type=param_type, default=default_value, help=f"{comment} (default: '{default_value}')")

    return parser



# If called from command line, run the script with command line arguments
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(seed=7)

    parser = generate_cli_from_function(run_pipeline_backward)
    args = parser.parse_args()
    run_pipeline_backward(**vars(args))


