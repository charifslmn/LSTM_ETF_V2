import pandas as pd
import pickle
import random
import numpy as np
import os
import itertools
from joblib import Parallel, delayed ,parallel_backend
from collections import defaultdict
import math
import torch.nn as nn

import random, io, base64, math
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
from IPython.display import Image, display ; import base64


from Equations_Run_Combo_V_2 import (

    run_combo_V_4,LSTM,
    TimeSeriesDataset,format_to_tensor, train_one_epoch,
    validate_one_epoch, evaluate_binary_0_1, evaluate_signed_neg1_1 ,
)



#### notice the eq below is also in the Equations_Run_Combo_V_2 file
def evaluate_binary_0_1_selective_ensemble(predicted_array_flat, actual_array_flat,do_print : bool):

    predicted_array_correction = []
    actual_array_correction = []
    actual_array_all = []
        
    for idx, (pred,act) in enumerate(zip(predicted_array_flat,actual_array_flat)):
        if not isinstance(pred, str) and not None :
            predicted_array_correction.append(pred)
            actual_array_correction.append(act)
        
        actual_array_all.append(act)

    # print(predicted_array_correction)
    # # print(predicted_array_correction)
    # # print(actual_array_correction)

    if not predicted_array_correction:      # if predicted_array_correction == [] or None:

        return {
        'accuracy': 'No Agreed Predictions',
        'precision_up': 'No Agreed Predictions',
        'recall_up': 'No Agreed Predictions',
        'precision_down': 'No Agreed Predictions',
        'recall_down': 'No Agreed Predictions',
    }
    
    else:
        # predicted_array_correction = [i for i in predicted_array_correction]
        # actual_array_correction = [i for i in actual_array_correction]

        # actual_array_all = [i for i in actual_array_all] ### FIX RECALL

        predicted_array_correction = np.array(predicted_array_correction)
        actual_array_correction = np.array(actual_array_correction)
        actual_array_all = np.array(actual_array_all) ### FIX RECALL

        pred_direction = (predicted_array_correction > 0.5).astype(int)
        actual_direction = (actual_array_correction > 0.5).astype(int)
        actual_all_direction = (actual_array_all > 0.5).astype(int) ### FIX RECALL


        correct = (pred_direction == actual_direction).astype(int)
        
        accuracy = correct.sum() / len(correct) * 100
        actual_ups = (actual_direction == 1)

        actual_all_ups = (actual_all_direction == 1) ### FIX RECALL

        predicted_ups = (pred_direction == 1)
        true_positives_up = (predicted_ups & actual_ups).sum()
        precision_up = true_positives_up / predicted_ups.sum() * 100 if predicted_ups.sum() > 0 else float('nan')
        recall_up = true_positives_up / actual_all_ups.sum() * 100 if actual_all_ups.sum() > 0 else float('nan')
        actual_downs = (actual_direction == 0)

        actual_all_downs = (actual_all_direction == 0) ### FIX RECALL

        predicted_downs = (pred_direction == 0)
        true_positives_down = (predicted_downs & actual_downs).sum()
        precision_down = true_positives_down / predicted_downs.sum() * 100 if predicted_downs.sum() > 0 else float('nan')
        recall_down = true_positives_down / actual_all_downs.sum() * 100 if actual_all_downs.sum() > 0 else float('nan')

        if actual_ups.sum() == 0 and predicted_ups.sum() == 0:
            precision_up = None
            recall_up = None

        if actual_ups.sum() == 0 and predicted_ups.sum() > 0:
            precision_up = 0
            recall_up = None      

        if actual_ups.sum() > 0 and predicted_ups.sum() == 0:
            precision_up = None
            recall_up = 0

            ####################################

        if actual_downs.sum() == 0 and predicted_downs.sum() == 0:
            precision_down = None
            recall_down = None

        if actual_downs.sum() == 0 and predicted_downs.sum() > 0:
            precision_down = 0
            recall_down = None
        
        if actual_downs.sum() > 0 and predicted_downs.sum() == 0:
            precision_down = None
            recall_down = 0


        # if do_print:
        #     print(f"Directional Accuracy: {accuracy:.2f}%")
        #     print(f'Up Precision: {precision_up:.2f}%')
        #     print(f'Up Recall:    {recall_up:.2f}%')
        #     print(f'Down Precision: {precision_down:.2f}%')
        #     print(f'Down Recall:    {recall_down:.2f}%')
        return {
            'accuracy': accuracy,
            'precision_up': precision_up,
            'recall_up': recall_up,
            'precision_down': precision_down,
            'recall_down': recall_down,
        }




def distribution_discovery(combo: dict, combo_index: int, number_of_seeds: int , store_model_weights : bool = False):
    """
    Run the same combo across many different seeds by overriding combo['seed_num'].
    Returns three plots (accuracy, precision_up, recall_up) with base64-encoded PNGs.
    Handles None values and records counts of None values in the plots.
    """
    print(f"****************** STARTED RUN FOR COMBO INDEX {combo_index}")

    # choose seeds
    seeds = random.sample(range(50000), number_of_seeds)
    if 42 not in seeds:
        seeds.append(42)

    def run_for_seed(seed_val):
        # override seed in combo (DO NOT mutate original)
        combo_seeded = dict(combo)
        combo_seeded['seed_num'] = seed_val
        combo_seeded['is_deterministic'] = True  # ensure deterministic path uses the provided seed

        # run
        result_entry = run_combo_V_4(0, combo_seeded, total_offset=0, use_print_acc_vs_pred=False , pred_threshold_sigmoid01_up_bool=False , store_model_weights = store_model_weights)

        return {
            'seed': seed_val,
            'result_entry': result_entry,
        }

    # parallel runs
    with parallel_backend("loky", n_jobs=35):
        per_seed = Parallel()(delayed(run_for_seed)(s) for s in seeds)

    return {
        'combo_index': combo_index,
        'combo': combo,
        'per_seed_all_results': per_seed,

    }



def avg_ensemble(combo_list , num_cv_sets , total_offset , INDEX, combo_numbers):



    # print('runnign combo -- ' , total_offset + INDEX)

    with parallel_backend("loky", n_jobs=-1):
        all_results  = Parallel()(
            delayed(run_combo)(i, combo, 0, use_print_acc_vs_pred=False)
            for i, combo in enumerate(combo_list)
        )

    results, weights  = zip(*all_results)

    all_model_preds = [res["all_preds"] for res in results]  # shape: (num_models, num_folds)

    transposed_preds = list(zip(*all_model_preds))  # shape: (num_folds, num_models)


    avg_preds_per_fold = [np.mean(np.stack(preds), axis=0) for preds in transposed_preds]

    predictions_folds  = avg_preds_per_fold
    actuals_folds = results[0]["all_actuals"]
    raw_actuals = results[0]["raw_actuals"]

    cv_data = {}  

    for set_idx , (pred_fold,act_fold) in enumerate(zip(predictions_folds , actuals_folds )):
        
        metrics = evaluate_binary_0_1(pred_fold, act_fold, one_fold=True , do_print=False)
        
        cv_data[f"set_{set_idx + 1}"] = metrics


    # === Compute overall average ===
    metrics_keys = cv_data[f"set_{set_idx + 1}"].keys()

    overall_avg = {}
    for k in metrics_keys:
        values = [cv_data[f"set_{i + 1}"][k] for i in range(num_cv_sets)]

        numeric_values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v)) and isinstance(v, (int, float))]
        if len(numeric_values) > 0:
            overall_avg[k] = np.mean(numeric_values)

        else:
            overall_avg[k] = None

    cv_data["avg_across_all_sets"] = overall_avg

        # --- NEW: overall metrics across ALL folds (handles "no agreement")
    # flat_preds   = [p for fold in predictions_folds for p in fold]
    # flat_actuals = [a for fold in actuals_folds     for a in fold]
    cv_data["overall_metrics"] = evaluate_binary_0_1(predictions_folds, actuals_folds, one_fold=False, do_print=False)

    result_entry = {
        "combo_number": total_offset + INDEX + 1,
        "parameters": combo_list,
        "cv_sets": cv_data,
        "all_preds" : predictions_folds ,
        "all_actuals" : actuals_folds,
        "raw_actuals" : raw_actuals ,
        "combo_numbers" : combo_numbers

    }

    return result_entry



def selective_ensemble_all_agree(existing_data: list ,combo_list , num_cv_sets , total_offset , INDEX , combo_numbers , 
                                 use_existing_data : bool) : 


    if not use_existing_data:

        with parallel_backend("loky", n_jobs=1):
            all_results  = Parallel()(
                delayed(run_combo)(i, combo, 0, use_print_acc_vs_pred=False)
                for i, combo in enumerate(combo_list)
            )

        results, weights  = zip(*all_results)

    if use_existing_data:
        results = existing_data

    all_model_preds = [res["all_preds"] for res in results]  # shape: (num_models, num_folds)


    transposed_preds = list(zip(*all_model_preds))  
    stacked = [np.stack(i) for i in transposed_preds]
    stacked_bool_int = [(array > 0.5).astype(int) for array in stacked]
    predictions_folds = []

    for fold in stacked_bool_int:
        
        agreement = (fold == fold[0]).all(axis=0)  
        agreed_vals = fold[0]  
    
        result = [
            agreed_vals[i] if agreement[i] else "no agreement"
            for i in range(fold.shape[1])
        ]
        
        predictions_folds.append(result)

    actuals_folds = results[0]["all_actuals"]
    raw_actuals_folds = results[0]["raw_actuals"]


    cv_data = {}  

    for set_idx , (pred_fold,act_fold) in enumerate(zip(predictions_folds , actuals_folds )):
        
        metrics = evaluate_binary_0_1_selective_ensemble(pred_fold, act_fold , do_print=False)
        
        cv_data[f"set_{set_idx + 1}"] = metrics       
        



    metrics_keys = cv_data[f"set_{set_idx + 1}"].keys()

    overall_avg = {}
    for k in metrics_keys:
        values = [cv_data[f"set_{i + 1}"][k] for i in range(num_cv_sets)]

        numeric_values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v)) and isinstance(v, (int, float))]
        if len(numeric_values) > 0:
            overall_avg[k] = np.mean(numeric_values)

        else:
            overall_avg[k] = None

    cv_data["avg_across_all_sets"] = overall_avg

        # --- NEW: overall metrics across ALL folds (handles "no agreement")
    flat_preds   = [p for fold in predictions_folds for p in fold]
    flat_actuals = [a for fold in actuals_folds     for a in fold]
    cv_data["overall_metrics"] = evaluate_binary_0_1_selective_ensemble(flat_preds, flat_actuals , do_print=False)


    result_entry = {
            "combo_number": total_offset + INDEX + 1,
            "parameters": combo_list,
            "cv_sets": cv_data,
            "all_preds" : predictions_folds ,
            "all_actuals" : actuals_folds,
            "raw_actuals" : raw_actuals_folds , 
            "combo_numbers" : combo_numbers }

    return result_entry





def selective_ensemble_threshold_agreement(combo_list, num_cv_sets, total_offset, INDEX, threshold_fraction ,combo_numbers  ): 
    """
    Ensemble that only makes predictions when a threshold fraction of models agree.
    
    Args:
        combo_list: List of model configurations
        num_cv_sets: Number of cross-validation sets
        total_offset: Offset for combo numbering
        INDEX: Current index
        threshold_fraction: Minimum fraction of models that must agree (default: 0.7)
        
    Returns:
        Dictionary with results similar to selective_ensemble_all_agree
    """
    
    with parallel_backend("loky", n_jobs=-1):
        all_results = Parallel()(
            delayed(run_combo)(i, combo, 0, use_print_acc_vs_pred=False)
            for i, combo in enumerate(combo_list)
        )

    results, weights = zip(*all_results)
    all_model_preds = [res["all_preds"] for res in results]  # shape: (num_models, num_folds)

    transposed_preds = list(zip(*all_model_preds))  
    stacked = [np.stack(i) for i in transposed_preds]
    stacked_bool_int = [(array > 0.5).astype(int) for array in stacked]
    predictions_folds = []
    
    num_models = len(combo_list)
    threshold_count = int(np.ceil(num_models * threshold_fraction))

    for fold in stacked_bool_int:
        # Count votes for each class (0 and 1)
        vote_count_1 = fold.sum(axis=0)  # how many models predicted 1
        vote_count_0 = num_models - vote_count_1  # how many predicted 0
        
        # Determine if either class meets the threshold
        meets_threshold_1 = (vote_count_1 >= threshold_count)
        meets_threshold_0 = (vote_count_0 >= threshold_count)
        
        # Create final predictions
        result = []
        for i in range(fold.shape[1]):
            if meets_threshold_1[i]:
                result.append(1)
            elif meets_threshold_0[i]:
                result.append(0)
            else:
                result.append("no agreement")
                
        predictions_folds.append(result)

    actuals_folds = results[0]["all_actuals"]
    raw_actuals_folds = results[0]["raw_actuals"]

    cv_data = {}  

    for set_idx, (pred_fold, act_fold) in enumerate(zip(predictions_folds, actuals_folds)):
        metrics = evaluate_binary_0_1_selective_ensemble(pred_fold, act_fold, do_print=False)
        cv_data[f"set_{set_idx + 1}"] = metrics       
        
    # Calculate average metrics across all sets
    metrics_keys = cv_data[f"set_{set_idx + 1}"].keys()
    overall_avg = {}
    for k in metrics_keys:
        values = [cv_data[f"set_{i + 1}"][k] for i in range(num_cv_sets)]
        numeric_values = [v for v in values if v is not None and 
                         not (isinstance(v, float) and np.isnan(v)) and 
                         isinstance(v, (int, float))]
        overall_avg[k] = np.mean(numeric_values) if numeric_values else None

    cv_data["avg_across_all_sets"] = overall_avg

    # Calculate overall metrics across ALL folds
    flat_preds = [p for fold in predictions_folds for p in fold]
    flat_actuals = [a for fold in actuals_folds for a in fold]
    cv_data["overall_metrics"] = evaluate_binary_0_1_selective_ensemble(
        flat_preds, flat_actuals, do_print=False)

    result_entry = {
        "combo_number": total_offset + INDEX + 1,
        "parameters": combo_list,
        "threshold_fraction": threshold_fraction,
        "cv_sets": cv_data,
        "all_preds": predictions_folds,
        "all_actuals": actuals_folds,
        "raw_actuals": raw_actuals_folds , 
        "combo_numbers" : combo_numbers
    }

    return result_entry




def selective_ensemble_majority_vote(combo_list ,  total_offset , INDEX , combo_numbers): 

    with parallel_backend("loky", n_jobs=-1):
        all_results = Parallel()(
            delayed(run_combo)(i, combo, 0, use_print_acc_vs_pred=False)
            for i, combo in enumerate(combo_list)
        )

    results, weights = zip(*all_results)
    all_model_preds = [res["all_preds"] for res in results]  # shape: (num_models, num_folds)

    transposed_preds = list(zip(*all_model_preds))  # shape: (num_folds, num_models)
    stacked = [np.stack(i) for i in transposed_preds]  # shape: (num_models, num_samples) per fold
    stacked_bool_int = [(array > 0.5).astype(int) for array in stacked]  # binarize

    predictions_folds = []

    for fold in stacked_bool_int:
        # fold: shape (num_models, num_samples)
        num_models = fold.shape[0]
        vote_count_1 = fold.sum(axis=0)  # how many models predicted 1
        vote_count_0 = num_models - vote_count_1
        final_vote_fold = (vote_count_1 > vote_count_0).astype(int)

        predictions_folds.append(final_vote_fold)

    actuals_folds = results[0]["all_actuals"]
    raw_actuals_folds = results[0]["raw_actuals"]

    cv_data = {}  

    for set_idx, (pred_fold, act_fold) in enumerate(zip(predictions_folds, actuals_folds)):
        metrics = evaluate_binary_0_1(pred_fold, act_fold, one_fold= True , do_print=False)
        cv_data[f"set_{set_idx + 1}"] = metrics

    metrics_keys = cv_data[f"set_{set_idx + 1}"].keys()
    num_cv_sets = 8
    overall_avg = {}
    for k in metrics_keys:
        values = [cv_data[f"set_{i + 1}"][k] for i in range(num_cv_sets)]
        numeric_values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v)) and isinstance(v, (int, float))]
        overall_avg[k] = np.mean(numeric_values) if numeric_values else None

    cv_data["avg_across_all_sets"] = overall_avg

        # --- NEW: overall metrics across ALL folds (handles "no agreement")
    flat_preds   = [p for fold in predictions_folds for p in fold]
    flat_actuals = [a for fold in actuals_folds     for a in fold]
    cv_data["overall_metrics"] = evaluate_binary_0_1(flat_preds, flat_actuals,one_fold= False , do_print=False)


    result_entry = {
            "combo_number": total_offset + INDEX + 1,
            "parameters": combo_list,
            "cv_sets": cv_data,
            "all_preds" : predictions_folds ,
            "all_actuals" : actuals_folds,
            "raw_actuals" : raw_actuals_folds , 
            "combo_numbers" : combo_numbers }

    return result_entry
