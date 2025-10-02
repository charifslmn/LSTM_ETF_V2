import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from itertools import combinations
from collections import Counter


from Equations_Ensembles_Dist import (

    evaluate_binary_0_1_selective_ensemble , distribution_discovery , avg_ensemble , selective_ensemble_all_agree
)


def add_threshold_metrics(data: List[Dict] ,threshold: float) -> None:
    """Add threshold metrics to result data"""
    for res in data:
        all_preds = res["all_preds"]
        actuals_per_fold = res["all_actuals"]
        actuals_per_fold_flat = [i for fold in res["all_actuals"] for i in fold]


        pred_threshold_sigmoid01_up = threshold
        all_actuals_threshold_per_fold = []  ; all_preds_threshold_per_fold = []


        for p_fold, a_fold in zip(all_preds, actuals_per_fold):
            new_p_fold = [] ; new_a_fold = [] 

            for p, a in zip(p_fold, a_fold):
                if p > 0.5 and p > pred_threshold_sigmoid01_up:
                    new_p_fold.append(p) ; new_a_fold.append(a)


                elif p > 0.5 and p <= pred_threshold_sigmoid01_up:
                    new_p_fold.append('below_threshold') ; new_a_fold.append('below_threshold')

                else:
                    new_p_fold.append(p) ; new_a_fold.append(a)


            all_preds_threshold_per_fold.append(new_p_fold) ; all_actuals_threshold_per_fold.append(new_a_fold)


        all_actuals_threshold_per_fold_flattened = [j for parts in all_actuals_threshold_per_fold for j in parts] 
        all_preds_threshold_per_fold_flattened = [j for parts in all_preds_threshold_per_fold for j in parts]


        res["all_preds_threshold"] = all_preds_threshold_per_fold
        res["all_actuals_threshold"] = all_actuals_threshold_per_fold


        threshold_metrics = evaluate_binary_0_1_selective_ensemble(
            all_preds_threshold_per_fold_flattened, 
            actuals_per_fold_flat, 
            do_print=False
        )
        
        threshold_metrics_renamed = {}
        for key, value in threshold_metrics.items():
            threshold_metrics_renamed[f"{key}_thresh"] = value
        
        res["overall_metrics_thresh"] = threshold_metrics_renamed


def flatten_results(result_list: List[Dict]) -> pd.DataFrame:
    """Flatten one list of results into a DataFrame without modifying the original."""
    flattened = []
    for entry in result_list:
        flat_entry = {k: v for k, v in entry.items() if k != "cv_sets"}
        flat_entry.update(entry["cv_sets"])
        flattened.append(flat_entry)
    return pd.DataFrame(flattened)

def add_up_prediction_counts(dfs: List[pd.DataFrame]) -> None:
    """Add number of up predictions for normal and threshold versions"""
    for df in dfs:
        no_up_list = []  
        no_up_thresh_list = []
        for row in df["all_preds"]:
            flat_preds = [p for fold in row for p in fold]
            no_up_list.append(sum(p > 0.5 for p in flat_preds))
        for row in df["all_preds_threshold"]:
            flat_preds = [p for fold in row for p in fold]
            no_up_thresh_list.append(sum(isinstance(p, (int, float)) and p > 0.5 for p in flat_preds))
        df["no_up_preds"] = no_up_list
        df["no_up_preds_thresh"] = no_up_thresh_list





def add_false_correct_up_stats(dfs: List[pd.DataFrame]) -> None:
    """Add false and correct up prediction statistics"""
    for df in dfs:
        false_up_preds_col_list_actual = []
        false_up_preds_col_probabs_list_actual = []
        false_up_preds_col_list_actual_thresh = []
        false_up_preds_col_probabs_list_actual_thresh = []
        correct_up_preds_col_list_actual = []
        correct_up_preds_col_probabs_list = []
        correct_up_preds_col_list_actual_thresh = []
        correct_up_preds_col_probabs_list_thresh = []


        for row_raw_actuals, row_01_actuals, row_preds, row_preds_thresh in zip(
            df["raw_actuals"], df["all_actuals"], df["all_preds"], df["all_preds_threshold"]
        ):


            false_up_preds_row_actual = []
            false_up_preds_row_probabs = []
            false_up_preds_row_actual_thresh = []
            false_up_preds_row_probabs_thresh = []
            correct_up_preds_row_actual = []
            correct_up_preds_row_probabs = []
            correct_up_preds_row_actual_thresh = []
            correct_up_preds_row_probabs_thresh = []


            row_raw_actuals_flattened = [p for fold in row_raw_actuals for p in fold]
            row_01_actuals_flattened = [p for fold in row_01_actuals for p in fold]
            row_preds_flattened = [p for fold in row_preds for p in fold]
            row_preds_thresh_flattened = [p for fold in row_preds_thresh for p in fold]


            # Normal version
            for entry_raw_actual, entry_01_actual, entry_pred in zip(row_raw_actuals_flattened, row_01_actuals_flattened, row_preds_flattened):
                if entry_pred > 0.5 and entry_01_actual < 0.5:
                    false_up_preds_row_actual.append(round(entry_raw_actual, 4)) 
                    false_up_preds_row_probabs.append(round(entry_pred,4))


            for entry_raw_actual, entry_01_actual, entry_pred in zip(row_raw_actuals_flattened, row_01_actuals_flattened, row_preds_flattened):
                if entry_pred > 0.5 and entry_01_actual > 0.5:
                    correct_up_preds_row_actual.append(round(entry_raw_actual, 4)) 
                    correct_up_preds_row_probabs.append(round(entry_pred,4))


            # Threshold version
            for entry_raw_actual, entry_01_actual, entry_pred in zip(row_raw_actuals_flattened, row_01_actuals_flattened, row_preds_thresh_flattened):
                if ( not isinstance(entry_pred, str)) and entry_pred > 0.5 and entry_01_actual < 0.5:
                    false_up_preds_row_actual_thresh.append(round(entry_raw_actual, 4))
                    false_up_preds_row_probabs_thresh.append(round(entry_pred,4))


            for entry_raw_actual, entry_01_actual, entry_pred in zip(row_raw_actuals_flattened, row_01_actuals_flattened, row_preds_thresh_flattened):
                
                if ( not isinstance(entry_pred, str) ) and entry_pred > 0.5 and entry_01_actual > 0.5:
                    correct_up_preds_row_actual_thresh.append(round(entry_raw_actual, 4))
                    correct_up_preds_row_probabs_thresh.append(round(entry_pred, 4))


            false_up_preds_col_list_actual.append(false_up_preds_row_actual)
            false_up_preds_col_probabs_list_actual.append(false_up_preds_row_probabs)
            false_up_preds_col_list_actual_thresh.append(false_up_preds_row_actual_thresh)
            false_up_preds_col_probabs_list_actual_thresh.append(false_up_preds_row_probabs_thresh)
            
            correct_up_preds_col_list_actual.append(correct_up_preds_row_actual)
            correct_up_preds_col_probabs_list.append(correct_up_preds_row_probabs)
            correct_up_preds_col_list_actual_thresh.append(correct_up_preds_row_actual_thresh)
            correct_up_preds_col_probabs_list_thresh.append(correct_up_preds_row_probabs_thresh)



        df["actuals_false_up"] = false_up_preds_col_list_actual
        df["false_up_preds"] = false_up_preds_col_probabs_list_actual
        df["actuals_false_up_thresh"] = false_up_preds_col_list_actual_thresh
        df["false_up_preds_thresh"] = false_up_preds_col_probabs_list_actual_thresh
        df["actuals_correct_up"] = correct_up_preds_col_list_actual
        df["correct_up_preds"] = correct_up_preds_col_probabs_list
        df["actuals_correct_up_thresh"] = correct_up_preds_col_list_actual_thresh
        df["correct_up_preds_thresh"] = correct_up_preds_col_probabs_list_thresh


################################################################## TESTING

def flatten_metrics_columns(dfs: List[pd.DataFrame]) -> None:
    """Flatten overall metrics columns"""
    for df in dfs:
        df.rename(columns={"overall_metrics": "overall"}, inplace=True)
    
    for df in dfs:
        if "overall" in df.columns:
            params_df = pd.json_normalize(df["overall"])
            params_df.columns = [f"OA_{col}" for col in params_df.columns]
            df[params_df.columns] = params_df
        
        if "overall_metrics_thresh" in df.columns:
            params_df = pd.json_normalize(df["overall_metrics_thresh"])
            params_df.columns = [f"OA_thresh_{col}" for col in params_df.columns]
            df[params_df.columns] = params_df



def process_parameters_and_merge(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Process parameters and create master DataFrame"""

    if len(dfs) == 2:
        concat_df_val = dfs[0]
        concat_df_test = dfs[1]
    if len(dfs) == 4:
        concat_df_val = pd.concat([dfs[0], dfs[1]], ignore_index=True)
        concat_df_test = pd.concat([dfs[2], dfs[3]], ignore_index=True)
    
    temp_dfs = [concat_df_val, concat_df_test]
    
    for df in temp_dfs:
        params_chosen = []
        for param_dict in df["parameters"]:
            param_dict_fix = {}
            for param_key in param_dict.keys():
                if param_key not in ['val_start_month', 'val_end_month']:
                    param_dict_fix[str(param_key)] = param_dict[param_key]
            params_chosen.append(str(param_dict_fix))
        df["params_fix"] = params_chosen

    dict_fix_params = {}
    for idx, p in enumerate(concat_df_test["params_fix"]):
        p_str = str(p).replace(' ', '').replace("'", '').replace('  ', '')
        dict_fix_params[p_str] = idx

    for df in temp_dfs:
        param_values = []
        for p in df["params_fix"]:
            p_str = str(p).replace(' ', '').replace("'", '')
            param_values.append(dict_fix_params.get(p_str, -1))
        df["param_int_value"] = param_values

    concat_df_val.rename(columns=lambda x: f"{x}_mac_val" if x != "params_fix" else x, inplace=True)
    concat_df_test.rename(columns=lambda x: f"{x}_mac_test" if x != "params_fix" else x, inplace=True)

    master_df = concat_df_val.merge(concat_df_test, on="params_fix", how="outer")
    
    master_df.columns = (
        master_df.columns
        .str.replace('precision', 'prec')
        .str.replace('recall', 'rec')
        .str.replace('down', '0')
        .str.replace('up', '1')
        .str.replace('accuracy', 'acc')
        .str.replace('test', 'T')
        .str.replace('val', 'V')
    )
    
    return master_df



############## NEW NEW NEW NEW 

def get_model_groups_in_corr_range_diff_params(master_df, machine, set_type, corr_range, group_size=2, 
                                 use_spearman_bool=False, num_diff_params=None):
    """
    Find groups of models (size=group_size) whose pairwise correlations
    of predictions fall within [low, high), with optional parameter difference filtering.

    Args:
        master_df (pd.DataFrame): your merged DF.
        machine (str): 'mac' or 'gc'.
        set_type (str): 'V' or 'T'.
        corr_range (tuple): (low, high) inclusive/exclusive as [low, high).
        group_size (int): number of models per group (>=2).
        use_spearman_bool (bool): if True, use Spearman correlation; else use Pearson.
        num_diff_params (int or None): Minimum number of different parameters required 
                                     between models in the group. If None, no filtering.

    Returns:
        list[tuple]: list of tuples of model IDs (param_int_Vue_...) meeting the criteria.
    """
    assert group_size >= 2, "group_size must be >= 2"

    preds_col = f"all_preds_{machine}_{set_type}"
    id_col = f"param_int_Vue_{machine}_{set_type}"
    params_col = f"parameters_{machine}_{set_type}"

    # Keep only rows with predictions, ID, and parameters
    block = master_df.loc[master_df[preds_col].notna() & 
                         master_df[id_col].notna() & 
                         master_df[params_col].notna(), 
                         [id_col, preds_col, params_col]]

    # Flatten predictions per model id and store parameters
    preds_by_id = {}
    params_by_id = {}
    
    for _, row in block.iterrows():
        mid = int(row[id_col])
        flat = [p for fold in row[preds_col] for p in fold]
        preds_by_id[mid] = np.asarray(flat, dtype=float)
        params_by_id[mid] = row[params_col]  # Store parameters

    if not preds_by_id:
        return []

    model_ids = sorted(preds_by_id.keys())

    # Build data matrix (n_models, n_samples)
    data = np.vstack([preds_by_id[mid] for mid in model_ids])

    # Correlation matrix
    if use_spearman_bool:
        from scipy.stats import spearmanr
        corr = spearmanr(data, axis=1).correlation
    else:
        corr = np.corrcoef(data)

    low, high = corr_range

    groups = []
    idx_map = {i: mid for i, mid in enumerate(model_ids)}

    # Fast path for pairs
    if group_size == 2:
        for i in range(len(model_ids)):
            for j in range(i+1, len(model_ids)):
                c = corr[i, j]
                if not np.isnan(c) and (low <= c < high):
                    groups.append((idx_map[i], idx_map[j]))
    else:
        # For k >= 3: check all combinations are within range
        for combo in combinations(range(len(model_ids)), group_size):
            ok = True
            for a, b in combinations(combo, 2):
                c = corr[a, b]
                if np.isnan(c) or not (low <= c < high):
                    ok = False
                    break
            if ok:
                groups.append(tuple(idx_map[i] for i in combo))

    # Filter by parameter differences if requested
    if num_diff_params is not None and groups:
        filtered_groups = []
        
        for group in groups:
            # Get parameters for all models in this group
            group_params = [params_by_id[mid] for mid in group]
            
            # Check if all models have the same parameter structure
            if not all(isinstance(params, dict) for params in group_params):
                continue
                
            # Count different parameters across all pairs in the group
            min_diff_params = float('inf')
            
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    params1 = group_params[i]
                    params2 = group_params[j]
                    
                    # Count different parameters between this pair
                    diff_count = 0
                    all_keys = set(params1.keys()) | set(params2.keys())
                    
                    for key in all_keys:
                        val1 = params1.get(key)
                        val2 = params2.get(key)
                        
                        # Consider parameters different if:
                        # 1. One has the parameter and the other doesn't
                        # 2. Both have the parameter but with different values
                        if (key not in params1 or key not in params2) or (val1 != val2):
                            diff_count += 1
                    
                    min_diff_params = min(min_diff_params, diff_count)
            
            # Keep group if minimum difference meets the threshold
            if min_diff_params >= num_diff_params:
                filtered_groups.append(group)
        
        return filtered_groups

    return groups






############## NEW NEW NEW NEW 





def get_model_groups_in_corr_range(master_df, machine, set_type, corr_range, group_size=2, use_spearman_bool=False):
    """
    Find groups of models (size=group_size) whose pairwise correlations
    of predictions fall within [low, high).

    Args:
        master_df (pd.DataFrame): your merged DF.
        machine (str): 'mac' or 'gc'.
        set_type (str): 'V' or 'T'.
        corr_range (tuple): (low, high) inclusive/exclusive as [low, high).
        group_size (int): number of models per group (>=2).
        use_spearman_bool (bool): if True, use Spearman correlation; else use Pearson.

    Returns:
        list[tuple]: list of tuples of model IDs (param_int_Vue_...) meeting the criterion.
    """
    # assert group_size >= 2, "group_size must be >= 2"

    preds_col = f"all_preds_{machine}_{set_type}"
    id_col    = f"param_int_Vue_{machine}_{set_type}"

    # Keep only rows with predictions and an ID
    block = master_df.loc[master_df[preds_col].notna() & master_df[id_col].notna(), [id_col, preds_col]]

    # Flatten predictions per model id
    preds_by_id = {}
    for _, row in block.iterrows():
        mid = int(row[id_col])
        flat = [p for fold in row[preds_col] for p in fold]
        preds_by_id[mid] = np.asarray(flat, dtype=float)

    if not preds_by_id:
        return []

    # Handle potential unequal lengths robustly:
    # lengths = [len(v) for v in preds_by_id.values()]
    # modal_len = Counter(lengths).most_common(1)[0][0]
    # Keep only models with the modal length to ensure comparable correlations
    # preds_by_id = {k: v for k, v in preds_by_id.items() if len(v) == modal_len}

    model_ids = sorted(preds_by_id.keys())


    # Build data matrix (n_models, n_samples)
    data = np.vstack([preds_by_id[mid] for mid in model_ids])

    # Correlation matrix - CHANGED: Added if statement for correlation type
    if use_spearman_bool:
        from scipy.stats import spearmanr
        corr = spearmanr(data, axis=1).correlation
    else:
        corr = np.corrcoef(data)

    low, high = corr_range

    groups = []
    idx_map = {i: mid for i, mid in enumerate(model_ids)}

    # Fast path for pairs
    if group_size == 2:
        for i in range(len(model_ids)):
            for j in range(i+1, len(model_ids)):
                c = corr[i, j]
                if not np.isnan(c) and (low <= c < high):
                    groups.append((idx_map[i], idx_map[j]))
        return groups

    # For k >= 3: check all combinations are within range
    for combo in combinations(range(len(model_ids)), group_size):
        ok = True
        for a, b in combinations(combo, 2):
            c = corr[a, b]
            if np.isnan(c) or not (low <= c < high):
                ok = False
                break
        if ok:
            groups.append(tuple(idx_map[i] for i in combo))

    return groups


##### NEW NEW NEW 


def get_model_groups_in_corr_range_diff_params_and_same_up_preds(master_df, machine, set_type, corr_range, group_size=2, 
                                 use_spearman_bool=False, num_diff_params=None, 
                                 min_same_up_preds=None, max_same_up_preds=None):
    """
    Find groups of models (size=group_size) whose pairwise correlations
    of predictions fall within [low, high), with optional parameter difference filtering
    and same "up" predictions filtering.

    Args:
        master_df (pd.DataFrame): your merged DF.
        machine (str): 'mac' or 'gc'.
        set_type (str): 'V' or 'T'.
        corr_range (tuple): (low, high) inclusive/exclusive as [low, high).
        group_size (int): number of models per group (>=2).
        use_spearman_bool (bool): if True, use Spearman correlation; else use Pearson.
        num_diff_params (int or None): Minimum number of different parameters required 
                                     between models in the group. If None, no filtering.
        min_same_up_preds (int or None): Minimum number of times both models predict "up" (>=0.5)
                                        for the same entry. If None, no filtering.
        max_same_up_preds (int or None): Maximum number of times both models predict "up" (>=0.5)
                                        for the same entry. If None, no filtering.

    Returns:
        list[tuple]: list of tuples of model IDs (param_int_Vue_...) meeting the criteria.
    """
    assert group_size >= 2, "group_size must be >= 2"

    preds_col = f"all_preds_{machine}_{set_type}"
    id_col = f"param_int_Vue_{machine}_{set_type}"
    params_col = f"parameters_{machine}_{set_type}"

    # Keep only rows with predictions, ID, and parameters
    block = master_df.loc[master_df[preds_col].notna() & 
                         master_df[id_col].notna() & 
                         master_df[params_col].notna(), 
                         [id_col, preds_col, params_col]]

    # Flatten predictions per model id and store parameters
    preds_by_id = {}
    params_by_id = {}
    
    for _, row in block.iterrows():
        mid = int(row[id_col])
        flat = [p for fold in row[preds_col] for p in fold]
        preds_by_id[mid] = np.asarray(flat, dtype=float)
        params_by_id[mid] = row[params_col]  # Store parameters

    if not preds_by_id:
        return []

    model_ids = sorted(preds_by_id.keys())

    # Build data matrix (n_models, n_samples)
    data = np.vstack([preds_by_id[mid] for mid in model_ids])

    # Correlation matrix
    if use_spearman_bool:
        from scipy.stats import spearmanr
        corr = spearmanr(data, axis=1).correlation
    else:
        corr = np.corrcoef(data)

    low, high = corr_range

    groups = []
    idx_map = {i: mid for i, mid in enumerate(model_ids)}

    # Fast path for pairs
    if group_size == 2:
        for i in range(len(model_ids)):
            for j in range(i+1, len(model_ids)):
                c = corr[i, j]
                if not np.isnan(c) and (low <= c < high):
                    groups.append((idx_map[i], idx_map[j]))
    else:
        # For k >= 3: check all combinations are within range
        for combo in combinations(range(len(model_ids)), group_size):
            ok = True
            for a, b in combinations(combo, 2):
                c = corr[a, b]
                if np.isnan(c) or not (low <= c < high):
                    ok = False
                    break
            if ok:
                groups.append(tuple(idx_map[i] for i in combo))

    # Filter by parameter differences if requested
    if num_diff_params is not None and groups:
        filtered_groups = []
        
        for group in groups:
            # Get parameters for all models in this group
            group_params = [params_by_id[mid] for mid in group]
            
            # Check if all models have the same parameter structure
            if not all(isinstance(params, dict) for params in group_params):
                continue
                
            # Count different parameters across all pairs in the group
            min_diff_params = float('inf')
            
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    params1 = group_params[i]
                    params2 = group_params[j]
                    
                    # Count different parameters between this pair
                    diff_count = 0
                    all_keys = set(params1.keys()) | set(params2.keys())
                    
                    for key in all_keys:
                        val1 = params1.get(key)
                        val2 = params2.get(key)
                        
                        # Consider parameters different if:
                        # 1. One has the parameter and the other doesn't
                        # 2. Both have the parameter but with different values
                        if (key not in params1 or key not in params2) or (val1 != val2):
                            diff_count += 1
                    
                    min_diff_params = min(min_diff_params, diff_count)
            
            # Keep group if minimum difference meets the threshold
            if min_diff_params >= num_diff_params:
                filtered_groups.append(group)
        
        groups = filtered_groups

    # Filter by same "up" predictions if requested (both min and max)
    if (min_same_up_preds is not None or max_same_up_preds is not None) and groups:
        filtered_groups = []
        
        for group in groups:
            # For groups of size 2, check the pair directly
            if group_size == 2:
                mid1, mid2 = group
                preds1 = preds_by_id[mid1]
                preds2 = preds_by_id[mid2]
                
                # Count number of times both predict "up" (>=0.5)
                both_up_count = np.sum((preds1 >= 0.5) & (preds2 >= 0.5))
                
                # Check both min and max constraints
                min_ok = (min_same_up_preds is None) or (both_up_count >= min_same_up_preds)
                max_ok = (max_same_up_preds is None) or (both_up_count <= max_same_up_preds)
                
                if min_ok and max_ok:
                    filtered_groups.append(group)
            
            # For groups larger than 2, check all pairs
            else:
                keep_group = True
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        mid1, mid2 = group[i], group[j]
                        preds1 = preds_by_id[mid1]
                        preds2 = preds_by_id[mid2]
                        
                        # Count number of times both predict "up" (>=0.5)
                        both_up_count = np.sum((preds1 >= 0.5) & (preds2 >= 0.5))
                        
                        # Check both min and max constraints for this pair
                        min_ok = (min_same_up_preds is None) or (both_up_count >= min_same_up_preds)
                        max_ok = (max_same_up_preds is None) or (both_up_count <= max_same_up_preds)
                        
                        if not (min_ok and max_ok):
                            keep_group = False
                            break
                    
                    if not keep_group:
                        break
                
                if keep_group:
                    filtered_groups.append(group)
        
        groups = filtered_groups

    return groups

##. NEW NEW NEW 

def create_pair_params_map(random_groups: list, master_df: pd.DataFrame, num_models: int, data_type: str = "V") -> dict:
    """
    Create parameter maps for model groups
    
    Args:
        random_groups: List of model ID tuples
        master_df: Master DataFrame containing model data
        num_models: Number of models in each group (2, 3, or 4)
        data_type: 'V' for validation or 'T' for test data
    
    Returns:
        Dictionary with model groups as keys and their parameters/predictions as values
    """
    pair_params_map = {}
    
    for pair in random_groups:
        pair_params_map[pair] = {
            "parameters": [
                master_df.loc[master_df[f"param_int_Vue_mac_{data_type}"] == model_id, f"parameters_mac_{data_type}"].iloc[0]
                for model_id in pair[:num_models]  # Take first num_models from the tuple
            ],
            "predictions": [
                master_df.loc[master_df[f"param_int_Vue_mac_{data_type}"] == model_id, f"all_preds_mac_{data_type}"].iloc[0]
                for model_id in pair[:num_models]
            ],
            "actuals": [
                master_df.loc[master_df[f"param_int_Vue_mac_{data_type}"] == model_id, f"all_actuals_mac_{data_type}"].iloc[0]
                for model_id in pair[:num_models]
            ],
            "raw_actuals": [
                master_df.loc[master_df[f"param_int_Vue_mac_{data_type}"] == model_id, f"raw_actuals_mac_{data_type}"].iloc[0]
                for model_id in pair[:num_models]
            ]
        }
    
    return pair_params_map




def selective_ensemble_all_agree_thresh(existing_data: list, combo_list, num_cv_sets, total_offset, INDEX, combo_numbers, 
                                 use_existing_data: bool): 

    if not use_existing_data:
        with parallel_backend("loky", n_jobs=1):
            all_results = Parallel()(
                delayed(run_combo)(i, combo, 0, use_print_acc_vs_pred=False)
                for i, combo in enumerate(combo_list)
            )
        results, weights = zip(*all_results)

    if use_existing_data:
        results = existing_data

    all_model_preds = [res["all_preds"] for res in results]  # shape: (num_models, num_folds)

    transposed_preds = list(zip(*all_model_preds))  
    
    # Handle string values in predictions (like 'below_threshold')
    stacked_bool_int = []
    for fold_preds in transposed_preds:
        fold_array = []
        for model_preds in fold_preds:
            # Convert each prediction, handling string values
            converted_preds = []
            for pred in model_preds:
                if isinstance(pred, str):
                    # For string values like 'below_threshold', treat as 0 (down prediction)
                    converted_preds.append(0)
                else:
                    converted_preds.append(1 if pred > 0.5 else 0)
            fold_array.append(converted_preds)
        stacked_bool_int.append(np.array(fold_array))
    
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

    for set_idx, (pred_fold, act_fold) in enumerate(zip(predictions_folds, actuals_folds)):
        metrics = evaluate_binary_0_1_selective_ensemble(pred_fold, act_fold, do_print=False)
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

    # Overall metrics across ALL folds
    flat_preds = [p for fold in predictions_folds for p in fold]
    flat_actuals = [a for fold in actuals_folds for a in fold]
    cv_data["overall_metrics"] = evaluate_binary_0_1_selective_ensemble(flat_preds, flat_actuals, do_print=False)

    result_entry = {
        "combo_number": total_offset + INDEX + 1,
        "parameters": combo_list,
        "cv_sets": cv_data,
        "all_preds": predictions_folds,
        "all_actuals": actuals_folds,
        "raw_actuals": raw_actuals_folds, 
        "combo_numbers": combo_numbers
    }

    return result_entry







def process_ensemble_groups(ensemble_config: list, master_df: pd.DataFrame, data_type: str = "T", 
                           use_threshold_data: bool = False) :
    """
    Process multiple ensemble groups with flexible configuration for both regular and threshold data
    
    Args:
        ensemble_config: List of tuples with (group_name, pair_map, num_models)
        master_df: Master DataFrame containing model data
        data_type: 'V' for validation or 'T' for test data
        use_threshold_data: True for threshold data, False for regular data
    
    Returns:
        Dictionary with ensemble results for all groups
    """
    ensemble_results = {}
    
    for group_name, pair_map, num_models in ensemble_config:
        ensemble_results[group_name] = []
        
        for pair, data in pair_map.items():
            existing_data = []
            individual_model_data = {}
            
            for i, model_id in enumerate(pair[:num_models]):
                # Determine which prediction column to use based on threshold flag
                if data_type == "T":
                    preds_column = "all_preds_threshold_mac_T" if use_threshold_data else "all_preds_mac_T"

                if data_type == "V": ##### ERROR this was missing before sept 14 !!!! so it was always jsut T set
                    preds_column = "all_preds_threshold_mac_V" if use_threshold_data else "all_preds_mac_V"

                # Get the data
                model_preds = master_df.loc[master_df[f"param_int_Vue_mac_{data_type}"] == model_id, preds_column].iloc[0]
                model_actuals = master_df.loc[master_df[f"param_int_Vue_mac_{data_type}"] == model_id, f"all_actuals_mac_{data_type}"].iloc[0]
                model_raw_actuals = master_df.loc[master_df[f"param_int_Vue_mac_{data_type}"] == model_id, f"raw_actuals_mac_{data_type}"].iloc[0]
                
                existing_data.append({
                    "all_preds": model_preds,
                    "all_actuals": model_actuals,
                    "raw_actuals": model_raw_actuals
                })
                
                # Store individual model data for the results
                individual_model_data[f"model_{model_id}_preds"] = model_preds


                ##### ERROR this was missing before sept 14 !!!! so it was always jsut T ser 
            if data_type == "V":
                num_cv_sets = 8 
            if data_type == "T":
                num_cv_sets = 4   
                ##### ERROR this was missing before sept 14 !!!! so it was always jsut T ser 

            
            result = selective_ensemble_all_agree_thresh(
                existing_data=existing_data,
                combo_list=None,
                num_cv_sets=num_cv_sets,
                total_offset=0,
                INDEX=len(ensemble_results[group_name]),
                combo_numbers=pair,
                use_existing_data=True,
            )
            
            # Add individual model data to the result
            result.update(individual_model_data)
            ensemble_results[group_name].append(result)
    
    return ensemble_results






import random



import pickle




def process_func_PLUS_return_analytics(master_df: pd.DataFrame, 
                                      groups_config: list, 
                                      data_type_corr_groups_creation: str = "V",
                                      data_type_ensemble: str = "T",
                                      use_threshold_data: bool = False, 
                                      seed = None, 
                                      no_maps_per_group: int = 10 , 
                                    filter_outliers: bool = False) -> dict:
    """
    Complete function that creates groups, processes ensembles, and returns analytics
    
    Args:
        master_df: Master DataFrame containing model data
        groups_config: List of tuples with (machine, data_type, corr_range, group_size)
        data_type_corr_groups_creation: Data type for correlation group creation ('V' or 'T')
        data_type_ensemble: Data type for ensemble processing ('V' or 'T')
        use_threshold_data: True for threshold data, False for regular data
        seed: Random seed for reproducibility
        no_maps_per_group: Number of pairs/triplets to sample per group
    
    Returns:
        Dictionary with detailed analytics and summary
    """


    # Set random seed
    if seed is not None:
        random.seed(seed)
    
    # Create groups and random samples
    all_pair_maps = {}
    
    for machine, corr_range, group_size in groups_config:
        # Create group name based on parameters
        group_name = f"pair_{group_size}_{corr_range[0]}_{corr_range[1]}".replace('.', '').replace('-', 'neg')
        
        # Get model groups
        groups = get_model_groups_in_corr_range(master_df, machine,data_type_corr_groups_creation , corr_range, group_size=group_size)
        
        if groups:

            # Randomly select from each group
            random_groups = random.sample(groups, min(no_maps_per_group, len(groups))) if groups else []
            
            # Create parameter map for both V and T data
            pair_params_map_V = create_pair_params_map(random_groups, master_df, num_models=group_size, data_type="V")
            pair_params_map_T = create_pair_params_map(random_groups, master_df, num_models=group_size, data_type="T")
            
            # Store both versions
            all_pair_maps[group_name] = {
                "V": pair_params_map_V,
                "T": pair_params_map_T,
                "random_groups": random_groups
            }

        else:
            print(f"No groups found for {group_name} with machine {machine} and data type {data_type_corr_groups_creation}.")

    # Create ensemble configuration using the specified data type
    ensemble_config = []
    for group_name, data in all_pair_maps.items():
        group_size = len(next(iter(data[data_type_ensemble].values()))["parameters"])  # Get group size from first item
        # group_size = len([ v for v in iter(data[data_type_ensemble].keys()) ][0])  # NOTE this is the same as the line above
        ensemble_config.append((group_name, data[data_type_ensemble], group_size))
    
    # Process the data
    ensemble_results = process_ensemble_groups(
        ensemble_config=ensemble_config,
        master_df=master_df,
        data_type=data_type_ensemble,
        use_threshold_data=use_threshold_data
    )

    # Initialize counters
    # total_ups = 0
    # total_correct_ups = 0
    # sum_actuals_ups = 0
    sum_actuals_ups_list_ALL = []


    group_results = {}
    detailed_results = {}




    # Process each group
    for ensemble_group_name, ensemble_group in ensemble_results.items():
        group_output = []
        group_ups = 0
        group_correct_ups = 0
        group_sum_actuals = 0
        group_details = []
        
        # Iterate through each ensemble result in the group
        for ensemble_result in ensemble_group:
            flatten_preds = [p for part in ensemble_result["all_preds"] for p in part]
            flatten_raw_actuals = [a for part in ensemble_result["raw_actuals"] for a in part]
            flatten_actuals = [a for part in ensemble_result["all_actuals"] for a in part]

            p_ups = sum(1 for i in flatten_preds if not isinstance(i, str) and i > 0.5)
            group_ups += p_ups
            # total_ups += p_ups

            up_vals_predicted_raw_val = []

            correct_ups = 0

            for p, a, actual_binary in zip(flatten_preds, flatten_raw_actuals, flatten_actuals):
                if not isinstance(p, str) and p > 0.5:
                    up_vals_predicted_raw_val.append(a)


                    group_sum_actuals += a
                    sum_actuals_ups_list_ALL.append(a)

                    # sum_actuals_ups += a
                    # Check if up prediction was correct (actual_bin > 0.5)
                    if actual_binary > 0.5:
                        correct_ups += 1
            
            group_correct_ups += correct_ups
            # total_correct_ups += correct_ups

            # Create output line
            output_line = (
                f"{ensemble_result['cv_sets']['overall_metrics']['precision_up']} - "
                f"{ensemble_result['cv_sets']['overall_metrics']['recall_up']} "
                f"{ensemble_result['combo_numbers']} - {p_ups} - Correct: {correct_ups} - {up_vals_predicted_raw_val} "
            )
            group_output.append(output_line)
            
            # Store detailed results
            group_details.append({
                "combo_numbers": ensemble_result["combo_numbers"],
                "precision_up": ensemble_result["cv_sets"]["overall_metrics"]["precision_up"],
                "recall_up": ensemble_result["cv_sets"]["overall_metrics"]["recall_up"],
                "up_predictions": p_ups,
                "correct_ups": correct_ups,
                "actual_returns": up_vals_predicted_raw_val,

            })
        
        group_results[ensemble_group_name] = group_output
        detailed_results[ensemble_group_name] = group_details
        
        # Add group summary
        group_results[f"{ensemble_group_name}_summary"] = [
            f"Total Up Predictions: {group_ups}",
            f"Total Correct Up Predictions: {group_correct_ups}",
            f"Sum of Actual Returns for Up Predictions: {group_sum_actuals:.3f}",
            f"Prec Up: {group_correct_ups/group_ups if group_ups > 0 else 0:.3f}"
        ]
        
    #     ## set outlier values past < -0.5 to -0.06
    # if filter_outliers:
    #     for i in range(len(sum_actuals_ups_list_ALL)):
    #         if sum_actuals_ups_list_ALL[i] < -0.5:
    #             sum_actuals_ups_list_ALL[i] = -0.1

    print("before set:", sum_actuals_ups_list_ALL)
    print("after set:", set(sum_actuals_ups_list_ALL))
    total_ups = len(set(sum_actuals_ups_list_ALL))
    total_correct_ups = len(set([val for val in sum_actuals_ups_list_ALL if val > 0.1]))  

    sum_actuals_ups = sum(set(sum_actuals_ups_list_ALL))


    # Create overall summary
    summary = {
        "Total Up Predictions": total_ups,
        "Total Correct Up Predictions": total_correct_ups,
        "Sum of Actual Returns for Up Predictions": sum_actuals_ups,

        "Prec Up": total_correct_ups/total_ups if total_ups > 0 else 0
    }

    # Return structured results
    return {
        "group_results": group_results,
        "detailed_results": detailed_results,
        "summary": summary,
        "total_metrics": {
            "total_ups": total_ups,
            "total_correct_ups": total_correct_ups,
            "sum_actuals_ups": sum_actuals_ups,

            "precision": total_correct_ups / total_ups if total_ups > 0 else 0
        },
        "config_info": {
            "groups_config": groups_config,
            "data_type_corr_groups_creation": data_type_corr_groups_creation,
            "data_type_ensemble": data_type_ensemble,
            "use_threshold_data": use_threshold_data,
            "seed": seed,
            "no_maps_per_group": no_maps_per_group
        }
    }




import numpy as np
from itertools import combinations
from collections import Counter
import random





# names_all = {'res_mac_L_val' : res_mac_L_val, 'res_mac_H_val' : res_mac_H_val, 
#              'res_mac_L_test' : res_mac_L_test, 'res_mac_H_test' : res_mac_H_test}

# names_test = {'res_mac_L_test' : res_mac_L_test, 'res_mac_H_test' : res_mac_H_test}
# names_val = {'res_mac_L_val' : res_mac_L_val, 'res_mac_H_val' : res_mac_H_val}



def process_func_PLUS_return_analytics_THRESH_var_included( #master_df: pd.DataFrame, 
    
    
                                    ## new params
                                    threshold : float,
                                    names_all: dict,

                                    #new arams 
                                
                                    groups_config: list, 
                                    data_type_corr_groups_creation: str = "V",
                                    data_type_ensemble: str = "T",
                                    use_threshold_data: bool = False, 
                                    seed = None, 
                                    no_maps_per_group: int = 10 , 
                                    filter_outliers: bool = False,
                                    use_spearman_corr: bool = False,

                                    use_corr_with_diff_params: bool =True, 
                                    min_diff_params : int = 0,
                                    
                                    use_corr_with_same_up_preds: bool = False,
                                    min_same_up_preds: int = None,
                                    max_same_up_preds: int = None,

                                    ) -> dict:
    



    for data in names_all.values():
        add_threshold_metrics(data, threshold = threshold)

    if len(names_all) == 4 :
 
        df_mac_L_val = flatten_results(names_all["res_mac_L_val"])
        df_mac_H_val = flatten_results(names_all["res_mac_H_val"])
        df_mac_L_test = flatten_results(names_all["res_mac_L_test"])
        df_mac_H_test = flatten_results(names_all["res_mac_H_test"])

                # Add H/L labels
        df_mac_L_val['H_L'] = 'L'
        df_mac_H_val['H_L'] = 'H'
        df_mac_L_test['H_L'] = 'L'
        df_mac_H_test['H_L'] = 'H'

        dfs = [df_mac_L_val, df_mac_H_val, df_mac_L_test, df_mac_H_test]


    if len(names_all) == 2 : ### use the names below for the top combos version 

        df_mac_H_val = flatten_results(names_all["res_mac_H_val"])
        df_mac_H_test = flatten_results(names_all["res_mac_H_test"])
        
                # Add H/L labels
        df_mac_H_val['H_L'] = 'H'
        df_mac_H_test['H_L'] = 'H'

        dfs = [ df_mac_H_val, df_mac_H_test]



    # Add various metrics
    add_up_prediction_counts(dfs)
    add_false_correct_up_stats(dfs)
    flatten_metrics_columns(dfs)

    master_df = process_parameters_and_merge(dfs)

    # Set random seed
    if seed is not None:
        random.seed(seed)
    

    # Create groups and random samples
    all_pair_maps = {}
    groups_data_for_output = {} #### NEW NEW NEW
    
    for machine, corr_range, group_size in groups_config:
        # Create group name based on parameters
        group_name = f"pair_{group_size}_{corr_range[0]}_{corr_range[1]}".replace('.', '').replace('-', 'neg')
        
        # Get model groups
        if use_corr_with_diff_params:
            groups = get_model_groups_in_corr_range_diff_params(master_df, machine, data_type_corr_groups_creation, corr_range, group_size=group_size, use_spearman_bool=use_spearman_corr, num_diff_params=min_diff_params)

        elif use_corr_with_same_up_preds:
            groups = get_model_groups_in_corr_range_diff_params_and_same_up_preds(master_df, machine, data_type_corr_groups_creation, corr_range, group_size=group_size, use_spearman_bool=use_spearman_corr , num_diff_params=None, min_same_up_preds=min_same_up_preds, max_same_up_preds=max_same_up_preds)
        
        else:
            groups = get_model_groups_in_corr_range(master_df, machine, data_type_corr_groups_creation, corr_range, group_size=group_size, use_spearman_bool=use_spearman_corr , num_diff_params=None)


        groups_data_for_output[group_name] = {}  #### NEW NEW NEW
        groups_data_for_output[group_name]['all_groups'] = groups  #### NEW NEW NEW
        groups_data_for_output[group_name]['num_groups'] = len(groups)  #### NEW NEW
        
        # Randomly select from each group
        random_groups = random.sample(groups, min(no_maps_per_group, len(groups))) if groups else []
        
        # Create parameter map for both V and T data
        pair_params_map_V = create_pair_params_map(random_groups, master_df, num_models=group_size, data_type="V")
        pair_params_map_T = create_pair_params_map(random_groups, master_df, num_models=group_size, data_type="T")
        
        # Store both versions
        all_pair_maps[group_name] = {
            "V": pair_params_map_V,
            "T": pair_params_map_T,
            "random_groups": random_groups
        }
    
    # Create ensemble configuration using the specified data type
# Create ensemble configuration using the specified data type
    ensemble_config = []
    for group_name, data in all_pair_maps.items():
        group = data.get(data_type_ensemble, {})          # safe: may be missing/empty

        first = next(iter(group.values()), None)          # safe: returns None if empty

        if not first or "parameters" not in first:
            print(f"[skip] {group_name}: no items or missing 'parameters' for '{data_type_ensemble}'")
            continue

        group_size = len(first["parameters"])
        print(f"[use] {group_name}: {len(group)} ensembles of size {group_size} for '{data_type_ensemble}'")
        
        ensemble_config.append((group_name, group, group_size))

    # Process the data
    ensemble_results = process_ensemble_groups(
        ensemble_config=ensemble_config,
        master_df=master_df,
        data_type=data_type_ensemble,
        use_threshold_data=use_threshold_data
    )

    # Initialize counters
    total_ups = 0
    total_correct_ups = 0
    # sum_actuals_ups = 0
    sum_actuals_ups_list_ALL = []


    group_results = {}
    detailed_results = {}



    # Process each group
    for ensemble_group_name, ensemble_group in ensemble_results.items():
        group_output = []
        group_ups = 0
        group_correct_ups = 0
        group_sum_actuals = 0
        group_details = []
        
        # Iterate through each ensemble result in the group
        for ensemble_result in ensemble_group:
            flatten_preds = [p for part in ensemble_result["all_preds"] for p in part]
            flatten_raw_actuals = [a for part in ensemble_result["raw_actuals"] for a in part]
            flatten_actuals = [a for part in ensemble_result["all_actuals"] for a in part]

            p_ups = sum(1 for i in flatten_preds if not isinstance(i, str) and i > 0.5)
            group_ups += p_ups
            total_ups += p_ups

            up_vals_predicted_raw_val = []

            correct_ups = 0

            for p, a, actual_binary in zip(flatten_preds, flatten_raw_actuals, flatten_actuals):
                if not isinstance(p, str) and p > 0.5:
                    up_vals_predicted_raw_val.append(a)


                    group_sum_actuals += a
                    sum_actuals_ups_list_ALL.append(a)

                    # sum_actuals_ups += a
                    # Check if up prediction was correct (actual_bin > 0.5)
                    if actual_binary > 0.5:
                        correct_ups += 1
            
            group_correct_ups += correct_ups
            total_correct_ups += correct_ups

            # Create output line
            output_line = (
                f"{ensemble_result['cv_sets']['overall_metrics']['precision_up']} - "
                f"{ensemble_result['cv_sets']['overall_metrics']['recall_up']} "
                f"{ensemble_result['combo_numbers']} - {p_ups} - Correct: {correct_ups} - {up_vals_predicted_raw_val} "
            )
            group_output.append(output_line)
            
            # Store detailed results
            group_details.append({
                "combo_numbers": ensemble_result["combo_numbers"],
                "precision_up": ensemble_result["cv_sets"]["overall_metrics"]["precision_up"],
                "recall_up": ensemble_result["cv_sets"]["overall_metrics"]["recall_up"],
                "up_predictions": p_ups,
                "correct_ups": correct_ups,
                "actual_returns": up_vals_predicted_raw_val,

            })
        
        group_results[ensemble_group_name] = group_output
        detailed_results[ensemble_group_name] = group_details
        
        # Add group summary
        group_results[f"{ensemble_group_name}_summary"] = [
            f"Total Up Predictions: {group_ups}",
            f"Total Correct Up Predictions: {group_correct_ups}",
            f"Sum of Actual Returns for Up Predictions: {group_sum_actuals:.3f}",
            f"Prec Up: {group_correct_ups/group_ups if group_ups > 0 else 0:.3f}"
        ]

    # ## set outlier values past < -0.5 to -0.06
    # if filter_outliers:
    #     for i in range(len(sum_actuals_ups_list_ALL)):
    #         if sum_actuals_ups_list_ALL[i] < -0.5:
    #             sum_actuals_ups_list_ALL[i] = -0.1

    total_ups = len(set(sum_actuals_ups_list_ALL))
    total_correct_ups = len(set([val for val in sum_actuals_ups_list_ALL if val > 0.1]))  

    sum_actuals_ups = sum(set(sum_actuals_ups_list_ALL))



    # Create overall summary
    summary = {
        "Total Up Predictions": total_ups,
        "Total Correct Up Predictions": total_correct_ups,
        "Sum of Actual Returns for Up Predictions": sum_actuals_ups,
        "Unique Actual Returns for Up Predictions": set(sum_actuals_ups_list_ALL),
        "Prec Up": total_correct_ups/total_ups if total_ups > 0 else 0
    }

    # Return structured results
    return {
        "group_results": group_results,
        "detailed_results": detailed_results,
        "summary": summary,
        "total_metrics": {
            "total_ups": total_ups,
            "total_correct_ups": total_correct_ups,
            "sum_actuals_ups": sum_actuals_ups,
            "unique_actuals_ups_list": set(sum_actuals_ups_list_ALL),

            "precision": total_correct_ups / total_ups if total_ups > 0 else 0
        },


        "config_info": {
            "groups_config": groups_config,
            "data_type_corr_groups_creation": data_type_corr_groups_creation,
            "data_type_ensemble": data_type_ensemble,
            "use_threshold_data": use_threshold_data,
            "seed": seed,
            "no_maps_per_group": no_maps_per_group
        },


       "groups_data": groups_data_for_output 

    }
