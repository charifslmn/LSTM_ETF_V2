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

        HOD_per_fold = res["HOD_values"]  ; UCO_per_fold = res["UCO_values"] ; HUC_per_fold = res["HUC_values"] ; 
        CRUD_per_fold = res["CRUD_values"] ; USO_per_fold = res["USO_values"] 

        HOD_actuals_per_fold_flat = [i for fold in res["HOD_values"] for i in fold]
        UCO_actuals_per_fold_flat = [i for fold in res["UCO_values"] for i in fold]
        HUC_actuals_per_fold_flat = [i for fold in res["HUC_values"] for i in fold]
        CRUD_actuals_per_fold_flat = [i for fold in res["CRUD_values"] for i in fold]
        USO_actuals_per_fold_flat = [i for fold in res["USO_values"] for i in fold]

        pred_threshold_sigmoid01_up = threshold
        all_actuals_threshold_per_fold = []  ; all_preds_threshold_per_fold = []
        HOD_actuals_threshold_per_fold = [] ; UCO_actuals_threshold_per_fold = []
        HUC_actuals_threshold_per_fold = [] ; CRUD_actuals_threshold_per_fold = [] ; USO_actuals_threshold_per_fold = []

        for p_fold, a_fold, HOD_a_fold, UCO_a_fold, HUC_a_fold, CRUD_a_fold, USO_a_fold in zip(all_preds, actuals_per_fold, HOD_per_fold, UCO_per_fold, HUC_per_fold, CRUD_per_fold, USO_per_fold):
            new_p_fold = [] ; new_a_fold = [] ; new_HOD_a_fold = [] 
            new_UCO_a_fold = [] ; new_HUC_a_fold = [] ; new_CRUD_a_fold = [] ; new_USO_a_fold = []

            for p, a, HOD_a, UCO_a, HUC_a, CRUD_a, USO_a in zip(p_fold, a_fold, HOD_a_fold, UCO_a_fold, HUC_a_fold, CRUD_a_fold, USO_a_fold):
                if p > 0.5 and p > pred_threshold_sigmoid01_up:
                    new_p_fold.append(p) ; new_a_fold.append(a)
                    new_HOD_a_fold.append(HOD_a) ; new_UCO_a_fold.append(UCO_a)
                    new_HUC_a_fold.append(HUC_a) ; new_CRUD_a_fold.append(CRUD_a) ; new_USO_a_fold.append(USO_a)

                elif p > 0.5 and p <= pred_threshold_sigmoid01_up:
                    new_p_fold.append('below_threshold') ; new_a_fold.append('below_threshold')
                    new_HOD_a_fold.append('below_threshold') ; new_UCO_a_fold.append('below_threshold')
                    new_HUC_a_fold.append('below_threshold') ; new_CRUD_a_fold.append('below_threshold') ; new_USO_a_fold.append('below_threshold')
                else:
                    new_p_fold.append(p) ; new_a_fold.append(a)
                    new_HOD_a_fold.append(HOD_a) ;new_UCO_a_fold.append(UCO_a)
                    new_HUC_a_fold.append(HUC_a) ; new_CRUD_a_fold.append(CRUD_a) ; new_USO_a_fold.append(USO_a)

            all_preds_threshold_per_fold.append(new_p_fold) ; all_actuals_threshold_per_fold.append(new_a_fold)
            HOD_actuals_threshold_per_fold.append(new_HOD_a_fold) ; UCO_actuals_threshold_per_fold.append(new_UCO_a_fold)
            HUC_actuals_threshold_per_fold.append(new_HUC_a_fold) ; CRUD_actuals_threshold_per_fold.append(new_CRUD_a_fold) ; USO_actuals_threshold_per_fold.append(new_USO_a_fold)

        all_actuals_threshold_per_fold_flattened = [j for parts in all_actuals_threshold_per_fold for j in parts] 
        all_preds_threshold_per_fold_flattened = [j for parts in all_preds_threshold_per_fold for j in parts]
        HOD_actuals_threshold_per_fold_flattened = [j for parts in HOD_actuals_threshold_per_fold for j in parts]
        UCO_actuals_threshold_per_fold_flattened = [j for parts in UCO_actuals_threshold_per_fold for j in parts]
        HUC_actuals_threshold_per_fold_flattened = [j for parts in HUC_actuals_threshold_per_fold for j in parts]
        CRUD_actuals_threshold_per_fold_flattened = [j for parts in CRUD_actuals_threshold_per_fold for j in parts]
        USO_actuals_threshold_per_fold_flattened = [j for parts in USO_actuals_threshold_per_fold for j in parts]

        res["all_preds_threshold"] = all_preds_threshold_per_fold
        res["all_actuals_threshold"] = all_actuals_threshold_per_fold
        res["HOD_actuals_threshold"] = HOD_actuals_threshold_per_fold
        res["UCO_actuals_threshold"] = UCO_actuals_threshold_per_fold
        res["HUC_actuals_threshold"] = HUC_actuals_threshold_per_fold
        res["CRUD_actuals_threshold"] = CRUD_actuals_threshold_per_fold
        res["USO_actuals_threshold"] = USO_actuals_threshold_per_fold

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

        #### ETF CODE NEW
        HOD_false_up_preds_col_list_actual = []
        UCO_false_up_preds_col_list_actual = []
        HUC_false_up_preds_col_list_actual = []
        CRUD_false_up_preds_col_list_actual = []
        USO_false_up_preds_col_list_actual = []

        HOD_correct_up_preds_col_list_actual = []
        UCO_correct_up_preds_col_list_actual = []
        HUC_correct_up_preds_col_list_actual = []
        CRUD_correct_up_preds_col_list_actual = []
        USO_correct_up_preds_col_list_actual = []


        #### ETF CODE NEW

        #### ETF CODE NEW THRESH
        HOD_false_up_preds_col_list_actual_thresh = []
        UCO_false_up_preds_col_list_actual_thresh = []
        HUC_false_up_preds_col_list_actual_thresh = []
        CRUD_false_up_preds_col_list_actual_thresh = []
        USO_false_up_preds_col_list_actual_thresh = []

        HOD_correct_up_preds_col_list_actual_thresh = []
        UCO_correct_up_preds_col_list_actual_thresh = []
        HUC_correct_up_preds_col_list_actual_thresh = []
        CRUD_correct_up_preds_col_list_actual_thresh = []
        USO_correct_up_preds_col_list_actual_thresh = []

        #### ETF CODE NEW THRESH


        for row_raw_actuals, row_01_actuals, row_preds, row_preds_thresh , row_HUC_vals , row_HOD_vals , row_UCO_vals , row_CRUD_vals , row_USO_vals in zip(
            df["raw_actuals"], df["all_actuals"], df["all_preds"], df["all_preds_threshold"], df["HUC_values"], df["HOD_values"], df["UCO_values"], df["CRUD_values"], df["USO_values"]
        ):


            false_up_preds_row_actual = []
            false_up_preds_row_probabs = []
            false_up_preds_row_actual_thresh = []
            false_up_preds_row_probabs_thresh = []
            correct_up_preds_row_actual = []
            correct_up_preds_row_probabs = []
            correct_up_preds_row_actual_thresh = []
            correct_up_preds_row_probabs_thresh = []
            #### ETF CODE NEW
            HOD_false_up_preds_col_row_actual = []
            UCO_false_up_preds_col_row_actual = []
            HUC_false_up_preds_col_row_actual = []
            CRUD_false_up_preds_col_row_actual = []
            USO_false_up_preds_col_row_actual = []

            HOD_correct_up_preds_col_row_actual = []
            UCO_correct_up_preds_col_row_actual = []
            HUC_correct_up_preds_col_row_actual = []
            CRUD_correct_up_preds_col_row_actual = []
            USO_correct_up_preds_col_row_actual = []
            #### ETF CODE NEW
            #### ETF CODE NEW THRESH
            HOD_false_up_preds_col_row_actual = []
            UCO_false_up_preds_col_row_actual = []
            HUC_false_up_preds_col_row_actual = []
            CRUD_false_up_preds_col_row_actual = []
            USO_false_up_preds_col_row_actual = []

            HOD_correct_up_preds_col_row_actual = []
            UCO_correct_up_preds_col_row_actual = []
            HUC_correct_up_preds_col_row_actual = []
            CRUD_correct_up_preds_col_row_actual = []
            USO_correct_up_preds_col_row_actual = []
            #### ETF CODE NEW
            #### ETF CODE NEW THRESH
            HOD_false_up_preds_col_row_actual_thresh = []
            UCO_false_up_preds_col_row_actual_thresh = []
            HUC_false_up_preds_col_row_actual_thresh = []
            CRUD_false_up_preds_col_row_actual_thresh = []
            USO_false_up_preds_col_row_actual_thresh = []

            HOD_correct_up_preds_col_row_actual_thresh = []
            UCO_correct_up_preds_col_row_actual_thresh = []
            HUC_correct_up_preds_col_row_actual_thresh = []
            CRUD_correct_up_preds_col_row_actual_thresh = []
            USO_correct_up_preds_col_row_actual_thresh = []
            #### ETF CODE NEW THRESH

            row_raw_actuals_flattened = [p for fold in row_raw_actuals for p in fold]
            row_01_actuals_flattened = [p for fold in row_01_actuals for p in fold]
            row_preds_flattened = [p for fold in row_preds for p in fold]
            row_preds_thresh_flattened = [p for fold in row_preds_thresh for p in fold]
            #### ETF CODE NEW
            HOD_actuals_raw_flattened = [p for fold in row_HOD_vals for p in fold]
            UCO_actuals_raw_flattened = [p for fold in row_UCO_vals for p in fold]
            HUC_actuals_raw_flattened = [p for fold in row_HUC_vals for p in fold]
            CRUD_actuals_raw_flattened = [p for fold in row_CRUD_vals for p in fold]
            USO_actuals_raw_flattened = [p for fold in row_USO_vals for p in fold]
            #### ETF CODE NEW

            #### ETF CODE NEW THRESH
            HOD_actuals_raw_flattened_thresh = [p for fold in row_HOD_vals for p in fold]
            UCO_actuals_raw_flattened_thresh = [p for fold in row_UCO_vals for p in fold]
            HUC_actuals_raw_flattened_thresh = [p for fold in row_HUC_vals for p in fold]
            CRUD_actuals_raw_flattened_thresh = [p for fold in row_CRUD_vals for p in fold]
            USO_actuals_raw_flattened_thresh = [p for fold in row_USO_vals for p in fold]
            #### ETF CODE NEW THRESH

            # Normal version
            for entry_raw_actual, entry_01_actual, entry_pred , entry_HOD_actual, entry_UCO_actual, entry_HUC_actual, entry_CRUD_actual, entry_USO_actual in zip(row_raw_actuals_flattened, row_01_actuals_flattened, row_preds_flattened \
                                                                      , HOD_actuals_raw_flattened, UCO_actuals_raw_flattened, HUC_actuals_raw_flattened, CRUD_actuals_raw_flattened, USO_actuals_raw_flattened):
                
                if entry_pred > 0.5 and entry_01_actual < 0.5:
                    false_up_preds_row_actual.append(round(entry_raw_actual, 4)) 
                    false_up_preds_row_probabs.append(round(entry_pred,4))
                    HOD_false_up_preds_col_row_actual.append(round(entry_HOD_actual, 4))
                    UCO_false_up_preds_col_row_actual.append(round(entry_UCO_actual, 4))
                    HUC_false_up_preds_col_row_actual.append(round(entry_HUC_actual, 4))

            for entry_raw_actual, entry_01_actual, entry_pred , entry_HOD_actual, entry_UCO_actual, entry_HUC_actual in zip(row_raw_actuals_flattened, row_01_actuals_flattened, row_preds_flattened \
                                                                     , HOD_actuals_raw_flattened, UCO_actuals_raw_flattened, HUC_actuals_raw_flattened):
                if entry_pred > 0.5 and entry_01_actual > 0.5:
                    correct_up_preds_row_actual.append(round(entry_raw_actual, 4)) 
                    correct_up_preds_row_probabs.append(round(entry_pred,4))
                    HOD_correct_up_preds_col_row_actual.append(round(entry_HOD_actual, 4))
                    UCO_correct_up_preds_col_row_actual.append(round(entry_UCO_actual, 4))
                    HUC_correct_up_preds_col_row_actual.append(round(entry_HUC_actual, 4))
                    CRUD_correct_up_preds_col_row_actual.append(round(entry_CRUD_actual, 4))
                    USO_correct_up_preds_col_row_actual.append(round(entry_USO_actual, 4))

            # Threshold version
            for entry_raw_actual, entry_01_actual, entry_pred , entry_HOD_actual, entry_UCO_actual, entry_HUC_actual, entry_CRUD_actual, entry_USO_actual in zip(row_raw_actuals_flattened, row_01_actuals_flattened, row_preds_thresh_flattened \
                                                                     , HOD_actuals_raw_flattened, UCO_actuals_raw_flattened, HUC_actuals_raw_flattened, CRUD_actuals_raw_flattened, USO_actuals_raw_flattened):
                
                if (not isinstance(entry_pred, str)) and entry_pred > 0.5 and entry_01_actual < 0.5:
                    false_up_preds_row_actual_thresh.append(round(entry_raw_actual, 4))
                    false_up_preds_row_probabs_thresh.append(round(entry_pred,4))
                    HOD_false_up_preds_col_row_actual_thresh.append(round(entry_HOD_actual, 4))
                    UCO_false_up_preds_col_row_actual_thresh.append(round(entry_UCO_actual, 4))
                    HUC_false_up_preds_col_row_actual_thresh.append(round(entry_HUC_actual, 4))
                    CRUD_false_up_preds_col_row_actual_thresh.append(round(entry_CRUD_actual, 4))
                    USO_false_up_preds_col_row_actual_thresh.append(round(entry_USO_actual, 4))

            for entry_raw_actual, entry_01_actual, entry_pred , entry_HOD_actual, entry_UCO_actual, entry_HUC_actual, entry_CRUD_actual, entry_USO_actual in zip(row_raw_actuals_flattened, row_01_actuals_flattened, row_preds_thresh_flattened \
                                                                     , HOD_actuals_raw_flattened, UCO_actuals_raw_flattened, HUC_actuals_raw_flattened, CRUD_actuals_raw_flattened, USO_actuals_raw_flattened):

                if (not isinstance(entry_pred, str)) and entry_pred > 0.5 and entry_01_actual > 0.5:
                    correct_up_preds_row_actual_thresh.append(round(entry_raw_actual, 4))
                    correct_up_preds_row_probabs_thresh.append(round(entry_pred, 4))
                    HOD_correct_up_preds_col_row_actual_thresh.append(round(entry_HOD_actual, 4))
                    UCO_correct_up_preds_col_row_actual_thresh.append(round(entry_UCO_actual, 4))
                    HUC_correct_up_preds_col_row_actual_thresh.append(round(entry_HUC_actual, 4))
                    CRUD_correct_up_preds_col_row_actual_thresh.append(round(entry_CRUD_actual, 4))
                    USO_correct_up_preds_col_row_actual_thresh.append(round(entry_USO_actual, 4))

            false_up_preds_col_list_actual.append(false_up_preds_row_actual)
            false_up_preds_col_probabs_list_actual.append(false_up_preds_row_probabs)
            false_up_preds_col_list_actual_thresh.append(false_up_preds_row_actual_thresh)
            false_up_preds_col_probabs_list_actual_thresh.append(false_up_preds_row_probabs_thresh)
            
            correct_up_preds_col_list_actual.append(correct_up_preds_row_actual)
            correct_up_preds_col_probabs_list.append(correct_up_preds_row_probabs)
            correct_up_preds_col_list_actual_thresh.append(correct_up_preds_row_actual_thresh)
            correct_up_preds_col_probabs_list_thresh.append(correct_up_preds_row_probabs_thresh)

            #####ETF 
            HOD_false_up_preds_col_list_actual.append(HOD_false_up_preds_col_row_actual)
            UCO_false_up_preds_col_list_actual.append(UCO_false_up_preds_col_row_actual)
            HUC_false_up_preds_col_list_actual.append(HUC_false_up_preds_col_row_actual)
            CRUD_false_up_preds_col_list_actual.append(CRUD_false_up_preds_col_row_actual)
            USO_false_up_preds_col_list_actual.append(USO_false_up_preds_col_row_actual)

            HOD_correct_up_preds_col_list_actual.append(HOD_correct_up_preds_col_row_actual)
            UCO_correct_up_preds_col_list_actual.append(UCO_correct_up_preds_col_row_actual)
            HUC_correct_up_preds_col_list_actual.append(HUC_correct_up_preds_col_row_actual)
            CRUD_correct_up_preds_col_list_actual.append(CRUD_correct_up_preds_col_row_actual)
            USO_correct_up_preds_col_list_actual.append(USO_correct_up_preds_col_row_actual)

            #####ETF 


            #####ETF THRESH
            HOD_false_up_preds_col_list_actual_thresh.append(HOD_false_up_preds_col_row_actual_thresh)
            UCO_false_up_preds_col_list_actual_thresh.append(UCO_false_up_preds_col_row_actual_thresh)
            HUC_false_up_preds_col_list_actual_thresh.append(HUC_false_up_preds_col_row_actual_thresh)
            CRUD_false_up_preds_col_list_actual_thresh.append(CRUD_false_up_preds_col_row_actual_thresh)
            USO_false_up_preds_col_list_actual_thresh.append(USO_false_up_preds_col_row_actual_thresh)

            HOD_correct_up_preds_col_list_actual_thresh.append(HOD_correct_up_preds_col_row_actual_thresh)
            UCO_correct_up_preds_col_list_actual_thresh.append(UCO_correct_up_preds_col_row_actual_thresh)
            HUC_correct_up_preds_col_list_actual_thresh.append(HUC_correct_up_preds_col_row_actual_thresh)
            CRUD_correct_up_preds_col_list_actual_thresh.append(CRUD_correct_up_preds_col_row_actual_thresh)
            USO_correct_up_preds_col_list_actual_thresh.append(USO_correct_up_preds_col_row_actual_thresh)

            #####ETF 

        df["actuals_false_up"] = false_up_preds_col_list_actual
        df["false_up_preds"] = false_up_preds_col_probabs_list_actual
        df["actuals_false_up_thresh"] = false_up_preds_col_list_actual_thresh
        df["false_up_preds_thresh"] = false_up_preds_col_probabs_list_actual_thresh
        df["actuals_correct_up"] = correct_up_preds_col_list_actual
        df["correct_up_preds"] = correct_up_preds_col_probabs_list
        df["actuals_correct_up_thresh"] = correct_up_preds_col_list_actual_thresh
        df["correct_up_preds_thresh"] = correct_up_preds_col_probabs_list_thresh

        ##### ETF
        df["HOD_actuals_false_up"] = HOD_false_up_preds_col_list_actual
        df["UCO_actuals_false_up"] = UCO_false_up_preds_col_list_actual
        df["HUC_actuals_false_up"] = HUC_false_up_preds_col_list_actual
        df["CRUD_actuals_false_up"] = CRUD_false_up_preds_col_list_actual
        df["USO_actuals_false_up"] = USO_false_up_preds_col_list_actual

        df["HOD_actuals_correct_up"] = HOD_correct_up_preds_col_list_actual
        df["UCO_actuals_correct_up"] = UCO_correct_up_preds_col_list_actual
        df["HUC_actuals_correct_up"] = HUC_correct_up_preds_col_list_actual
        df["CRUD_actuals_correct_up"] = CRUD_correct_up_preds_col_list_actual
        df["USO_actuals_correct_up"] = USO_correct_up_preds_col_list_actual
        ##### ETF


        ##### ETF THRESH
        df["HOD_actuals_false_up_thresh"] = HOD_false_up_preds_col_list_actual_thresh
        df["UCO_actuals_false_up_thresh"] = UCO_false_up_preds_col_list_actual_thresh
        df["HUC_actuals_false_up_thresh"] = HUC_false_up_preds_col_list_actual_thresh
        df["CRUD_actuals_false_up_thresh"] = CRUD_false_up_preds_col_list_actual_thresh
        df["USO_actuals_false_up_thresh"] = USO_false_up_preds_col_list_actual_thresh

        df["HOD_actuals_correct_up_thresh"] = HOD_correct_up_preds_col_list_actual_thresh
        df["UCO_actuals_correct_up_thresh"] = UCO_correct_up_preds_col_list_actual_thresh
        df["HUC_actuals_correct_up_thresh"] = HUC_correct_up_preds_col_list_actual_thresh
        df["CRUD_actuals_correct_up_thresh"] = CRUD_correct_up_preds_col_list_actual_thresh
        df["USO_actuals_correct_up_thresh"] = USO_correct_up_preds_col_list_actual_thresh
        ##### ETF THRESH


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
                if param_key not in ['num_folds', 'end_value_train_set_fraction', 'val_set_fraction']:
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
                # Determine which prediction column to use based on threshold flag
                if data_type == "T":
                    preds_column = "all_preds_threshold_mac_T" if use_threshold_data else "all_preds_mac_T"

                if data_type == "V": ##### NEW NEW ERROR this was missing before SEPT 14 !!!! so it was always jsut T set
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



                ##### NEW NEW ERROR this was missing before SEPT 14 !!!! so it was always jsut T ser 
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
    HOD_sum_actuals_ups_list_ALL = []
    UCO_sum_actuals_ups_list_ALL = []
    HUC_sum_actuals_ups_list_ALL = []
    CRUD_sum_actuals_ups_list_ALL = []
    USO_sum_actuals_ups_list_ALL = []

    group_results = {}
    detailed_results = {}



    import pickle
    with open('ETF_dfs_WTI_preds_analysis/df_HUC_T_folds_values', 'rb') as f:
        df_HUC_T_folds_values = pickle.load(f)

    with open('ETF_dfs_WTI_preds_analysis/df_UCO_T_folds_values', 'rb') as f:
        df_UCO_T_folds_values = pickle.load(f)

    with open('ETF_dfs_WTI_preds_analysis/df_HOD_T_folds_values', 'rb') as f:
        df_HOD_T_folds_values = pickle.load(f)

    with open('ETF_dfs_WTI_preds_analysis/df_CRUD_T_folds_values', 'rb') as f:
        df_CRUD_T_folds_values = pickle.load(f)

    with open('ETF_dfs_WTI_preds_analysis/df_USO_T_folds_values', 'rb') as f:
        df_USO_T_folds_values = pickle.load(f)



    with open('ETF_dfs_WTI_preds_analysis/df_HUC_V_folds_values', 'rb') as f:
        df_HUC_V_folds_values = pickle.load(f)

    with open('ETF_dfs_WTI_preds_analysis/df_UCO_V_folds_values', 'rb') as f:
        df_UCO_V_folds_values = pickle.load(f)

    with open('ETF_dfs_WTI_preds_analysis/df_HOD_V_folds_values', 'rb') as f:
        df_HOD_V_folds_values = pickle.load(f)

    with open('ETF_dfs_WTI_preds_analysis/df_CRUD_V_folds_values', 'rb') as f:
        df_CRUD_V_folds_values = pickle.load(f)

    with open('ETF_dfs_WTI_preds_analysis/df_USO_V_folds_values', 'rb') as f:
        df_USO_V_folds_values = pickle.load(f)




    if data_type_ensemble == "T":
        flatten_HOD_values = [p for part in df_HOD_T_folds_values for p in part]
        flatten_HUC_values = [p for part in df_HUC_T_folds_values for p in part]
        flatten_UCO_values = [p for part in df_UCO_T_folds_values for p in part]
        flatten_CRUD_values = [p for part in df_CRUD_T_folds_values for p in part]
        flatten_USO_values = [p for part in df_USO_T_folds_values for p in part]

    if data_type_ensemble == "V":
        flatten_HOD_values = [p for part in df_HOD_V_folds_values for p in part]
        flatten_HUC_values = [p for part in df_HUC_V_folds_values for p in part]
        flatten_UCO_values = [p for part in df_UCO_V_folds_values for p in part]
        flatten_CRUD_values = [p for part in df_CRUD_V_folds_values for p in part]
        flatten_USO_values = [p for part in df_USO_V_folds_values for p in part]

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
            HOD_up_vals_predicted_raw_val = []
            UCO_up_vals_predicted_raw_val = []
            HUC_up_vals_predicted_raw_val = []
            CRUD_up_vals_predicted_raw_val = []
            USO_up_vals_predicted_raw_val = []
            correct_ups = 0

            for p, a, actual_binary, HOD_val, UCO_val, HUC_val, CRUD_val, USO_val in zip(flatten_preds, flatten_raw_actuals, flatten_actuals, flatten_HOD_values, flatten_UCO_values, flatten_HUC_values, flatten_CRUD_values, flatten_USO_values):
                if not isinstance(p, str) and p > 0.5:
                    up_vals_predicted_raw_val.append(a)
                    HOD_up_vals_predicted_raw_val.append(HOD_val)
                    UCO_up_vals_predicted_raw_val.append(UCO_val)
                    HUC_up_vals_predicted_raw_val.append(HUC_val)
                    CRUD_up_vals_predicted_raw_val.append(CRUD_val)
                    USO_up_vals_predicted_raw_val.append(USO_val)

                    group_sum_actuals += a
                    sum_actuals_ups_list_ALL.append(a)
                    HOD_sum_actuals_ups_list_ALL.append(HOD_val)
                    UCO_sum_actuals_ups_list_ALL.append(UCO_val)
                    HUC_sum_actuals_ups_list_ALL.append(HUC_val)
                    CRUD_sum_actuals_ups_list_ALL.append(CRUD_val)
                    USO_sum_actuals_ups_list_ALL.append(USO_val)
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
                f"{ensemble_result['combo_numbers']} - {p_ups} - Correct: {correct_ups} - {up_vals_predicted_raw_val} - HOD vals: {HOD_up_vals_predicted_raw_val} \
                    - UCO vals: {UCO_up_vals_predicted_raw_val} - HUC vals: {HUC_up_vals_predicted_raw_val} - CRUD vals: {CRUD_up_vals_predicted_raw_val} - USO vals: {USO_up_vals_predicted_raw_val}"
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
                "HOD_values_returns": HOD_up_vals_predicted_raw_val,
                "UCO_values_returns": UCO_up_vals_predicted_raw_val,
                "HUC_values_returns": HUC_up_vals_predicted_raw_val,
                "CRUD_values_returns": CRUD_up_vals_predicted_raw_val,
                "USO_values_returns": USO_up_vals_predicted_raw_val
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
        
        ## set outlier values past < -0.5 to -0.06
    if filter_outliers:
        for i in range(len(sum_actuals_ups_list_ALL)):
            if sum_actuals_ups_list_ALL[i] < -0.5:
                sum_actuals_ups_list_ALL[i] = -0.1

    print("before set:", sum_actuals_ups_list_ALL)
    print("after set:", set(sum_actuals_ups_list_ALL))
    total_ups = len(set(sum_actuals_ups_list_ALL))
    total_correct_ups = len(set([val for val in sum_actuals_ups_list_ALL if val < - 0.05]))  

    sum_actuals_ups = sum(set(sum_actuals_ups_list_ALL))
    HOD_sum_actuals_ups = sum(set(HOD_sum_actuals_ups_list_ALL))
    UCO_sum_actuals_ups = sum(set(UCO_sum_actuals_ups_list_ALL))
    HUC_sum_actuals_ups = sum(set(HUC_sum_actuals_ups_list_ALL))
    CRUD_sum_actuals_ups = sum(set(CRUD_sum_actuals_ups_list_ALL))
    USO_sum_actuals_ups = sum(set(USO_sum_actuals_ups_list_ALL))

    # Create overall summary
    summary = {
        "Total Up Predictions": total_ups,
        "Total Correct Up Predictions": total_correct_ups,
        "Sum of Actual Returns for Up Predictions": sum_actuals_ups,
        "Sum of HOD Actual Returns for Up Predictions": HOD_sum_actuals_ups,
        "Sum of UCO Actual Returns for Up Predictions": UCO_sum_actuals_ups,
        "Sum of HUC Actual Returns for Up Predictions": HUC_sum_actuals_ups,
        "Sum of CRUD Actual Returns for Up Predictions": CRUD_sum_actuals_ups,
        "Sum of USO Actual Returns for Up Predictions": USO_sum_actuals_ups,
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
            "HOD_sum_actuals_ups": HOD_sum_actuals_ups,
            "UCO_sum_actuals_ups": UCO_sum_actuals_ups,
            "HUC_sum_actuals_ups": HUC_sum_actuals_ups,
            "CRUD_sum_actuals_ups": CRUD_sum_actuals_ups,
            "USO_sum_actuals_ups": USO_sum_actuals_ups,
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
                                    names_val: dict,
                                    names_test: dict,
                                    etf_data_T_dict: dict,
                                    etf_data_V_dict: dict,
                                    #new arams 
                                    
                                    groups_config: list, 
                                    data_type_corr_groups_creation: str = "V",
                                    data_type_ensemble: str = "T",
                                    use_threshold_data: bool = False, 
                                    seed = None, 
                                    no_maps_per_group: int = 10 , 
                                    filter_outliers: bool = False,
                                    use_spearman_corr: bool = False
                                    ) -> dict:
    

    etf_data_T = etf_data_T_dict
    etf_data_V = etf_data_V_dict

    def add_etf_values(data , etf_data): 
        for res in data:
            for key in etf_data.keys():
                res[f"{key}_values"] = etf_data[key]


    for data in names_val.values():
        add_etf_values(data, etf_data_V)

    for data in names_test.values():
        add_etf_values(data, etf_data_T)

    for data in names_all.values():
        add_threshold_metrics(data, threshold = threshold)


    # Create DataFrames
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
    
    for machine, corr_range, group_size in groups_config:
        # Create group name based on parameters
        group_name = f"pair_{group_size}_{corr_range[0]}_{corr_range[1]}".replace('.', '').replace('-', 'neg')
        
        # Get model groups
        groups = get_model_groups_in_corr_range(master_df, machine, data_type_corr_groups_creation, corr_range, group_size=group_size, use_spearman_bool=use_spearman_corr)
        
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
    total_ups = 0
    total_correct_ups = 0
    # sum_actuals_ups = 0
    sum_actuals_ups_list_ALL = []
    HOD_sum_actuals_ups_list_ALL = []
    UCO_sum_actuals_ups_list_ALL = []
    HUC_sum_actuals_ups_list_ALL = []
    CRUD_sum_actuals_ups_list_ALL = []
    USO_sum_actuals_ups_list_ALL = []

    group_results = {}
    detailed_results = {}



    import pickle
    with open('ETF_dfs_WTI_preds_analysis/df_HUC_T_folds_values', 'rb') as f:
        df_HUC_T_folds_values = pickle.load(f)

    with open('ETF_dfs_WTI_preds_analysis/df_UCO_T_folds_values', 'rb') as f:
        df_UCO_T_folds_values = pickle.load(f)

    with open('ETF_dfs_WTI_preds_analysis/df_HOD_T_folds_values', 'rb') as f:
        df_HOD_T_folds_values = pickle.load(f)

    with open('ETF_dfs_WTI_preds_analysis/df_CRUD_T_folds_values', 'rb') as f:
        df_CRUD_T_folds_values = pickle.load(f)

    with open('ETF_dfs_WTI_preds_analysis/df_USO_T_folds_values', 'rb') as f:
        df_USO_T_folds_values = pickle.load(f)



    with open('ETF_dfs_WTI_preds_analysis/df_HUC_V_folds_values', 'rb') as f:
        df_HUC_V_folds_values = pickle.load(f)

    with open('ETF_dfs_WTI_preds_analysis/df_UCO_V_folds_values', 'rb') as f:
        df_UCO_V_folds_values = pickle.load(f)

    with open('ETF_dfs_WTI_preds_analysis/df_HOD_V_folds_values', 'rb') as f:
        df_HOD_V_folds_values = pickle.load(f)

    with open('ETF_dfs_WTI_preds_analysis/df_CRUD_V_folds_values', 'rb') as f:
        df_CRUD_V_folds_values = pickle.load(f)

    with open('ETF_dfs_WTI_preds_analysis/df_USO_V_folds_values', 'rb') as f:
        df_USO_V_folds_values = pickle.load(f)




    if data_type_ensemble == "T":
        flatten_HOD_values = [p for part in df_HOD_T_folds_values for p in part]
        flatten_HUC_values = [p for part in df_HUC_T_folds_values for p in part]
        flatten_UCO_values = [p for part in df_UCO_T_folds_values for p in part]
        flatten_CRUD_values = [p for part in df_CRUD_T_folds_values for p in part]
        flatten_USO_values = [p for part in df_USO_T_folds_values for p in part]

    if data_type_ensemble == "V":
        flatten_HOD_values = [p for part in df_HOD_V_folds_values for p in part]
        flatten_HUC_values = [p for part in df_HUC_V_folds_values for p in part]
        flatten_UCO_values = [p for part in df_UCO_V_folds_values for p in part]
        flatten_CRUD_values = [p for part in df_CRUD_V_folds_values for p in part]
        flatten_USO_values = [p for part in df_USO_V_folds_values for p in part]

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
            HOD_up_vals_predicted_raw_val = []
            UCO_up_vals_predicted_raw_val = []
            HUC_up_vals_predicted_raw_val = []
            CRUD_up_vals_predicted_raw_val = []
            USO_up_vals_predicted_raw_val = []
            correct_ups = 0

            for p, a, actual_binary, HOD_val, UCO_val, HUC_val, CRUD_val, USO_val in zip(flatten_preds, flatten_raw_actuals, flatten_actuals, flatten_HOD_values, flatten_UCO_values, flatten_HUC_values, flatten_CRUD_values, flatten_USO_values):
                if not isinstance(p, str) and p > 0.5:
                    up_vals_predicted_raw_val.append(a)
                    HOD_up_vals_predicted_raw_val.append(HOD_val)
                    UCO_up_vals_predicted_raw_val.append(UCO_val)
                    HUC_up_vals_predicted_raw_val.append(HUC_val)
                    CRUD_up_vals_predicted_raw_val.append(CRUD_val)
                    USO_up_vals_predicted_raw_val.append(USO_val)

                    group_sum_actuals += a
                    sum_actuals_ups_list_ALL.append(a)
                    HOD_sum_actuals_ups_list_ALL.append(HOD_val)
                    UCO_sum_actuals_ups_list_ALL.append(UCO_val)
                    HUC_sum_actuals_ups_list_ALL.append(HUC_val)
                    CRUD_sum_actuals_ups_list_ALL.append(CRUD_val)
                    USO_sum_actuals_ups_list_ALL.append(USO_val)
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
                f"{ensemble_result['combo_numbers']} - {p_ups} - Correct: {correct_ups} - {up_vals_predicted_raw_val} - HOD vals: {HOD_up_vals_predicted_raw_val} \
                    - UCO vals: {UCO_up_vals_predicted_raw_val} - HUC vals: {HUC_up_vals_predicted_raw_val} - CRUD vals: {CRUD_up_vals_predicted_raw_val} - USO vals: {USO_up_vals_predicted_raw_val}"
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
                "HOD_values_returns": HOD_up_vals_predicted_raw_val,
                "UCO_values_returns": UCO_up_vals_predicted_raw_val,
                "HUC_values_returns": HUC_up_vals_predicted_raw_val,
                "CRUD_values_returns": CRUD_up_vals_predicted_raw_val,
                "USO_values_returns": USO_up_vals_predicted_raw_val
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

    ## set outlier values past < -0.5 to -0.06
    if filter_outliers:
        for i in range(len(sum_actuals_ups_list_ALL)):
            if sum_actuals_ups_list_ALL[i] < -0.5:
                sum_actuals_ups_list_ALL[i] = -0.1

    total_ups = len(set(sum_actuals_ups_list_ALL))
    total_correct_ups = len(set([val for val in sum_actuals_ups_list_ALL if val < -0.05]))  

    sum_actuals_ups = sum(set(sum_actuals_ups_list_ALL))
    HOD_sum_actuals_ups = sum(set(HOD_sum_actuals_ups_list_ALL))
    UCO_sum_actuals_ups = sum(set(UCO_sum_actuals_ups_list_ALL))
    HUC_sum_actuals_ups = sum(set(HUC_sum_actuals_ups_list_ALL))
    CRUD_sum_actuals_ups = sum(set(CRUD_sum_actuals_ups_list_ALL))
    USO_sum_actuals_ups = sum(set(USO_sum_actuals_ups_list_ALL))

    # Create overall summary
    summary = {
        "Total Up Predictions": total_ups,
        "Total Correct Up Predictions": total_correct_ups,
        "Sum of Actual Returns for Up Predictions": sum_actuals_ups,
        "Sum of HOD Actual Returns for Up Predictions": HOD_sum_actuals_ups,
        "Sum of UCO Actual Returns for Up Predictions": UCO_sum_actuals_ups,
        "Sum of HUC Actual Returns for Up Predictions": HUC_sum_actuals_ups,
        "Sum of CRUD Actual Returns for Up Predictions": CRUD_sum_actuals_ups,
        "Sum of USO Actual Returns for Up Predictions": USO_sum_actuals_ups,
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
            "HOD_sum_actuals_ups": HOD_sum_actuals_ups,
            "UCO_sum_actuals_ups": UCO_sum_actuals_ups,
            "HUC_sum_actuals_ups": HUC_sum_actuals_ups,
            "CRUD_sum_actuals_ups": CRUD_sum_actuals_ups,
            "USO_sum_actuals_ups": USO_sum_actuals_ups,
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