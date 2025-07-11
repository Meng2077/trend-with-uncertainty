import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from eumap.misc import find_files, ttprint, nan_percentile, GoogleSheet
from eumap.raster import read_rasters, save_rasters
import warnings
import multiprocess as mp
import time
from scipy.special import expit, logit
import warnings
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, cross_val_score, HalvingGridSearchCV, KFold, GroupKFold
from sklearn.model_selection import RandomizedSearchCV, HalvingRandomSearchCV, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
import joblib
import pickle
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
# from cubist import Cubist
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import os
from scipy.stats import randint, uniform
# import mathtil
from datetime import datetime
import random
import math
import seaborn as sns

# Prototype pipeline for global soil macrofauna mapping
# Developed by OpenGeoHub team (Xuemeng Tian, Martijn Witjes, and Davide Consoli)

def read_features(file_path):
    with open(file_path, 'r') as file:
        features = [line.strip() for line in file.readlines()]
    return features

def find_knee(df):
    slopes = (df['accum'].diff(-1)) / (df['freq'].diff(-1))*(-1)
    knee_index = slopes.idxmax()
    return knee_index

def calc_ccc(y_true, y_pred):
    if len(y_true) <= 1 or len(y_pred) <= 1:
        return np.nan  # Return NaN if there is insufficient data

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    try:
        cov_matrix = np.cov(y_true, y_pred)
        covariance = cov_matrix[0, 1]
        var_true = cov_matrix[0, 0]
        var_pred = cov_matrix[1, 1]
    except Warning:
        warnings.warn("Covariance calculation encountered an issue.")
        return np.nan  # Return NaN if covariance calculation fails

    if var_true + var_pred + (mean_true - mean_pred) ** 2 == 0:
        return np.nan  # Avoid division by zero

    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in multiply")

def calc_metrics(y_true, y_pred, space):
    if space == 'normal':
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        bias = np.nanmean(y_pred-y_true)
        ccc = calc_ccc(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
    else:
        ccc = calc_ccc(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # report MAE and MAPE in original scale
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        bias = np.nanmean(y_pred-y_true)
    return rmse, mae, medae, mape, ccc, r2, bias


def cfi_calc(data, tgt, prop, space, output_folder, version, covs_all):
    data = data.dropna(subset=covs_all,how='any')
    n_bootstrap=20
    ntrees = 100
    
    runs = []
    feature_importances = []
    
    random.seed(42)
    rn = [random.randint(0, 100) for _ in range(20)]

    print(f'start bootstrap on different subset...')
    for k in range(n_bootstrap):
        
        np.random.seed(k)
        train, test = train_test_split(
            data,
            test_size=0.3,
            random_state=rn[k]
        )
        
        ttprint(f'{k} iteration, training size: {len(train)}')
        rf = RandomForestRegressor(random_state=41, n_jobs=80, n_estimators=ntrees)
        rf.fit(train[covs_all], train[tgt])
        
        # impurity-based feature importance
        feature_importances.append(rf.feature_importances_)
        
        # # feature permutation
        # result_fi = permutation_importance(rf, test[covs_all], test[tgt], n_jobs=80, n_repeats=10, random_state=42)
        # feature_importances.append(result_fi.importances_mean)
            
    result = pd.DataFrame(feature_importances, columns=covs_all)
        
    sorted_importances = result.mean(axis=0).sort_values(ascending=False)
    sorted_importances = sorted_importances.reset_index()
    sorted_importances.columns = ['feature', 'cfi']
    # version = datetime.today().strftime('%Y%m%d')
    sorted_importances.to_csv(f'{output_folder}/feature_cfi_{prop}_{version}.csv',index=False)
    
    return sorted_importances
    
def rscfi(data, tgt, prop, space, output_folder, version, covs_all, sorted_importances, threshold_num=[50,100], step_size=0.0002):
    min_num, max_num = threshold_num
    max_threshold = sorted_importances['cfi'].max()
    min_threshold = sorted_importances['cfi'].min()
    thresholds = np.arange(min_threshold, max_threshold + step_size, step_size)
    if 0 not in thresholds:
        idx = np.searchsorted(thresholds, 0)  
        thresholds = np.insert(thresholds, idx, 0)

    previous_feature_set = set([])
    results = []
    data = data.dropna(subset=covs_all,how='any')
    
    n_splits = 5
    ntrees = 100
    spatial_cv_column='tile_id'
    groups = data[spatial_cv_column].unique()
    
    for threshold in thresholds:
        current_features = sorted_importances.loc[sorted_importances['cfi'] >= threshold, 'feature'].tolist()
        
        if set(current_features) == previous_feature_set:
            continue  # Skip if feature set doesn't change
        previous_feature_set = set(current_features)

        if len(current_features)<2:
            break  # Stop if limited (<2) features are left

        ttprint(f'processing {threshold} ...')
        rf = RandomForestRegressor(random_state=41, n_jobs=80, n_estimators=ntrees)
        group_kfold = GroupKFold(n_splits=n_splits)

        groups = data[spatial_cv_column].values
        y_pred = cross_val_predict(rf, data[current_features], data[tgt], cv=group_kfold, groups=groups, n_jobs=-1)
        y_true = data[tgt]

        metrics = calc_metrics(y_true, y_pred, space)
        results.append((threshold, len(current_features), *metrics))

    results_df = pd.DataFrame(results, columns=['Threshold', 'Num_Features', 'RMSE', 'MAE', 'MedAE','MAPE','CCC', 'R2', 'bias'])
    results_df = results_df.drop(columns=['MAE', 'MedAE','MAPE','R2'])
    # results_df['MAE_Rank'] = results_df['MAE'].rank(ascending=True)
    results_df['RMSE_Rank'] = results_df['RMSE'].rank(ascending=True)
    # results_df['MedAE_Rank'] = results_df['MedAE'].rank(ascending=True)
    results_df['bias_Rank'] = results_df['bias'].abs().rank(ascending=True)
    results_df['CCC_Rank'] = results_df['CCC'].rank(ascending=False)
    # results_df['R2_Rank'] = results_df['R2'].rank(ascending=False)
    results_df['Combined_Rank'] = results_df['RMSE_Rank'] + results_df['CCC_Rank']# + results_df['bias_Rank']
    
    # select threshold
    results_df = results_df.sort_values(by=['Combined_Rank', 'Num_Features'], ascending=[True, True]).reset_index(drop=True)
    # version = datetime.today().strftime('%Y%m%d')
    results_df.to_csv(f'{output_folder}/feature_metrics.elimination_{prop}_{version}.csv', index=False)
    
    for index, row in results_df.iterrows():
        if index == 0:
            best_threshold = row['Threshold']
        if (row['Num_Features'] <= max_num) & (row['Num_Features'] >= min_num):
            selected_threshold = row['Threshold']
            break
    # Best combined rank
    best_num_features = results_df.loc[results_df['Threshold'] == best_threshold, 'Num_Features'].values[0]
    # results_df.loc[best_combined_rank_index, 'Num_Features']
    selected_num_features = results_df.loc[results_df['Threshold'] == selected_threshold, 'Num_Features'].values[0]
          
    
    features_df = sorted_importances[sorted_importances['cfi'] >= selected_threshold]
    covs = features_df['feature'].tolist()
    
    # if 'hzn_dep' not in covs:
    #     print(f'{prop} model did not select depth as covs, adding it')
    #     covs.append('hzn_dep')
        
    print(f'**{len(covs)} features selected for {prop}, cfi threshold: {selected_threshold}**')
    with open(f'{output_folder}/feature_selected_{prop}_{version}.txt', 'w') as file:
        for item in covs:
            file.write(f"{item}\n")
        
    results_df = results_df.sort_values(by='Threshold')
    results_df['scaled_RMSE'] = (results_df['RMSE'] - results_df['RMSE'].min()) / (results_df['RMSE'].max() - results_df['RMSE'].min())

    # plot feature elimination analysis
    fig, ax1 = plt.subplots(figsize=(11, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Feature Importance Threshold', fontsize=16)
    ax1.set_ylabel('Number of Features', color=color, fontsize=16)
    line1 = ax1.plot(results_df['Threshold'], results_df['Num_Features'], color=color, marker='o', label='Num_Feat')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Evaluation Metrics', color=color, fontsize=16)
    line2 = ax2.plot(results_df['Threshold'], results_df['CCC'], color='tab:green', marker='x', linestyle='-', linewidth=2, label='CCC')
    line3 = ax2.plot(results_df['Threshold'], results_df['scaled_RMSE'], color='tab:orange', marker='^', linestyle='-', linewidth=2, label='scaled RMSE')

    ax2.tick_params(axis='y', labelcolor=color, labelsize=14)

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to avoid cutting off the title

    # Combine legends
    lines = line1 + line2 + line3# + line4 #+ line5
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.15, 0.95), fontsize=14, framealpha=0.5)

    # Vertical line for the best and selected threshold
    ax1.axvline(x=best_threshold, color='grey', linestyle='--', label='Best Threshold')
    ax1.axvline(x=selected_threshold, color='cyan', linestyle='--', label='Selected Threshold')
    
    # Update the legend to include the vertical line label
    lines += [ax1.axvline(x=best_threshold, color='grey', linestyle='--')]
    labels += ['Best Threshold']
    lines += [ax1.axvline(x=selected_threshold, color='cyan', linestyle='--')]
    labels += ['Selected Threshold']
    ax1.legend(lines, labels, loc='upper right', fontsize=14, framealpha=0.5)#, bbox_to_anchor=(0.15, 0.95)

    plt.title(f'{prop}\nbest feature number: {best_num_features}, select {selected_num_features}', fontsize=16)
    plt.savefig(f'{output_folder}/feature_plot.elimination_{prop}_{version}.pdf')
    plt.show()
    
    return covs

    
def parameter_fine_tuning(cal, covs, tgt, prop, output_folder, version):
    models = [] #[rf, ann, lgb, rf_weighted, lgb_weighted] #cubist, cubist_weighted, 
    model_names = [] #['rf', 'ann', 'lgb', 'rf_weighted', 'lgb_weighted'] # 'cubist',, 'cubist_weighted'
    cal = cal.dropna(subset=covs,how='any')

    ### parameter fine tuning
    spatial_cv_column = 'tile_id'
    cv = GroupKFold(n_splits=5)
    ccc_scorer = make_scorer(calc_ccc, greater_is_better=True)
    fitting_score = ccc_scorer
    
    ## no weights version
    # random forest
    ttprint('----------------------rf------------------------')
    param_rf = {
        'n_estimators': [80, 100, 120],
        "criterion": ['squared_error'], #['squared_error', 'absolute_error', 'poisson', 'friedman_mse'],
        'max_depth': [10, 20, 30],
        'max_features': [0.3, 0.5, 0.7, 'log2', 'sqrt'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    tune_rf = HalvingGridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid=param_rf,
        scoring=fitting_score,
        n_jobs=90, 
        cv=cv,
        verbose=1,
        random_state = 1992
    )
    tune_rf.fit(cal[covs], cal[tgt], groups=cal[spatial_cv_column])
    warnings.filterwarnings('ignore')
    rf = tune_rf.best_estimator_
    joblib.dump(rf, f'{output_folder}/model_rf.{prop}_ccc_{version}.joblib')
    models.append(rf)
    model_names.append('rf')
    
    # # # lightGBM
    # import lightgbm as lgb
    # ttprint('----------------------lightGBM------------------------')
    # def clean_feature_names(df):
    #     df.columns = [col.replace('{', '').replace('}', '').replace(':', '').replace(',', '').replace('"', '') for col in df.columns]
    #     return df
    # from sklearn.preprocessing import FunctionTransformer
    # clean_names_transformer = FunctionTransformer(clean_feature_names, validate=False)
    # pipeline = Pipeline([
    #     ('clean_names', clean_names_transformer),  # Clean feature names
    #     ('lgbm', lgb.LGBMRegressor(random_state=35,verbose=-1))         # Replace with any model you intend to use
    # ])
    # param_lgb = {
    #     'lgbm__n_estimators': [80, 100, 120],  # Lower initial values for quicker testing
    #     'lgbm__max_depth': [3, 5, 7],  # Lower maximum depths
    #     'lgbm__num_leaves': [20, 31, 40],  # Significantly fewer leaves
    #     'lgbm__learning_rate': [0.01, 0.05, 0.1],  # Fine as is, covers a good range
    #     'lgbm__min_child_samples': [20, 30, 50],  # Much lower values to accommodate small data sets
    #     'lgbm__subsample': [0.8, 1.0],  # Reduced range, focusing on higher subsampling
    #     'lgbm__colsample_bytree': [0.8, 1.0],  # Less variation, focus on higher values
    #     'lgbm__verbosity': [-1]
    # }
    # tune_lgb = HalvingGridSearchCV(
    #     estimator=pipeline,
    #     param_grid=param_lgb,
    #     scoring=fitting_score,
    #     n_jobs=90,
    #     cv=cv,
    #     verbose=1,
    #     random_state=1994
    # )
    # tune_lgb.fit(cal[covs], cal[tgt], groups=cal[spatial_cv_column])
    # lgbmd = tune_lgb.best_estimator_
    # joblib.dump(lgbmd, f'{output_folder}/model_lgb.{prop}_ccc.joblib')
    # models.append(lgbmd)
    # model_names.append('lgb')
    
    return models, model_names

from matplotlib.colors import LinearSegmentedColormap

# Define the custom CET-L19 colormap
cet_l19_cmap = LinearSegmentedColormap.from_list(
    "CET-L19", ["#abdda4", "#ffffbf", "#fdae61", "#d7191c"]
)

def accuracy_plot(y_test, y_pred, prop, space, mdl, test_type, output_folder):
    rmse, mae, medae, mape, ccc, r2, bias = calc_metrics(y_test, y_pred, space)
    
    show_range = [
        math.floor(np.min([y_test.min(), y_pred.min()])),
        math.ceil(np.max([y_test.max(), y_pred.max()]))]
    vmax = 0.4 * len(y_test)
    
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(8, 7))
    
    ax.set_title(f'{test_type} of {mdl} in {space} scale using {len(y_test)} data\nRMSE={rmse:.2f}, CCC={ccc:.2f}, bias={bias:.2f}')
    
    # Use the CET-L19 colorblind-friendly colormap
    hb = ax.hexbin(y_pred, y_test, gridsize=(20, 20), cmap=cet_l19_cmap, mincnt=2, vmax=vmax, bins='log')
    ax.set_xlabel(f'Predicted {prop}')
    ax.set_ylabel(f'Observed {prop}')
    ax.set_aspect('auto', adjustable='box')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(show_range, show_range, "-k", alpha=.5)
    
    # Create a colorbar with proper spacing
    cax = fig.add_axes([ax.get_position().x1 + 0.05, ax.get_position().y0, 0.02, ax.get_position().height])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label('Count')
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])  # Adjust the right margin to make room for colorbar
    plt.savefig(f'{output_folder}/plot_accuracy.{test_type}_{mdl}.{prop}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    return rmse, mae, medae, mape, ccc, r2, bias

def accuracy_strata_plot(metric, strata_df, prop, mdl):

    plt.figure(figsize=(10, 6))
    pivot_data = strata_df.pivot(index='clm_class', columns='hzn_dep_bin', values=metric)

    pivot_data = pivot_data.reindex(['0-30', '60-100', '30-60', '>100'],axis=1)

    ax = sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".2f",
        cmap=cet_l19_cmap,
        cbar_kws={'label': metric},
        annot_kws={"fontsize": 11},  # Smaller font size for annotations
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)  # Font size for colorbar ticks
    cbar.set_label(metric, size=14)  # Font size for colorbar label

    for i in range(pivot_data.shape[0] + 1):  # Horizontal grid lines
        ax.axhline(i, color='black', linewidth=1)
    for j in range(pivot_data.shape[1] + 1):  # Vertical grid lines
        ax.axvline(j, color='black', linewidth=1)

    ax.set_title(f'{prop}, {mdl}, {metric}', fontsize=14)
    ax.set_xlabel('Climate class', fontsize=14)
    ax.set_ylabel('Soil depth', fontsize=14)

    ax.set_xticks(np.arange(len(pivot_data.columns)) + 1)
    ax.set_xticklabels(pivot_data.columns, rotation=30, ha='right', fontsize=14)

    ax.set_yticklabels(pivot_data.index, rotation=60, fontsize=14)

    plt.tight_layout()
    plt.show()



def separate_data(prop, space, output_folder, version, df, strata_col): 
    # df = pd.read_csv(f'/home/opengeohub/xuemeng/work_xuemeng/soc/data/002_data_whole.csv',low_memory=False) 
    os.makedirs(output_folder, exist_ok=True)
    
    ### data set preparation
    # clean the data according to each properties
    ini_len = len(df)
    if prop+'_qa' in df.columns:
        df = df.loc[df[prop].notna() & (df[prop+'_qa']>2)]
        print(f'drop invalid {ini_len-len(df)} records')
    # df[prop].hist(bins=40)
    
    # set target variable
    if space=='log1p':
        df.loc[:,f'{prop}_log1p'] = np.log1p(df[prop])
        tgt = f'{prop}_log1p'
    else:
        tgt = prop
       
    if len(strata_col)>1:
        df['strata'] = df[strata_col].astype(str).agg(','.join, axis=1)
    else:
        df['strata'] = df[strata_col[0]]
    strata_counts = df['strata'].value_counts()

    # Step 1: Separate small and large strata
    min_strata_count = 10
    strata_counts = df['strata'].value_counts()
    small_strata = strata_counts[strata_counts < min_strata_count].index
    large_strata = strata_counts[strata_counts >= min_strata_count].index
    small_strata_data = df[df['strata'].isin(small_strata)]
    large_strata_data = df[df['strata'].isin(large_strata)]

    # Step 2: Handle small strata with randomness
    len_group_strata = len(small_strata_data['strata'].unique())
    len_data_strata = len(small_strata_data)
    print(f'{len_group_strata} small strata groups, with {len_data_strata} data records')
    if len_group_strata<20:
        print('small strata groups:',small_strata_data['strata'].unique().tolist())
    
    test_small = []
    calibration_small = []
    train_small = []

    for stratum in small_strata:
        subset = small_strata_data[small_strata_data['strata'] == stratum]
        if len(subset) < 3:
            # Randomly assign to train, test, or calibration
            assignments = np.random.choice(['train', 'test', 'calibration'], size=len(subset), replace=False)
            for idx, assignment in zip(subset.index, assignments):  # Use subset index to align with assignments
                row = subset.loc[[idx]]  # Wrap row in a DataFrame
                if assignment == 'train':
                    train_small.append(row)
                elif assignment == 'test':
                    test_small.append(row)
                else:
                    calibration_small.append(row)
        else:
            # Randomly pick one for test and one for calibration, the rest for training
            subset = subset.sample(frac=1, random_state=42)  # Shuffle rows
            test_small.append(subset.iloc[[0]])              # First random row for test
            calibration_small.append(subset.iloc[[1]])       # Second random row for calibration
            train_small.append(subset.iloc[2:])             # Remaining rows for training

    # Combine small strata data
    train_small = pd.concat(train_small, ignore_index=True)
    test_small = pd.concat(test_small, ignore_index=True)
    calibration_small = pd.concat(calibration_small, ignore_index=True)


    # Step 3: Split Large Strata Data into Training and Temporary (Test + Calibration)
    temp_large, test_large = train_test_split(
        large_strata_data,
        test_size=min(0.1, 4000 / len(large_strata_data)),
        stratify=large_strata_data['strata'],
        random_state=42
    )
    
    # Split Temporary Data into Test and Calibration
    train_large, calibration_large = train_test_split(
        temp_large,
        test_size=min(0.125, 8000 / len(temp_large)),
        stratify=temp_large['strata'],
        random_state=42
    )

    # Step 4: Combine all splits
    train = pd.concat([train_large, train_small])
    test = pd.concat([test_large, test_small])
    cal = pd.concat([calibration_large, calibration_small])

    # Drop the temporary strata column (optional)
    train = train.drop(columns=['strata'])
    test = test.drop(columns=['strata'])
    cal = cal.drop(columns=['strata'])
    
    # verbose
    lsum = len(cal)+len(train)+len(test)
    print(f'size: calibration {len(cal)}, training {len(train)}, test {len(test)}')
    print(f'ratio: calibration {len(cal)/lsum:.2f}, training {len(train)/lsum:.2f}, test {len(test)/lsum:.2f}')
    print(f'sum {lsum}, df {len(df)}')
    
    # name with version
    # version = datetime.today().strftime('%Y%m%d')
    cal.to_parquet(f'{output_folder}/data_cal_{prop}_{version}.pq')
    train.to_parquet(f'{output_folder}/data_train_{prop}_{version}.pq')
    test.to_parquet(f'{output_folder}/data_test_{prop}_{version}.pq')
    return cal, train, test


import matplotlib.pyplot as plt
def plot_top_features(prop, mdl, data_path, top_n=10):
    """
    Plots the top N features by importance in descending order.
    
    Parameters:
    - feature_importance_df (pd.DataFrame): A DataFrame with two columns: 
      'feature' and 'importance', sorted in descending order of importance.
    - top_n (int): Number of top features to plot. Default is 15.
    """
    feature_importance_df = pd.read_csv(f'{data_path}/{prop}/feature.importances_{prop}_{mdl}.txt', delimiter='\t') 
    # Ensure the DataFrame is sorted by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    # Select the top N features
    top_features = feature_importance_df.head(top_n)
    
    # edit the feature names to make them less lengthy
    names = []
    for ii in top_features['feature'].to_list():
        if len(ii.split('_'))>5:
            kk = ii.split('_')[0] + '_' + ii.split('_')[1] + '_' + ii.split('_')[2] 
            if ('km_' in ii.split('_')[3]) & ('0m_' in ii.split('_')[3]):
                kk = kk + '_' + ii.split('_')[3]
        elif len(ii.split('_'))>2:
            kk = ii.split('_')[0] + '_' + ii.split('_')[1]
        else:
            kk = ii
            
        names.append(kk)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(names, top_features['importance'], color='skyblue')
    plt.xlabel('Feature importance')
    plt.title(f'{prop}\ntop {top_n} most important features')
    plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
    # plt.show()
    
    plt.tight_layout()
    plt.savefig(f'{data_path}/{prop}/plot_feature.importance_{prop}.{mdl}.pdf')
    plt.show()  # Display the plot
    plt.close()
    
def plot_histogram(df, prop, space, data_path):
    if space == 'normal':
        plt.figure(figsize=(10, 6))
        plt.hist(df[prop], bins=40, alpha=0.75)
        plt.title(f'Histogram of {prop}\nin normal scale')
        # plt.xlabel(prop)
        plt.ylabel('Count')
        plt.savefig(f'{data_path}/{prop}/plot_histogram_{prop}.pdf', bbox_inches='tight')
        plt.show()
        plt.close()
    elif space == 'log1p':
        # Create the log1p transformed column
        df[prop + '_log1p'] = np.log1p(df[prop])

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        # Original data histogram
        axes[0].hist(df[prop], bins=40, alpha=0.75)
        axes[0].set_title('In normal scale')
        # axes[0].set_xlabel(prop)

        # Transformed data histogram
        axes[1].hist(df[prop + '_log1p'], bins=40, alpha=0.75)
        axes[1].set_title('In log1p scale')
        # axes[1].set_xlabel(f'{prop}_log1p')

        # Setting a shared y-axis label
        fig.text(0.04, 0.5, 'Count', va='center', rotation='vertical', fontsize=18)
        fig.suptitle(f'Histograms of {prop}', fontsize=20)

        # Adjust layout and save to PDF
        plt.tight_layout(rect=[0.05, 0, 1, 1])  # Adjust layout to make room for the y-label
        plt.savefig(f'{data_path}/{prop}/plot_histogram_{prop}.pdf', format='pdf')
        plt.show()
        plt.close()
        
def pdp_hexbin(df, prop, space, mdl, data_path, fn = 3, grid_resolution=50, bins = None, cmap=cet_l19_cmap):
    """
    Generate a partial dependence hexbin heatmap for a single feature.

    Parameters:
    - model: Trained model (should support `predict`).
    - df: DataFrame of input features.
    - feature: Feature name for which to generate PDP.
    - grid_resolution: Number of points to evaluate PDP along the feature's range.
    - bins: Binning method for hexbin ('log', 'linear', etc.).
    - cmap: Colormap for hexbin.
    - output_path: Path to save the plot. If None, the plot will be shown.
    """
    # get the top features that we would like to plot pdp for
    feature_importance_df = pd.read_csv(f'{data_path}/{prop}/feature.importances_{prop}_{mdl}.txt', delimiter='\t') 
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    top_features =  feature_importance_df.head(fn)['feature'].to_list()
    
    # load the model to do the pdp generation
    model = joblib.load(f'{data_path}/{prop}/model_rf.{prop}_ccc.joblib')
    
    # read in all the features
    covs = read_features(f'{data_path}/{prop}/benchmark_selected.covs_{prop}.txt')
    
    # prepare the data
    df = df.dropna(subset = covs+[prop])
    if space=='log1p':
        df.loc[:,f'{prop}_log1p'] = np.log1p(df[prop])
        tgt = f'{prop}_log1p'
    else:
        tgt = prop
        
    # start plotting
    idx = 0
    for feature in top_features:
        idx = idx+1
        feature_values = np.linspace(df[feature].min(), df[feature].max(), grid_resolution)
        pd_values = []

        # Iterate over the feature grid, calculate predictions while fixing the feature value
        for val in feature_values:
            # Copy the original data to avoid modifying it
            X_temp = df.copy()

            # Set the feature of interest to the current grid value for all samples
            X_temp[feature] = val

            # Predict and calculate the mean prediction for the fixed feature value
            mean_prediction = model.predict(X_temp[covs]).mean()
            pd_values.append(mean_prediction)
            
        if len(feature.split('_'))>5:
            fname = feature.split('_')[0] + '_' + feature.split('_')[1] + '_' + feature.split('_')[2] 
            if ('km_' in feature.split('_')[3]) & ('0m_' in feature.split('_')[3]):
                fname = fname + '_' + feature.split('_')[3]
        elif len(feature.split('_'))>2:
            fname = feature.split('_')[0] + '_' + feature.split('_')[1]
        else:
            fname = feature

        plt.figure(figsize=(8, 6))
        plt.hexbin(feature_values, pd_values, gridsize=30, mincnt=1, cmap=cmap, bins=bins if bins else None)
        plt.colorbar(label='Density')
        plt.xlabel(fname)
        plt.ylabel(prop)
        plt.title(f'{mdl} PDP')
        plt.savefig(f"{data_path}/{prop}/plot_pdp_top{idx}.{prop}.pdf", format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()
        
        
        
def textures_fw_transform(sand, silt, clay, k=1, a=100):
    """
    Forward transformation from sand, silt, and clay fractions to texture_1 and texture_2
    using logarithm base 2.

    Parameters:
    - sand, silt, clay: Soil texture fractions (absolute values, not normalized)
    - k: Small offset to avoid division by zero
    - a: Normalization factor (typically 100)

    Returns:
    - texture_1, texture_2: Transformed soil texture values
    """
    texture_1 = np.log2((sand / a + k) / (clay / a + k))
    texture_2 = np.log2((silt / a + k) / (clay / a + k))
    
    return texture_1, texture_2

    
def textures_bw_transform(texture_1, texture_2, k=1, a=100):
    """
    Backward transformation from texture_1 and texture_2 back to sand, silt, and clay fractions.

    Parameters:
    - texture_1, texture_2: Transformed soil texture values
    - k: Small offset used in the forward transform
    - a: Normalization factor (typically 100)

    Returns:
    - sand, silt, clay: Reconstructed soil texture fractions (absolute values)
    """
    # Invert log2 transformation
    x1 = 2 ** texture_1
    x2 = 2 ** texture_2

    # Solve for clay fraction
    C = (1 - (x1 + x2 - 2) * k) / (x1 + x2 + 1)

    # Solve for sand and silt fractions
    S = x1 * C + x1 * k - k
    L = x2 * C + x2 * k - k

    # Convert back to absolute values
    sand = a * S
    silt = a * L
    clay = a * C

    return sand, silt, clay

from scipy.spatial import distance_matrix

def pairwise_var(gdft):
    coords = np.array([[geom.x, geom.y] for geom in gdft.geometry])
    values = gdft['std_rsd'].to_numpy()

    # Compute pairwise distances (in km)
    dist_matrix = distance_matrix(coords, coords) / 1000  

    # Compute squared differences for all pairs
    diff_matrix = (values[:, None] - values[None, :]) ** 2  

    # Extract upper triangular part (excluding diagonal)
    i, j = np.triu_indices(len(gdft), k=1)
    all_dists = dist_matrix[i, j]
    all_diffs = diff_matrix[i, j]

    # Sort based on distances
    sort_idx = np.argsort(all_dists)
    return all_dists[sort_idx], all_diffs[sort_idx]



def bin_semivar(all_dists, all_diffs, bins_size):
    # bin size in km
    # truncate distance in km
    bin_edges = np.arange(0, max(all_dists), bins_size)  
    bin_indices = np.digitize(all_dists, bin_edges)

    # initialize the variogram bin sizes
    binned_variograms = np.zeros(len(bin_edges) - 1)
    binned_dist = np.zeros(len(bin_edges) - 1)
    binned_number = np.zeros(len(bin_edges) - 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # midpoint of each bin

    for i in range(1, len(bin_edges)):
        # Select data in the current bin
        mask = bin_indices == i
        binned_number[i - 1] = np.sum(mask)
        if np.sum(mask) > 0:
            binned_dist[i - 1] = np.nanmean(all_dists[mask])  # Average semi-variance for this bin
            binned_variograms[i - 1] = np.nanmean(all_diffs[mask])  # Average semi-variance for this bin

    # Removing bins with zero count
    mask_nonzero = binned_number > 0

    # Filter the data
    filtered_binned_variograms = binned_variograms[mask_nonzero]
    filtered_binned_dist = binned_dist[mask_nonzero]
    filtered_binned_number = binned_number[mask_nonzero]
    filtered_bin_centers = bin_centers[mask_nonzero]
    
    return filtered_binned_dist, filtered_binned_variograms, filtered_bin_centers, filtered_binned_number

from scipy.optimize import curve_fit
# Define variogram models
def spherical_model(h, nugget, sill, range_):
    return np.where(h < range_, nugget + (sill - nugget) * (1.5 * (h / range_) - 0.5 * (h / range_)**3), sill)

def exponential_model(h, nugget, sill, range_):
    return nugget + (sill - nugget) * (1 - np.exp(-h / range_))

def gaussian_model(h, nugget, sill, range_):
    return nugget + (sill - nugget) * (1 - np.exp(- (h**2) / (range_**2)))

# Use curve_fit to fit the empirical variogram data to a model
def fit_variogram_model(model, h, gamma):
    # Initial guess for parameters: nugget, sill, range
    initial_params = [0.1, 1.0, 10.0]
    params, _ = curve_fit(model, h, gamma, p0=initial_params)
    
    return params