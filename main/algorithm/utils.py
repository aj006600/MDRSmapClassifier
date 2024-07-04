import pandas as pd
import numpy as np
from pyEDM import *
import scipy.stats as stats
from scipy.stats import kendalltau
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import MinMaxScaler

def ConcateLibPred(Library, Prediction, th): 
    row_to_add = Prediction.iloc[th]
    Lib_Pred_df = pd.concat([Library, row_to_add.to_frame().T], ignore_index=True)
    Lib_Pred_df[Library.columns.to_list()] = Lib_Pred_df[Library.columns.to_list()].apply(pd.to_numeric, errors='coerce')
    Lib_Pred_df['Date'] = range(1, len(Lib_Pred_df) + 1)

    return Lib_Pred_df

def DataNormalize(Lib_Pred_df):
    feature_to_standardize = Lib_Pred_df.columns.to_list()
    feature_to_standardize.remove(Lib_Pred_df.columns[0])
    scaler = MinMaxScaler()
    Lib_Pred_df_ = Lib_Pred_df.copy()
    scaler.fit(Lib_Pred_df_[feature_to_standardize])
    Lib_Pred_df[feature_to_standardize] = scaler.transform(Lib_Pred_df[feature_to_standardize])
    return Lib_Pred_df

def EDM_FeatureProcessing(data, target):
    df_columns = list(data.columns)
    train_feature = df_columns.copy()
    train_feature.remove('Date')
    formatted_columns = ' '.join(df_columns[1:])
    train_feature.remove(target)
    return formatted_columns, train_feature

def EDM_TargetOED(data, target, valid_interval):
    TargetOED = EmbedDimension(dataFrame=data, columns=target, showPlot=False,
                                lib=f'1 {len(data)}', pred=f'{len(data) -valid_interval-1} {len(data)-1}')
    TargetOED_rho = TargetOED['rho'].max()
    TargetOED = int(TargetOED['E'][TargetOED['rho'] == TargetOED['rho'].max()].iloc[0])
    return TargetOED, TargetOED_rho

def EDM_FeatureSelection(data, ticker, target, train_feature, TargetOED, E_max):
    crirho = stats.t.ppf(0.95, len(data) - 1) / (len(data) - 2 + stats.t.ppf(0.95, len(data) - 1) ** 2)
    ccm_libSizes = f'{TargetOED+10} {len(data)-10} 10'
    f_selec = pd.DataFrame(columns=train_feature)
    for train in train_feature:

        ccm_E_termRHO = pd.DataFrame(columns=['E', 'term_rho'])
        """
        Here, use term_rho to select ccm_OED.
        """
        for e in range(1, E_max+1):
            ccm_result = CCM(dataFrame=data, E=e, columns=train, target=target,
                             libSizes=ccm_libSizes, random=False, showPlot=False)
            new_data = {'E':e, 'term_rho': ccm_result[f'{target}:{train}'].iloc[-1]}
            ccm_E_termRHO.loc[len(ccm_E_termRHO)] = new_data

        max_term_rho_index = ccm_E_termRHO['term_rho'].idxmax()
        ccm_OED = ccm_E_termRHO.at[max_term_rho_index, 'E']

        ccm_result = CCM(dataFrame=data, E=ccm_OED, columns=train, target=target, 
                         libSizes=ccm_libSizes, random=False, showPlot=False)
        """
        Here, use target:train, LibSize to perform Kendall tau test.
        """
        ccm_result = ccm_result[['LibSize', f'{target}:{train}']]
        ccm_result[f'{target}:{train}'][ccm_result[f'{target}:{train}'] < 0] = 0
        term_rho = ccm_result[f'{target}:{train}'].iloc[-1]

        tau, p_value = kendalltau(ccm_result['LibSize'], ccm_result[f'{target}:{train}']) 
        alpha = 0.05
        if (p_value < alpha) and (term_rho > crirho):
            f_selec[train] = [term_rho]
        else:
            f_selec[train] = [0]

    f_selec.index = pd.Index([f'{ticker}']) 
    return f_selec

def EDM_CreateNewDF(data, target, f_selec, max_lag):

    non_zero_columns = f_selec.loc[:, (f_selec != 0).any(axis = 0)]
    train_feature_ls = list(non_zero_columns.columns)
    formatted_columns = ' '.join(train_feature_ls)
    columns_to_lag = formatted_columns + f' {target}'

    Embed_df = Embed(dataFrame=data, E=max_lag, tau=-1, columns=columns_to_lag)
    Embed_df['Date'] = data['Date']
    Embed_df.dropna(inplace=True)
    Embed_df = Embed_df.reset_index(drop=True)
    Embed_df = Embed_df[['Date'] + [col for col in Embed_df.columns if col != 'Date']]

    ML_df = Embed_df.copy()
    ML_df = ML_df.filter(like="(t-0)")
    return Embed_df, ML_df

def EDM_RandomSimplex(Embed_df, target, targetOED, valid_interval, kmax, kn):    
    np.random.seed(87)
    Embed_for_train = Embed_df.drop(columns='Date')
    Embed_for_train = Embed_for_train.drop(columns=f'{target}(t-0)')
    train_f_ls = list(Embed_for_train.columns)
    train_f_num = len(Embed_for_train.columns)

    rho_feature_view = pd.DataFrame(columns=['rho'])
    new_column = pd.DataFrame(columns=['feature_' + str(i) for i in range(1, targetOED+1)])
    rho_feature_view = pd.concat([rho_feature_view, new_column], axis=1)

    k = 1
    while k <= kmax:
        random_pick_train = np.random.choice(train_f_num, targetOED, replace=False)
        train_f_ls = np.array(train_f_ls)
        select_train_f = train_f_ls[random_pick_train]
        formatted_random_columns = ' '.join(select_train_f)
        simp = Simplex(dataFrame=Embed_df, columns=formatted_random_columns, target=f'{target}(t-0)',
                       lib=f'1 {len(Embed_df)}', pred = f'{len(Embed_df) -valid_interval-1} {len(Embed_df)-1}', 
                       embedded=True, showPlot=False)
        sub_simp = simp[['Observations', 'Predictions']]
        rho = sub_simp['Observations'].corr(sub_simp['Predictions'])
        rho_feature_view.loc[len(rho_feature_view), 'rho'] = rho
        rho_feature_view.loc[len(rho_feature_view)-1, rho_feature_view.columns[1:]] = select_train_f
        k += 1

    rs_score = rho_feature_view.sort_values(by='rho', ascending=False).head(kn)
    rs_score = rs_score.reset_index(drop=True)
    return rs_score

def EDM_WeightedDistanceMatrix(Embed_df, rs_score):
    ww = rs_score['rho'] / rs_score['rho'].sum()
    dmatrix_ls = []
    for j in range(rs_score.shape[0]):
        view_feature = rs_score.iloc[j, 1:]
        view_feature = np.array(view_feature)
        view_feature_value = Embed_df[view_feature]
        view_matrix = view_feature_value.to_numpy()
        view_matrix = np.vstack(view_matrix)

        Dx_t2 = pdist(view_matrix, metric='euclidean') * ww[j]
        Dx_t2 = squareform(Dx_t2)
        dmatrix_ls.append(Dx_t2)

    dmatrix = np.sum(dmatrix_ls, axis=0)
    return dmatrix

def EDM_ModelSelection(ML_df, target, dmatrix, theta_seq, Tp, valid_interval):
    result_ls = pd.DataFrame(columns=['Theta', 'Score', 'Param'])

    tp = len(ML_df) -1
    tp_distence = dmatrix[tp]
    mask = np.ones(len(tp_distence), dtype=bool)
    mask[tp] = False
    dpar = np.mean(tp_distence[mask])

    for theta in theta_seq:
        w_tp = np.exp(-theta * tp_distence / dpar)
        w_tp = np.sqrt(w_tp)

        ML_df_new = ML_df.copy()
        ML_df_new[f'ans(t-0)'] = ML_df_new[f'{target}(t-0)'].shift(-Tp)
        ML_df_new = ML_df_new.multiply(w_tp, axis=0) 
        ML_df_new[f'{target}(t-0)'] = ML_df_new[f'{target}(t-0)'].apply(lambda x: 1.0 if x != 0 else x)
        ML_df_new['ans(t-0)'] = ML_df_new['ans(t-0)'].apply(lambda x: 1.0 if x != 0 else x)
        ML_df_new = ML_df_new[:-(Tp+1)]

        X = ML_df_new.iloc[:, :-1]
        y = ML_df_new.iloc[:, -1]
        val_fold = [-1] * (len(X)-valid_interval) + [0] * valid_interval
        ps = PredefinedSplit(test_fold=val_fold)
        logistic_elastic_net = LogisticRegression(
                                                  class_weight='balanced',
                                                  penalty='elasticnet', 
                                                  solver='saga',
                                                  fit_intercept=True,
                                                  intercept_scaling=0.1,
                                                  )
        param_grid = {
                      'l1_ratio': [0.9, 0.7, 0.5, 0.3, 0.1],
                      'C': [100, 50, 10, 1, 0.1],
                      }
        grid_search = GridSearchCV(estimator=logistic_elastic_net,
                                   param_grid=param_grid, 
                                   cv=ps,
                                   scoring='accuracy', 
                                   return_train_score=True,
                                   )       
        grid_search.fit(X, y)

        result_ls.loc[len(result_ls), 'Theta'] = theta
        result_ls.loc[len(result_ls)-1, 'Score'] = grid_search.best_score_
        result_ls.loc[len(result_ls)-1, 'Param'] = [grid_search.best_params_]
        theta = result_ls['Theta'][result_ls['Score'].idxmax()]
        param = result_ls['Param'][result_ls['Score'].idxmax()][0]
    return result_ls, theta, param

def EDM_TrainningModel(ML_df, target, dmatrix, theta, param, Tp):
    tp = len(ML_df) -1
    tp_distence = dmatrix[tp]
    mask = np.ones(len(tp_distence), dtype=bool)
    mask[tp] = False
    dpar = np.mean(tp_distence[mask])
    w_tp = np.exp(-theta * tp_distence / dpar)
    w_tp = np.sqrt(w_tp)

    ML_df_new = ML_df.copy()
    ML_df_new[f'ans(t-0)'] = ML_df_new[f'{target}(t-0)'].shift(-Tp)
    ML_df_new = ML_df_new.multiply(w_tp, axis=0)
    ML_df_new[f'{target}(t-0)'] = ML_df_new[f'{target}(t-0)'].apply(lambda x: 1.0 if x != 0 else x)
    ML_df_new['ans(t-0)'] = ML_df_new['ans(t-0)'].apply(lambda x: 1.0 if x != 0 else x)
    ML_df_new = ML_df_new[:-(Tp+1)]
    
    X = ML_df_new.iloc[:, :-1]
    y = ML_df_new.iloc[:, -1]
    model = LogisticRegression(
                               class_weight='balanced',
                               penalty='elasticnet', 
                               solver='saga',
                               fit_intercept=True,
                               intercept_scaling=0.1,                     
                               **param
                               )
    model.fit(X, y)
    return model