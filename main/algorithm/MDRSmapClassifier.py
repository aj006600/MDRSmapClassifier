import os
import numpy as np
import pandas as pd
import joblib
from .utils import *

import warnings
warnings.filterwarnings('ignore')

# Create a dataframe for prediction results #
def create_df_for_result(actual_start_pred_date, origi_data, target):
    Date = origi_data['Date'][origi_data['Date']>=actual_start_pred_date].reset_index(drop=True)
    Observations = origi_data[target][origi_data['Date']>=actual_start_pred_date].reset_index(drop=True)
    EDM_result = pd.DataFrame(Date)
    EDM_result['Observations'] = Observations
    EDM_result['Predictions'] = None  
    return EDM_result

# Perform the first prediction and find the hyperparameters #
def first_pred(Library, Prediction, target, valid_interval, tp, ticker, E_max, max_lag, kn, theta_seq):
    # Concate Library and single prediction data
    th=0
    Lib_Pred_df = ConcateLibPred(Library=Library, Prediction=Prediction, th=0)
    Lib_Pred_df = DataNormalize(Lib_Pred_df)
    # Find optimal embedding dimenson
    _, train_feature = EDM_FeatureProcessing(data=Lib_Pred_df, target=target)
    TargetOED, TargetOED_rho = EDM_TargetOED(data=Lib_Pred_df, target=target, valid_interval=valid_interval)
    print(f'optimal embedding dimenson has been found: {TargetOED}\n')
    # CCM feature selection
    print('now applying CCM feature selection...')
    f_selec = EDM_FeatureSelection(data=Lib_Pred_df, ticker=ticker, target=target, train_feature=train_feature, TargetOED=TargetOED, E_max=E_max)
    # Muti-view embedding Simplex projection random search
    print('now applying Muti-view random search...')
    Embed_df, ML_df = EDM_CreateNewDF(data=Lib_Pred_df, target=target, f_selec=f_selec, max_lag=max_lag)
    rs_score = EDM_RandomSimplex(Embed_df=Embed_df, target=target, targetOED=TargetOED, valid_interval=valid_interval, kmax=10000, kn=kn)
    # S-map
    print('now applying S-map...')
    dmatrix = EDM_WeightedDistanceMatrix(Embed_df=Embed_df, rs_score=rs_score)
    # SAGA logistic (elastic-net)
    print('now applying SAGA logistic(elastic-net)...\n')
    _, theta, param = EDM_ModelSelection(ML_df=ML_df, target=target, dmatrix=dmatrix, theta_seq=theta_seq, Tp=tp, valid_interval=valid_interval)
    model = EDM_TrainningModel(ML_df=ML_df, target=target, dmatrix=dmatrix, theta=theta, param=param, Tp=tp)
    # Prediction 
    X_pred = np.array(ML_df.iloc[-1]).reshape(1, -1) # when choose "NOT" to delete target feature in step.5 from EDM_ModelSelection()
    y_pred = model.predict(X_pred)
    y_pred = y_pred[0]
    return f_selec, rs_score, param, theta, th, y_pred

# Record prediction results #
def record_results(EDM_result, th, y_pred):
    EDM_result.loc[th, 'Predictions'] = y_pred
    return EDM_result

# use the hyperparameters for the remaining predictions #
def other_preds_with_hp(Library, Prediction, f_selec, rs_score, param, target, max_lag, theta, tp, EDM_result):
    for th in range(1, len(Prediction)):
        Lib_Pred_df = ConcateLibPred(Library=Library, Prediction=Prediction, th=th)
        Lib_Pred_df = DataNormalize(Lib_Pred_df=Lib_Pred_df)
        Embed_df, ML_df = EDM_CreateNewDF(data=Lib_Pred_df, target=target, f_selec=f_selec, max_lag=max_lag)
        dmatrix = EDM_WeightedDistanceMatrix(Embed_df=Embed_df, rs_score=rs_score)
        model = EDM_TrainningModel(ML_df=ML_df, target=target, dmatrix=dmatrix, theta=theta, param=param, Tp=tp)

        X_pred = np.array(ML_df.iloc[-1]).reshape(1, -1)
        y_pred = model.predict(X_pred)
        y_pred = y_pred[0]
        EDM_result = record_results(EDM_result, th, y_pred)
    y_preds = EDM_result['Predictions']
    return y_preds, EDM_result

# Evaluate prediction results #
def eval_results(EDM_result, ticker, backtrade_end_date):
    EDM_result.dropna(inplace=True)
    EDM_result = EDM_result[EDM_result['Date'] <= backtrade_end_date]
    ACC = len(EDM_result[EDM_result['Predictions'] == EDM_result['Observations']]) / len(EDM_result['Observations'])
    print(f'\n {ticker} ACC: {ACC}\n')
    return EDM_result

# MDRSmapClassifier whole pipeline #
def pipeline(pred_type, actual_start_pred_date, origi_data, backtrade_end_date, 
             Library, Prediction, target, valid_interval, tp, ticker, E_max, max_lag, kn, theta_seq):
    if pred_type == 'single_pred':
        _, _, _, _, _, y_pred = first_pred(Library, Prediction, target, valid_interval, tp, ticker, E_max, max_lag, kn, theta_seq)
        print(f'prediction: {y_pred}')
        return y_pred
    elif pred_type == 'multi_pred':
        EDM_result = create_df_for_result(actual_start_pred_date, origi_data, target)
        f_selec, rs_score, param, theta, th, y_pred = first_pred(Library, Prediction, target, valid_interval, tp, ticker, E_max, max_lag, kn, theta_seq)
        EDM_result = record_results(EDM_result, th, y_pred)
        y_preds, _ = other_preds_with_hp(Library, Prediction, f_selec, rs_score, param, target, max_lag, theta, tp, EDM_result)
        print(f'predictions: {y_preds}')
        return y_preds
    elif pred_type == 'backtrade':
        EDM_result = create_df_for_result(actual_start_pred_date, origi_data, target)
        f_selec, rs_score, param, theta, th, y_pred = first_pred(Library, Prediction, target, valid_interval, tp, ticker, E_max, max_lag, kn, theta_seq)
        EDM_result = record_results(EDM_result, th, y_pred)
        _, EDM_result = other_preds_with_hp(Library, Prediction, f_selec, rs_score, param, target, max_lag, theta, tp, EDM_result)
        EDM_result = eval_results(EDM_result, ticker, backtrade_end_date)
        return EDM_result

# Create a dataframe for realtime prediction results #
def realtime_result_df():
    EDM_result = pd.DataFrame(columns=['Predictions'])
    return EDM_result

# find MDRSmapClassifier realtime parameter #
def realtime_find_hp(Library, Prediction, target, valid_interval, tp, ticker, E_max, max_lag, kn, theta_seq):
    f_selec, rs_score, param, theta, th, y_pred = first_pred(Library, Prediction, target, valid_interval, tp, ticker, E_max, max_lag, kn, theta_seq)
    print(f'prediction: {y_pred}')
    return f_selec, rs_score, param, theta, th, y_pred

# Save MDRSmapClassifier realtime parameter #
def save_parameters(f_selec, rs_score, param, theta, ticker, tp):
    current_dir = os.path.dirname(__file__)
    file_dir = os.path.join(current_dir, '..', 'models', f'{ticker}_{tp}_param.pkl')
    joblib.dump({
        'f_selec': f_selec,
        'rs_score': rs_score,
        'param': param,
        'theta': theta
    }, file_dir)
    print(f"Parameter saved to {file_dir}")
    return file_dir

# Load MDRSmapClassifier realtime parameter #
def load_parameter(file_dir):
    parameter = joblib.load(file_dir)
    print(f"Parameter loaded from {file_dir}")
    return parameter

# MDRSmapClassifier realtime training function #
def realtime_train(Library, Prediction, target, valid_interval, tp, ticker, E_max, max_lag, kn, theta_seq):
    EDM_result = realtime_result_df()
    f_selec, rs_score, param, theta, th, y_pred = realtime_find_hp(Library, Prediction, target, valid_interval, tp, ticker, E_max, max_lag, kn, theta_seq)
    EDM_result = record_results(EDM_result, th, y_pred)
    file_dir = save_parameters(f_selec, rs_score, param, theta, ticker, tp)
    return y_pred, file_dir, EDM_result

# MDRSmapClassifier realtime predictive function #
def realtime_pred(Prediction, file_dir,
                  Library, target, max_lag, tp, EDM_result):
    parameter = load_parameter(file_dir)
    y_pred, _ = other_preds_with_hp(Library, Prediction, parameter['f_selec'], parameter['rs_score'], parameter['param'], parameter['theta'], target, max_lag, tp, EDM_result)
    print(f'prediction: {y_pred[0]}')
    return y_pred