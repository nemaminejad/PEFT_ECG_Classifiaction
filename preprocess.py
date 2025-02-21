import csv
import pandas as pd
import os
import numpy as np
import re
import heartpy as hp
import copy
import wfdb
import ast

"""
This code is a modification to the code adapted from the following GitHub repository:
ECGCAD repository (https://github.com/MediaBrain-SJTU/ECGAD/tree/main) by Aofan Jiang, Chaoqin Huang, et al
"""


def normalize(X_train_unnormal:np.ndarray) -> np.ndarray:
    """
    normalize to [-1,1]
    """

    X_train = copy.deepcopy(X_train_unnormal)
    for count in range(X_train.shape[0]):
        for j in range(12):
            seq = X_train[count][:,j]
            X_train[count][:,j] = 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1
    return X_train

def hp_preprocess(X_train:np.ndarray)-> np.ndarray:
    """
    high pass filtering
    """
    train_data = []
    for i in range(X_train.shape[0]):
        ecg_tmp = []
        for lead in range(12):
            ecg = X_train[i][:,lead]

            filtered = hp.filter_signal(ecg,sample_rate=500, filtertype="highpass", cutoff=1)
            filt = hp.filter_signal(filtered, sample_rate=500, cutoff=35 ,filtertype="notch")
            y = hp.filter_signal(filt, sample_rate=500, filtertype="lowpass", cutoff=25)

            ecg_tmp.append(y)

        ecg_tmp = np.array(ecg_tmp).T
        train_data.append(ecg_tmp)

    train_data = np.array(train_data)

    return train_data

def load_save_raw_data(df:pd.DataFrame, path:str, fold:str,sampling_rate: int) -> None:
    """
    Load and save raw data
    
    df (pd.DataFrame): corresponding dataframe for each fold
    path (str): path/to/data
    fold (str): fold must be one of ('train', 'test', 'val')
    sampling_rate (int): sampling rate (100Hz or 500 Hz)
    
    """
  for idx, row in df.iterrows():
    if not os.path.exists(os.path.join(path,'PTB-XL-npy',fold,str(row.diagnostic_superclass),f'{idx}.npy')):
      data = np.array([])
      if sampling_rate == 100 and os.path.exists(path+row.filename_lr+'.hea'):
        data = [wfdb.rdsamp(path+row.filename_lr)]
        data = np.array([signal for signal, meta in data])
      else:
        if os.path.exists(path+row.filename_hr+'.hea'):
          data = [wfdb.rdsamp(path+row.filename_hr)]
          data = np.array([signal for signal, meta in data])
      if data.size > 0:
        denoised_data = hp_preprocess(data)
        # denoised_data = normalize(denoised_data) # Normalization not used here as it was used in the step of saving ECG plots. Users may uncomment this per they need.
        np.save(os.path.join(path,'PTB-XL-npy',fold,str(row.diagnostic_superclass),f'{idx}.npy'),denoised_data)
        del denoised_data
        del data




def preprocess_ptbxl(path: str, scp_df:pd.DataFrame, agg_df:pd.DataFrame, sampling_rate: int = 500) -> None:
  """
  load and convert annotation data
  path (str) : /path/to/data
  scp_df (pd.DataFrame): data from 'ptbxl_datasbase.csv' 
  agg_df (pd.DataFrame) : data from 'scp_statements.csv'
  sampling_rate (int) : sampling rate is one of: [100, 500]
  """

  #Obtain diagnostic superclass
  scp_df.scp_codes = scp_df.scp_codes.apply(lambda x: ast.literal_eval(x))

  agg_df = agg_df[agg_df.diagnostic == 1]
  def aggregate_diagnostic(y_dic):
      tmp = []
      for key in y_dic.keys():
          if key in agg_df.index:
              tmp.append(agg_df.loc[key].diagnostic_class)
      return list(set(tmp))
      
  # Apply diagnostic superclass
  scp_df['diagnostic_superclass'] = scp_df.scp_codes.apply(aggregate_diagnostic)
  scp_df.diagnostic_superclass = scp_df.diagnostic_superclass.apply(lambda x:x[0] if isinstance(x, list) and len(x) > 0 else x)
  scp_df.loc[:,'diagnostic_superclass'] = scp_df['diagnostic_superclass'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
  scp_df = scp_df[scp_df.diagnostic_superclass != ()]


  # create index numbers for each scp_code
  label_dict = {}
  label_idx = 0
  for k in ['NORM' ,'STTC', 'CD' ,'HYP' ,'MI']:
    label_dict[k] = label_idx
    label_idx +=1

  # Split data into train and test and validation folds
  # fold numbers acquired from original physionet splits provided on https://physionet.org/content/ptb-xl/1.0.3/
  test_fold = 10
  val_fold = 9
  train_fold = [1,2,3,4,5,6,7,8]
  
  #Identify train fold
  Y_train_df = scp_df[(scp_df.strat_fold.isin(train_fold))]
  Y_train_df.diagnostic_superclass = Y_train_df.diagnostic_superclass.apply(lambda x: label_dict[x])
  y_train = Y_train_df.diagnostic_superclass
  #Identify test fold
  Y_test_df = scp_df[(scp_df.strat_fold == test_fold)]
  Y_test_df.diagnostic_superclass = Y_test_df.diagnostic_superclass.apply(lambda x: label_dict[x])
  y_test = Y_test_df.diagnostic_superclass
  #Identify validation fold
  Y_val_df = scp_df[(scp_df.strat_fold == val_fold)]
  Y_val_df.diagnostic_superclass = Y_val_df.diagnostic_superclass.apply(lambda x: label_dict[x])
  y_val = Y_val_df.diagnostic_superclass

  # Load raw signal data
  print('loading')

  for label in label_dict.values():
    os.makedirs(os.path.join(path, 'PTB-XL-npy','train', str(label)), exist_ok = True)
    os.makedirs(os.path.join(path, 'PTB-XL-npy','test', str(label)), exist_ok = True)
    os.makedirs(os.path.join(path, 'PTB-XL-npy','val', str(label)), exist_ok = True)

  y_train.to_csv(os.path.join(path,'PTB-XL-npy','train_label.csv'))
  y_test.to_csv(os.path.join(path,'PTB-XL-npy','test_label.csv'))
  y_val.to_csv(os.path.join(path,'PTB-XL-npy','val_label.csv'))
  # Save raw data
  print('saving')
  sampling_rate = 500
  load_save_raw_data(Y_train_df, path, 'train',sampling_rate)
  load_save_raw_data(Y_test_df, path, 'test',sampling_rate)
  load_save_raw_data(Y_val_df, path, 'val',sampling_rate)



if __name__ =='__main__':
    
  path = './data/PTB-XL/physionet.org/files/ptb-xl/1.0.3/'
  scp_df = pd.read_csv(path+'ptbxl_datasbase.csv', index_col='ecg_id')
  #Load scp_statements.csv for diagnostic aggregation
  agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
  # run code
  preprocess_ptbxl(path, ptbxl_csv, scp_statements, sampling_rate = 500)
  