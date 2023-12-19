import logging
import math
import os
from typing import List, Tuple
from data_process import preprocess, to_vector
import numpy as np
import scipy.sparse as ss
# from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.utils import Sequence
from spectrum_utils.spectrum import MsmsSpectrum
import re

import sklearn
import pandas as pd

'''
data generator
data loader for model
get_spectrum : get spectrum from mgf file 
get_data : input : dataframe and mgf file,
            return : preprocessed spectrum, one hot encoded sequence, sequence
class gen() : data loader. Returns batch size of spectrum sequence pair and labels.
'''

        
        
AA_LIST = [['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'] ,]
MAX_LEN= 50

# f = open("/home/workplace/ms2/embedding/LIBRARY_AUGMENT-adfc8252-download_filtered_mgf_library-main.mgf")
# mgf = f.readlines()
# f.close()

logger = logging.getLogger('gleams')

def get_spectrum(mgf, begin, end):
    spectrum = mgf[begin + 16: end]
    mz = [float(i.split("\t")[0].strip()) for i in spectrum]
    intensity = [float(i.split("\t")[1].strip()) for i in spectrum]
    return mz, intensity


def get_data(df, mgf):
    ms2_mz = [get_spectrum(mgf, start, end)[0] for start, end in zip(df.begin.values, df.end.values)]
    ms2_int = [get_spectrum(mgf, start, end)[1] for start, end in zip(df.begin.values, df.end.values)]
    # ms2_spec = [get_spectrum(mgf, start, end) for start, end in zip(df.begin_index.values, df.end_idx.values)]
    
    #need binning
    # ms2_spec = preprocess(ms2_spec)
    
    charge = [2 for a in df.begin.values]
    precursor_mz = [a for a in df.pepmass.values]
    sequence = [re.sub('[+|-|.|[0-9]]*','', a) for a in df.sequence.values]
    spec_sequence = []
    if "spec_sequence" in df.columns:
        
        spec_sequence = [re.sub('[+|-|.|[0-9]]*','', a) for a in df.spec_sequence.values]
    else:
        spec_sequence = sequence
        # print(sequence)
    valid_sequence = []
    valid_spec_sequence = []
    preprocessed_ms2_spec = [preprocess(MsmsSpectrum(pre_seq, pre_mz, pre_charge, pre_ms2_mz, pre_ms2_int)) for pre_seq, pre_mz, pre_charge, pre_ms2_mz, pre_ms2_int in zip(sequence, precursor_mz, charge, ms2_mz, ms2_int)]
    # ms2_vector = [to_vector(spec.mz, spec._intensity) for spec in preprocessed_ms2_spec if spec.is_valid == True and spec.is_processed == True]
    # seq_pad = [seq[:MAX_LEN] if len(seq) > MAX_LEN else seq + '-' * (MAX_LEN - len(seq)) for seq in sequence]
    ms2_vector = []
    seq_pad = []
    # ms2_vector,seq_pad = [to_vector(spec.mz, spec._intensity) for spec in preprocessed_ms2_spec if spec.is_valid == True and spec.is_processed == True]
    # seq_pad = [seq[:MAX_LEN] if len(seq) > MAX_LEN else seq + '-' * (MAX_LEN - len(seq)) for seq in sequence]
    
    for spec_ele, seq_ele, spec_seq_ele in zip(preprocessed_ms2_spec, sequence, spec_sequence):
        if spec_ele.is_valid == True and spec_ele.is_processed == True:
            ms2_vector.append(to_vector(spec_ele.mz, spec_ele._intensity))
            valid_sequence.append(seq_ele) 
            valid_spec_sequence.append(spec_seq_ele)
            if len(seq_ele) > MAX_LEN:
                seq_pad.append(seq_ele[:MAX_LEN])
                   
            else:
                seq_pad.append(seq_ele + '-' * (MAX_LEN - len(seq_ele)))
            
    
    ohe_list = []
    for seq_ele in seq_pad:
        seq_df = pd.DataFrame([{"aa" : x} for i, x in enumerate(seq_ele)]) 
        seq_ohe = sklearn.preprocessing.OneHotEncoder(categories=AA_LIST,sparse_output=False).fit_transform(seq_df)
        ohe_list.append(seq_ohe)
    
    return np.array(ms2_vector), np.array(ohe_list), [valid_sequence, valid_spec_sequence] 


class gen(Sequence):
    
    def __init__(self, file_name_pos, file_name_neg, batch_size, mgf,shuffle =True):
        self.file_pos = file_name_pos
        self.file_neg = file_name_neg
        self.batch_size = batch_size
        self.mgf = mgf
      
    def __len__(self):
        return math.ceil(len(self.file_pos)/(self.batch_size//10+16))
    
    def __getitem__(self, idx : int):
        batch_pairs, batch_y = [], []
        pos_batch_size = self.batch_size//10
        neg_batch_size = (self.batch_size*9)//10
        pos_batch_size += self.batch_size - pos_batch_size - neg_batch_size
        
        batch_start_i_pos = idx * pos_batch_size
        batch_start_i_neg = idx * neg_batch_size
        
        if idx == 0:
            self.file_pos = self.file_pos.sample(frac=1).reset_index(drop=True)
            self.file_neg = self.file_neg.sample(frac=1).reset_index(drop=True)
        batch_stop_i_pos = batch_start_i_pos + pos_batch_size
        batch_stop_i_neg = batch_start_i_neg + neg_batch_size
        
        #여기서 vector화 해야함
        batch_df_pos = self.file_pos[batch_start_i_pos:batch_stop_i_pos]
        batch_df_neg = self.file_neg[batch_start_i_neg:batch_stop_i_neg]
        
        batch_vector_pos, batch_ohe_pos, batch_seq_pos = get_data(df=batch_df_pos, mgf=self.mgf)
        batch_vector_neg, batch_ohe_neg, batch_seq_neg = get_data(df=batch_df_neg, mgf=self.mgf)
        # print(f"pos df shape = {batch_df_pos.shape}, neg df shape = {batch_df_neg.shape}")
        # print(f"idx = {idx}pos start idx = {batch_start_i_pos}, pos end idx = {batch_stop_i_pos}, neg start idx = {batch_start_i_neg}, neg_stop = {batch_stop_i_neg}")
        # print(f"vector pos shape = {batch_vector_pos.shape}, vector neg_shape = {batch_vector_neg.shape}")
        
        # print(len(batch_seq_neg))
        
        # print(len(batch_seq_pos))
        batch_vector = np.concatenate((batch_vector_pos ,batch_vector_neg))
        batch_ohe = np.concatenate((batch_ohe_pos,batch_ohe_neg))
        batch_y = np.concatenate((np.ones(len(batch_vector_pos), dtype=np.int8), np.zeros(len(batch_vector_neg),dtype=np.int8)), axis = 0)
        batch_seq = np.concatenate((batch_seq_pos[0], batch_seq_neg[0]))
        batch_spec_seq = np.concatenate((batch_seq_pos[1], batch_seq_neg[1]))
        # print(f"seq= {len(batch_seq)} spec_seq = {len(batch_spec_seq)}")

        final_batch_seq = [batch_seq, batch_spec_seq]
        # print(f"final len = {len(final_batch_seq)} seq= {len(batch_seq)} spec_seq = {len(batch_spec_seq)}")
        return ([batch_vector, batch_ohe], batch_y)
        # return ([batch_vector, batch_ohe], batch_y, final_batch_seq)


class gen_test(Sequence):
    
    def __init__(self, file_name_pos, file_name_neg, batch_size, mgf,shuffle =True):
        self.file_pos = file_name_pos
        self.file_neg = file_name_neg
        self.batch_size = batch_size
        self.mgf = mgf
      
    def __len__(self):
        return math.ceil(len(self.file_pos)/(self.batch_size//10+16))
    
    def __getitem__(self, idx : int):
        batch_pairs, batch_y = [], []
        pos_batch_size = self.batch_size//10
        neg_batch_size = (self.batch_size*9)//10
        pos_batch_size += self.batch_size - pos_batch_size - neg_batch_size
        
        batch_start_i_pos = idx * pos_batch_size
        batch_start_i_neg = idx * neg_batch_size
        
        if idx == 0:
            self.file_pos = self.file_pos.sample(frac=1).reset_index(drop=True)
            self.file_neg = self.file_neg.sample(frac=1).reset_index(drop=True)
        batch_stop_i_pos = batch_start_i_pos + pos_batch_size
        batch_stop_i_neg = batch_start_i_neg + neg_batch_size
        
        #여기서 vector화 해야함
        batch_df_pos = self.file_pos[batch_start_i_pos:batch_stop_i_pos]
        batch_df_neg = self.file_neg[batch_start_i_neg:batch_stop_i_neg]
        
        batch_vector_pos, batch_ohe_pos, batch_seq_pos = get_data(df=batch_df_pos, mgf=self.mgf)
        batch_vector_neg, batch_ohe_neg, batch_seq_neg = get_data(df=batch_df_neg, mgf=self.mgf)
        # print(f"pos df shape = {batch_df_pos.shape}, neg df shape = {batch_df_neg.shape}")
        # print(f"idx = {idx}pos start idx = {batch_start_i_pos}, pos end idx = {batch_stop_i_pos}, neg start idx = {batch_start_i_neg}, neg_stop = {batch_stop_i_neg}")
        # print(f"vector pos shape = {batch_vector_pos.shape}, vector neg_shape = {batch_vector_neg.shape}")
        
        # print(len(batch_seq_neg))
        
        # print(len(batch_seq_pos))
        batch_vector = np.concatenate((batch_vector_pos ,batch_vector_neg))
        batch_ohe = np.concatenate((batch_ohe_pos,batch_ohe_neg))
        batch_y = np.concatenate((np.ones(len(batch_vector_pos), dtype=np.int8), np.zeros(len(batch_vector_neg),dtype=np.int8)), axis = 0)
        batch_seq = np.concatenate((batch_seq_pos[0], batch_seq_neg[0]))
        batch_spec_seq = np.concatenate((batch_seq_pos[1], batch_seq_neg[1]))
        # print(f"seq= {len(batch_seq)} spec_seq = {len(batch_spec_seq)}")

        final_batch_seq = [batch_seq, batch_spec_seq]
        # print(f"final len = {len(final_batch_seq)} seq= {len(batch_seq)} spec_seq = {len(batch_spec_seq)}")
        # return ([batch_vector, batch_ohe], batch_y)
        return ([batch_vector, batch_ohe], batch_y, final_batch_seq)
