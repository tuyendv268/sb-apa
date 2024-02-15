from time import time
import subprocess
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import shutil
import torch
import base64
import pickle
import json
import yaml
import sys
import os

def generate_df_phones_pure(phones_path, phones_to_pure_int_path, phones_pure_path, final_mdl_path, transition_dir):
    phones_df = pd.read_csv(
        phones_path, sep="\s", 
        names=["phone", "id"], 
        dtype={"phone":str, "id":str}, engine='python')
    phones_df.set_index("id", inplace=True)
    phones_dict = phones_df["phone"].to_dict()

    phones_pure_df = pd.read_csv(
        phones_pure_path, sep="\s", 
        names=["phone", "id", "type"], 
        dtype={"phone":str, "id":str, "type":str}, engine='python')
    phones_pure_df.set_index("id", inplace=True)
    phones_pure_dict = phones_pure_df["phone"].to_dict()

    output = []
    phone_to_pure_df = pd.read_csv(
        phones_to_pure_int_path, sep="\s", 
        names=["phone", "phone_pure"], 
        dtype={"phone":str, "phone_pure":str}, engine='python')
    
    phone_to_pure_df["phone_name"] = phone_to_pure_df["phone"].apply(lambda phn: phones_dict[phn])
    
    phone_to_pure_df["phone_pure_name"] = phone_to_pure_df["phone_pure"].apply(lambda phn: phones_pure_dict[phn])

    path_show_transitions = os.path.join(transition_dir, "transitions.txt")
    if not os.path.exists(path_show_transitions):
        os.system(f'show-transitions {phones_path} {final_mdl_path} > {path_show_transitions}')  

    df_transitions = load_transitions(path_show_transitions)
    df_transitions = df_transitions.set_index('phone_name').join(phone_to_pure_df.set_index('phone_name'))
    df_transitions = df_transitions.reset_index().set_index("phone")

    return df_transitions

def load_transitions(path):
    transitions_dict = {}

    f = open(path, "r")
    phone = -1
    data = []
    for line in f:
        line_array = line.split(' ')
        if line_array[0] == 'Transition-state':
            data = []
            transition_state = line_array[1].split(":")[0]
            phone = line_array[4]
            hmm_state = line_array[7]
            forward_pdf = line_array[10]
            forward_pdf = forward_pdf.split('\n')
            forward_pdf = forward_pdf[0]
            self_pdf = line_array[13]
            self_pdf = self_pdf.split('\n')
            self_pdf = self_pdf[0]

            data.append(transition_state)
            data.append(phone)
            data.append(hmm_state)
            data.append(forward_pdf)
            data.append(self_pdf)
        if line_array[1] == 'Transition-id':
            transition_id = line_array[3]
            transitions_dict[transition_id] = data + [transition_id]

    df_transitions = pd.DataFrame.from_dict(
        transitions_dict, orient='index', 
        columns=['transition_state', 'phone_name', 'hmm_state', 'forward_pdf', 'self_pdf', 'transition_id']
    )

    return df_transitions

def removeSymbols(str, symbols):
    for symbol in symbols:
        str = str.replace(symbol,'')
    return str

def get_alignments(alignments_path):
    alignments_dict = {}
    for l in open(alignments_path, 'r', encoding="utf8").readlines():
        l=l.split()
        #Get transitions alignments
        if len(l) > 3 and l[1] == 'transitions':
            waveform_name = l[0]
            transition_lists = []
            transitions = []
            #alignments_dict[waveform_name] = {}
            for i in range(2, len(l)):
                transition_id = int(removeSymbols(l[i],['[',']',',']))
                transitions.append(transition_id)
                if ']' in l[i]:
                    transition_lists.append(transitions)
                    transitions = []
                    current_phone_transition = transition_id
            alignments_dict[waveform_name]['transitions'] = transition_lists

        #Get phones alignments
        if len(l) > 3 and l[1] == 'phones':
            waveform_name = l[0]
            phones = []
            alignments_dict[waveform_name] = {}
            for i in range(2, len(l),3):
                current_phone = removeSymbols(l[i],['[',']',',',')','(','\''])
                phones.append(current_phone)
            alignments_dict[waveform_name]['phones'] = phones
    return alignments_dict


def generate_df_alignments(gop_dir, alignments_dir_path):
    alignments_dict = get_alignments(alignments_dir_path)

    alignments_name = os.path.basename(alignments_dir_path).rstrip(".out")
    alignments_path = gop_dir + f'/{alignments_name}.pkl'
    with open(alignments_path, 'wb') as handle:
        pickle.dump(alignments_dict, handle)

    return alignments_path

def prepare_dataframes(phones_path, phones_to_pure_int_path, 
    phones_pure_path, final_mdl_path, gop_dir, alignments_path):

    df_transitions = generate_df_phones_pure(
        phones_path, phones_to_pure_int_path, 
        phones_pure_path, final_mdl_path, gop_dir)

    alignments_path = generate_df_alignments(gop_dir, alignments_path)

    df_phones_pure = df_transitions
    df_phones_pure = df_phones_pure.reset_index()

    df_alignments = pd.read_pickle(alignments_path)
    df_alignments = pd.DataFrame.from_dict(df_alignments)

    return df_phones_pure, df_alignments

def get_pdfs_for_pure_phone(df_phones_pure, phone):
    pdfs = list(df_phones_pure.loc[(df_phones_pure['phone_pure'] == str(phone+1) )].forward_pdf)
    pdfs = pdfs + list(df_phones_pure.loc[(df_phones_pure['phone_pure'] == str(phone+1) )].self_pdf)
    pdfs = set(pdfs)
    return pdfs

def matrix_gop_robust(df_phones_pure, number_senones, batch_size):            
    pdfs_to_phone_pure_mask = []

    for phone_pure in range(0, len(list(df_phones_pure.phone_pure.unique()))):
        pdfs = get_pdfs_for_pure_phone(df_phones_pure, phone_pure)
        pdfs_to_phone_pure_file = np.zeros(number_senones)

        for pdf in pdfs:
            pdfs_to_phone_pure_file[int(pdf)] = 1.0 
        
        pdfs_to_phone_pure_mask.append(pdfs_to_phone_pure_file)
                                
    pdfs_to_phone_pure_mask_3D = []

    for i in range(0, batch_size):                
        pdfs_to_phone_pure_mask_3D.append(pdfs_to_phone_pure_mask)
    
    return pdfs_to_phone_pure_mask_3D

def gop_robust_with_matrix(df_scores, df_phones_pure, number_senones, batch_size, output_gop_dict):
    mask_score = matrix_gop_robust(df_phones_pure, number_senones, batch_size)

    mask_score = np.array(mask_score)
    mask_score = mask_score.transpose(0, 2, 1)

    scores = np.array(df_scores.p.tolist())

    scores_phone_pure = np.matmul(scores, mask_score)
    logids = df_scores.index
    
    for j in range(0, len(df_scores)):
        phones = df_scores.phones[j]
        transitions = df_scores.transitions[j]
        logid = logids[j]

        ti = 0
        phones_pure = []
        features = []
        for i in range(0, len(transitions)):
            transitions_by_phone = transitions[i]
            tf = ti + len(transitions_by_phone) - 1
            try:
                np.seterr(all = "raise") 
                lpp = sum(np.log(scores_phone_pure[j][ti:tf+1]))/(tf-ti+1)
            except FloatingPointError as e:
                raise Exception("Floating Point Error !!!")
            phone_pure = df_phones_pure.loc[(df_phones_pure['phone_name'] == str(phones[i]) )].phone_pure.unique()[0]
            
            lpr = lpp - lpp[int(phone_pure)-1]
                        
            phones_pure.append(phone_pure)
            features.append(np.concatenate([lpp, lpr]))
            
            ti = tf + 1

        out_phones = " ".join(df_scores.phones[j])
        out_phones_pure = " ".join(phones_pure)
        gop_features = np.vstack(features)

        output_gop_dict[logid] = {
            'phones-pure': out_phones_pure, 
            'phones':out_phones, 
            'gop-features':gop_features
        }        
    return output_gop_dict

def load_config(path):
    config_fh = open(path, "r")
    config_dict = yaml.safe_load(config_fh)
    return config_dict

def load_ivector_period_from_conf(conf_path):
    conf_fh = open(conf_path + '/ivector_extractor.conf', 'r')
    ivector_period_line = conf_fh.readlines()[1]
    ivector_period = int(ivector_period_line.split('=')[1])
    return ivector_period

def prepare_data_in_kaldi_format(data_dir, text, wav_path, ID=8888):
    with open(f'{data_dir}/wav.scp', "w", encoding="utf-8") as wavscp_file:
        wavscp_file.write(f'{ID}\t{wav_path}\n')

    with open(f'{data_dir}/text', "w", encoding="utf-8") as text_file:
        text_file.write(f'{ID}\t{text}\n')

    with open(f'{data_dir}/spk2utt', "w", encoding="utf-8") as spk2utt_file:
        spk2utt_file.write(f'{ID}\t{ID}\n')

    with open(f'{data_dir}/utt2spk', "w", encoding="utf-8") as utt2spk_file:
        utt2spk_file.write(f'{ID}\t{ID}\n')
    
    print(f'###saved data to {data_dir} ')

def extract_features_using_kaldi(conf_path, data_dir):
    wav_scp_path = f'{data_dir}/wav.scp'
    spk2utt_path = f'{data_dir}/spk2utt'
    mfcc_path = f'{data_dir}/mfcc.ark'
    ivectors_path = f'{data_dir}/ivector.ark'
    feats_scp_path = f'{data_dir}/feat.scp'

    start_time = time()
    os.system(
        'compute-mfcc-feats --config='+conf_path+'/mfcc_hires.conf \
            scp,p:' + wav_scp_path+' ark:- | copy-feats \
            --compress=true ark:- ark,scp:' + mfcc_path + ',' + feats_scp_path)

    os.system(
        'ivector-extract-online2 --config='+ conf_path +'/ivector_extractor.conf ark:'+ spk2utt_path + '\
            scp:' + feats_scp_path + ' ark:' + ivectors_path)    
    end_time = time()

    print("###duration: ", end_time-start_time)

