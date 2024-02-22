from kaldi.util.table import RandomAccessMatrixReader
from kaldi.util.table import DoubleMatrixWriter
from kaldi.alignment import MappedAligner
from kaldi.fstext import SymbolTable
from kaldi.matrix import Matrix
from kaldi.lat.align import (
    WordBoundaryInfoNewOpts, 
    WordBoundaryInfo)
    
from scipy.special import softmax
from time import time
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
import argparse
import shutil
import torch
import base64
import uuid
import json
import yaml
import sys
import os

from src.model.acoustic_model import FTDNNAcoustic

from kaldi.alignment import GmmAligner
from kaldi.fstext import SymbolTable
from kaldi.lat.align import WordBoundaryInfoNewOpts, WordBoundaryInfo
from kaldi.util.table import SequentialMatrixReader

from src.utils.aligner import (
    load_ivector_period_from_conf,
    extract_features_using_kaldi,
    load_config
)

class GMM_Aligner(object):
    def __init__(self, configs) -> None:
        self.data_dir = configs["data-dir"]
        
        self.conf_path = configs["gmm"]["conf-path"]
        self.kaldi_gmm_mdl_path = configs["gmm"]["kaldi-gmm-mdl-path"]
        self.kaldi_final_mat_path = configs["gmm"]["kaldi-mat-matrix-path"]
        self.tree_path = configs["gmm"]["tree-path"]
        self.phones_path = configs["gmm"]["phones-path"]
        
        self.disam_path = configs['gmm']['data']['disambig-path']
        self.word_boundary_path = configs['gmm']['data']['word-boundary-path']
        self.lang_graph_path = configs['gmm']['data']['lang-graph-path']
        self.words_path = configs['gmm']['data']['words-path']

        self.init_directories()
        
        self.model, self.phones, self.wb_info = self.initialize(
            self.kaldi_gmm_mdl_path, self.tree_path, self.lang_graph_path, 
            self.words_path, self.disam_path, self.phones_path, 
            self.word_boundary_path
        )
        
    def initialize(self, kaldi_gmm_mdl_path, tree_path, lang_graph_path, words_path, \
        disam_path, phones_path, word_boundary_path):
        aligner = GmmAligner.from_files(
            f"gmm-boost-silence --boost=1.0 1 {kaldi_gmm_mdl_path} - |",
            tree_path, lang_graph_path, words_path, 
            disam_path, self_loop_scale=0.1)
        
        phones = SymbolTable.read_text(phones_path)
        wb_info = WordBoundaryInfo.from_file(
            WordBoundaryInfoNewOpts(),
            word_boundary_path
        )
        
        return aligner, phones, wb_info
        
    def init_directories(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def prepare_data(self, data_dir, batch):
        wavscp_file = open(f'{data_dir}/wav.scp', "w", encoding="utf-8")
        text_file = open(f'{data_dir}/text', "w", encoding="utf-8")
        spk2utt_file = open(f'{data_dir}/spk2utt', "w", encoding="utf-8")
        utt2spk_file = open(f'{data_dir}/utt2spk', "w", encoding="utf-8")

        for index in range(len(batch)):
            id = batch[index]["id"]
            wav_path = batch[index]["wav_path"]
            transcript = batch[index]["transcript"]
            
            wavscp_file.write(f'{id}\t{wav_path}\n')
            text_file.write(f'{id}\t{transcript}\n')
            spk2utt_file.write(f'{id}\t{id}\n')
            utt2spk_file.write(f'{id}\t{id}\n')
        
        wavscp_file.close()
        text_file.close()
        spk2utt_file.close()
        utt2spk_file.close()

        print(f'###saved data to {data_dir}')
        
    def run(self, batch):
        """
        batch = [
            {
                "id": "8888",
                "wav_path": wav_path,
                "transcript": transcript
            }
        ]
        """
        data_dir = f'{self.data_dir}/{os.getpid()}'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        self.prepare_data(
            data_dir=data_dir, batch=batch,
        )

        output = self.run_align(
            batch=batch, data_dir=data_dir)
        
        output = self.post_process(output)

        return output

    def run_align(self, batch, data_dir):
        wav_scp_path = f'{data_dir}/wav.scp'
        text_path = f'{data_dir}/text'
        
        feats_rspecifier = (
            f"ark:compute-mfcc-feats --config={self.conf_path}/mfcc.conf --allow-downsample=true scp:{wav_scp_path} ark:- |"
            " apply-cmvn-sliding --cmn-window=10000 --center=true ark:- ark:- |"
            " splice-feats --left-context=3 --right-context=3 ark:- ark:- |"
            f" transform-feats {self.kaldi_final_mat_path} ark:- ark:- |"
            )
                
        id2ali = {}
        with SequentialMatrixReader(feats_rspecifier) as f, open(text_path) as t:
            for (fkey, feats), line in zip(f, t):
                tkey, text = line.strip().split(None, 1)
                output = self.model.align(feats, text)
                phone_alignment = self.model.to_phone_alignment(output["alignment"], self.phones)
                # print(fkey, phone_alignment, flush=True)
                id2ali[fkey] = phone_alignment
        
        for index, sample in enumerate(batch):
            batch[index]["alignment"] = id2ali[str(sample["id"])]
            
        return batch
    
    def post_process(self, batch):
        for index, sample in enumerate(batch):
            _alignment = []
                        
            for _phone in sample["alignment"]:
                if _phone[0] in ["SIL", "SPN"]:
                    continue
                _phone = list(_phone)
                _phone[1] = round(_phone[1]*0.01, 2)
                _phone[2] = round(_phone[1] + _phone[2]*0.01, 2)
                _alignment.append(_phone)
            
            batch[index]["alignment"] = _alignment
            
        return batch

    
class Nnet3_Aligner(object):
    def __init__(self, configs):
        self.device = configs["device"]
        self.data_dir = configs["data-dir"]
        
        self.wav_scp_path = f'{self.data_dir}/wav.scp'
        self.text_path = f'{self.data_dir}/text'
        self.spk2utt_path = f'{self.data_dir}/spk2utt'
        self.mfcc_path = f'{self.data_dir}/mfcc.ark'
        self.ivectors_path = f'{self.data_dir}/ivectors.ark'
        self.feats_scp_path = f'{self.data_dir}/feats.scp'

        self.acoustic_model_path = configs['acoustic-model-path']
        
        self.disam_path = configs["nnet3"]['data']['disambig-path']
        self.word_boundary_path = configs["nnet3"]['data']['word-boundary-path']
        self.lang_graph_path = configs["nnet3"]['data']['lang-graph-path']
        self.words_path = configs["nnet3"]['data']['words-path']
        
        self.num_senones = configs["nnet3"]["num_senones"]
        self.conf_path = configs["nnet3"]["conf-path"]
        self.final_mdl_path = configs["nnet3"]["kaldi-chain-mdl-path"]
        self.transition_model_path = configs["nnet3"]['transition-model-path']
        self.tree_path = configs["nnet3"]['tree-path']
        self.phones_path = configs["nnet3"]['kaldi-phones-path']

        self.lexicon = self.load_lexicon(configs['lexicon-path'])
        
        self.init_directories()

        self.aligner, self.phones, self.word_boundary_info, self.acoustic_model = \
            self.initialize(
                transition_model_path=self.transition_model_path, 
                tree_path=self.tree_path, 
                lang_graph_path=self.lang_graph_path, 
                words_path=self.words_path, 
                disam_path=self.disam_path, 
                phones_path=self.phones_path, 
                word_boundary_path=self.word_boundary_path, 
                acoustic_model_path=self.acoustic_model_path,
                num_senones=self.num_senones
            )
        self.ivector_period = load_ivector_period_from_conf(self.conf_path)
        self.acoustic_model.eval().to(self.device)
        
        
    def init_directories(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
                    
    def initialize(self, transition_model_path, tree_path, lang_graph_path, \
        words_path, disam_path, phones_path, word_boundary_path, acoustic_model_path, num_senones):

        aligner = MappedAligner.from_files(
            transition_model_path, tree_path, 
            lang_graph_path, words_path,
            disam_path, beam=40.0, acoustic_scale=1.0)
        
        phones  = SymbolTable.read_text(phones_path)
        word_boundary_info = WordBoundaryInfo.from_file(
            WordBoundaryInfoNewOpts(),
            word_boundary_path)

        acoustic_model = FTDNNAcoustic(num_senones=num_senones, device_name=self.device)
        acoustic_model.load_state_dict(torch.load(acoustic_model_path))
        acoustic_model.eval()

        print(f'load_state_dict from {acoustic_model_path}')

        return aligner, phones, word_boundary_info, acoustic_model
    
    def load_lexicon(self, path):
        lexicon = pd.read_csv(
            path, names=["word", "arapa"], sep="\t")
        
        vocab = set(lexicon["word"].to_list())

        return vocab
        
    def prepare_data(self, data_dir, batch):
        wavscp_file = open(f'{data_dir}/wav.scp', "w", encoding="utf-8")
        text_file = open(f'{data_dir}/text', "w", encoding="utf-8")
        spk2utt_file = open(f'{data_dir}/spk2utt', "w", encoding="utf-8")
        utt2spk_file = open(f'{data_dir}/utt2spk', "w", encoding="utf-8")

        for index in range(len(batch)):
            id = batch[index]["id"]
            wav_path = batch[index]["wav_path"]
            transcript = batch[index]["transcript"]
            
            wavscp_file.write(f'{id}\t{wav_path}\n')
            text_file.write(f'{id}\t{transcript}\n')
            spk2utt_file.write(f'{id}\t{id}\n')
            utt2spk_file.write(f'{id}\t{id}\n')
        
        wavscp_file.close()
        text_file.close()
        spk2utt_file.close()
        utt2spk_file.close()

        print(f'###saved data to {data_dir}')
    
    def run(self, batch):
        """
        batch = [
            {
                "id": "8888",
                "wav_path": wav_path,
                "transcript": transcript
            }
        ]
        """
        data_dir = f'{self.data_dir}/{os.getpid()}'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        self.prepare_data(data_dir=data_dir, batch=batch)
        extract_features_using_kaldi(
            conf_path=self.conf_path, data_dir=data_dir)
        
        output = self.run_align(batch=batch, data_dir=data_dir)
        output = self.post_process(output)

        return output
    
    def post_process(self, batch):
        for index, sample in enumerate(batch):
            _alignment = []
                        
            for _phone in sample["alignment"]:
                if _phone[0] in ["SIL", "SPN"]:
                    continue
                _phone = list(_phone)
                _phone[1] = _phone[1]*0.01
                _phone[2] = _phone[1] + _phone[2]*0.01
                
                _alignment.append(_phone)
                
            
            batch[index]["alignment"] = _alignment
            
        return batch

    def pad_1d(self, inputs, max_length=None, pad_value=0.0):
        if max_length is None:
            max_length = max([sample.shape[0] for sample in inputs])     
            
        attention_masks = []
        for i in range(len(inputs)):
            if inputs[i].shape[0] < max_length:
                attention_mask = [1]*inputs[i].shape[0] + [0]*(max_length-inputs[i].shape[0])
                
                padding = torch.ones(
                    (max_length-inputs[i].shape[0], inputs[i].shape[-1]))
                
                inputs[i] = torch.cat((inputs[i], padding), dim=0)

            elif inputs[i].shape[0] >= max_length:
                inputs[i] = inputs[i][:, 0:max_length]

                attention_mask = [1]*max_length

            attention_mask = torch.tensor(attention_mask)
            attention_masks.append(attention_mask)
        
        return {
            "inputs": torch.stack(inputs),
            "attention_mask": torch.vstack(attention_masks)
        }

    def run_align(self, batch, data_dir):
        mfccs_rspec = ("ark:" + f'{data_dir}/mfcc.ark')
        ivectors_rspec = ("ark:" + f'{data_dir}/ivector.ark')

        features = []
        mfccs_reader = RandomAccessMatrixReader(mfccs_rspec)
        ivectors_reader = RandomAccessMatrixReader(ivectors_rspec)

        for sample in batch:
            mfccs = mfccs_reader[sample["id"]]
            ivectors = ivectors_reader[sample["id"]]

            ivectors = np.repeat(ivectors, self.ivector_period, axis=0) 
            ivectors = ivectors[:mfccs.shape[0],:]
            x = np.concatenate((mfccs,ivectors), axis=1)
            feats = torch.from_numpy(x)

            features.append(feats)

        padded = self.pad_1d(inputs=features, pad_value=0.0)

        features = padded["inputs"].to(self.device)
        attention_mask = padded["attention_mask"].to(self.device)
        lengths = attention_mask.sum(1).cpu()

        with torch.no_grad():
            logits = self.acoustic_model(features)

        logits = logits.cpu()

        for index, sample in enumerate(batch):
            logit = logits[index].detach().numpy()
            length = lengths[index]

            logit = logit[0:length]
            transcript = sample["transcript"]

            log_likes = Matrix(logit)
            output = self.aligner.align(log_likes, transcript)

            alignment = self.aligner.to_phone_alignment(
                output["alignment"], self.phones)
            
            batch[index]["alignment"] = alignment
        
        return batch
    
    
if __name__ == "__main__":
    configs = load_config("config.yml")

    wav_path = "/workspace/align/tmp/5580072.wav"
    transcript = "She's paying for some jewelry at a check out".upper()

    nnet3_aligner = Nnet3_Aligner(configs=configs)
    # gmm_aligner = GMM_Aligner(configs=configs)
    
    sample = [
        {
            "id": "8888",
            "wav_path": wav_path,
            "transcript": transcript
        }
    ]

    batch = nnet3_aligner.run(sample)
    
    print(batch)
    
