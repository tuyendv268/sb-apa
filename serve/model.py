from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
import torch
import json
import sys
import os

from glob import glob
from tqdm import tqdm
import json

from g2p_en import G2p

from core import ScorerWav2vec2
from nemo_text_processing.text_normalization.normalize import Normalizer
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch

from data import infer_dataio_prep
from utils import (
    load_lexicon
)
import re

def load_state_dict(hparams):
    wav2vec2_state_dict = torch.load(f'{hparams["ckpt_path"]}/wav2vec2.ckpt')
    hparams["wav2vec2"].load_state_dict(wav2vec2_state_dict)

    model_state_dict = torch.load( f'{hparams["ckpt_path"]}/model.ckpt')
    hparams["model"].load_state_dict(model_state_dict)

    model_scorer_state_dict = torch.load(f'{hparams["ckpt_path"]}/model_scorer.ckpt')
    hparams["model_scorer"].load_state_dict(model_scorer_state_dict)

    return hparams

def init_model(hparams, run_opts):
    model = ScorerWav2vec2(
            modules=hparams["modules"],
            hparams=hparams,
            run_opts=run_opts
        )

    hparams = load_state_dict(hparams)
    
    for key, value in hparams["modules"].items():
        value.eval()

    label_encoder = sb.dataio.encoder.CTCTextEncoder.from_saved(
        hparams["label_encoder_path"])
    
    return model, label_encoder, hparams


class SB_PREP_Model():
    def __init__(self, hparams, run_opts) -> None:
        self.hparams = hparams        
        self.lexicon = load_lexicon(hparams["lexicon_path"])
        
        self.prep_model, self.label_encoder, self.hparams = init_model(hparams, run_opts)
        
        self.START_POSITION_ID = hparams["START_POSITION_ID"]
        self.PAD_POSITION_ID = hparams["PAD_POSITION_ID"]
        self.INNER_POSITION_ID = hparams["INNER_POSITION_ID"]
        self.END_POSITION_ID = hparams["END_POSITION_ID"]
        self.g2p = G2p()
        self.normalizer = Normalizer(input_case='cased', lang='en')
        
    def normalize(self, text):
        text = self.normalizer.normalize(
            text, verbose=True, punct_post_process=True)

        text = re.sub(
            r'[\!@#$%^&*\(\)\\\.\"\,\?\;\:\+\-\_\/\|~`]', ' ', text)
        
        text = re.sub('\s+', ' ', text)
        text = text.lower().strip()
        return text
            
    def prepare_input_text(self, transcript):
        transcript = self.normalize(transcript)
        words = transcript.split()
        
        word_ids = []
        phones, rel_positions = [], []
        for _word_id, _word in enumerate(words):
            try:
                _phones = self.lexicon[_word].split()
            except:
                print("### Out Of Vocabulary !!!")
                _phones = []
                for _phone in self.g2p(_word):
                    if _phone == "AH0":
                        _phone = "AX"
                    _phone = re.sub("\d", "", _phone).lower()
                    _phones.append(_phone)

            for _index, _phone in enumerate(_phones):
                if _index == 0:
                    _rel_position = self.START_POSITION_ID
                    
                elif _index == len(_phones) - 1:
                    _rel_position = self.END_POSITION_ID
                    
                else:
                    _rel_position = self.INNER_POSITION_ID

                rel_positions.append(str(_rel_position))
                word_ids.append(str(_word_id))

            phones += _phones

        assert len(phones) == len(rel_positions)
        phones, rel_positions, word_ids = " ".join(phones), \
            " ".join(rel_positions), " ".join(word_ids)

        return transcript, words, phones, rel_positions, word_ids
    
    def prepare_data(self, batch):
        dataset = {}
        for sample in batch:
            _id = sample["id"]
            _audio = sample["audio"]
            _transcript = sample["transcript"]
            
            _transcript, _words, _phones, _rel_positions, \
                _word_ids = self.prepare_input_text(_transcript)

            dataset[_id] = {
                "phn": _phones,
                "utt": _transcript,
                "wrd": _words,
                "phn_canonical": _phones,
                "rel_pos": _rel_positions,
                "wav": _audio,
                "wrd_id": _word_ids
            }

        return dataset
    
    
    def run(self, batch):
        """
            batch = [
                {
                    "id": ...
                    "audio": ...
                    "transcript": ...
                }, 
            ]
        """
        dataset = self.prepare_data(batch)
        outputs = self.inference(dataset=dataset)
        outputs = [outputs[sample["id"]] for index, sample in enumerate(batch)]
        
        return outputs
        
    @torch.no_grad()
    def inference(self, dataset):
        print(f'###batch_size: {len(dataset)}')
        dataset = infer_dataio_prep(dataset, self.label_encoder)
        dataloader = SaveableDataLoader(
            dataset, batch_size=self.hparams["batch_size"], 
            collate_fn=PaddedBatch, shuffle=False
        )

        pred_ids, wrd_ids= [], []
        utterances, words, phones = [], [], []
        phn_acc_scores, wrd_acc_scores, utt_acc_scores = [], [], []
        for batch in dataloader:
            ids = batch.id
            wavs, wav_lens = batch.sig
            rel_pos, _ = batch.rel_pos_list
            wrd_id, _ = batch.wrd_id_list
            phns, phn_lens = batch.phn_canonical_encoded
            phns_canonical_bos, _ = batch.phn_canonical_encoded_bos
            phns_canonical_eos, _ = batch.phn_canonical_encoded_eos

            wavs, wav_lens, rel_pos, phns_canonical_bos, phns_canonical_eos = \
                wavs.cuda(), wav_lens.cuda(), rel_pos.cuda(), phns_canonical_bos.cuda(), phns_canonical_eos.cuda()

            utt_acc_score, phn_acc_score, wrd_acc_score = self.prep_model.infer(
                wavs, wav_lens, rel_pos, phns_canonical_bos, phns_canonical_eos)
            
            # rescale score 
            utt_acc_score = (utt_acc_score * 100).cpu().round().int()
            wrd_acc_score = (wrd_acc_score * 100).cpu().round().int()
            phn_acc_score = (phn_acc_score[:, :-1] * 100).cpu().round().int()
            
            # unbatching
            phn_acc_score = self.prep_model.get_real_length_sequences(phn_acc_score, phn_lens)        
            wrd_acc_score = self.prep_model.get_real_length_sequences(wrd_acc_score, phn_lens)
            wrd_id = self.prep_model.get_real_length_sequences(wrd_id, phn_lens)
            
            # convert to list
            pred_ids += ids.tolist()
            wrd_ids += [score.tolist() for score in wrd_id]
            phn_acc_scores += [score.tolist() for score in phn_acc_score]
            wrd_acc_scores += [score.tolist() for score in wrd_acc_score]
            utt_acc_scores += utt_acc_score.cpu().tolist()

            phones += batch.phn
            utterances += batch.utt
            words += batch.wrd
            
        outputs = {}
        for index in range(len(pred_ids)):
            _pred_ids = pred_ids[index] 
            _wrd_ids = wrd_ids[index] 
            _utterances = utterances[index] 
            _words = words[index] 
            _phones = phones[index] 
            _phn_acc_scores = phn_acc_scores[index] 
            _wrd_acc_scores = wrd_acc_scores[index] 
            _utt_acc_scores = utt_acc_scores[index] 
            
            outputs[_pred_ids] = {
                "id": _pred_ids,
                "wrd_id": _wrd_ids,
                "utt": _utterances,
                "wrd": _words,
                "phn": _phones,
                "phn_acc_score": _phn_acc_scores,
                "wrd_acc_score": _wrd_acc_scores,
                "utt_acc_score": _utt_acc_scores,
                
            }
            
        return outputs

if __name__ == "__main__":
    argv = [
        'hparams.yml', 
        ]
    hparams_file, run_opts, overrides = sb.parse_arguments(argv)
    
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    data_dir = "/data/codes/sb-apa/wav"
    wav_files = glob(f'{data_dir}/*wav')
    transcripts = [
        os.path.basename(path).split(".wav")[0] for path in wav_files
    ]
    
    import librosa
    
    batch = []
    for index, (wav_path, transcript) in enumerate(zip(wav_files, transcripts)):
        wav, sr = librosa.load(wav_path, sr=16000)
        sample = {
            "id": index,
            "audio": wav.tolist(),
            "transcript": transcript
        }
        
        batch.append(sample)
    
    model = SB_PREP_Model(hparams=hparams, run_opts=run_opts)
    outputs = model.run(
        batch=batch
    )
            
    print(json.dumps(outputs[0], indent=4, ensure_ascii=False))
