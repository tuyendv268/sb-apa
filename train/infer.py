from src.brain import get_brain_class
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
import torch
import json
import sys
import os

from glob import glob
from tqdm import tqdm
import json

from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
from utils.arpa import arpa_to_ipa
import pandas as pd
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
    brain_class = get_brain_class(hparams)

    model = brain_class(
            modules=hparams["modules"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )

    hparams = load_state_dict(hparams)
    
    for key, value in hparams["modules"].items():
        value.eval()

    label_encoder = sb.dataio.encoder.CTCTextEncoder.from_saved(
        hparams["label_encoder_path"])
    
    return model, label_encoder, hparams

def prepare_transcipt(transcript):
    transcript = normalize(transcript)

    phones, rel_positions = [], []
    word_ids = []
    words = transcript.split()
    for _word_id, _word in enumerate(words):
        _phones = lexicon[_word].split()

        for _index, _phone in enumerate(_phones):
            if _index == 0:
                _rel_position = START_POSITION_ID
            elif _index == len(_phones) - 1:
                _rel_position = END_POSITION_ID
            else:
                _rel_position = INNER_POSITION_ID

            rel_positions.append(str(_rel_position))
            word_ids.append(str(_word_id))

        phones += _phones

    assert len(phones) == len(rel_positions)
    phones, rel_positions, word_ids = " ".join(phones), \
        " ".join(rel_positions), " ".join(word_ids)

    return transcript, words, phones, rel_positions, word_ids

def prepare_data(wav_files, transcripts, ids=None):
    dataset = {}
    for index in range(len(wav_files)):
        if ids is not None:
            _id = ids[index]
        else:
            _id = index
        
        _audio_path = wav_files[index]
        _transcript = transcripts[index]
        
        _transcript, _words, _phones, _rel_positions, _word_ids = prepare_transcipt(_transcript)

        dataset[_id] = {
            "phn": _phones,
            "utt": _transcript,
            "wrd": _words,
            "phn_canonical": _phones,
            "rel_pos": _rel_positions,
            "wav": _audio_path,
            "wrd_id": _word_ids
        }

    return dataset

def load_lexicon(path):
    lexicon = pd.read_csv(path, names=["word", "arpa"], sep="\t")

    lexicon.dropna(inplace=True)
    lexicon["word"] = lexicon.word.apply(lambda x: x.lower())
    lexicon["arpa"] = lexicon.arpa.apply(lambda x: re.sub("\d", "", x).lower())

    lexicon.word.drop_duplicates(inplace=True)
    lexicon.set_index("word", inplace=True)
    lexicon = lexicon.to_dict()["arpa"]

    return lexicon

def normalize(text):
    text = re.sub(
        r'[\!@#$%^&*\(\)\\\.\"\,\?\;\:\+\-\_\/\|~`]', ' ', text)
    
    text = re.sub('\s+', ' ', text)
    text = text.lower().strip()
    return text

def infer_dataio_prep(wav_files, transcripts, ids):
    dataset = prepare_data(
        wav_files=wav_files,
        transcripts=transcripts,
        ids=ids
    )

    dataset = sb.dataio.dataset.DynamicItemDataset(dataset)
    dataset = [dataset, ]

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig
    
    sb.dataio.dataset.add_dynamic_item(dataset, audio_pipeline)

    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides(
        "phn_encoded",
        "phn_encoded_eos",
        "phn_encoded_bos",
    )
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded
        phn_encoded_eos = torch.LongTensor(
            label_encoder.append_eos_index(phn_encoded_list)
        )
        yield phn_encoded_eos
        phn_encoded_bos = torch.LongTensor(
            label_encoder.prepend_bos_index(phn_encoded_list)
        )
        yield phn_encoded_bos

    sb.dataio.dataset.add_dynamic_item(dataset, text_pipeline)

    @sb.utils.data_pipeline.takes(
        "utt", "wrd", "phn", "wrd_id", "rel_pos"
        )
    @sb.utils.data_pipeline.provides(
        "utt",
        "wrd",
        "phn",
        "phn_canonical_encoded",
        "phn_canonical_encoded_eos",
        "phn_canonical_encoded_bos",
        "wrd_id_list",
        "rel_pos_list"
    )
    def text_canonical_pipeline(utt, wrd, phn, wrd_id, rel_pos):
        yield utt
        yield wrd

        phn_list = phn.strip().split()
        yield phn_list

        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded

        phn_encoded_eos = torch.LongTensor(
            label_encoder.append_eos_index(phn_encoded_list)
        )
        yield phn_encoded_eos

        phn_encoded_bos = torch.LongTensor(
            label_encoder.prepend_bos_index(phn_encoded_list)
        )
        yield phn_encoded_bos

        wrd_id_list = [int(ele) for ele in wrd_id.strip().split()]
        wrd_encoded_list = torch.LongTensor(wrd_id_list)
        yield wrd_encoded_list
        
        rel_pos_list = [int(ele) for ele in rel_pos.strip().split()]
        rel_pos_list = torch.LongTensor(rel_pos_list)
        yield rel_pos_list
    sb.dataio.dataset.add_dynamic_item(dataset, text_canonical_pipeline)

    output_keys = [
        "id",
        "sig",
        "utt",
        "wrd",
        "phn",
        "phn_canonical_encoded",
        "phn_canonical_encoded_bos",
        "phn_canonical_encoded_eos",
        "wrd_id_list",
        "rel_pos_list",
    ]

    sb.dataio.dataset.set_output_keys(dataset, output_keys)
    return dataset[0]

@torch.no_grad()
def inference(wav_files, transcripts, sample_ids, batch_size=4, shuffle=False):
    assert shuffle == False
    dataset = infer_dataio_prep(
        wav_files=wav_files, transcripts=transcripts, ids=sample_ids)
    dataloader = SaveableDataLoader(
        dataset, batch_size=batch_size, collate_fn=PaddedBatch, shuffle=shuffle)

    pred_ids, wrd_ids= [], []
    utterances, words, phones = [], [], []
    phn_acc_scores, wrd_acc_scores, utt_acc_scores = [], [], []
    for batch in tqdm(dataloader, desc="Infer"):
        ids = batch.id
        wavs, wav_lens = batch.sig
        rel_pos, _ = batch.rel_pos_list
        wrd_id, _ = batch.wrd_id_list
        phns, phn_lens = batch.phn_canonical_encoded
        phns_canonical_bos, _ = batch.phn_canonical_encoded_bos
        phns_canonical_eos, _ = batch.phn_canonical_encoded_eos

        wavs, wav_lens, rel_pos, phns_canonical_bos, phns_canonical_eos = \
            wavs.cuda(), wav_lens.cuda(), rel_pos.cuda(), phns_canonical_bos.cuda(), phns_canonical_eos.cuda()

        utt_acc_score, phn_acc_score, wrd_acc_score = prep_model.infer(
            wavs, wav_lens, rel_pos, phns_canonical_bos, phns_canonical_eos)
            
        utt_acc_score = (utt_acc_score * 100).cpu().round().int()
        wrd_acc_score = (wrd_acc_score * 100).cpu().round().int()
        phn_acc_score = (phn_acc_score * 100).cpu().round().int()
            
        phn_acc_score = prep_model.get_real_length_sequences(phn_acc_score, phn_lens)        
        wrd_acc_score = prep_model.get_real_length_sequences(wrd_acc_score, phn_lens)
        wrd_id = prep_model.get_real_length_sequences(wrd_id, phn_lens)
        
        pred_ids += ids.tolist()
        wrd_ids += [score.tolist() for score in wrd_id]
        phn_acc_scores += [score.tolist() for score in phn_acc_score]
        wrd_acc_scores += [score.tolist() for score in wrd_acc_score]
        utt_acc_scores += utt_acc_score.cpu().tolist()

        phones += batch.phn
        utterances += batch.utt
        words += batch.wrd

    if sample_ids is not None:
        assert len(pred_ids) == len(sample_ids)
        for _id, _pred_id in zip(sample_ids, pred_ids):
            assert _id == _pred_id

    return pred_ids, wrd_ids, utterances, words, phones, phn_acc_scores, wrd_acc_scores, utt_acc_scores

if __name__ == "__main__":
    PAD_POSITION_ID = 0
    START_POSITION_ID = 1
    INNER_POSITION_ID = 2
    END_POSITION_ID = 3

    DATA_DIR = "data"
    APR_DATA_FOLDER = f'{DATA_DIR}/apr/'

    RESULTS_FOLDER = f'{DATA_DIR}/results/'
    EXP_METADATA_FILE = f'{RESULTS_FOLDER}/exp_metadata.csv'
    PREP_SCORING_RESULTS_FILE = f'{RESULTS_FOLDER}/results_scoring.csv'
    EPOCH_RESULTS_DIR = f'{RESULTS_FOLDER}/epoch_results'
    PARAMS_DIR= f'{RESULTS_FOLDER}/params'

    SCORING_MODEL_DIR = f"results/scoring"
    PRETRAINED_MODEL_DIR = f"results/apr"
    SCORING_HPARAM_FILE = f"hparams/scoring.yml"

    argv = [
        SCORING_HPARAM_FILE,
        "--data_folder", APR_DATA_FOLDER,
        "--batch_size", "4",
        "--pretrained_model_folder", PRETRAINED_MODEL_DIR,
        "--use_augmentation", "True",
        "--exp_folder", SCORING_MODEL_DIR,
        "--exp_metadata_file", EXP_METADATA_FILE,
        "--results_file", PREP_SCORING_RESULTS_FILE,
        "--epoch_results_dir", EPOCH_RESULTS_DIR,
        "--params_dir", PARAMS_DIR
        ]

    hparams_file, run_opts, overrides = sb.parse_arguments(argv)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    LEXICON_PATH = "lexicon"
    CKPT_PATH = "results/scoring/save/best"
    LABLE_ENCODER_PATH = "results/scoring/save/label_encoder.txt"

    hparams["ckpt_path"] = CKPT_PATH
    hparams["label_encoder_path"] = LABLE_ENCODER_PATH

    lexicon = load_lexicon(LEXICON_PATH)
    prep_model, label_encoder, hparams = init_model(hparams, run_opts)

    data_dir = "/data/codes/sb-apa/wav"
    wav_files = glob(f'{data_dir}/*wav')
    transcripts = [
        os.path.basename(path).split(".wav")[0] for path in wav_files
    ]
    sample_ids = [i for i in range(len(wav_files))]

    inference(
        wav_files=wav_files,
        transcripts=transcripts,
        sample_ids=sample_ids
    )


