from src.brain import get_brain_class
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
import torch
import json
import sys
import os

from utils.arpa import arpa_to_ipa
import pandas as pd
import re

def load_state_dict(hparams):
    wav2vec2_ckpt_path = f'{ckpt_path}/wav2vec2.ckpt'
    model_ckpt_path = f'{ckpt_path}/model.ckpt'
    model_scorer_ckpt_path = f'{ckpt_path}/model_scorer.ckpt'

    wav2vec2_state_dict = torch.load(wav2vec2_ckpt_path)
    model_state_dict = torch.load(model_ckpt_path)
    model_scorer_state_dict = torch.load(model_scorer_ckpt_path)

    hparams["wav2vec2"].load_state_dict(wav2vec2_state_dict)
    hparams["model"].load_state_dict(model_state_dict)
    hparams["model_scorer"].load_state_dict(model_scorer_state_dict)

    return hparams

def init_model(hparams):
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
    
    return model, hparams

def convert_word_to_arpa(word):
    word = lexicon[word].lower()
    word = word.replace("ax", "ah")
    word = word.split()

    return word

def prepare_input(audio_path, transcript):
    words = normalize(transcript).split()

    metadata = pd.DataFrame(
        {
            "word": words,
            "word-id": range(len(words))
        }
    )
    metadata["phone"] = metadata["word"].apply(convert_word_to_arpa)
    metadata = metadata.explode(column="phone")
    metadata = metadata.reset_index().drop(columns=["index"])
    
    metadata["rel-pos"] = 0
    curr_word_id = -1
    for index in metadata[:-1].index:
        if metadata["word-id"][index] != curr_word_id:
            rel_pos = 1
        elif metadata["word-id"][index] != metadata["word-id"][index+1]:
            rel_pos = 3
        else:
            rel_pos = 2
            
        curr_word_id = metadata["word-id"][index]
        metadata["rel-pos"][index] = rel_pos
    
    metadata.loc[metadata.index[-1], "rel-pos"] = 3

    return audio_path, metadata


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

def prepare_batch(audio_path, phn_canonical_list, rel_pos):
    phn_encoded_list = label_encoder.encode_sequence(phn_canonical_list)
    phn_canonical_encoded = torch.LongTensor(phn_encoded_list)
    phn_canonical_encoded_eos = torch.LongTensor(label_encoder.append_eos_index(phn_encoded_list))
    phn_canonical_encoded_bos = torch.LongTensor(label_encoder.prepend_bos_index(phn_encoded_list))
    

    wavs = sb.dataio.dataio.read_audio(audio_path)
    wavs = wavs.unsqueeze(0).cuda()
    wav_lens = torch.tensor([wavs.shape[1]]).cuda()
    rel_pos = torch.LongTensor(rel_pos).cuda()
    phns_canonical_bos = phn_canonical_encoded_bos.unsqueeze(0).cuda()
    phns_canonical_eos = phn_canonical_encoded_eos.unsqueeze(0).cuda()

    return wavs, wav_lens, rel_pos, phns_canonical_bos, phns_canonical_eos

def post_process(metadata, phn_acc_score, utt_acc_score):
    utt_acc_score = utt_acc_score.tolist()[0]
    phn_acc_score = phn_acc_score.tolist()[:-1]
    
    metadata["phone-score"] = phn_acc_score
    metadata["start-time"] = 0
    metadata["end-time"] = 0
    metadata["start-index"] = 0
    metadata["end-index"] = 0
    metadata["ipa"] = metadata["phone"]
    metadata["sound_most_like"] = metadata["phone"]
    
    print(utt_acc_score)

    sentence = {
        "duration": 0,
        "text": "",
        "score": utt_acc_score,
        "ipa": "",
        "words": [],
    }

    sentence["words"] = [None] * (metadata["word-id"].max() + 1)    
    for (word, word_id), group in metadata.groupby(["word", "word-id"]):
        group = group.reset_index()

        word = {
            "start_time": 0,
            "end_time": 0,
            "start_index": 0,
            "end_index": 0,
            "text": word,
            "arpabet": " ".join(group["phone"].tolist()),
            "ipa": " ".join(group["phone"].tolist()),
            "score": 0,
            "phonemes": []
        }
        for phone_index in group.index:
            phone = {
                "start_time": int(group["start-time"][phone_index]),
                "end_time": int(group["end-time"][phone_index]),
                "start_index": int(group["start-index"][phone_index]),
                "end_index": int(group["end-index"][phone_index]),
                "arpabet": group["phone"][phone_index],
                "ipa": arpa_to_ipa(group["phone"][phone_index]),
                "sound_most_like": group["sound_most_like"][phone_index],
                "score": int(group["phone-score"][phone_index])
            }

            word["phonemes"].append(phone)

        sentence["words"][word_id] = word

    sent_ipa = []
    for word in sentence["words"]:
        for phone in word["phonemes"]:
            sent_ipa.append(phone["ipa"])

    sentence["ipa"] = " ".join(sent_ipa)

    return {
        "version": "v0.1.0",
        "utterance": [sentence, ]
    }


def infer(audio_path, transcript):
    audio_path, metadata = prepare_input(
        audio_path=audio_path,
        transcript=transcript
    )
    
    rel_pos = metadata["rel-pos"].tolist()
    phn_canonical_list = metadata["phone"].tolist()

    wavs, wav_lens, rel_pos, phns_canonical_bos, phns_canonical_eos = prepare_batch(audio_path, phn_canonical_list, rel_pos)

    with torch.no_grad():
        utt_acc_score, phn_acc_score, word_acc_score = prep_model.infer(
            wavs, wav_lens, rel_pos, phns_canonical_bos, phns_canonical_eos)

        phn_acc_score = (phn_acc_score * 100).round()
        utt_acc_score = (utt_acc_score * 100).round()
        
        phn_acc_score = phn_acc_score.detach().cpu()[0]
        utt_acc_score = utt_acc_score.detach().cpu()[0]

    result = post_process(metadata, phn_acc_score, utt_acc_score)

    return result

import soundfile as sf
from flask import (
    Flask, 
    request
)
import io

app = Flask(__name__)

@app.route('/infer', methods=['GET', 'POST'])
def api_endpoint():
    wav_file = request.files.get('file')
    waveform, samplerate = sf.read(io.BytesIO(wav_file.read()))

    transcript = request.form['transcript']
    audio_path = "temp.wav"

    sf.write(audio_path, waveform, samplerate=samplerate)

    result = infer(audio_path, transcript)
    return result


if __name__ == "__main__":
    DATA_DIR = "data"
    PREP_DATA_FOLDER = f'{DATA_DIR}/apr/'

    RESULTS_FOLDER = f'{DATA_DIR}/results/'
    EXP_METADATA_FILE = f'{RESULTS_FOLDER}/exp_metadata.csv'
    PREP_SCORING_RESULTS_FILE = f'{RESULTS_FOLDER}/results_scoring.csv'
    EPOCH_RESULTS_DIR = f'{RESULTS_FOLDER}/epoch_results'
    PARAMS_DIR= f'{RESULTS_FOLDER}/params'


    MODEL_TYPE = "w2v2"
    SCORING_TYPE=""

    SCORING_MODEL_DIR = f"results/scoring"
    PRETRAINED_MODEL_DIR = f"results/apr"
    SCORING_HPARAM_FILE = f"hparams/scoring.yml"

    argv = [
        SCORING_HPARAM_FILE,
        "--data_folder", PREP_DATA_FOLDER,
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

    lexicon_path = "resources/lexicon"
    ckpt_path = "results/scoring/save/CKPT+2024-02-04+11-46-32+00"
    label_encoder_path = "results/scoring/save/label_encoder.txt"
    
    hparams["ckpt_path"] = ckpt_path
    hparams["label_encoder_path"] = label_encoder_path
    label_encoder_path = hparams["label_encoder_path"]

    prep_model, hparams = init_model(hparams)
    label_encoder = sb.dataio.encoder.CTCTextEncoder.from_saved(label_encoder_path)
    lexicon = load_lexicon(lexicon_path)

    app.run(host='0.0.0.0', port=6868)
