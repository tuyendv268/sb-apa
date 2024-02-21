from starlette.requests import Request
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from ray import serve
import librosa
import torch

from src.core.scorer import Prep_Scorer

import pandas as pd
from src.utils.arpa import arpa_to_ipa

def parse_result(sample):
    length = len(sample["wrd_id"])
    duration = sample["duration"]

    sample["id"] = [sample["id"], ]*length
    sample["utt"] = [sample["utt"], ]*length
    sample["wrd"] = [sample["wrd"][wrd_id] for wrd_id in sample["wrd_id"]]
    sample["phn"] = sample["phn"].split()
    sample["utt_acc_score"] = sample["utt_acc_score"]*length

    metadata = pd.DataFrame(sample)

    metadata["ipa"] = metadata["phn"].apply(arpa_to_ipa)
    metadata["sound_most_like"] = metadata["phn"]

    sentence = {
        "duration": None,
        "text": None,
        "score": None,
        "ipa": None,
        "words": None,
    }

    sentence_words = [None] * (metadata["wrd_id"].max() + 1)
    sentence_score = round(metadata["utt_acc_score"].mean())
    sentence_text = []
    sentence_ipas = []
    sentence_duration = duration

    current_word_index, current_phone_index = -1, -1
    for (word, word_id), group in metadata.groupby(["wrd", "wrd_id"], sort=False):
        group = group.reset_index().sort_values(by="start_time")

        word_text = word
        word_ipa = "".join(group["ipa"].tolist())
        word_start_time = group.loc[group.index[0], "start_time"]
        word_end_time = group.loc[group.index[-1], "end_time"]
        word_start_index = current_word_index + 1
        word_end_index = word_start_index + len(word_ipa) - 1
        word_score = round(group["wrd_acc_score"].mean())
        word_arpabet = " ".join(group["phn"].tolist())

        current_word_index = word_end_index + 1

        word = {
            "start_time": word_start_time,
            "end_time": word_end_time,
            "start_index": word_start_index,
            "end_index": word_end_index,
            "text": word_text,
            "arpabet": word_arpabet,
            "ipa": word_ipa,
            "score": word_score,
            "phonemes": []
        }
        for phone_index in group.index:
            phone_start_time = group["start_time"][phone_index]
            phone_end_time = group["end_time"][phone_index]
            phone_arpabet = group["phn"][phone_index]
            phone_start_index = current_phone_index + 1
            phone_end_index = phone_start_index + len(phone_arpabet) - 1
            phone_ipa = group["ipa"][phone_index]
            phone_sound_most_like = group["sound_most_like"][phone_index]
            phone_score = int(group["phn_acc_score"][phone_index])

            current_phone_index = phone_end_index

            phone = {
                "start_time": phone_start_time,
                "end_time": phone_end_time,
                "start_index": phone_start_index,
                "end_index": phone_end_index,
                "arpabet": phone_arpabet,
                "ipa": phone_ipa,
                "sound_most_like": phone_sound_most_like,
                "score": phone_score
            }

            word["phonemes"].append(phone)

        sentence_words[word_id] = word
        sentence_text.append(word["text"])
        sentence_ipas.append(word["ipa"])

    sentence["duration"] = sentence_duration
    sentence["text"] = " ".join(sentence_text)
    sentence["score"] = sentence_score
    sentence["words"] = sentence_words
    sentence["ipa"] = " ".join(sentence_ipas)
    
    return {
        "version": "v0.1.0",
        "utterance": [sentence, ]
    }


@serve.deployment(
    num_replicas=1, 
    max_concurrent_queries=64,
    ray_actor_options={
        "num_cpus": 0.1, "num_gpus": 0.2
        }
    )
class WavLM_Model:
    def __init__(self):
        argv = [
            'hparams.yml', 
            ]
        hparams_file, run_opts, overrides = sb.parse_arguments(argv)
        
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)
        
        self.model = Prep_Scorer(
            hparams=hparams, run_opts=run_opts
        )
        
    def parse_alignment(self, alignment):
        rel2id = {"B":1, "I": 2, "E":3, "S": 2}

        _transcript, _rel_position, _word_id, _alignment = [], [], [], []
        
        curr_word_id = -1
        for _phone in alignment:
            prefix, postfix = _phone[0].split("_")
            if postfix == "B" or postfix == "S":
                curr_word_id += 1

            _transcript.append(prefix)
            _rel_position.append(rel2id[postfix])
            _word_id.append(curr_word_id)
            _alignment.append([_phone[1], _phone[2]])
            
        return _transcript, _rel_position, _word_id, _alignment
        
    def prep_batch(self, batch):
        dataset = []
        for index, sample in enumerate(batch):
            _audio_path = sample["wav_path"]
            _alignment =  sample["alignment"]
            _transcript = sample["transcript"]
            
            _transcript_arpabet, _rel_position, _word_id, _alignment = self.parse_alignment(_alignment)
            _audio, _sr = librosa.load(_audio_path, sr=16000)
            _duration = _audio.shape[0] / _sr
                 
            dataset.append(
                {
                    "id": index,
                    "audio": _audio,
                    "duration": _duration,
                    "transcript": _transcript,
                    "transcript_arpabet": _transcript_arpabet,
                    "rel_position": _rel_position,
                    "word_id": _word_id,
                    "alignment": _alignment
                    }
                )
        
        return dataset

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def run(self, batch):
        batch = self.prep_batch(batch)
        output = self.model.run(batch)
        output = [parse_result(sample) for sample in output]
        
        return output

    async def __call__(self, http_request: Request):
        sample = await http_request.json() 
        
        return await self.run(sample)