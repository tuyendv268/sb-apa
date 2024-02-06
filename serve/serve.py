from starlette.requests import Request
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from ray import serve
import torch

from model import SB_PREP_Model

import pandas as pd
from arpa import arpa_to_ipa

def parse_result(sample):
    length = len(sample["wrd_id"])
    
    sample["id"] = [sample["id"], ]*length
    sample["utt"] = [sample["utt"], ]*length
    sample["wrd"] = [sample["wrd"][wrd_id] for wrd_id in sample["wrd_id"]]
    sample["phn"] = sample["phn"].split()
    sample["utt_acc_score"] = sample["utt_acc_score"]*length
    
    metadata = pd.DataFrame(sample)
    
    metadata["start-time"] = 0
    metadata["end-time"] = 0
    metadata["start-index"] = 0
    metadata["end-index"] = 0
    metadata["ipa"] = metadata["phn"].apply(arpa_to_ipa)
    metadata["sound_most_like"] = metadata["phn"]
    
    sentence = {
        "duration": 0,
        "text": [],
        "score": round(metadata["utt_acc_score"].mean()),
        "ipa": [],
        "words": [],
    }

    sentence["words"] = [None] * (metadata["wrd_id"].max() + 1)    
    for (word, word_id), group in metadata.groupby(["wrd", "wrd_id"]):
        group = group.reset_index()

        word = {
            "start_time": 0,
            "end_time": 0,
            "start_index": 0,
            "end_index": 0,
            "text": word,
            "arpabet": " ".join(group["phn"].tolist()),
            "ipa": " ".join(group["ipa"].tolist()),
            "score": round(group["wrd_acc_score"].mean()),
            "phonemes": []
        }
        for phone_index in group.index:
            phone = {
                "start_time": int(group["start-time"][phone_index]),
                "end_time": int(group["end-time"][phone_index]),
                "start_index": int(group["start-index"][phone_index]),
                "end_index": int(group["end-index"][phone_index]),
                "arpabet": group["phn"][phone_index],
                "ipa": group["ipa"][phone_index],
                "sound_most_like": group["sound_most_like"][phone_index],
                "score": int(group["phn_acc_score"][phone_index])
            }

            word["phonemes"].append(phone)
            sentence["ipa"].append(phone["ipa"])

        sentence["words"][word_id] = word
        sentence["text"].append(word["text"])

    sentence["ipa"] = " ".join(sentence["ipa"])
    sentence["text"] = " ".join(sentence["text"])
    
    return {
        "version": "v0.1.0",
        "utterance": [sentence, ]
    }
    

@serve.deployment(
    num_replicas=1, 
    max_concurrent_queries=64,
    route_prefix="/scoring",
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
        
        self.model = SB_PREP_Model(
            hparams=hparams, run_opts=run_opts
        )
        
    def prep_batch(self, batch, audio_name="audio", transcript_name="transcript"):
        
        dataset = []
        for index, sample in enumerate(batch):
            _transcript = sample[transcript_name]
            _audio = sample[audio_name]
            
            dataset.append(
                {
                    "id": index,
                    "audio": _audio,
                    "transcript": _transcript,
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
    
app = WavLM_Model.bind()
