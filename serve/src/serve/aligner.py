from src.utils.aligner import load_config
from src.interface.aligner import Nnet3_Aligner, GMM_Aligner

from ray.serve.handle import DeploymentHandle
from starlette.requests import Request
import soundfile as sf
from ray import serve
import numpy as np
import asyncio
import re
import os

def normalize(text):
    text = re.sub(r'[\!@#$%^&*\(\)\\\.\"\,\?\;\:\+\-\_\/\|~`]', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.upper().strip()
    return text

@serve.deployment(
    # num_replicas=2, 
    # max_concurrent_queries=64,
    num_replicas=1, 
    max_concurrent_queries=16,
    # autoscaling_config={
    #     "min_replicas": 1,
    #     "initial_replicas": 2,
    #     "max_replicas": 4,
    #     "target_num_ongoing_requests_per_replica": 10
    # },
    # health_check_period_s=10,
    # health_check_timeout_s=30,
    # graceful_shutdown_timeout_s=20,
    # graceful_shutdown_wait_loop_s=2,
    ray_actor_options={
        "num_cpus": 0.2, "num_gpus": 0.2
        }
    )
class Nnet3_Forced_Aligner:
    def __init__(self, configs):
        self.model = Nnet3_Aligner(configs=configs)
        self.configs = configs

        self.init_dir()

    def init_dir(self):
        if not os.path.exists(self.configs["data-dir"]):
            os.mkdir(self.configs["data-dir"])

        data_dir = f'{self.configs["data-dir"]}/{os.getpid()}'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        audio_dir = f'{data_dir}/wav'
        if not os.path.exists(audio_dir):
            os.mkdir(audio_dir)
        
        self.audio_dir = audio_dir

    def preprocess(self, batch):        
        processed_batch = []
        for _id, _sample in enumerate(batch):
            _id = f'audio-{_id}'
            _transcript = normalize(_sample["transcript"])
            _audio_path = f'{self.audio_dir}/{_id}.wav'
            
            waveform = np.array(_sample["audio"])
            sf.write(_audio_path, waveform, samplerate=16000)

            processed_batch.append(
                {
                    "id": _id,
                    "transcript": _transcript,
                    "wav_path": _audio_path,
                }
            )

        return processed_batch

    @serve.batch(max_batch_size=128, batch_wait_timeout_s=0.1)
    async def run(self, batch):
        """
        batch = [
            {
                "id": _id,
                "transcript": _transcript,
                "wav_path": _audio_path
            }
        ]
        """
        batch = self.preprocess(batch)
        batch = self.model.run(batch)
        output = self.postprocess(batch)

        return output
    
    async def __call__(self, http_request: Request):
        sample = await http_request.json() 
        
        return await self.run(sample)


    def postprocess(self, batch):

        return batch

    
@serve.deployment(
    # num_replicas=2, 
    # max_concurrent_queries=64,
    num_replicas=1, 
    max_concurrent_queries=16,
    # autoscaling_config={
    #     "min_replicas": 1,
    #     "initial_replicas": 2,
    #     "max_replicas": 4,
    #     "target_num_ongoing_requests_per_replica": 10
    # },
    # health_check_period_s=10,
    # health_check_timeout_s=30,
    # graceful_shutdown_timeout_s=20,
    # graceful_shutdown_wait_loop_s=2,
    ray_actor_options={
        "num_cpus": 0.2, "num_gpus": 0.2
        }
    )
class GMM_Forced_Aligner:
    def __init__(self, configs):
        self.model = GMM_Aligner(configs=configs)
        self.configs = configs

        self.init_dir()

    def init_dir(self):
        if not os.path.exists(self.configs["data-dir"]):
            os.mkdir(self.configs["data-dir"])

        data_dir = f'{self.configs["data-dir"]}/{os.getpid()}'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        audio_dir = f'{data_dir}/wav'
        if not os.path.exists(audio_dir):
            os.mkdir(audio_dir)
        
        self.audio_dir = audio_dir

    def preprocess(self, batch):        
        processed_batch = []
        for _id, _sample in enumerate(batch):
            _id = f'audio-{_id}'
            _transcript = normalize(_sample["transcript"])
            _audio_path = f'{self.audio_dir}/{_id}.wav'
            
            waveform = np.array(_sample["audio"])
            sf.write(_audio_path, waveform, samplerate=16000)

            processed_batch.append(
                {
                    "id": _id,
                    "transcript": _transcript,
                    "wav_path": _audio_path
                }
            )

        return processed_batch

    @serve.batch(max_batch_size=128, batch_wait_timeout_s=0.1)
    async def run(self, batch):
        """
        batch = [
            {
                "id": _id,
                "transcript": _transcript,
                "wav_path": _audio_path
            }
        ]
        """
        batch = self.preprocess(batch)
        batch = self.model.run(batch)
        output = self.postprocess(batch)

        return output
    
    async def __call__(self, http_request: Request):
        sample = await http_request.json() 
        
        return await self.run(sample)


    def postprocess(self, batch):

        return batch

@serve.deployment(
    # num_replicas=2,
    # max_concurrent_queries=128,
    num_replicas=1,
    max_concurrent_queries=16,
    # health_check_period_s=10,
    # health_check_timeout_s=30,
    # graceful_shutdown_timeout_s=20,
    # graceful_shutdown_wait_loop_s=2,
    ray_actor_options={
        "num_cpus": 0.05,
        }
)
class Forced_Aligner:
    def __init__(self, gmm_aligner, nnet3_aligner):
        self.gmm_aligner: DeploymentHandle = gmm_aligner.options(
            use_new_handle_api=True,
        )
        self.nnet3_aligner: DeploymentHandle = nnet3_aligner.options(
            use_new_handle_api=True,
        )

    async def run_gmm_aligner(self, sample):
        return await self.gmm_aligner.run.remote(
            sample
        )

    async def run_nnet3_aligner(self, sample):
        return await self.nnet3_aligner.run.remote(
            sample
        )
        
    async def run(self, sample):
        if len(sample["transcript"].split()) > 2:
            print("###Run NNet3 aligner")
            return await self.run_nnet3_aligner(
                 sample
            )
        else:
            print("###Run GMM aligner")
            return await self.run_gmm_aligner(
                 sample
            ) 

    async def __call__(self, http_request: Request):
        sample = await http_request.json() 

        if len(sample["transcript"].split()) > 2:
            print("###Run NNet3 aligner")
            return await self.run_nnet3_aligner(
                 sample
            )
        else:
            print("###Run GMM aligner")
            return await self.run_gmm_aligner(
                 sample
            ) 

# configs = load_config("config.yml")
# nnet3_app = Nnet3_Forced_Aligner.bind(configs)
# gmm_app = GMM_Forced_Aligner.bind(configs)

# app = Forced_Aligner.bind(gmm_app, nnet3_app)
