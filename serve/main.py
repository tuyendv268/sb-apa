from src.utils.aligner import load_config

from src.serve.aligner import (
    Forced_Aligner,
    Nnet3_Forced_Aligner,
    GMM_Forced_Aligner
)

from src.serve.scorer import WavLM_Model

from starlette.requests import Request
from ray.serve.handle import DeploymentHandle

from ray import serve
import asyncio


from src.utils.aligner import (
    load_config
)

@serve.deployment(
    # num_replicas=2,
    # max_concurrent_queries=128,
    num_replicas=1,
    max_concurrent_queries=16,
    # health_check_period_s=10,
    # health_check_timeout_s=30,
    # graceful_shutdown_timeout_s=20,
    # graceful_shutdown_wait_loop_s=2,
    route_prefix="/scoring",
    ray_actor_options={
        "num_cpus": 0.05,
        }
)
class Main:
    def __init__(self, aligner, scorer):
        self.aligner: DeploymentHandle = aligner.options(
            use_new_handle_api=True,
        )
        self.scorer: DeploymentHandle = scorer.options(
            use_new_handle_api=True,
        )

    async def run_align(self, sample):
        align_outputs = await self.aligner.run.remote(
            sample
        )

        return align_outputs

    async def run_scoring(self, sample):
        score_outputs = await self.scorer.run.remote(
            sample
        )
        
        return score_outputs
    
    async def __call__(self, http_request: Request):
        sample = await http_request.json() 

        outputs = await self.run_align(sample)
        outputs = await self.run_scoring(outputs)
        
        return outputs


configs = load_config("config.yml")
nnet3_app = Nnet3_Forced_Aligner.bind(configs)
gmm_app = GMM_Forced_Aligner.bind(configs)

aligner_app = Forced_Aligner.bind(gmm_app, nnet3_app)
scorer_app = WavLM_Model.bind()

app = Main.bind(aligner_app, scorer_app)
