from src.models.apr_model import APRWav2vec2
from src.models.scoring_model import ScorerWav2vec2

brain_classes = {
    "apr": {
        "wav2vec2.0": {
            "pure": APRWav2vec2,
        },
    },
    "scoring": {
        "wav2vec2.0": {
            "pure": ScorerWav2vec2,
        },
    }
}

def get_brain_class(hparams):
    model_task = hparams["model_task"]
    model_type = hparams["model_type"]
    if model_task == "apr":
        return brain_classes[model_task][model_type]["pure"]

    else:
        return brain_classes[model_task][model_type]["pure"]
