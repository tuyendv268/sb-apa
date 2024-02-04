"""This lobe is the interface for huggingface transformers models
It enables loading config and model via AutoConfig & AutoModel.

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Titouan Parcollet 2021, 2022, 2023
 * Mirco Ravanelli 2021
 * Boumadane Abdelmoumene 2021
 * Ju-Chieh Chou 2021
 * Artem Ploujnikov 2021, 2022
 * Abdel Heba 2021
 * Aku Rouhe 2022
 * Arseniy Gorin 2022
 * Ali Safaya 2022
 * Benoit Wang 2022
 * Adel Moumen 2022, 2023
 * Andreas Nautsch 2022, 2023
 * Luca Della Libera 2022
 * Heitor Guimarães 2022
 * Ha Nguyen 2023
"""
import os
import torch
import logging
import pathlib
from torch import nn
from typing import Union
from huggingface_hub import model_info
from speechbrain.pretrained.fetching import fetch
from speechbrain.dataio.dataio import length_to_mask


# We check if transformers is installed.
try:
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        AutoFeatureExtractor,
        AutoModelForPreTraining,
        # AutoProcessor,
        AutoModel,
    )

except ImportError:
    MSG = "Please install transformers from HuggingFace to use wav2vec2 / Hubert\n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)

logger = logging.getLogger(__name__)


class HuggingFaceTransformer(nn.Module):
    """This lobe provides an interface for integrating any HuggingFace transformer model within SpeechBrain
    We use AutoClasses for loading any model from the hub and its necessary components.
    For example, we build HuggingFaceWav2Vec2 class which inherits HuggingFaceTransformer for working with HuggingFace's wav2vec models
    While HuggingFaceWav2Vec2 can enjoy some already built features like modeling loading, pretrained weights loading, all weights freezing,
    feature_extractor loading, etc.
    Users are expected to override the essential forward() function to fit their specific needs.
    Depending on the HuggingFace transformer model in question, one can also modify the state_dict by overwriting the _modify_state_dict() method,
    or adapting their config by modifying override_config() method, etc.
    See:
    https://huggingface.co/docs/transformers/model_doc/auto
    https://huggingface.co/docs/transformers/autoclass_tutorial

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        save directory of the downloaded model.
    for_pretraining: bool (default: False)
        If True, build the model for pretraining
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    cache_dir: str or Path (default: None)
        Location of HuggingFace cache for storing pre-trained models, to which symlinks are created.

    Example
    -------
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "tmp"
    >>> model = HuggingFaceTransformer(model_hub, save_path=save_path)
    """

    def __init__(
        self,
        source,
        save_path,
        for_pretraining=False,
        freeze=False,
        cache_dir: Union[str, pathlib.Path, None] = "pretrained_models",
        **kwarg,
    ):
        super().__init__()

        self.load_feature_extractor(source, cache_dir=save_path, **kwarg)

        # Fetch config
        self.config, _unused_kwargs = AutoConfig.from_pretrained(
            source, cache_dir=save_path, return_unused_kwargs=True,
        )

        self.config = self.override_config(self.config)

        self.for_pretraining = for_pretraining
        if for_pretraining:
            model = AutoModelForPreTraining.from_config(self.config)
        else:
            model = AutoModel.from_config(self.config)

        # Download model
        self._from_pretrained(
            source,
            config=self.config,
            model=model,
            save_path=save_path,
            cache_dir=cache_dir,
        )

        # Prepare for training, fine-tuning, or inference
        self.freeze = freeze
        if self.freeze:
            logger.warning(
                "speechbrain.lobes.models.HuggingFaceTransformer is frozen."
            )
            self.freeze_model(self.model)
        else:
            self.model.gradient_checkpointing_disable()  # Required by DDP
            self.model.train()

    def _from_pretrained(
        self, source, config, model, save_path, cache_dir,
    ):
        """This function manages the source checking and loading of the params.
        # 1. Is the model from HF or a local path
        # 2. Is the model pretrained with HF or SpeechBrain
        # 3. Download (if appropriate) and load with respect to 1. and 2.
        """
        is_sb, ckpt_file, is_local = self._check_model_source(source, save_path)
        if is_sb:
            config = config.from_pretrained(source, cache_dir=save_path)
            self.model = model(config)
            self.model.gradient_checkpointing_disable()  # Required by DDP
            # fetch the checkpoint file
            ckpt_full_path = fetch(
                filename=ckpt_file,
                source=source,
                savedir=save_path,
                cache_dir=cache_dir,
            )
            # We transfer the parameters from the checkpoint.
            self._load_sb_pretrained_parameters(
                ckpt_full_path,
                modify_state_dict_partial_fn=self._modify_state_dict_partial_fn,
            )
        else:
            if self.for_pretraining:
                # For now, we don't need to load pretrained model for pretraining
                # To be modified in the future to support more complicated scenerios
                # For example fine-tuning in the SSL manner
                self.model = model
            else:
                self.model = model.from_pretrained(
                    source, config=config, cache_dir=save_path
                )

    def _check_model_source(self, path, save_path):
        """Checks if the pretrained model has been trained with SpeechBrain and
        is hosted locally or on a HuggingFace hub.
        Called as static function in HuggingFaceTransformer._from_pretrained.
        Arguments
        ---------
        path : str
            Used as "source"; local path or HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
        save_path : str
            norm_output (dir) of the downloaded model.
        Returns
        -------
        is_sb : bool
            Whether/not the model is deserializable w/ SpeechBrain or not (then, model conversion is needed).
        checkpoint_filename : str
            as of HuggingFace documentation: file name relative to the repo root (guaranteed to be here).
        """
        checkpoint_filename = ""
        source = pathlib.Path(path)
        is_local = True

        # If path is a huggingface hub.
        if not source.exists():
            is_local = False

        # Check if source is downloaded already
        sink = pathlib.Path(
            save_path + "/models--" + path.replace("/", "--") + "/snapshots"
        )
        if sink.exists():
            sink = (
                sink / os.listdir(str(sink))[0]
            )  # there's a hash-id subfolder
            if any(
                File.endswith(".bin") or File.endswith(".ckpt")
                for File in os.listdir(str(sink))
            ):
                is_local = True
                local_path = str(sink)
            else:
                local_path = path
        else:
            local_path = path

        if is_local:
            # Test for HuggingFace model
            if any(File.endswith(".bin") for File in os.listdir(local_path)):
                is_sb = False
                return is_sb, checkpoint_filename, is_local

            # Test for SpeechBrain model and get the filename.
            for File in os.listdir(local_path):
                if File.endswith(".ckpt"):
                    checkpoint_filename = os.path.join(path, File)
                    is_sb = True
                    return is_sb, checkpoint_filename, is_local
        else:
            files = model_info(
                path
            ).siblings  # get the list of files of the Hub

            # Test if it's an HuggingFace model or a SB one
            for File in files:
                if File.rfilename.endswith(".ckpt"):
                    checkpoint_filename = File.rfilename
                    is_sb = True
                    return is_sb, checkpoint_filename, is_local

            for File in files:
                if File.rfilename.endswith(".bin"):
                    checkpoint_filename = File.rfilename
                    is_sb = False
                    return is_sb, checkpoint_filename, is_local

        err_msg = f"{path} does not contain a .bin or .ckpt checkpoint !"
        raise FileNotFoundError(err_msg)

    def _modify_state_dict(self):
        """"A custom loading ensures SpeechBrain compatibility for pretrain and model
        For example, wav2vec2 model pretrained with SB (HuggingFaceWav2Vec2Pretrain) has slightly different keys from HuggingFaceWav2Vec2.
        This method handle the compatibility between the two.
        Users should modify this function according to their own tasks."""
        raise NotImplementedError

    def _load_sb_pretrained_parameters(
        self, path, modify_state_dict_partial_fn
    ):
        """Loads the parameter of a HuggingFace model pretrained with SpeechBrain
        and the HuggingFace Pretrain Object. It is necessary to perform a custom
        loading because HuggingFace adds a level to the checkpoint when storing
        the model breaking the compatibility Pretrain and model de/serialization.

        For example, a typical HuggingFaceWav2Vec2 checkpoint for a given parameter
        would be: model.conv.weight.data while for HuggingFaceWav2Vec2Pretrain it
        is: model.wav2vec2.weight.data (wav2vec2 must be removed before loading).
        """
        if modify_state_dict_partial_fn is not None:
            modified_state_dict = modify_state_dict_partial_fn(path)
        else:
            modified_state_dict = torch.load(path, map_location="cpu")

        incompatible_keys = self.model.load_state_dict(
            modified_state_dict, strict=False
        )
        for missing_key in incompatible_keys.missing_keys:
            logger.warning(
                f"During parameter transfer to {self.model} loading from "
                + f"{path}, the transferred parameters did not have "
                + f"parameters for the key: {missing_key}"
            )
        for unexpected_key in incompatible_keys.unexpected_keys:
            logger.warning(
                f"The param with the key: {unexpected_key} is discarded as it "
                + "is useless for finetuning this HuggingFaceTransformer."
            )

    def forward(self, **kwargs):
        """Users should modify this function according to their own tasks."""
        raise NotImplementedError

    def forward_encoder(self, **kwargs):
        """Users should modify this function according to their own tasks."""
        raise NotImplementedError

    def forward_decoder(self, **kwargs):
        """Users should modify this function according to their own tasks."""
        raise NotImplementedError

    def decode(self, **kwargs):
        """Might be useful for models like mbart, which can exploit SB's beamsearch for inference
        Users should modify this function according to their own tasks."""
        raise NotImplementedError

    def encode(self, **kwargs):
        """Customed encoding for inference
        Users should modify this function according to their own tasks."""
        raise NotImplementedError

    def freeze_model(self, model):
        """
        Freezes parameters of a model.
        This should be overrided too, depending on users' needs, for example, adapters use.

        Arguments
        ---------
        model : from AutoModel.from_config
            Valid HuggingFace transformers model object.
        """
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def override_config(self, config):
        """Users should modify this function according to their own tasks."""
        return config

    def load_feature_extractor(self, source, cache_dir, **kwarg):
        """Load model's feature_extractor from the hub"""
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            source, cache_dir=cache_dir, **kwarg
        )

    def load_tokenizer(self, source, **kwarg):
        """Load model's tokenizer from the hub"""
        self.tokenizer = AutoTokenizer.from_pretrained(source, **kwarg)


def make_padding_masks(src, wav_len=None, pad_idx=0):
    """This method generates the padding masks.
    Arguments
    ---------
    src : tensor
        The sequence to the encoder (required).
    wav_len : tensor
        The relative length of the wav given in SpeechBrain format.
    pad_idx : int
        The index for <pad> token (default=0).
    """
    src_key_padding_mask = None
    if wav_len is not None:
        abs_len = torch.round(wav_len * src.shape[1])
        src_key_padding_mask = length_to_mask(abs_len).bool()

    return src_key_padding_mask


def make_masks(src, wav_len=None, pad_idx=0):
    """This method generates the padding masks.
    Arguments
    ---------
    src : tensor
        The sequence to the encoder (required).
    wav_len : tensor
        The relative length of the wav given in SpeechBrain format.
    pad_idx : int
        The index for <pad> token (default=0).
    """
    src_key_padding_mask = None
    if wav_len is not None:
        abs_len = torch.round(wav_len * src.shape[1])
        src_key_padding_mask = length_to_mask(abs_len).bool()

    return src_key_padding_mask
