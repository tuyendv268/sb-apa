import json
import logging
import os
import random
import speechbrain as sb
import torch

from collections import defaultdict
from speechbrain.dataio.dataio import read_audio
from typing import Dict, List, Tuple


random.seed(10)
logger = logging.getLogger(__name__)
SAMPLERATE = 16000
PHONE_LIST = [
    'V', 'AE', 'Y', 'EH', 'DH', 'AW', 'P', 
    'JH', 'M', 'UW', 'B', 'R', 'TH', 'G', 
    'CH', 'UH', 'EY', 'D', 'L', 'AO', 'Z', 
    'W', 'IH', 'ER', 'AA', 'AY', 'HH', 'S', 
    'F', 'N', 'ZH', 'K', 'OW', 'NG', 'OY', 
    'AH', 'SH', 'T', 'IY', 'DX', 'AX'
]

# Phone-level scores
MIN_PHONE_ACCURACY_SCORE = 0.0
MAX_PHONE_ACCURACY_SCORE = 2.0
# Word-level scores
MIN_WORD_ACCURACY_SCORE = 0.0
MAX_WORD_ACCURACY_SCORE = 10.0
MIN_WORD_STRESS_SCORE = 5.0
MAX_WORD_STRESS_SCORE = 10.0
MIN_WORD_TOTAL_SCORE = 0.0
MAX_WORD_TOTAL_SCORE = 10.0
# Sentence-level scores
MIN_SENTENCE_ACCURACY_SCORE = 0.0
MAX_SENTENCE_ACCURACY_SCORE = 10.0
MIN_SENTENCE_COMPLETENESS_SCORE = 0.0
MAX_SENTENCE_COMPLETENESS_SCORE = 10.0
MIN_SENTENCE_FLUENCY_SCORE = 0.0
MAX_SENTENCE_FLUENCY_SCORE = 10.0
MIN_SENTENCE_PROSODIC_SCORE = 0.0
MAX_SENTENCE_PROSODIC_SCORE = 10.0
MIN_SENTENCE_TOTAL_SCORE = 0.0
MAX_SENTENCE_TOTAL_SCORE = 10.0

def create_score_tensor_from_string(scores_string, min_score_limit, max_score_limit):
    return torch.FloatTensor(
        [
            (float(s) - min_score_limit) / max_score_limit 
            for s in scores_string.strip().split()
            ]
        )

def dataio_prep(hparams, label_encoder=None):
    dataio_prep_funcs = {
        "apr": apr_dataio_prep,
        "scoring": scoring_dataio_prep
        }
    
    dataio_prep_func = dataio_prep_funcs[hparams["dataset_name"]]
    if label_encoder is None:
        return dataio_prep_func(hparams)
    else:
        return dataio_prep_func(hparams, label_encoder)

def apr_dataio_prep(hparams, label_encoder=None):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    if label_encoder is None:
        label_encoder = sb.dataio.encoder.CTCTextEncoder()
        special_labels = {
            "bos_label": hparams["bos_index"],
            "eos_label": hparams["eos_index"],
            "blank_label": hparams["blank_index"],
        }
    else:
        special_labels = {}

    data_folder = hparams["data_folder"]
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides(
        "phn_list",
        "phn_encoded_list",
        "phn_encoded",
        "phn_encoded_eos",
        "phn_encoded_bos",
    )
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        yield phn_encoded_list
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

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Fit encoder:
    # Load or compute the label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="phn_list",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 5. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "phn_encoded", "phn_encoded_eos", "phn_encoded_bos"],
    )

    return train_data, valid_data, test_data, label_encoder

def scoring_dataio_prep(hparams, label_encoder=None):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    data_folder = hparams["data_folder"]
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

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

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    @sb.utils.data_pipeline.takes(
        "phn_canonical", "wrd_id", "rel_pos"
        )
    @sb.utils.data_pipeline.provides(
        "phn_canonical_list",
        "phn_canonical_encoded",
        "phn_canonical_encoded_eos",
        "phn_canonical_encoded_bos",
        "wrd_id_list",
        "rel_pos_list"
    )
    def text_canonical_pipeline(phn_canonical, wrd_id, rel_pos):
        phn_list = phn_canonical.strip().split()
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

    sb.dataio.dataset.add_dynamic_item(datasets, text_canonical_pipeline)

    # 3. Fit encoder (if needed) and save:
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    if label_encoder is None:
        special_labels = {
            "bos_label": hparams["bos_index"],
            "eos_label": hparams["eos_index"],
            "blank_label": hparams["blank_index"],
        }
        label_encoder = sb.dataio.encoder.CTCTextEncoder()
        label_encoder.load_or_create(
            path=lab_enc_file,
            from_didatasets=[train_data],
            output_key="phn_list",
            special_labels=special_labels,
            sequence_input=True,
        )
    else:
        label_encoder.load_or_create(
            path=lab_enc_file,
            output_key="phn_list",
        )

    # 4. Define phone scores pipeline:
    @sb.utils.data_pipeline.takes(
        "phn_score", "wrd_score", "utt_score",
    )
    @sb.utils.data_pipeline.provides(
        "phn_score_list", "wrd_score_list", "utt_score_list",
    )
    def scores_pipeline(
            phn_score, wrd_score, utt_score,
    ):
        # The returned sequence has the same length as phn_canonical_encoded (i.e.: one-less element than
        # phn_canonical_encoded_bos and phn_canonical_encoded_eos.
        yield create_score_tensor_from_string(phn_score, MIN_PHONE_ACCURACY_SCORE,
                                              MAX_PHONE_ACCURACY_SCORE)

        yield create_score_tensor_from_string(wrd_score, MIN_PHONE_ACCURACY_SCORE,
                                              MAX_PHONE_ACCURACY_SCORE)

        yield create_score_tensor_from_string(utt_score, MIN_PHONE_ACCURACY_SCORE,
                                              MAX_PHONE_ACCURACY_SCORE)

    sb.dataio.dataset.add_dynamic_item(datasets, scores_pipeline)

    # 6. Define alignments pipeline:
    if "network_type" in hparams and hparams["network_type"] == "lstm":
        @sb.utils.data_pipeline.takes("phn_ali", "phn_ali_start", "phn_ali_duration")
        @sb.utils.data_pipeline.provides("phn_ali_list", "phn_ali_encoded", "phn_ali_start_list",
                                         "phn_ali_duration_list")
        def alignments_pipeline(phn_ali, phn_ali_start, phn_ali_duration):
            phn_ali_list = phn_ali.strip().split()
            yield phn_ali_list
            phn_ali_encoded_list = label_encoder.encode_sequence(phn_ali_list)
            phn_ali_encoded = torch.LongTensor(phn_ali_encoded_list)
            yield phn_ali_encoded
            phn_ali_start_list = [float(i) for i in phn_ali_start.strip().split()]
            yield phn_ali_start_list
            phn_ali_duration_list = [float(i) for i in phn_ali_duration.strip().split()]
            yield phn_ali_duration_list

        sb.dataio.dataset.add_dynamic_item(datasets, alignments_pipeline)

    output_keys = ["id",
                   "sig",
                   "phn_canonical_list",
                   "phn_canonical_encoded",
                   "phn_canonical_encoded_bos",
                   "phn_canonical_encoded_eos",
                   "wrd_id_list",
                   "rel_pos_list",
                   "phn_score_list", 
                   "wrd_score_list", 
                   "utt_score_list",
                   ]

    if "model_task" in hparams and hparams["model_task"] == "apr":
        output_keys.extend(["phn_encoded", "phn_encoded_eos", "phn_encoded_bos"])

    if "network_type" in hparams and hparams["network_type"] == "lstm":
        output_keys.extend(["phn_ali_list", "phn_ali_encoded", "phn_ali_start_list", "phn_ali_duration_list"])

    # 7. Set output:
    sb.dataio.dataset.set_output_keys(datasets, output_keys)

    return train_data, valid_data, test_data, label_encoder
