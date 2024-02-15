import speechbrain as sb
import torch

def infer_dataio_prep(dataset, label_encoder):
    dataset = sb.dataio.dataset.DynamicItemDataset(dataset)
    dataset = [dataset, ]

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        # sig = sb.dataio.dataio.read_audio(wav)
        sig = torch.Tensor(wav)
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
        "utt", "wrd", "phn", "wrd_id", "rel_pos", "alignment"
        )
    @sb.utils.data_pipeline.provides(
        "utt",
        "wrd",
        "phn",
        "phn_canonical_encoded",
        "phn_canonical_encoded_eos",
        "phn_canonical_encoded_bos",
        "wrd_id_list",
        "rel_pos_list",
        "alignment_list"
    )
    def text_canonical_pipeline(utt, wrd, phn, wrd_id, rel_pos, alignment):
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
        
        
        alignment_list = [ele for ele in alignment]
        yield alignment_list
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
        "alignment_list"
    ]

    sb.dataio.dataset.set_output_keys(dataset, output_keys)
    return dataset[0]
