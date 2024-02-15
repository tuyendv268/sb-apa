import torch
import speechbrain as sb

MAX_SCORE = 2.0

class ScorerWav2vec2(sb.Brain):
    def __init__(
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
        profiler=None,
    ):
        super(ScorerWav2vec2, self).__init__(
            modules, opt_class, hparams, run_opts, checkpointer, profiler)
    
    def infer(self, wavs, wav_lens, rel_pos, phns_canonical_bos, phns_canonical_eos):
        feats = self.hparams.wav2vec2(wavs)
        x = self.modules.enc(feats)

        e_in_canonical = self.modules.emb(phns_canonical_bos)
        h_scoring, _ = self.modules.dec(e_in_canonical, x, wav_lens)

        phone_rep_pred = self.modules.scorer_nn(h_scoring)
        emb_actual = self.modules.emb_scorer(phns_canonical_eos)
        emb_actual = self.modules.scorer_nn(emb_actual)

        utt_acc_score, _, word_acc_score = self.modules.prep_scorer(
            h_scoring[:, :-1].detach().clone(), phns_canonical_eos[:, :-1], rel_pos
        )
        phone_acc_score = torch.nn.functional.cosine_similarity(
            phone_rep_pred, emb_actual, dim=len(phone_rep_pred.shape) - 1)

        word_acc_score = word_acc_score.squeeze(2)

        return utt_acc_score, phone_acc_score, word_acc_score
        
    def rescale_scores(self, scores):
        return MAX_SCORE * scores

    def get_real_length_sequences(self, seq, lens):
        seqs = []
        for i in range(len(lens)):
            seq_len = round((lens[i] * seq.shape[1]).item())
            seqs.append(seq[i, :seq_len].squeeze())
        return seqs

    