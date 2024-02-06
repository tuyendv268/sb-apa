import logging
import speechbrain as sb
import torch

from src.logger import ResultLogger
from speechbrain.utils import hpopt as hp

logger = logging.getLogger(__name__)

MAX_SCORE = 2.0

import time
import torch
import logging
from tqdm.contrib import tqdm
from speechbrain.utils.distributed import run_on_main
from speechbrain.core import Stage

class ScorerWav2vec2(sb.Brain):
    def __init__(  # noqa: C901
            self,
            modules=None,
            opt_class=None,
            hparams=None,
            run_opts=None,
            checkpointer=None,
            profiler=None,
    ):
        super().__init__(modules, opt_class, hparams, run_opts, checkpointer, profiler)
        self.result_logger = ResultLogger(self.hparams)

    def compute_forward(self, batch, stage):
        """Given an input batch it computes the phoneme probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        rel_pos, rel_lens = batch.rel_pos_list
        wrd_ids, wrd_id_lens = batch.wrd_id_list
        phns_canonical_bos, _ = batch.phn_canonical_encoded_bos
        phns_canonical_eos, _ = batch.phn_canonical_encoded_eos

        if stage == sb.Stage.TRAIN:
            if self.hparams.use_augmentation is True and hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.wav2vec2(wavs)
        x = self.modules.enc(feats)

        e_in_canonical = self.modules.emb(phns_canonical_bos)
        h_scoring, _ = self.modules.dec(e_in_canonical, x, wav_lens)

        # Computing phone representations for pronounced and canonical phones
        phone_rep_pred = self.modules.scorer_nn(h_scoring)
        emb_actual = self.modules.emb_scorer(phns_canonical_eos)
        emb_actual = self.modules.scorer_nn(emb_actual)

        utt_acc_score, phone_acc_score, word_acc_score = self.modules.prep_scorer(
            h_scoring[:, :-1].detach().clone(), phns_canonical_eos[:, :-1], rel_pos
        )

        # # Computing similarity
        if self.hparams.similarity_calc == "cosine":
            # Cosine similarity
            scores_pred = torch.nn.functional.cosine_similarity(
                phone_rep_pred, emb_actual, dim=len(phone_rep_pred.shape) - 1)
        elif self.hparams.similarity_calc == "euclidean":
            # Normalized Euclidean similarity (NES)
            scores_pred = 1.0 - 0.5 * (phone_rep_pred - emb_actual).var(dim=2) / \
                          (phone_rep_pred.var(dim=2) + emb_actual.var(dim=2))
        else:
            scores_pred = self.modules.scorer_similarity_nn(torch.concat([phone_rep_pred, emb_actual], dim=2)) \
                .view(self.hparams.batch_size, emb_actual.shape[1])

        phone_acc_score = phone_acc_score.squeeze(2)
        word_acc_score = word_acc_score.squeeze(2)

        return utt_acc_score, scores_pred, word_acc_score
    
    def infer(self, wavs, wav_lens, rel_pos, phns_canonical_bos, phns_canonical_eos):
        feats = self.hparams.wav2vec2(wavs)
        x = self.modules.enc(feats)

        e_in_canonical = self.modules.emb(phns_canonical_bos)
        h_scoring, _ = self.modules.dec(e_in_canonical, x, wav_lens)

        # Computing phone representations for pronounced and canonical phones
        print(h_scoring.shape)
        phone_rep_pred = self.modules.scorer_nn(h_scoring)
        emb_actual = self.modules.emb_scorer(phns_canonical_eos)
        emb_actual = self.modules.scorer_nn(emb_actual)

        utt_acc_score, phone_acc_score, word_acc_score = self.modules.prep_scorer(
            h_scoring[:, :-1].detach().clone(), phns_canonical_eos[:, :-1], rel_pos
        )

        # # Computing similarity
        if self.hparams.similarity_calc == "cosine":
            # Cosine similarity
            scores_pred = torch.nn.functional.cosine_similarity(
                phone_rep_pred, emb_actual, dim=len(phone_rep_pred.shape) - 1)
        elif self.hparams.similarity_calc == "euclidean":
            # Normalized Euclidean similarity (NES)
            scores_pred = 1.0 - 0.5 * (phone_rep_pred - emb_actual).var(dim=2) / \
                          (phone_rep_pred.var(dim=2) + emb_actual.var(dim=2))
        else:
            scores_pred = self.modules.scorer_similarity_nn(torch.concat([phone_rep_pred, emb_actual], dim=2)) \
                .view(self.hparams.batch_size, emb_actual.shape[1])

        phone_acc_score = phone_acc_score.squeeze(2)
        word_acc_score = word_acc_score.squeeze(2)

        return utt_acc_score, scores_pred, word_acc_score
        
    def rescale_scores(self, scores):
        """Rescales scores from range [0, 1] to range [0, 2]."""
        return MAX_SCORE * scores

    def round_scores(self, scores):
        """Rescales scores to the nearest integer."""
        return torch.round(torch.minimum(torch.maximum(scores, torch.full_like(scores, 0)), torch.full_like(scores, 2)))

    def get_real_length_sequences(self, seq, lens):
        """Return the sequences with their real length."""
        seqs = []
        for i in range(len(lens)):
            seq_len = round((lens[i] * seq.shape[1]).item())
            seqs.append(seq[i, :seq_len].squeeze())
        return seqs

    def compute_objectives(self, predictions, batch, stage):
        pred_utt_acc_score, pred_phn_acc_score, pred_wrd_acc_score = predictions
        ids = batch.id
        wrd_ids, wrd_id_lens = batch.wrd_id_list
        phn, phn_lens = batch.phn_canonical_encoded

        label_phn_acc_score, _ = batch.phn_score_list
        label_wrd_acc_score, _ = batch.wrd_score_list
        label_utt_acc_score, _ = batch.utt_score_list

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)

        label_phn_acc_score = label_phn_acc_score.unsqueeze(2)
        pred_phn_acc_score = pred_phn_acc_score[:, :-1].unsqueeze(2)

        phn_loss = self.hparams.score_cost(label_phn_acc_score, pred_phn_acc_score, phn_lens)
        wrd_loss = self.hparams.score_cost(label_wrd_acc_score, pred_wrd_acc_score, phn_lens)
        utt_loss = self.hparams.score_cost(label_utt_acc_score, pred_utt_acc_score)

        loss = phn_loss + wrd_loss + utt_loss
        
        train_log = {
            "phn_loss": phn_loss.cpu().item(),
            "wrd_loss": wrd_loss.cpu().item(),
            "utt_loss": utt_loss.cpu().item(),
        }

        # Rescale and round scores for final evaluation.
        pred_phn_acc_score = self.rescale_scores(pred_phn_acc_score)
        label_phn_acc_score = self.rescale_scores(label_phn_acc_score)
        
        pred_wrd_acc_score = self.rescale_scores(pred_wrd_acc_score)
        label_wrd_acc_score = self.rescale_scores(label_wrd_acc_score)

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            self.distance_scoring_metrics.append(ids, pred_phn_acc_score, label_phn_acc_score, phn, phn_lens,
                                                 self.label_encoder.decode_ndim)

        # Save predictions to compute MSE and PCC in the end of the stage.
        real_length_phn_prediction_seq = self.get_real_length_sequences(pred_phn_acc_score, phn_lens)
        real_length_phn_scores = self.get_real_length_sequences(label_phn_acc_score, phn_lens)
        
        real_length_phn_prediction_seq = torch.concat(real_length_phn_prediction_seq, 0)
        real_length_phn_scores = torch.concat(real_length_phn_scores, 0)
        
        real_length_wrd_prediction_seq = self.get_real_length_sequences(pred_wrd_acc_score, wrd_id_lens)
        real_length_wrd_scores = self.get_real_length_sequences(label_wrd_acc_score, wrd_id_lens)
        real_length_wrd_ids = self.get_real_length_sequences(wrd_ids, wrd_id_lens)
        
        for index in range(len(real_length_wrd_ids)):
            _real_length_wrd_ids = real_length_wrd_ids[index] - 1
            _real_length_wrd_scores = real_length_wrd_scores[index]
            _real_length_wrd_prediction_seq = real_length_wrd_prediction_seq[index]
            
            indices = torch.nn.functional.one_hot(
                _real_length_wrd_ids, num_classes=int(_real_length_wrd_ids.max().item())+1).cuda()
            indices = indices / indices.sum(0, keepdim=True)
    
            _real_length_wrd_scores = torch.matmul(indices.transpose(0, 1), _real_length_wrd_scores)
            _real_length_wrd_prediction_seq = torch.matmul(indices.transpose(0, 1), _real_length_wrd_prediction_seq)
            
            real_length_wrd_scores[index] = _real_length_wrd_scores
            real_length_wrd_prediction_seq[index] = _real_length_wrd_prediction_seq
                    
        real_length_wrd_prediction_seq = torch.concat(real_length_wrd_prediction_seq, 0)
        real_length_wrd_scores = torch.concat(real_length_wrd_scores, 0)
        
        self.stage_wrd_preds.append(real_length_wrd_prediction_seq.detach().cpu())
        self.stage_wrd_scores.append(real_length_wrd_scores.detach().cpu())

        self.stage_utt_preds.append(pred_utt_acc_score.detach().cpu())
        self.stage_utt_scores.append(label_utt_acc_score.detach().cpu())
        
        self.stage_phn_preds.append(real_length_phn_prediction_seq.detach().cpu())
        self.stage_phn_scores.append(real_length_phn_scores.detach().cpu())
        
        self.stage_phn_preds_rounded.append(self.round_scores(real_length_phn_prediction_seq).detach().cpu())
        self.stage_phn_scores_rounded.append(self.round_scores(real_length_phn_scores).detach().cpu())

        return loss, train_log

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        # Managing automatic mixed precision
        if self.auto_mix_prec:

            self.wav2vec_optimizer.zero_grad()
            self.asr_optimizer.zero_grad()
            self.scorer_optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss, train_log = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            if self.optimizer_step > self.hparams.warmup_steps_wav2vec:
                self.scaler.unscale_(self.wav2vec_optimizer)
            if self.optimizer_step > self.hparams.warmup_steps_asr:
                self.scaler.unscale_(self.asr_optimizer)
            self.scaler.unscale_(self.scorer_optimizer)

            if self.check_gradients(loss):
                if self.optimizer_step > self.hparams.warmup_steps_wav2vec and not self.hparams.wav2vec2.freeze:
                    self.scaler.step(self.wav2vec_optimizer)
                if self.optimizer_step > self.hparams.warmup_steps_asr:
                    self.scaler.step(self.asr_optimizer)
                self.scaler.step(self.scorer_optimizer)

            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            loss, train_log = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            loss.backward()

            if self.check_gradients(loss):
                if self.optimizer_step > self.hparams.warmup_steps_wav2vec:
                    self.wav2vec_optimizer.step()
                if self.optimizer_step > self.hparams.warmup_steps_asr:
                    self.asr_optimizer.step()
                self.scorer_optimizer.step()

            self.wav2vec_optimizer.zero_grad()
            self.asr_optimizer.zero_grad()
            self.scorer_optimizer.zero_grad()

        self.optimizer_step += 1
        return loss.detach().cpu(), train_log

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss, train_log = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called when a stage (either training, validation, test) starts."""
        self.score_metrics_mse = self.hparams.score_stats_mse()
        self.stage_phn_preds = []
        self.stage_phn_preds_rounded = []
        self.stage_phn_scores = []
        self.stage_phn_scores_rounded = []
        
        self.stage_utt_preds = []
        self.stage_utt_scores = []
        
        self.stage_wrd_preds = []
        self.stage_wrd_scores = []

        if stage != sb.Stage.TRAIN:
            self.distance_scoring_metrics = self.hparams.scoring_stats_dist()
            
    def _fit_train(self, train_set, epoch, enable):
        # Training stage
        self.on_stage_start(Stage.TRAIN, epoch)
        self.modules.train()
        self.zero_grad()

        # Reset nonfinite count to 0 each epoch
        self.nonfinite_count = 0

        if self.train_sampler is not None and hasattr(
            self.train_sampler, "set_epoch"
        ):
            self.train_sampler.set_epoch(epoch)

        # Time since last intra-epoch checkpoint
        last_ckpt_time = time.time()
        with tqdm(
            train_set,
            initial=self.step,
            dynamic_ncols=True,
            disable=not enable,
            colour=self.tqdm_barcolor["train"],
        ) as t:
            for batch in t:
                if self._optimizer_step_limit_exceeded:
                    logger.info("Train iteration limit exceeded")
                    break
                self.step += 1
                loss, train_log = self.fit_batch(batch)
                self.avg_train_loss = self.update_average(
                    loss, self.avg_train_loss
                )
                t.set_postfix(
                    train_loss=self.avg_train_loss, 
                    phn_loss=round(train_log["phn_loss"],4),
                    wrd_loss=round(train_log["wrd_loss"],4), 
                    utt_loss=round(train_log["utt_loss"],4)
                )

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

                if (
                    self.checkpointer is not None
                    and self.ckpt_interval_minutes > 0
                    and time.time() - last_ckpt_time
                    >= self.ckpt_interval_minutes * 60.0
                ):
                    # This should not use run_on_main, because that
                    # includes a DDP barrier. That eventually leads to a
                    # crash when the processes'
                    # time.time() - last_ckpt_time differ and some
                    # processes enter this block while others don't,
                    # missing the barrier.
                    if sb.utils.distributed.if_main_process():
                        self._save_intra_epoch_ckpt()
                    last_ckpt_time = time.time()

        # Run train "on_stage_end" on all processes
        self.zero_grad(set_to_none=True)  # flush gradients
        self.on_stage_end(Stage.TRAIN, self.avg_train_loss, epoch)
        self.avg_train_loss = 0.0
        self.step = 0

    def _fit_valid(self, valid_set, epoch, enable):
        # Validation stage
        if valid_set is not None:
            self.on_stage_start(Stage.VALID, epoch)
            self.modules.eval()
            avg_valid_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(
                    valid_set,
                    dynamic_ncols=True,
                    disable=not enable,
                    colour=self.tqdm_barcolor["valid"],
                ):
                    self.step += 1
                    loss = self.evaluate_batch(batch, stage=Stage.VALID)
                    avg_valid_loss = self.update_average(loss, avg_valid_loss)

                    # Profile only if desired (steps allow the profiler to know when all is warmed up)
                    if self.profiler is not None:
                        if self.profiler.record_steps:
                            self.profiler.step()

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                # Only run validation "on_stage_end" on main process
                self.step = 0
                run_on_main(
                    self.on_stage_end,
                    args=[Stage.VALID, avg_valid_loss, epoch],
                )

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        stage_preds = torch.concat(self.stage_phn_preds, 0)
        stage_scores = torch.concat(self.stage_phn_scores, 0)
        
        stage_preds_rounded = torch.concat(self.stage_phn_preds_rounded, 0)
        stage_scores_rounded = torch.concat(self.stage_phn_scores_rounded, 0)
        
        stage_utt_preds = torch.concat(self.stage_utt_preds, 0).squeeze(-1)
        stage_utt_scores = torch.concat(self.stage_utt_scores, 0).squeeze(-1)
        
        stage_wrd_preds = torch.concat(self.stage_wrd_preds, 0)
        stage_wrd_scores = torch.concat(self.stage_wrd_scores, 0)
        
        stage_wrd_pcc = torch.corrcoef(torch.stack([stage_wrd_preds, stage_wrd_scores]))[0, 1].item()
        stage_wrd_mse = torch.nn.functional.mse_loss(stage_wrd_preds, stage_wrd_scores).item()
                
        stage_utt_pcc = torch.corrcoef(torch.stack([stage_utt_preds, stage_utt_scores]))[0, 1].item()
        stage_utt_mse = torch.nn.functional.mse_loss(stage_utt_preds, stage_utt_scores).item()
        
        stage_pcc = torch.corrcoef(torch.stack([stage_preds, stage_scores]))[0, 1].item()
        stage_mse = torch.nn.functional.mse_loss(stage_preds, stage_scores).item()
        stage_pcc_rounded = torch.corrcoef(torch.stack([stage_preds_rounded, stage_scores_rounded]))[0, 1].item()
        stage_mse_rounded = torch.nn.functional.mse_loss(stage_preds_rounded, stage_scores_rounded).item()

        results_to_log = dict()

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_pcc = stage_pcc
            self.train_mse = stage_mse
        else:
            stats = {"loss": stage_loss, "error": stage_mse}

        if stage == sb.Stage.VALID:
            scoring_error = stage_mse
            old_lr_asr, new_lr_asr = self.hparams.lr_annealing_asr(scoring_error)
            old_lr_scorer, new_lr_scorer = self.hparams.lr_annealing_scorer(scoring_error)
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(scoring_error)
            sb.nnet.schedulers.update_learning_rate(self.asr_optimizer, new_lr_asr)
            sb.nnet.schedulers.update_learning_rate(self.scorer_optimizer, new_lr_scorer)
            sb.nnet.schedulers.update_learning_rate(self.wav2vec_optimizer, new_lr_wav2vec)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr_asr": old_lr_asr, "lr_scorer": old_lr_scorer,
                            "lr_wav2vec": old_lr_wav2vec},
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": stage_loss,
                    "MSE (phone)": stage_mse,
                    "MSE (word)": stage_wrd_mse,
                    "MSE (utterance)": stage_utt_mse,
                    "PCC (phone)": stage_pcc,
                    "PCC (word)": stage_wrd_pcc,
                    "PCC (utterance)": stage_utt_pcc,
                },
            )
            if self.hparams.ckpt_enable:
                self.checkpointer.save_and_keep_only(
                    meta={"scoring_error": scoring_error}, min_keys=["scoring_error"]
                )

            results_to_log["train_loss"] = self.train_loss
            results_to_log["valid_loss"] = stage_loss
            results_to_log["valid_pcc"] = stage_pcc
            results_to_log["valid_mse"] = stage_mse
            results_to_log["valid_pcc_rounded"] = stage_pcc_rounded
            results_to_log["valid_mse_rounded"] = stage_mse_rounded

            print("Reporting the following stats to hpopt", stats)
            if hasattr(self.hparams, "optimizing_hps") and self.hparams.optimizing_hps == True:
                hp.report_result(stats)

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={
                    "loss": stage_loss,
                    "MSE (phone)": stage_mse,
                    "MSE (word)": stage_wrd_mse,
                    "MSE (utterance)": stage_utt_mse,
                    "PCC (phone)": stage_pcc,
                    "PCC (word)": stage_wrd_pcc,
                    "PCC (utterance)": stage_utt_pcc,
                    },
            )
            with open(self.hparams.scoring_dist_file, "w") as w:
                w.write("Score loss stats:\n")
                self.distance_scoring_metrics.write_stats(w)
                logger.info(f"Scoring stats written to file {self.hparams.scoring_dist_file}")

            results_to_log["test_loss"] = stage_loss
            results_to_log["test_pcc"] = stage_pcc
            results_to_log["test_mse"] = stage_mse
            results_to_log["test_pcc_rounded"] = stage_pcc_rounded
            results_to_log["test_mse_rounded"] = stage_mse_rounded

        self.result_logger.log_results(stage, results_to_log)

    def init_optimizers(self):
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.asr_optimizer = self.hparams.asr_opt_class(
            self.hparams.model.parameters()
        )
        self.scorer_optimizer = self.hparams.scorer_opt_class(
            self.hparams.model_scorer.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("wav2vec_opt", self.wav2vec_optimizer)
            self.checkpointer.add_recoverable("asr_opt", self.asr_optimizer)
            self.checkpointer.add_recoverable("scorer_opt", self.scorer_optimizer)
