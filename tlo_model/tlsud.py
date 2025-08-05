from typing import Dict, Any, List, Optional, Union

import torch
import torch.nn as nn
from nemo.collections.asr.models.asr_model import ASRModel

from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.core.classes.mixins import AccessMixin
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.utils import logging, model_utils
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from omegaconf import OmegaConf

from lightning.pytorch.utilities import CombinedLoader   # 2.x

from torch.nn import MSELoss

__all__ = ['JointSSLCTCKDModel']

class KDAdapter(torch.nn.Module):
    """Adapter for knowledge distillation. This module adapts the teacher's features to match the student's features."""

    def __init__(self, in_channels, out_channels, kernel_size=1):
        """
        Args:
            in_channels: the number of input channels from the teacher model.
            out_channels: the number of output channels for the student model.
            kernel_size: the size of the convolution kernel to use for adaptation.
        """
        super().__init__()
        self.adapter = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.kd_loss_fn = MSELoss(reduction="mean")

    def forward(self, t_feats, s_feats):
        x = t_feats.permute(0, 2, 1)
        x = self.adapter(x)
        x = torch.nn.functional.interpolate(x, size=s_feats.size(1), mode='linear', align_corners=False)
        proj_t = x.permute(0, 2, 1)
        return self.kd_loss_fn(proj_t, s_feats)

class JointSSLCTCKDModel(EncDecCTCModelBPE):
    def __init__(self,
        kd_cfg,
        ssl_cfg,
        ctc_cfg,
        trainer,
        init_alpha_ssl=0.0,
        increase_rate_alpha_ssl=0.0000167,
        kd_teacher_path=None,
        init_alpha_kd=0.0,
        increase_rate_alpha_kd=0.0333,
        **trainer_kwargs):

        ctc_cfg = model_utils.convert_model_config_to_dict_config(ctc_cfg)
        ctc_cfg = model_utils.maybe_update_config_version(ctc_cfg)

        super().__init__(cfg=ctc_cfg, trainer=trainer)

        self.hparams.clear()
        self.save_hyperparameters(
            {
                "kd_cfg":  OmegaConf.to_container(kd_cfg, resolve=True),
                "ctc_cfg":  OmegaConf.to_container(ctc_cfg, resolve=True),
                "ssl_cfg":  OmegaConf.to_container(ssl_cfg, resolve=True),
                "init_alpha_ssl": init_alpha_ssl,
                "increase_rate_alpha_ssl": increase_rate_alpha_ssl,
                "kd_teacher_path": kd_teacher_path,
                "init_alpha_kd": init_alpha_kd,
                "increase_rate_alpha_kd": increase_rate_alpha_kd,
            }
        )

        # SSL HEAD
        self.ssl_head = SpeechEncDecSelfSupervisedModel(ssl_cfg, trainer=trainer)
        self.ssl_head.encoder = self.encoder

        ## KD HEAD
        self.kd_teacher_path = kd_teacher_path
        self.kd_teacher = ASRModel.restore_from(restore_path=self.kd_teacher_path)
        self.kd_teacher.eval()
        for p in self.kd_teacher.parameters():
            p.requires_grad = False

        self.kd_pairs = kd_cfg.get("kd_pairs", None)
        if self.kd_pairs is None:
            raise ValueError("`kd_pairs` cannot be None")
        
        self.teacher_outputs = {}
        self.student_outputs = {}

        # Register forward hooks for specified layers
        t_modules = dict(self.kd_teacher.named_modules())
        s_modules = dict(self.named_modules())
        for pair in self.kd_pairs:
            t_name = pair["teacher"]
            s_name = pair["student"]
            if t_name not in t_modules:
                raise ValueError(f"Teacher layer '{t_name}' not found.")
            if s_name not in s_modules:
                raise ValueError(f"Student layer '{s_name}' not found.")
            t_modules[t_name].register_forward_hook(self._get_hook(self.teacher_outputs, t_name))
            s_modules[s_name].register_forward_hook(self._get_hook(self.student_outputs, s_name))

        self.adapters = torch.nn.ModuleDict()
        for pair in self.kd_pairs:
            self.adapters[pair["student"].replace(".", "_")] = KDAdapter(pair["t_dim"], pair["s_dim"])

    # -------------------------------------------------
    # get_hook
    # -------------------------------------------------
    def _get_hook(self, output_dict, name):
        """Returns a hook that stores the layer's output in output_dict."""

        def hook(module, input, output):
            output_dict[name] = output

        return hook
    
    # -------------------------------------------------
    # DataLoader：Lightning 允許一次回傳多個 loader；
    # -------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        self.setup_training_data(self._cfg.train_ds)
        self.sup_train_dl = self._setup_dataloader_from_config(config = self._cfg.train_ds)

        self.ssl_head.setup_training_data(self.ssl_head.cfg.train_ds)
        self.unsup_train_dl = self.ssl_head._setup_dataloader_from_config(config = self.ssl_head.cfg.train_ds)
        
        self.setup_training_data(self._cfg.kd_train_ds)
        self.kd_train_dl = self._setup_dataloader_from_config(config = self._cfg.kd_train_ds)

    def train_dataloader(self):
        loaders = {
            'sup': self.sup_train_dl,
            'unsup': self.unsup_train_dl,
            'kd':  self.kd_train_dl,
        }
        return CombinedLoader(loaders, mode='min_size')

    
    # -------------------------------------------------
    # -------------- on_train_epoch_start -------------
    # -------------------------------------------------
    def on_train_epoch_start(self):
        if self.trainer.current_epoch > 0:
            self.loss_alpha_ssl = self.trainer.current_epoch * self.increase_rate_alpha_ssl
            self.loss_alpha_kd = self.trainer.current_epoch * self.increase_rate_alpha_kd
        else:
            self.loss_alpha_ssl = self.init_alpha_ssl
            self.loss_alpha_kd = self.init_alpha_kd
        """
        Decoder with CUDA graphs does not release memory, thus we disable it for training epoch.
        EncDecRNNTModel.decoding.decoding is the inference class with CUDA graphs
        """
        WithOptionalCudaGraphs.disable_cuda_graphs_recursive(self, attribute_path="decoding.decoding")
    
    # -------------------------------------------------
    # ----------------- Training step -----------------
    # -------------------------------------------------
    def training_step(self, batch, batch_nb):
        sup_batch  = batch['sup']
        unsup_batch = batch['unsup']
        kd_batch   = batch['kd']

        # sup_batch
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, transcript, transcript_len = sup_batch
        
        if isinstance(sup_batch, DALIOutputs) and sup_batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)

        ctc_loss = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )

        # unsup_batch
        ssl_signal, ssl_signal_len, ssl_targets, ssl_target_lengths = unsup_batch
        if isinstance(unsup_batch, DALIOutputs) and unsup_batch.has_processed_signal:
            spectrograms, spec_masks, encoded, encoded_len = self.ssl_head.forward(
                processed_signal=ssl_signal,
                processed_signal_length=ssl_signal_len,
            )
        else:
            spectrograms, spec_masks, encoded, encoded_len = self.ssl_head.forward(
                input_signal=ssl_signal,
                input_signal_length=ssl_signal_len,
            )

        if self.ssl_head.decoder_losses is not None:
            for dec_loss_name, dec_loss in self.ssl_head.decoder_losses.items():
                self.ssl_head.decoder_losses_active[dec_loss_name] = self.ssl_head.trainer.global_step >= self.ssl_head.start_step[dec_loss_name]
                loss = dec_loss['loss']
                if hasattr(loss, "set_num_updates"):
                    loss.set_num_updates(self.ssl_head.trainer.global_step)
        else:
            if hasattr(self.ssl_head.loss, "set_num_updates"):
                self.ssl_head.loss.set_num_updates(self.ssl_head.trainer.global_step)

        ssl_loss_value, ssl_loss_val_dict = self.ssl_head.decoder_loss_step(
            spectrograms, spec_masks, encoded, encoded_len, ssl_targets, ssl_target_lengths
        )

        for loss_name, ssl_loss_value in ssl_loss_val_dict.items():
            tensorboard_logs['train_' + loss_name] = ssl_loss_value

        if self.ssl_head.feat_pen:
            unsup_loss += self.ssl_head.feat_pen

        # Reset access registry
        self.ssl_head.reset_registry()

        # kd_batch
        kd_signal, kd_signal_len, kd_transcript, kd_transcript_len = kd_batch

        # clear previous hook outputs
        self.teacher_outputs.clear()
        self.student_outputs.clear()

        # process and specaug before forward
        kd_processed_signal, kd_processed_signal_length = self.preprocessor(
            input_signal=kd_signal,
            length=kd_signal_len,
        )

        kd_processed_signal = self.spec_augmentation(input_spec=kd_processed_signal, length=kd_processed_signal_length)

        kd_log_probs, kd_encoded_len, kd_predictions = super().forward(processed_signal=kd_processed_signal, processed_signal_length=kd_processed_signal_length)

        with torch.no_grad():
            self.kd_teacher.forward(processed_signal=kd_processed_signal, processed_signal_length=kd_processed_signal_length)

        kd_loss = 0.0
        for pair in self.kd_pairs:
            t_feat = self.teacher_outputs[pair["teacher"]]
            s_feat = self.student_outputs[pair["student"]]
            kd_loss += self.adapters[pair["student"].replace(".", "_")](t_feat, s_feat)

        # Add auxiliary losses, if registered
        loss_value = ctc_loss + unsup_loss * self.loss_alpha_ssl + kd_loss * (self.loss_alpha_kd / 2)
        loss_value = self.add_auxiliary_losses(loss_value)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        # only computing WER when requested in the logs (same as done for final-layer WER below)
        loss_value, tensorboard_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=((batch_nb + 1) % log_every_n_steps == 0)
        )

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        tensorboard_logs.update(
            {
                'train_loss': ctc_loss,
                'kd_loss': kd_loss,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }
        )

        if (batch_nb + 1) % log_every_n_steps == 0:
            self.wer.update(
                predictions=log_probs,
                targets=transcript,
                targets_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            wer, _, _ = self.wer.compute()
            self.wer.reset()
            tensorboard_logs.update({'training_batch_wer': wer})

        return {'loss': loss_value, 'log': tensorboard_logs}