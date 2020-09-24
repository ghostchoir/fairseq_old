import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.logging.meters import safe_round

@register_criterion('abc')
class ABCCriterion(FairseqCriterion):

    def __init__(self, task, loss_weights=None, log_keys=None):
        super().__init__(task)
        
        self.loss_weights = None if loss_weights is None else eval(loss_weights)
        self.log_keys = [] if log_keys is None else eval(log_keys)
        
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--loss-weights', type=str, default=None,
                            help='weights for additional loss terms (not first one)')
        parser.add_argument('--log-keys', type=str, default=None,
                            help='output keys to log')
        
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
        # B x T x C
        result_0, result_1, result_target_0, result_target_1 = model(**sample(['net_input']))
        
        online_pred_0 = result_0["x"]
        online_pred_1 = result_1["x"]
        
        target_proj_0 = result_target_0["x"].detach()
        target_proj_1 = result_target_1["x"].detach()
        
        online_pred_0_norm = F.normalize(online_pred_0, dim=-1)
        online_pred_1_norm = F.normalize(online_pred_1, dim=-1)
        
        target_proj_0_norm = F.normalize(target_proj_0, dim=-1)
        target_proj_1_norm = F.normalize(target_proj_1, dim=-1)
        
        loss_ab = F.mse_loss(online_pred_0_norm, target_proj_0_norm, reduction="sum" if reduce else "none",)
        loss_ba = F.mse_loss(online_pred_1_norm, target_proj_1_norm, reduction="sum" if reduce else "none",)
        
        loss = loss_ab + loss_ba
        
        losses = []
        
        sample_size = target_proj_0_norm.numel()
        losses.append(loss.detach().clone())
        
        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(self.loss_weights), f'{len(extra_losses)}, {len(self.loss_weights)}'
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)
                    
        logging_output = {
            'loss': loss.item() if reduce else loss,
            'ntokens': sample_size,
            'nsentences': sample['id'].numel(),
            'sample_size': sample_size,
        }
                    
        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f'loss_{i}'] = l.item()
                
        return loss, sample_size, logging_output
    
    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('ntokens', ntokens)
        metrics.log_scalar('nsentences', nsentences)

        builtin_keys = {'loss', 'ntokens', 'nsentences', 'sample_size'}

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs) / len(logging_outputs)
                if k.startswith('loss'):
                    metrics.log_scalar(k, val / sample_size / math.log(2), sample_size)
                else:
                    metrics.log_scalar(k, val, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False