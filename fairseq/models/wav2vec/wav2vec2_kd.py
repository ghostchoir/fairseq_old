# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils, tasks, utils

from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)

from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer


def add_w2v_args(parser):
    parser.add_argument(
        "--extractor-mode",
        choices=["default", "layer_norm"],
        help="mode for feature extractor. default has a single group norm with d groups in the first conv block, whereas layer_norm has layer norms in every block (meant to use with --normalize)",
    )

    parser.add_argument(
        "--encoder-layers",
        type=int,
        metavar="L",
        help="num encoder layers in the transformer",
    )
    parser.add_argument(
        "--encoder-embed-dim",
        type=int,
        metavar="H",
        help="encoder embedding dimension",
    )
    parser.add_argument(
        "--encoder-ffn-embed-dim",
        type=int,
        metavar="F",
        help="encoder embedding dimension for FFN",
    )
    parser.add_argument(
        "--encoder-attention-heads",
        type=int,
        metavar="A",
        help="num encoder attention heads",
    )
    parser.add_argument(
        "--activation-fn",
        choices=utils.get_available_activation_fns(),
        help="activation function to use",
    )

    parser.add_argument(
        "--final-dim",
        type=int,
        metavar="D",
        help="project final representations and targets to this many dimensions",
    )

    parser.add_argument(
        "--layer-norm-first",
        action="store_true",
        help="apply layernorm first in the transformer",
    )

    parser.add_argument(
        "--encoder-layerdrop",
        type=float,
        help="probability of dropping a tarnsformer layer",
    )

    parser.add_argument(
        "--conv-feature-layers",
        type=str,
        metavar="EXPR",
        help="convolutional feature extraction layers [(dim, kernel_size, stride), ...]",
    )

    parser.add_argument(
        "--logit-temp", type=float, help="temperature to divide logits by"
    )

    parser.add_argument(
        "--quantize-targets", action="store_true", help="use quantized targets"
    )

    parser.add_argument(
        "--quantize-input", action="store_true", help="use quantized inputs"
    )

    parser.add_argument(
        "--same-quantizer",
        action="store_true",
        help="use same quantizer for inputs and targets",
    )

    parser.add_argument(
        "--latent-vars",
        type=int,
        metavar="N",
        help="number of latent variables V in each group of the codebook",
    )

    parser.add_argument(
        "--latent-groups",
        type=int,
        metavar="N",
        help="number of groups G of latent variables in the codebook",
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        metavar="N",
        help="if set, uses this dimensionality for latent variables. otherwise uses final_dim / latent_groups",
    )

    parser.add_argument(
        "--mask-min-space",
        type=int,
        help="min space between spans (if no overlap is enabled)",
    )

    parser.add_argument(
        "--mask-channel-min-space",
        type=int,
        help="min space between spans (if no overlap is enabled)",
    )

    parser.add_argument(
        "--dropout-features",
        type=float,
        metavar="D",
        help="dropout to apply to the features (after feat extr)",
    )

    parser.add_argument(
        "--num-negatives", type=int, metavar="N", help="number of negative examples"
    )

    parser.add_argument(
        "--negatives-from-everywhere",
        action="store_true",
        help="sample negatives from everywhere, not just masked states",
    )

    parser.add_argument(
        "--cross-sample-negatives",
        type=int,
        metavar="N",
        help="num of cross sampled negatives",
    )

    parser.add_argument(
        "--codebook-negatives",
        type=int,
        metavar="N",
        help="num of codebook sampled negatives",
    )

    parser.add_argument(
        "--conv-pos",
        type=int,
        metavar="N",
        help="number of filters for convolutional positional embeddings",
    )

    parser.add_argument(
        "--conv-pos-groups",
        type=int,
        metavar="N",
        help="number of groups for convolutional positional embedding",
    )

    parser.add_argument(
        "--latent-temp",
        type=str,
        metavar="D",
        help="temperature for latent variable sampling. can be tuple of 3 values (start, end, decay)",
    )

    parser.add_argument(
        "--target-glu", action="store_true", help="adds projection + glu to targets"
    )

    parser.add_argument(
        "--conv-bias", action="store_true", help="include bias in conv encoder"
    )


def add_common_args(parser):
    parser.add_argument("--w2v-path", help="path to wav2vec 2.0 model")
    parser.add_argument(
        "--no-pretrained-weights",
        action="store_true",
        help="if true, does not load pretrained weights",
    )
    parser.add_argument(
        "--dropout-input",
        type=float,
        metavar="D",
        help="dropout to apply to the input (after feat extr)",
    )
    parser.add_argument(
        "--final-dropout",
        type=float,
        metavar="D",
        help="dropout after transformer and before final projection",
    )
    parser.add_argument(
        "--apply-mask", action="store_true", help="apply masking during fine-tuning"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        metavar="D",
        help="dropout probability inside wav2vec 2.0 model",
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        metavar="D",
        help="dropout probability for attention weights inside wav2vec 2.0 model",
    )
    parser.add_argument(
        "--activation-dropout",
        "--relu-dropout",
        type=float,
        metavar="D",
        help="dropout probability after activation in FFN inside wav2vec 2.0 model",
    )

    parser.add_argument(
        "--mask-length", type=int, help="repeat the mask indices multiple times"
    )

    parser.add_argument(
        "--mask-prob", type=float, help="probability of replacing a token with mask"
    )

    parser.add_argument(
        "--mask-selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        help="how to choose masks",
    )

    parser.add_argument(
        "--mask-other",
        type=float,
        help="stdev of the mask length in case of 'normal' selection strategy",
    )

    parser.add_argument(
        "--no-mask-overlap",
        action="store_true",
        help="whether to allow masks to overlap",
    )

    parser.add_argument(
        "--mask-channel-length", type=int, help="repeat the mask indices multiple times"
    )

    parser.add_argument(
        "--mask-channel-prob",
        type=float,
        help="probability of replacing a token with mask",
    )

    parser.add_argument(
        "--mask-channel-selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        help="how to choose masks",
    )

    parser.add_argument(
        "--mask-channel-other",
        type=float,
        help="stdev of the mask length in case of 'normal' selection strategy",
    )

    parser.add_argument(
        "--no-mask-channel-overlap",
        action="store_true",
        help="whether to allow masks to overlap",
    )

    parser.add_argument(
        "--freeze-finetune-updates",
        default=0,
        type=int,
        help="dont finetune wav2vec for this many updates",
    )

    parser.add_argument(
        "--feature-grad-mult",
        default=None,
        type=float,
        help="reset feature grad mult in wav2vec 2.0 to this",
    )

    parser.add_argument(
        "--layerdrop",
        default=0.0,
        type=float,
        help="probability of dropping a layer in wav2vec 2.0",
    )


@register_model("wav2vec_kd")
class Wav2VecKD(BaseFairseqModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        add_common_args(parser)
        #add_w2v_args(parser)

    def __init__(self, teacher, student, args):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.args = args

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        w2v_path = args.w2v_path
        args.w2v_path = None
        cp = torch.load(w2v_path)
        args.w2v_args = cp["args"].w2v_args
        
        sd = cp["model"]
        sd_new = {}
        
        for key, value in sd.items():
            sd_new[key.replace('w2v_encoder.', '')] = value
        
        teacher = Wav2VecEncoder(args, task.target_dictionary)
        teacher.load_state_dict(sd_new, strict=True)
        s_args = copy.deepcopy(args)
        w2v_base_architecture(s_args)
        args.w2v_args = s_args
        student = Wav2VecEncoder(args, task.target_dictionary)
        return cls(teacher, student, args)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)
    
    def forward(self, **kwargs):
        
        x = self.student(**kwargs)
        with torch.no_grad():
            y = self.teacher(**kwargs)
            
        #print(x["encoder_out"].size(), y["encoder_out"].size())
        return x, y

    # def max_positions(self):
    #     return None
    

class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, args, tgt_dict=None):
        self.apply_mask = args.apply_mask

        arg_overrides = {
            "dropout": args.dropout,
            "activation_dropout": args.activation_dropout,
            "dropout_input": args.dropout_input,
            "attention_dropout": args.attention_dropout,
            "mask_length": args.mask_length,
            "mask_prob": args.mask_prob,
            "mask_selection": args.mask_selection,
            "mask_other": args.mask_other,
            "no_mask_overlap": args.no_mask_overlap,
            "mask_channel_length": args.mask_channel_length,
            "mask_channel_prob": args.mask_channel_prob,
            "mask_channel_selection": args.mask_channel_selection,
            "mask_channel_other": args.mask_channel_other,
            "no_mask_channel_overlap": args.no_mask_channel_overlap,
            "encoder_layerdrop": args.layerdrop,
            "feature_grad_mult": args.feature_grad_mult,
        }
        
        if getattr(args, "w2v_args", None) is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.w2v_path, arg_overrides
            )
            w2v_args = state["args"]
        else:
            state = None
            w2v_args = args.w2v_args
            w2v_args.arch = "wav2vec2"

        assert args.normalize == w2v_args.normalize, 'Fine-tuning works best when data normalization is the same'
        
        w2v_args.data = args.data
        task = tasks.setup_task(w2v_args)
        
        model = task.build_model(w2v_args)

        if state is not None and not args.no_pretrained_weights:
            model.load_state_dict(state["model"], strict=True)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = w2v_args.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(args.final_dropout)
        self.freeze_finetune_updates = args.freeze_finetune_updates
        self.num_updates = 0

        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(args, 'decoder_embed_dim', d) != d:
            self.proj = Linear(d, args.decoder_embed_dim)
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

def w2v_base_architecture(args):
    args.extractor_mode = getattr(args, "extractor_mode", "default")

    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)

    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)

    args.final_dim = getattr(args, "final_dim", 128)

    args.layer_norm_first = getattr(args, "layer_norm_first", False)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.05)

    conv_feature_layers = "[(256, 10, 5)]"
    conv_feature_layers += " + [(256, 3, 2)] * 4"
    conv_feature_layers += " + [(256, 2, 2)] * 2"
    args.conv_feature_layers = getattr(args, "conv_feature_layers", conv_feature_layers)

    args.logit_temp = getattr(args, "logit_temp", 0.1)

    args.quantize_targets = getattr(args, "quantize_targets", True)
    args.quantize_input = getattr(args, "quantize_input", False)
    args.same_quantizer = getattr(args, "same_quantizer", False)

    args.feature_grad_mult = getattr(args, "feature_grad_mult", 0.1)

    args.latent_vars = getattr(args, "latent_vars", 160)
    args.latent_groups = getattr(args, "latent_groups", 2)
    args.latent_dim = getattr(args, "latent_dim", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.5)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_min_space = getattr(args, "mask_min_space", 1)

    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.5)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)
    args.mask_channel_min_space = getattr(args, "mask_channel_min_space", 1)

    args.dropout_input = getattr(args, "dropout_input", 0.1)
    args.dropout_features = getattr(args, "dropout_features", 0.1)

    args.num_negatives = getattr(args, "num_negatives", 100)
    args.negatives_from_everywhere = getattr(args, "negatives_from_everywhere", False)
    args.cross_sample_negatives = getattr(args, "cross_sample_negatives", 0)
    args.codebook_negatives = getattr(args, "codebook_negatives", 0)

    args.conv_pos = getattr(args, "conv_pos", 128)
    args.conv_pos_groups = getattr(args, "conv_pos_groups", 16)

    args.latent_temp = getattr(args, "latent_temp", "(2,0.5,0.999995)")

    args.target_glu = getattr(args, "target_glu", False)

    args.conv_bias = getattr(args, "conv_bias", False)
    
    args.infonce = getattr(args, "infonce", True)

    
@register_model_architecture("wav2vec_kd", "wav2vec_kd")
def base_architecture(args):
    args.no_pretrained_weights = getattr(args, "no_pretrained_weights", False)
    args.dropout_input = getattr(args, "dropout_input", 0)
    args.final_dropout = getattr(args, "final_dropout", 0)
    args.apply_mask = getattr(args, "apply_mask", False)
    args.dropout = getattr(args, "dropout", 0)
    args.attention_dropout = getattr(args, "attention_dropout", 0)
    args.activation_dropout = getattr(args, "activation_dropout", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.5)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.5)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)

    args.freeze_finetune_updates = getattr(args, "freeze_finetune_updates", 0)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 0)
    args.layerdrop = getattr(args, "layerdrop", 0.0)