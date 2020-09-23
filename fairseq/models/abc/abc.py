import logging
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models import BaseFairseqModel, register_model, register_model_architecture
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange


@register_model("abc")
class ABCModel(BaseFairseqModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

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
            "--dropout",
            type=float,
            metavar="D",
            help="dropout probability for the transformer",
        )

        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )

        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
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
            "--feature-grad-mult",
            type=float,
            help="multiply feature extractor var grads by this",
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

        parser.add_argument("--mask-length", type=int, help="mask length")

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
            help="secondary mask argument (used for more complex distributions), see help in compute_mask_indices",
        )

        parser.add_argument(
            "--no-mask-overlap",
            action="store_true",
            help="whether to allow masks to overlap",
        )

        parser.add_argument(
            "--mask-min-space",
            type=int,
            help="min space between spans (if no overlap is enabled)",
        )

        parser.add_argument(
            "--mask-channel-length",
            type=int,
            help="repeat the mask indices multiple times",
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
            help="secondary mask argument (used for more complex distributions), see help in compute_mask_indices",
        )

        parser.add_argument(
            "--no-mask-channel-overlap",
            action="store_true",
            help="whether to allow masks to overlap",
        )

        parser.add_argument(
            "--mask-channel-min-space",
            type=int,
            help="min space between spans (if no overlap is enabled)",
        )

        parser.add_argument(
            "--dropout-input",
            type=float,
            metavar="D",
            help="dropout to apply to the input (after feat extr)",
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
        
        #parser.add_argument(
        #    "--byol-all",
        #    action="store_true",
        #    help="apply byol to whole network"
        #)
        
        parser.add_argument(
            "--base-decay",
            type=float,
            metavar="D",
            help="base decay value of target network"
        )
        
        parser.add_argument(
            "--projection-dim",
            type=int,
            metavar="N",
            help="byol projection network's output dimension"
        )
        
        parser.add_argument(
            "--prediction-dim",
            type=int,
            metavar="N",
            help="byol prediction network's output dimension"
        )
        
        parser.add_argument(
            "--byol-hidden-dim",
            type=int,
            metavar="N",
            help="hidden dimension of MLPs used for byol"
        )
        
        parser.add_argument(
            "--shared-quantizer",
            action="store_true",
            help="share quantizer with target network"
        )
        
        parser.add_argument(
            "--shared-emb",
            action="store_true",
            help="share mask embedding parameter"
        )
        
        parser.add_argument(
            "--mlp-prediction",
            action="store_true",
            help="use MLP as prediction network"
        )
        
        parser.add_argument(
            "--mlp-encoder",
            action="store_true",
            help="use MLP as encoder network"
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
        
    def __init__(self, args):
        super().__init__()
        self.args = args

        feature_enc_layers = eval(args.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=args.extractor_mode,
            conv_bias=args.conv_bias,
        )
        
        self.target_params = []
        
        self.feature_extractor_target = copy.deepcopy(self.feature_extractor)
        
        self.target_params += list(self.feature_extractor_target.parameters())

        self.post_extract_proj = (
            nn.Linear(self.embed, args.encoder_embed_dim)
            if self.embed != args.encoder_embed_dim and not args.quantize_input
            else None
        )
        
        if self.embed != args.encoder_embed_dim and not args.quantize_input:
            self.post_extract_proj_target = (
                nn.Linear(self.embed, args.encoder_embed_dim)
            )
            self.target_params += list(self.post_extract_proj_target.parameters())

        self.mask_prob = args.mask_prob
        self.mask_selection = args.mask_selection
        self.mask_other = args.mask_other
        self.mask_length = args.mask_length
        self.no_mask_overlap = args.no_mask_overlap
        self.mask_min_space = args.mask_min_space

        self.mask_channel_prob = args.mask_channel_prob
        self.mask_channel_selection = args.mask_channel_selection
        self.mask_channel_other = args.mask_channel_other
        self.mask_channel_length = args.mask_channel_length
        self.no_mask_channel_overlap = args.no_mask_channel_overlap
        self.mask_channel_min_space = args.mask_channel_min_space

        self.dropout_input = nn.Dropout(args.dropout_input)
        self.dropout_features = nn.Dropout(args.dropout_features)
        
        self.base_decay = args.base_decay
        self.step = 0
        self.total_steps = args.total_num_update

        self.feature_grad_mult = args.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None
        self.shared_quantizer = args.shared_quantizer
        
        final_dim = args.final_dim if args.final_dim > 0 else args.encoder_embed_dim

        if args.quantize_targets:
            vq_dim = args.latent_dim if args.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=args.latent_vars,
                temp=eval(args.latent_temp),
                groups=args.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            
            if not args.shared_quantizer:
                self.quantizer_target = copy.deepcopy(self.quantizer)
                
                self.target_params += list(self.quantizer_target.parameters())
                
            ### TODO: separate project_q?
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)
            
        if args.quantize_input:
            if args.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
                if not args.shared_quantizer:
                    self.input_quantizer = self.quantizer_target
            else:
                vq_dim = (
                    args.latent_dim if args.latent_dim > 0 else args.encoder_embed_dim
                )
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=args.latent_vars,
                    temp=eval(args.latent_temp),
                    groups=args.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                )
                
                if not args.shared_quantizer:
                    self.input_quantizer_target = copy.deepcopy(self.input_quantizer)
                    self.target_params += list(self.input_quantizer_target.parameters())
                    
            self.project_inp = nn.Linear(vq_dim, args.encoder_embed_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(args.encoder_embed_dim).uniform_()
        )
        
        if not args.shared_emb:
            self.mask_emb_target = copy.deepcopy(self.mask_emb)
            self.target_params += list(self.mask_emb_target.parameters())

        if args.mlp_encoder:
            self.encoder = nn.Sequential(
                nn.Linear(args.encoder_embed_dim, args.byol_hidden_dim),
                nn.BatchNorm1d(args.byol_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linaer(args.byol_hidden_dim, final_dim)
            )
        else:
            self.encoder = TransformerEncoder(args)
        
        ### TODO: Transformer predictor?
        if args.mlp_predictor:
            self.predictor = nn.Sequential(
                nn.Linear(final_dim, args.byol_hidden_dim),
                nn.BatchNorm1d(args.byol_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linaer(args.byol_hidden_dim, final_dim)
            )
        else:
            self.predictor = nn.Sequential(
                nn.Linear(final_dim, args.byol_hidden_dim),
                nn.BatchNorm1d(args.byol_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linaer(args.byol_hidden_dim, final_dim)
            )
            
        self.encoder_target = copy.deepcopy(self.encoder)
        self.target_params += list(self.encoder_target.parameters())
        
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if args.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )
            self.target_glu_target = copy.deepcopy(self.target_glu)
            self.target_params += list(self.target_glu_target.parameters())

        self.final_proj = nn.Linear(args.encoder_embed_dim, final_dim)
        self.final_proj_target = copy.deepcopy(self.final_proj)
        
        self.target_params += list(self.final_proj_target.parameters())
    
    def update_target(self):
        decay = 1 - (1 - self.base_decay) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        
        for online_param, target_param in zip(self.online_params, self.target_params):
            target_old = target_param.data
            target_param.data = decay * target_old + (1 - decay) * online_param.data
            
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict
    
    @classmethod
    def build_model(cls, args, task=None):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        return cls(args)
    
    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices
    
    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, source, padding_mask, mask=False):
        res = self.forward(source[0], padding_mask, mask=mask, features_only=True)
        return res["x"], res["padding_mask"]
    
    def prediction(self, source, padding_mask=None, mask=True):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None
            
        x = self.encoder(x, padding_mask=padding_mask)

        if features_only:
            return {"x": x, "padding_mask": padding_mask}

        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.project_q(y)

        else:
            y = self.project_q(y)

        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        #x = self.compute_preds(x, y, negs)

        result = {"x": x, "padding_mask": padding_mask, "features_pen": features_pen}

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result
    
    def target_prediction(self, source, padding_mask=None, mask=True):
        
        with torch.no_grad():
            if self.feature_grad_mult > 0:
                features = self.feature_extractor_target(source)
                if self.feature_grad_mult != 1.0:
                    features = GradMultiply.apply(features, self.feature_grad_mult)
            else:
                with torch.no_grad():
                    features = self.feature_extractor_target(source)
    
            features_pen = features.float().pow(2).mean()
    
            features = features.transpose(1, 2)
            features = self.layer_norm(features)
            unmasked_features = features.clone()
    
            if padding_mask is not None:
                extra = padding_mask.size(1) % features.size(1)
                if extra > 0:
                    padding_mask = padding_mask[:, :-extra]
                padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
                padding_mask = padding_mask.all(-1)
    
            if self.post_extract_proj is not None:
                features = self.post_extract_proj_target(features)
    
            features = self.dropout_input(features)
            unmasked_features = self.dropout_features(unmasked_features)
    
            num_vars = None
            code_ppl = None
            prob_ppl = None
            curr_temp = None
    
            if self.input_quantizer:
                if self.shared_quantizer:
                    q = self.input_quantizer(features, produce_targets=False)
                else:
                    q = self.input_quantizer(features, produce_targets=False)
                features = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]
                if self.shared_quantizer:
                    features = self.project_inp(features)
                else:
                    features = self.project_inp_target(features)
    
            if mask:
                x, mask_indices = self.apply_mask(features, padding_mask)
                if mask_indices is not None:
                    y = unmasked_features[mask_indices].view(
                        unmasked_features.size(0), -1, unmasked_features.size(-1)
                    )
                else:
                    y = unmasked_features
            else:
                x = features
                y = unmasked_features
                mask_indices = None
                
            x = self.encoder_target(x, padding_mask=padding_mask)
    
            if self.quantizer:
                if self.shared_quantizer:
                    q = self.quantizer(y, produce_targets=False)
                else:
                    q = self.quantizer_target(y, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]
    
                if self.shared_quantizer:
                    y = self.project_q(y)
                else:
                    y = self.project_q_target(y)
    
            else:
                y = self.project_q(y)
    
            x = x[mask_indices].view(x.size(0), -1, x.size(-1))
    
            if self.target_glu:
                y = self.target_glu_target(y)
                negs = self.target_glu(negs)
    
            x = self.final_proj_target(x)
            #x = self.compute_preds(x, y, negs)
    
            result = {"x": x, "padding_mask": padding_mask, "features_pen": features_pen}
    
            if prob_ppl is not None:
                result["prob_perplexity"] = prob_ppl
                result["code_perplexity"] = code_ppl
                result["num_vars"] = num_vars
                result["temp"] = curr_temp
    
            return result
    
    #def forward(self, source, padding_mask=None, mask=True, features_only=True):
    #    result = self.prediction(source[0], padding_mask, mask, features_only)
    #    
    #    return result["x"], result["padding_mask"]
    
    def forward(self, source, padding_mask=None, mask=True, features_only=False):
        result_0 = self.prediction(source[0], padding_mask, mask, features_only)
        result_1 = self.prediction(source[1], padding_mask, mask, features_only)
        result_target_0 = self.target_prediction(source[0], padding_mask, mask, features_only)
        result_target_1 = self.target_prediction(source[1], padding_mask, mask, features_only)
        
        return result_0, result_1, result_target_0, result_target_1
    
    
class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, args, embed_dim=None):
        super().__init__()

        self.dropout = args.dropout
        if embed_dim is None:
            self.embedding_dim = args.encoder_embed_dim
        else:
            self.embedding_dim = embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )
                for _ in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None):
        x = self.extract_features(x, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x, padding_mask=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x += x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                layer_results.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn
    
    
@register_model_architecture("abc", "abc")
def base_architecture(args):
    args.extractor_mode = getattr(args, "extractor_mode", "default")

    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)

    args.final_dim = getattr(args, "final_dim", 0)

    args.layer_norm_first = getattr(args, "layer_norm_first", False)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)

    conv_feature_layers = "[(512, 10, 5)]"
    conv_feature_layers += " + [(512, 8, 4)]"
    conv_feature_layers += " + [(512, 4, 2)] * 3"
    conv_feature_layers += " + [(512, 1, 1)]"
    args.conv_feature_layers = getattr(args, "conv_feature_layers", conv_feature_layers)

    args.logit_temp = getattr(args, "logit_temp", 0.1)

    args.quantize_targets = getattr(args, "quantize_targets", False)
    args.quantize_input = getattr(args, "quantize_input", False)
    args.same_quantizer = getattr(args, "same_quantizer", False)

    args.feature_grad_mult = getattr(args, "feature_grad_mult", 1.0)

    args.latent_vars = getattr(args, "latent_vars", 320)
    args.latent_groups = getattr(args, "latent_groups", 2)
    args.latent_dim = getattr(args, "latent_dim", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.65)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_min_space = getattr(args, "mask_min_space", 1)

    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)
    args.mask_channel_min_space = getattr(args, "mask_channel_min_space", 1)

    args.dropout_input = getattr(args, "dropout_input", 0)
    args.dropout_features = getattr(args, "dropout_features", 0)

    args.num_negatives = getattr(args, "num_negatives", 100)
    args.negatives_from_everywhere = getattr(args, "negatives_from_everywhere", False)
    args.cross_sample_negatives = getattr(args, "cross_sample_negatives", 0)
    args.codebook_negatives = getattr(args, "codebook_negatives", 0)

    #args.byol_all = getattr(args, "byol_all", True)
    args.base_decay = getattr(args, "base_decay", 0.996)
    args.projection_dim = getattr(args, "projection_dim", 256)
    args.prediction_dim = getattr(args, "prediction_dim", 256)
    args.byol_hidden_dim = getattr(args, "byol_hidden_dim", 4096)
    
    args.shared_quantizer = getattr(args, "shared_quantizer", True)
    args.shared_emb = getattr(args, "shared_emb", True)
    args.mlp_prediction = getattr(args, "mlp_prediction", True)
    args.mlp_encoder = getattr(args, "mlp_encoder", False)
    
    args.conv_pos = getattr(args, "conv_pos", 128)
    args.conv_pos_groups = getattr(args, "conv_pos_groups", 16)

    args.latent_temp = getattr(args, "latent_temp", "(2,0.5,0.999995)")

    args.target_glu = getattr(args, "target_glu", False)

    args.conv_bias = getattr(args, "conv_bias", False)
