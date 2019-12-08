# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    Linear,
)
import random
from .combine_transformer import TransformerCombineEncoder, TransformerCombineDecoder
from .transformer_bm import TransformerBMEncoder, TransformerBMDecoder

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('transformer')
class TransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')

        # entmax
        parser.add_argument('--use_attn_default', '-use_attn_default', type=int, default=1,
                            help="")
        parser.add_argument('--entmax', type=int, default=0, help='1 for sparsemax, 2 entmax15 3 entmax_bisect')
        parser.add_argument('--distance',type=int, default=0)
        parser.add_argument('--non_global_attn', '-nga', type=int, default=0, help='layer of non global attention')
        parser.add_argument('--max_relative_positions', '-max_relative_positions', type=int, default=0,
                            help="Maximum distance between inputs in relative positions representations.  For more "
                                 "detailed information, see: https://arxiv.org/pdf/1803.02155.pdf 8 is good")
        # qk dim  self attn
        # parser.add_argument('--qk_dim', type=float, default=1, help='qk_dim*=dim')
        # parser.add_argument('--qk_dropout', type=float, default=1, help='cand 0.5')
        # parser.add_argument('--qk_same_rand', type=int, default=1)
        # parser.add_argument('--qk_sort_rand', type=int, default=1) # seems no effect always true
        # parser.add_argument('--qk_bagging', type=int, default=0)
        # parser.add_argument('--bagging_type', type=int, default=0, help='0:mean,1:max')
        parser.add_argument('--ds', type=int, default=0,help='dimension self attention cands 0 1')
        parser.add_argument('--ds_ds', type=int, default=0, help='dimension self attention in decoder self attn cands 0 1 2')
        parser.add_argument('--ds_dim', type=int, default=0, help='')
        parser.add_argument('--ds_heads', type=int, default=1, help='')
        parser.add_argument('--window_size', type=int, default=8)
        parser.add_argument('--stride_rho', type=float, default=0, help='cand 0.5')
        parser.add_argument('--window_padding', type=int, default=0, help='cands 0 1 0 half,1 sequential')
        parser.add_argument('--dim_group_size', type=int, default=32, help='cand 16 32 64 ')
        parser.add_argument('--ds_k', type=int, default=0, help='k max pooling along sequential positions for dimension self attention')

        # eight path results bad than mha
        parser.add_argument('--xt_attn', type=int,default=0,help='0 default, 1out of multi-head.  2 bypass xt  in multi head3 default xs in multi head')
        parser.add_argument('--xt_attn_heads', type=int, default=4,
                            help='')
        parser.add_argument('--xt_in_attn_heads', type=int, default=1,
                            help='')
        parser.add_argument('--xt_merge', type=int, default=0,
                            help='0: add,1:mean,2:attn,3 transpose,4transpose_linear')
        parser.add_argument('--xt_relu1',type=int,default=0)
        parser.add_argument('--xt_relu2', type=int, default=0)
        # attempt relu or bypass fnn
        parser.add_argument('--v_relu', type=int, default=0,
                            help='')
        parser.add_argument('--qk_relu', type=int, default=0,
                            help='')
        parser.add_argument('--att_relu', type=int, default=0,
                            help='')
        parser.add_argument('--bypass_ffn', type=int, default=0, help='currently only work for encoder 0 defaulf  '
                                                                      '1bypass with mha 2 enc,dec 3 encoder vlinear and ffn variants')
        parser.add_argument('--vl', default='linear',type=str, help='none,linear,ffn,hffn')
        parser.add_argument('--ffn', default='ffn', type=str, help='none,linear,ffn')

        # layer memory unused
        parser.add_argument('--mem_q', type=bool, default=False, help='rnn,lstm,gru,dnc,dncr,rmc')
        parser.add_argument('--memory_type', type=str, default='no', help='no,rnn,lstm,gru,dnc,dncr,rmc')
        parser.add_argument('--memory_position', type=str, default='before', help='before,after')
        parser.add_argument('--rnn_ratio', type=float,default=1, help='rnn_memory_size= rnn_ratio*emb_dim')
        parser.add_argument('--rnn_init_type',type=str,default='zeros')
        parser.add_argument('--rnn_integrate_type', type=str, default='add', help='add,concat_linear,update')
        parser.add_argument('--rnn_integrate_attn', type=str, default='no', help= 'no, multi_head_attn, separate_multi_head_attn')
        parser.add_argument('--m_rnn_head', type=int, default=8)

        # rn in multi-head attn
        parser.add_argument('--use_rn', type=int, default=0, help='')
        parser.add_argument('--rn_dim', type=int, default=64, help='')
        parser.add_argument('--g_fc_layers', type=int, default=3, help='')
        parser.add_argument('--f_fc_layers', type=int, default=1, help='')
        parser.add_argument('--g_ln', type=int, default=0, help='')
        parser.add_argument('--f_ln', type=int, default=0, help='')
        parser.add_argument('--rn_conv_k', type=int, default=3, help='')
        parser.add_argument('--rn_conv_s', type=int, default=3, help='')
        parser.add_argument('--share_conv', type=int, default=0, help='')
        parser.add_argument('--rn_dropout', type=float, default=0, help='')
        parser.add_argument('--rn_time', type=int, default=0, help='')
        parser.add_argument('--rn_head', type=int,default=1,help='')

        # # se
        # parser.add_argument('--fc0_before', type=int, default=0)
        # parser.add_argument('--fc0_relu', type=int, default=0)
        # parser.add_argument('--fc0_ln_relu', type=int, default=0)
        # parser.add_argument('--se_activation', type=str, default='none',
        #                     help="none, sigmoid, softmax")
        # parser.add_argument('--se_f', type=str, default='avg', help='avg,rn0,rn1,avg_rn0,avg_rn1')
        parser.add_argument('--se_ratio', type=int, default=4, help='')
        parser.add_argument('--use_se', type=int, default=0, help='')
        parser.add_argument('--se_inner_ln', type=int, default=0)
        parser.add_argument('--se_res', type=int, default=1)
        parser.add_argument('--se_softmax_t', type=float, default=0)
        parser.add_argument('--se_head', type=int, default=0)
        parser.add_argument('--se_head_type', type=int, default=0)
        parser.add_argument('--se_position', type=int, default=0, help='0 inter block, 1 after fc,2- inner block,3 decoder inter block,4- all inner  encoder block')
        parser.add_argument('--se_softmax_group',type=int, default=0,help='4')
        parser.add_argument('--se_vl', type=int, default=0, help=' se net with bypass value linear, 1 bleus bad ')
        # added in 9/10 for SEScores in cmb exp
        parser.add_argument('--se_activation', type=str, default='none',
                            help="none, sigmoid, softmax")
        parser.add_argument('--se_list',type=str, nargs='+',
                            default=[],
                            help="'ffn', 'self_mha' , 'context_mha' 'cmb'")
        parser.add_argument('--se_para',type=int,default=0)
        parser.add_argument('--se_on_x',type=int,default=0, help='0 on ouput 1 onx ')
        parser.add_argument('--se_scale', type=float, default=1, help='enlarge the se output if it is too small')
        parser.add_argument('--se_avg_type', type=int, default=0, help='0-avg,1-attn,2-avg_time,3-attn_simply,4-no_avg')
        parser.add_argument('--se_nonlinear', type=int, default=0)
        parser.add_argument('--se_add', type=int, default=0)
        # parser.add_argument('--se_cmb',type=int,default=0)

        # init and layer norm variations
        parser.add_argument('--init_method', type=str, default='xavier',
                            help='xavier,km，xi')
        # add init scale

        parser.add_argument('--qkv_a', type=float, default=5, )
        parser.add_argument('--attn_a', type=float, default=1, )
        parser.add_argument('--fc1_a', type=float, default=5, )
        parser.add_argument('--fc2_a', type=float, default=5, )

        # adanorm
        parser.add_argument('--lnv', type=str, default='origin', help='origin, no_norm, topk, adanorm,nowb')
        parser.add_argument('--sigma', type=float, default=0.005,)
        parser.add_argument('--adanorm_scale', type=float, default=2.0, help='')
        parser.add_argument('--nowb_scale', type=float, default=1.0, help='')
        parser.add_argument('--mean_detach', type=int, default=0, help='')
        parser.add_argument('--std_detach', type=int, default=0, help='')

        #  special attn select
        parser.add_argument('--use_att', type=str, nargs='+',
                            default=['es', 'ds', 'dc', ],
                            help='which attn  do we apply rn or se or sp to ')
        # sparse transformer
        parser.add_argument('--div', type=int, default=0,
                            help='control the attention sparsity')
        parser.add_argument('--lb', type=int, default=0,
                            help='the lower bound of the attention sparsity')

        # combine information in transformer model
        parser.add_argument('--combine', type=int, default=0, help='0 as usual  1 combine residual')
        parser.add_argument('--combine_linear',type=int,default=0,help='combine multi head attn and ffn by dense added in 8/17')
        parser.add_argument('--cmb_gate', type=int, default=0, help='0:no gate | 1 gate dim | 2 gate scalar 3 dim parameter ')
        parser.add_argument('--cmb_gate_nonlinear',type=int, default=0)
        parser.add_argument('--cmb_gate_avgpos',type=int,default=0)
        parser.add_argument('--info_linear', type=int, default=0,
                            help='extract info from ffn and multi head attn')
        parser.add_argument('--il_relu',type=int,default=0,help='0,nothing,1 relu 2 leaky relu 3 tanh')
        parser.add_argument('--il_relu_linear', type=int, default=0)
        parser.add_argument('--il_2act', type=int, default=0, help='0 nothing 1 relu 3 tanh')
        parser.add_argument('--il_ratio', type=float, default=0)
        parser.add_argument('--enc_il_inc', type=int,default=0,help='only for encoder')
        parser.add_argument('--dec_info_linear',type=int,default=0)
        parser.add_argument('--dec_il_list', type=str, nargs='+',
                            default=['self_mha', 'context_mha', 'ffn'],
                            help="['residual', 'self_mha', 'context_mha', 'ffn']")
        parser.add_argument('--enc_il_dropout', type=float, default=0)
        parser.add_argument('--dec_il_dropout', type=float, default=0)
        parser.add_argument('--reslayer', type=str, default='combine',help='combine,dense1,cache,cache_linear1 3*4,cache_linear2 4*3')
        parser.add_argument('--cache_block_valia_attn',type=int,default=0,help='added in 9.1 ,if true, cmb enc does not include ffn ')
        parser.add_argument('--ffn_nr', type=int, default=0, help='ffn no residual,exp on the regular transformer')
        # parser.add_argument('--encr',type=int, default=0,help='if use enc ratio')
        # parser.add_argument('--decr', type=int, default=0, help='if use dec ratio')

        parser.add_argument('--enc_ratio',type=float, default=0, help='0.5 means x+=0.5*x1 + 0.5 *x2')
        parser.add_argument('--dec_ratio',type=float, default=0, help='0.5 means x+=0.5*(x1+x2+x3);>1 x = (self.'
                                                                     'dec_ratio- 1 ) * (residual + x1 + x2 + x3),>2 ode')
        parser.add_argument('--attn_ratio', type=float, default=1)

        # reslayer = mix_conv
        parser.add_argument('--kernel_size', type=int, default=0, help='for dynamic in ffn or conv, 0 to use depth kernels')
        parser.add_argument('--attn_dynamic_type', type=int, default=0,
                            help='0: no use,1 use static kernel(k>0) or depth kernel(k==0) 2. use  wide kernel ')
        parser.add_argument('--attn_cat_relu',type=int,default=0)
        parser.add_argument('--attn_wide_kernels', type=lambda x: options.eval_str_list(x, int),
                            help='list of kernel size (default: "[3,9]") for wide and gate')
        parser.add_argument('--weight-dropout', type=float, metavar='D',
                            help='dropout probability for conv weights')
        parser.add_argument('--dynamic_gate',type=int,default=0,help='0,1')
        parser.add_argument('--dynamic_depth_kernels', type=lambda x: options.eval_str_list(x, int),
                            help='list of kernel size (default: "[3,3,3,7,7,7,7,7,7,15,15,15]"),for ffn or attn')
        parser.add_argument('--dynamic_padding', type=int, default=0, help='padding before dynamic conv')
        parser.add_argument('--attn_dynamic_cat', type=int, default=0)
        parser.add_argument('--attn_dynamic_indie_v', type=int, default=0,help='whether  there is indepent v for dynamic')
        parser.add_argument('--attn1', type=str, default='origin', help='origin,mha,')
        parser.add_argument('--attn2', type=str, default='none', help='origin,none')
        parser.add_argument('--conv2', type=str, default='ffn', help='none,fc,hffn,ffn, biggerffn, 2ffn,ffn_ln,ffn_tanh,'
                                                                     'ffn_relutanh,conv,conv_after,dynamic_conv,add_dynamic_conv')
        parser.add_argument('--conv3', type=str, default='none', help='none,ffn')
        # ffn like dynamic conv
        parser.add_argument('--dynamic_inner_dim_ratio',type=int, default=2)
        parser.add_argument('--dc_relu', type=int, default=0, help='for use in dynamic conv')
        parser.add_argument('--dcb_relu',type=int,default=0, help='relu before dynamic')
        parser.add_argument('--mha_act', type=str, default='none',help='none,relu,tanh')
        parser.add_argument('--mha_2fc', type=int, default=0,help='0:nothing 1:another fc')
        parser.add_argument('--enc_tanh_list', type=str, nargs='+',
                            default=[],
                            help=" ['self_mha', 'ffn']")
        parser.add_argument('--dec_tanh_list',type=str, nargs='+',
                            default=[],
                            help=" ['self_mha', 'context_mha', 'ffn']")
        parser.add_argument('--dec_nffn',type=int,default=0,help='no ffn in decoder ,to demonstate the importance for tanh in remedy ffn')

        parser.add_argument('--sep_out_proj', type=int, default=0, help='sep_out_proj for dynamic and attn')
        parser.add_argument('--gate_inc_attn', type=int, default=0, help=' whether include attention for gate multiply in the. 2019/9/20 ')
        parser.add_argument('--gate_before_proj',type=int, default=0)

        # reduction
        parser.add_argument('--reduction_ffn_mode', type=int, default=0, help='0 concat fc; 1 concat fc act fc ; 2 fc act concat fc')
        parser.add_argument('--dense_activate', type=int, default=0, help='0:no act,1 relu 2 tanh, added in 9.1  used for one or two step reduction')
        parser.add_argument('--dense_dropout', type=float, default=0,)
        parser.add_argument('--cd_dropout', type=float, default=0,help='for caches out')

        parser.add_argument('--cache_residual', type=int,default=0,help=' whether to pass residual across cache')
        parser.add_argument('--inc_cur', type=int, default=0)
        parser.add_argument('--reduction_mode', type=str, default='linear', help='linear, attn,rnn')
        parser.add_argument('--cmb_mode', type=str, default='add', help='concat_linear, add')
        parser.add_argument('--layer_mha_dim', type=int, default=512, help=' hidden model dim in layer mha')
        parser.add_argument('--layer_mha_head', type=int, default=1)
        parser.add_argument('--layer_memory_size', type=int, default=512)
        parser.add_argument('--reduction_rnn', type=str, default='gru', help='rnn,gru,lstm')

        # combine or divide  步子迈太大了
        parser.add_argument('--cmb_2hffn', type=int, default=0)
        parser.add_argument('--linear_divide', type=int, default=0)

        parser.add_argument('--cache_size',type=int,default=3)
        parser.add_argument('--caches_dense', type=int, default=0,help='把每个大cache的结果总结起来')
        parser.add_argument('--caches_cat', type=int, default=0)
        parser.add_argument('--cache_norms', type=int, default=0)

        parser.add_argument('--init_topk_rho', type=float, default=0)
        parser.add_argument('--big_km', type=int, default=0)
        parser.add_argument('--big_km_list', type=str, nargs='+',
                            default=['in', 'out', 'fc1', 'fc2', 'qkv', 'attn_out'],
                            help='')
        # for version control
        parser.add_argument('--layer_version', type=str, default='820', help='default,...')
        parser.add_argument('--k_sampling', type=int,default=0, help='added in 0:13 24/8 2019 for aggregation features in k different ways')

        # for big matrix cmba
        parser.add_argument('--bm', type=int, default=0, help='whether to use transformer_bm')
        parser.add_argument('--bm_in_a', type=float, default=5, help='sqrt(6/(1+a)),-1 for xavier')
        parser.add_argument('--bm_out_a', type=float, default=5, help='sqrt(6/(1+a)), -1 for xavier')
        parser.add_argument('--bm_fc3', type=float, default=0, help='')  # 虚惊一场
        parser.add_argument('--bm_fc4', type=float, default=0, help='')
        parser.add_argument('--bm_norm', type=int, default=0)  # 0 no norm 1 layer norm
        parser.add_argument('--bm_ffn_norm', type=int, default=0)  # 这个不建议调
        # parser.add_argument('--bm_prenorm',type=int,default=0)

        # for bagging in bm
        parser.add_argument('--qk_bagging', type=int,default=0,help='0 nothing,1 max,2 mean')
        parser.add_argument('--qk_dropout', type=float, default=0, help='')
        parser.add_argument('--qk_big', type=int, default=1, help='1,2,4')
        parser.add_argument('--bagging_num', type=int, default=1, help='')

        # glu for conv in both for bm and cmba
        parser.add_argument('--conv_in_glu', type=int, default=0, help='0 nothing, 1 for conv in only 2 for attn and conv both') # 觉得conv 和attn 必须在同一空间

        parser.add_argument('--input_dropout', type=float, default=0, help='')

        # entmax
        # parser.add_argument('--entmax', type=int, default=0, help='1 for sparsemax, 2 entmax15 3 entmax_bisect')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, 'max_source_positions', None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        if args.combine:
            encoder = TransformerCombineEncoder(args, src_dict, encoder_embed_tokens)
            decoder = TransformerCombineDecoder(args, tgt_dict, decoder_embed_tokens)
        elif args.bm:
            encoder = TransformerBMEncoder(args, src_dict, encoder_embed_tokens)
            decoder = TransformerBMDecoder(args, tgt_dict, decoder_embed_tokens)
        else:
            encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
            decoder = cls.build_encoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
        )


@register_model('transformer_align')
class TransformerAlignModel(TransformerModel):
    """
    See "Jointly Learning to Align and Translate with Transformer
    Models" (Garg et al., EMNLP 2019).
    """

    def __init__(self, encoder, decoder, args):
        super().__init__(args, encoder, decoder)
        self.alignment_heads = args.alignment_heads
        self.alignment_layer = args.alignment_layer
        self.full_context_alignment = args.full_context_alignment

    @staticmethod
    def add_args(parser):
        # fmt: off
        super(TransformerAlignModel, TransformerAlignModel).add_args(parser)
        parser.add_argument('--alignment-heads', type=int, metavar='D',
                            help='Number of cross attention heads per layer to supervised with alignments')
        parser.add_argument('--alignment-layer', type=int, metavar='D',
                            help='Layer number which has to be supervised. 0 corresponding to the bottommost layer.')
        parser.add_argument('--full-context-alignment', type=bool, metavar='D',
                            help='Whether or not alignment is supervised conditioned on the full target context.')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        # set any default arguments
        transformer_align(args)

        transformer_model = TransformerModel.build_model(args, task)
        return TransformerAlignModel(transformer_model.encoder, transformer_model.decoder, args)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths)
        return self.forward_decoder(prev_output_tokens, encoder_out)

    def forward_decoder(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args,
    ):
        attn_args = {'alignment_layer': self.alignment_layer, 'alignment_heads': self.alignment_heads}
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out,
            **attn_args,
            **extra_args,
        )

        if self.full_context_alignment:
            attn_args['full_context_alignment'] = self.full_context_alignment
            _, alignment_out = self.decoder(
                prev_output_tokens, encoder_out, features_only=True, **attn_args, **extra_args,
            )
            decoder_out[1]['attn'] = alignment_out['attn']

        return decoder_out


EncoderOut = namedtuple('TransformerEncoderOut', [
    'encoder_out',  # T x B x C
    'encoder_padding_mask',  # B x T
    'encoder_embedding',  # B x T x C
    'encoder_states',  # List[T x B x C]
])


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, 'layernorm_embedding', False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(self, src_tokens, src_lengths, cls_input=None, return_all_hiddens=False, **unused):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    encoder_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.index_select(1, new_order)
            )
        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(0, new_order)
            )
        if encoder_out.encoder_embedding is not None:
            encoder_out = encoder_out._replace(
                encoder_embedding=encoder_out.encoder_embedding.index_select(0, new_order)
            )
        if encoder_out.encoder_states is not None:
            for idx, state in enumerate(encoder_out.encoder_states):
                encoder_out.encoder_states[idx] = state.index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                print('deleting {0}'.format(weights_key))
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, self.padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, 'layernorm_embedding', False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def forward(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            **extra_args
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        full_context_alignment=False,
        alignment_layer=None,
        alignment_heads=None,
        **unused,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = len(self.layers) - 1

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn = None
        inner_states = [x]
        for idx, layer in enumerate(self.layers):
            encoder_state = None
            if encoder_out is not None:
                if self.layer_wise_attention:
                    encoder_state = encoder_out.encoder_states[idx]
                else:
                    encoder_state = encoder_out.encoder_out

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, layer_attn = layer(
                    x,
                    encoder_state,
                    encoder_out.encoder_padding_mask if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=(idx == alignment_layer),
                    need_head_weights=(idx == alignment_layer),
                )
                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float()

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, '_future_mask')
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m




@register_model_architecture('transformer', 'transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.no_cross_attention = getattr(args, 'no_cross_attention', False)
    args.cross_self_attention = getattr(args, 'cross_self_attention', False)
    args.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', False)


@register_model_architecture('transformer', 'transformer_iwslt_de_en')
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)


@register_model_architecture('transformer', 'transformer_wmt_en_de')
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture('transformer', 'transformer_vaswani_wmt_en_de_big')
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)


@register_model_architecture('transformer', 'transformer_vaswani_wmt_en_fr_big')
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('transformer', 'transformer_wmt_en_de_big')
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('transformer', 'transformer_wmt_en_de_big_t2t')
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('transformer_align', 'transformer_align')
def transformer_align(args):
    args.alignment_heads = getattr(args, 'alignment_heads', 1)
    args.alignment_layer = getattr(args, 'alignment_layer', 4)
    args.full_context_alignment = getattr(args, 'full_context_alignment', False)
    base_architecture(args)


@register_model_architecture('transformer_align', 'transformer_wmt_en_de_big_align')
def transformer_wmt_en_de_big_align(args):
    args.alignment_heads = getattr(args, 'alignment_heads', 1)
    args.alignment_layer = getattr(args, 'alignment_layer', 4)
    transformer_wmt_en_de_big(args)
