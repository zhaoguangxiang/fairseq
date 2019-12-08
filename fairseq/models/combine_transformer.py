import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    MultiheadAttention,
    MultiheadAttention820,
    Linear,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    DynamicConv1dTBC
)
# import numpy as np
import math
from .layer_mha import MultiHeadedAttention


class fc_act_cat_fc(nn.Module):
    def __init__(self, num, args,):
        super().__init__()
        self.num = num
        self.dim = args.encoder_embed_dim
        self.act = args.dense_activate
        self.fcs = nn.ModuleList([nn.Linear(self.dim, self.dim) for i in range(num-1)])
        self.fcs.append(nn.Linear(self.dim, args.encoder_ffn_embed_dim))
        self.fc2 = nn.Linear((num-1) * self.dim + args.encoder_ffn_embed_dim, self.dim)

    def forward(self, x):
        xs = list(torch.split(x, self.dim, dim=-1))
        xi_list = []
        for i in range(len(xs)):
            fci = self.fcs[i]
            xi = fci(xs[i])
            if self.act == 1:
                xi = F.relu(xi)
            elif self.act == 2:
                xi = F.tanh(xi)
            xi_list.append(xi)
        x = torch.cat(xi_list, dim=-1)
        x = self.fc2(x) # RuntimeError: size mismatch, m1: [3648 x 384], m2: [1152 x 384] at /pytorch/aten/src/THC/generic/THCTensorMathBlas.cu:273
        return x

class TransformerCombineEncoder(FairseqEncoder):
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
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.out_linear = None
        self.reslayer = args.reslayer
        self.dense_dropout = args.dense_dropout if 'dense_dropout' in args else 0
        if args.reslayer == 'combine':
            self.layers.extend([
                TransformerCombineEncoderLayer(layer_id=i, args=args)
                for i in range(args.encoder_layers)
            ])
        elif args.reslayer == '2L':
            self.layers.extend([TransformerCombine2LEncoderLayer(layer_id=i,args=args) for i in  range(int(args.encoder_layers/2))])
        elif args.reslayer == 'mixconv':
            self.layers.extend([
                TransformerCombineMixconvEncoderLayer(layer_id=i, args=args)
                for i in range(args.encoder_layers)
            ])

        elif args.reslayer == 'valina_attn':
            self.layers.extend(
                [TransformerAttnEncoderLayer(layer_id=i, args=args, ) for i in range(args.encoder_layers)])
        elif args.reslayer == 'dense1':
            self.layers.extend(
                [TransformerDense1EncoderLayer(layer_id=i, args=args, in_planes=(i + 1) * args.encoder_embed_dim) for i
                 in range(args.encoder_layers)])
            self.out_linear = nn.Linear((args.encoder_layers+1) * args.encoder_embed_dim, args.encoder_embed_dim)
        elif args.reslayer == 'res_flow':  # 只是把residual 放到网络外面，但也让最后一层隔几层总结下，效果不行
            self.layers.extend(
                [TransformerCacheEncoderLayer(layer_id=i, args=args, ) for i in range(args.encoder_layers)])
            if args.cache_size != 0:
                self.cache_size = args.cache_size
                num_cache = int(args.encoder_layers/args.cache_size)
                self.caches_dense = nn.Linear(num_cache*args.encoder_embed_dim, args.encoder_embed_dim)
        elif args.reslayer == 'cache':  # 所有的redsidual 都存在一个cache里
            # 0. 定义基本操作的层
            self.layers.extend([TransformerCacheEncoderLayer(layer_id=i, args=args, ) for i in range(args.encoder_layers)])
            # 1. 定义将之前层残差组合的方法
            self.reduction_layers = None
            self.mha = None
            self.rnn = None
            self.inc_cur = args.inc_cur  # attn concat linear  什么的包不包括当前的
            if args.reduction_mode == 'linear':
                self.reduction_layers = nn.ModuleList([])
                if self.inc_cur:
                    self.reduction_layers.extend(
                        [nn.Linear((i + 2) * args.encoder_embed_dim, args.encoder_embed_dim) for i in
                         range(args.encoder_layers)])
                else:
                    self.reduction_layers.extend([nn.Linear((i + 1) * args.encoder_embed_dim,args.encoder_embed_dim)
                                                  for i in range(args.encoder_layers)])
            elif args.reduction_mode == 'attn':
                self.mha = MultiHeadedAttention(head_count=args.layer_mha_head, input_dim=args.encoder_embed_dim,
                                                model_dim=args.layer_mha_dim)
            elif args.reduction_mode == 'rnn':  # rnn目前只考虑不包含当前的输出的版本
                self.reduction_rnn = args.reduction_rnn
                if self.reduction_rnn == 'rnn':
                    self.rnn = torch.nn.RNNCell(args.encoder_embed_dim, args.layer_memory_size, bias=True)
                elif self.reduction_rnn == 'lstm':
                    self.rnn = torch.nn.LSTMCell(args.encoder_embed_dim, args.layer_memory_size, bias=True)
                else:
                    self.rnn = torch.nn.GRUCell(args.encoder_embed_dim, args.layer_memory_size, bias=True)
            # 2. 定义如果只组合残差不考虑当前输出，组合的残差和当前的输出的结合方式，默认是add
            if not self.inc_cur:
                self.cmb_mode = args.cmb_mode  # add; concat_linear
                if self.cmb_mode == 'concat_linear':
                    self.cmb_reduction = nn.Linear(2 * args.encoder_embed_dim, args.encoder_embed_dim)
                else:  # add
                    self.cmb_reduction = None
        elif args.reslayer == 'cache_block':  # 2*6 3*4 4*3
            self.dense_activate = args.dense_activate if 'dense_activate' in args else 0
            # self.cd_dropout = args.cd_dropout if 'cd_dropout' in args else 0
            num_cache = int(args.encoder_layers / args.cache_size)
            if args.caches_dense:
                self.caches_dense = nn.Sequential(nn.Linear(num_cache*args.encoder_embed_dim,args.encoder_embed_dim), nn.Dropout(args.cd_dropout))
                if args.caches_cat:
                    self.caches_out_cmb = nn.Linear(2*args.encoder_embed_dim, args.encoder_embed_dim)
                else:
                    self.caches_out_cmb = None
            else:
                self.caches_dense = None
            self.num_cache = num_cache
            self.cache_size = args.cache_size
            # 0. 定义基本操作的层
            if args.cache_block_valia_attn:
                self.layers.extend(
                    [TransformerAttnEncoderLayer(layer_id=i, args=args, ) for i in range(args.encoder_layers)])
            else:
                self.layers.extend(
                [TransformerCacheEncoderLayer(layer_id=i, args=args, ) for i in range(args.encoder_layers)])
            # 1. 定义将之前层残差组合的方法
            self.reduction_layers = nn.ModuleList([])
            self.inc_cur = args.inc_cur  # attn concat linear  什么的包不包括当前的
            self.reduction_ffn_mode = args.reduction_ffn_mode
            for i in range(num_cache):
                if args.reduction_ffn_mode == 1:  # cat fc act fc
                    if self.dense_activate == 1:
                        inner_act = nn.ReLU()
                    elif self.dense_activate == 2:
                        inner_act = nn.Tanh()
                    else:
                        inner_act = nn.Sequential()
                    if not self.inc_cur:
                            self.reduction_layers.extend([nn.Sequential(nn.Linear((j + 1) * args.encoder_embed_dim, args.encoder_embed_dim),
                                                                        inner_act, nn.Linear(args.encoder_embed_dim,args.encoder_embed_dim))
                                                          for j in range(args.cache_size)])
                    else:
                            self.reduction_layers.extend([nn.Sequential(nn.Linear((j + 2) * args.encoder_embed_dim, args.encoder_embed_dim),
                                                                        inner_act, nn.Linear(args.encoder_embed_dim,args.encoder_embed_dim))
                                                          for j in range(args.cache_size)])
                elif args.reduction_ffn_mode == 2:  # fc_act_cat_fc similar to ffn
                    # 暂不用于inc_cur 情形
                    assert not self.inc_cur
                    self.reduction_layers.extend([fc_act_cat_fc(num=j+1, args=args) for j in range(args.cache_size)])
                    # else:
                    #     self.reduction_layers.extend([fc_act_cat_fc(num=j+2, act=self.inner_act,args=args)
                    #                                       for j in range(args.cache_size)])
                else:  # cat fc act
                    if not self.inc_cur:
                        self.reduction_layers.extend([nn.Linear((j + 1) * args.encoder_embed_dim, args.encoder_embed_dim)
                                                      for j in range(args.cache_size)])
                    else:
                        self.reduction_layers.extend([nn.Linear((j + 2) * args.encoder_embed_dim, args.encoder_embed_dim)
                                                      for j in range(args.cache_size)])
            self.cmb_mode = args.cmb_mode  # add; concat_linear
            self.cmb_reduction = None
            if not self.inc_cur and args.cmb_mode == 'concat_linear':
                self.cmb_reduction = nn.ModuleList([nn.Sequential(nn.Linear(2 * args.encoder_embed_dim, args.encoder_embed_dim)) for i in range(num_cache)])
            self.cache_residual = args.cache_residual
            if args.cache_norms:  # only use for cache residual
                self.cache_norms = nn.ModuleList([LayerNorm(embed_dim, args=args) for _ in range(num_cache)])
            else:
                self.cache_norms = None

        elif args.reslayer == 'latest_res':
            if args.cache_block_valia_attn:
                self.layers.extend(
                    [TransformerAttnEncoderLayer(layer_id=i, args=args, ) for i in range(args.encoder_layers)])
            else:
                self.layers.extend(
                [TransformerCacheEncoderLayer(layer_id=i, args=args, ) for i in range(args.encoder_layers)])
            # 1. 定义将之前层残差组合的方法
            self.reduction_layers = nn.ModuleList([])
            self.cache_size = args.cache_size
            self.reduction_ffn_mode = args.reduction_ffn_mode
            self.dense_activate=args.dense_activate
            if args.reduction_ffn_mode == 1:  # cat fc act fc
                if self.dense_activate == 1:
                    inner_act = nn.ReLU()
                elif self.dense_activate == 2:
                    inner_act = nn.Tanh()
                else:
                    inner_act = nn.Sequential()
            for i in range(args.cache_size-1):
                if args.reduction_ffn_mode == 1:
                    self.reduction_layers.append(nn.Sequential(nn.Linear((i+1) * args.encoder_embed_dim, args.encoder_embed_dim),
                                                                inner_act, nn.Linear(args.encoder_embed_dim,args.encoder_embed_dim)))
                elif args.reduction_ffn_mode == 2:  # fc_act_cat_fc similar to ffn
                    self.reduction_layers.append(fc_act_cat_fc(num=(i+1), args=args))
                else:  # cat fc act
                    self.reduction_layers.append(nn.Sequential(nn.Linear((i+1) * args.encoder_embed_dim,
                                                                         args.encoder_embed_dim), inner_act))
            for i in range(args.encoder_layers-args.cache_size+1):
                if args.reduction_ffn_mode == 1:
                    self.reduction_layers.append(nn.Sequential(nn.Linear(args.cache_size * args.encoder_embed_dim, args.encoder_embed_dim),
                                                                inner_act, nn.Linear(args.encoder_embed_dim,args.encoder_embed_dim)))
                elif args.reduction_ffn_mode == 2:  # fc_act_cat_fc similar to ffn
                    self.reduction_layers.append(fc_act_cat_fc(num=args.cache_size, args=args))
                else:  # cat fc act
                    self.reduction_layers.append(nn.Sequential(nn.Linear(args.cache_size * args.encoder_embed_dim,
                                                                         args.encoder_embed_dim), inner_act))

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim, args=args)

    def m_rnn(self, x, rnn, rnn_hidden):
        origin_bsz, x_len, dim = x.size()
        in_x = x.reshape(origin_bsz*x_len, dim)
        if self.reduction_rnn in ['rnn', 'gru']:
            hx = rnn(in_x, rnn_hidden)
            m_output = hx  # bsz, self.rnn_memory_size
            rnn_hidden = hx
        else:  # self.reduction_rnn == 'lstm':
            hx, cx = rnn(in_x, rnn_hidden)
            m_output = hx  # bsz, self.rnn_memory_size
            rnn_hidden = (hx, cx)
        return m_output, rnn_hidden

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        if self.reslayer == 'cache':
            cache = []
            rnn_hidden = None
            for i in range(len(self.layers)):
                # 0. 和原来一样计算layer(x)
                layer = self.layers[i]
                residual = x  # batch, src_len, embed_dim
                cache.append(residual)
                x = layer.maybe_layer_norm(layer.self_attn_layer_norm, x, before=True)
                x = layer(x, encoder_padding_mask)

                # 1.计算残差
                bsz, seq_len, dim = x.size()
                num_res = len(cache)
                if self.inc_cur:  # 把当前的输出一起考虑进来做类似concat linear attn 之类的 做linear可能比较好，但是attn肯能还是分开加起来比较好
                    # res_x = copy.deepcopy(cache)
                    res_x = [x for x in cache]
                    res_x.append(x)  # 问题在这里
                    res_x = torch.stack(res_x, dim=2)  # bsz, src_len ,i+2,embed_dim
                    if self.reduction_layers is not None:
                        res_x = torch.reshape(res_x, (bsz, seq_len, (num_res + 1) * dim))
                        reduction = self.reduction_layers[i]
                        x = reduction(res_x)
                    elif self.mha is not None:
                        res_x = res_x.reshape((bsz * seq_len, (num_res + 1), dim))
                        query = x.reshape((bsz * seq_len, dim)).unsqueeze(1)
                        reduction_res, _ = self.mha(key=res_x, value=res_x, query=query, mask=None)
                        x = torch.reshape(reduction_res, (bsz, seq_len, dim))
                else:
                    past_res = torch.stack(cache, dim=2)  # bsz, src_len ,i+1,embed_dim
                    if self.reduction_layers is not None:
                        reduction = self.reduction_layers[i]
                        past_res = torch.reshape(past_res, (bsz, seq_len, num_res * dim))
                        reduction_res = reduction(past_res)
                    elif self.mha is not None:
                        past_res = past_res.reshape((bsz * seq_len, num_res, dim))
                        query = x.reshape((bsz * seq_len, dim)).unsqueeze(1)
                        reduction_res, _ = self.mha(key=past_res, value=past_res, query=query, mask=None)
                        reduction_res = torch.reshape(reduction_res, (bsz, seq_len, dim))
                    elif self.rnn is not None:
                        reduction_res, rnn_hidden = self.m_rnn(residual, self.rnn, rnn_hidden)
                        reduction_res = torch.reshape(reduction_res, (bsz, seq_len, dim))
                    if self.cmb_mode == 'concat_linear':
                        x = self.cmb_reduction(torch.cat([reduction_res, x], -1))
                    else:
                        x += reduction_res
                x = layer.maybe_layer_norm(layer.self_attn_layer_norm, x, after=True)
        elif self.reslayer == 'cache_block':
            if self.caches_dense:  # cmb with output
                caches = []
            else:
                caches = None
            for i in range(self.num_cache):
                cache = []
                for j in range(self.cache_size):
                    # 0. 和原来一样计算layer(x)
                    layer = self.layers[i*self.cache_size+j]
                    residual = x  # batch, src_len, embed_dim
                    cache.append(residual)
                    x = layer.maybe_layer_norm(layer.self_attn_layer_norm, x, before=True)
                    x = layer(x, encoder_padding_mask)
                    # 1.计算残差
                    bsz, seq_len, dim = x.size()
                    num_res = len(cache)
                    past_res = torch.stack(cache, dim=2)  # bsz, src_len ,i+1,embed_dim
                    reduction = self.reduction_layers[i*self.cache_size+j]
                    past_res = torch.reshape(past_res, (bsz, seq_len, num_res * dim))
                    if not self.inc_cur:
                        reduction_res = reduction(past_res)
                        if self.dense_activate and self.reduction_ffn_mode == 0:
                            if self.dense_activate == 1:
                                reduction_res = F.relu(reduction_res)
                            elif self.dense_activate == 2:
                                reduction_res = F.tanh(reduction_res)
                        reduction_res = F.dropout(reduction_res, p=self.dense_dropout, training=self.training) # 2019/9/4
                        if self.cmb_mode == 'concat_linear':
                            cmb_reduction_i = self.cmb_reduction[i]
                            x = cmb_reduction_i(torch.cat([reduction_res, x], -1))
                        else:
                            x += reduction_res
                    else:
                        x = reduction(torch.cat((past_res, x), dim=-1))
                        x = F.dropout(x, p=self.dense_dropout, training=self.training)
                        if self.dense_activate and self.reduction_ffn_mode == 0:
                            if self.dense_activate == 1:
                                x = F.relu(x)
                            elif self.dense_activate == 2:
                                x = F.tanh(x)
                    x = layer.maybe_layer_norm(layer.self_attn_layer_norm, x, after=True) # 820 发现cache的都忘了加，所以其实是没有norm 怪不得，这些说不定可以试incur了
                if caches is not None:
                    caches.append(cache[0])
                if self.cache_residual:
                    x += cache[0]
                    if self.cache_norms:
                        cache_norm = self.cache_norms[i]
                        x = cache_norm(x)
            if caches is not None:
                dense_caches = self.caches_dense(torch.cat(caches,dim=-1))
                if self.caches_out_cmb:
                    x = self.caches_out_cmb(torch.cat((dense_caches, x), dim=-1))
                else:
                    x = x + dense_caches
        elif self.reslayer == 'res_flow':
            caches = []
            for i in range(len(self.layers)):
                # 0. 和原来一样计算layer(x)
                layer = self.layers[i]
                residual = x  # batch, src_len, embed_dim
                if self.cache_size != 0:
                    if i % self.cache_size == 0:
                        caches.append(residual)
                x = layer.maybe_layer_norm(layer.self_attn_layer_norm, x, before=True)
                x = layer(x, encoder_padding_mask)
                x = x + residual
                x = layer.maybe_layer_norm(layer.self_attn_layer_norm, x, after=True)
            if self.cache_size != 0:
                dense_caches = self.caches_dense(torch.cat(caches, dim=-1))
                x = x + dense_caches
        elif self.reslayer in ['valina_attn']:
            for i in range(len(self.layers)):
                # 0. 和原来一样计算layer(x)
                layer = self.layers[i]
                residual = x  # batch, src_len, embed_dim
                x = layer.maybe_layer_norm(layer.self_attn_layer_norm, x, before=True)
                x = layer(x, encoder_padding_mask)
                x = x + residual
                x = layer.maybe_layer_norm(layer.self_attn_layer_norm, x, after=True)
        elif self.reslayer == 'latest_res':
            caches = []
            for i in range(len(self.layers)):
                # 0. 和原来一样计算layer(x)
                layer = self.layers[i]
                residual = x  # batch, src_len, embed_dim
                caches.append(residual)
                x = layer.maybe_layer_norm(layer.self_attn_layer_norm, x, before=True)
                x = layer(x, encoder_padding_mask)
                reduction = self.reduction_layers[i]
                # print('i=',i,'len cache',len(caches),'reduct i',reduction)
                reduction_res = reduction(torch.cat(caches[-self.cache_size:], -1))
                x = x + residual + reduction_res
                x = layer.maybe_layer_norm(layer.self_attn_layer_norm, x, after=True)
        else:
            for layer in self.layers:
                x = layer(x, encoder_padding_mask)
            if self.out_linear:
                x = self.out_linear(x)
        if self.normalize:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, f"{name}.layers.{i}")

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerCombineEncoderLayer(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention820(
                    self.embed_dim, args.encoder_attention_heads, layer_id=layer_id, args=args,
                    dropout=args.attention_dropout, cur_attn_type='es'
                )

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, args=args)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
                activation=getattr(args, 'activation_fn', 'relu')
            )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim, layer_id=layer_id, cur_linear='fc1', args=args)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args)
        self.cmb_2hffn= args.cmb_2hffn if 'cmb_2hffn' in args else 0
        if self.cmb_2hffn:
            self.fc1 = Linear(self.embed_dim, int(args.encoder_ffn_embed_dim/2), layer_id=layer_id, cur_linear='fc1', args=args)
            self.fc2 = Linear(int(args.encoder_ffn_embed_dim / 2),self.embed_dim , layer_id=layer_id, cur_linear='fc2',
                              args=args)
            self.fc3 = Linear(self.embed_dim, int(args.encoder_ffn_embed_dim/2), layer_id=layer_id, cur_linear='fc3', args=args)
            self.fc4 = Linear(int(args.encoder_ffn_embed_dim / 2),self.embed_dim , layer_id=layer_id, cur_linear='fc4',
                              args=args)
        self.dropout = args.dropout
        if args.combine_linear:
            self.combine_linear = Linear(2*self.embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='combine_linearr', args=args)
        else:
            self.combine_linear = None
        # added in 9.1 cmb gate for add ffn and mha
        self.cmb_gate = args.cmb_gate if 'cmb_gate' in args else 0
        self.enc_ratio = args.enc_ratio if 'enc_ratio' in args else 0
        self.gate_linear = None
        if self.cmb_gate in [1,2]:
            self.cmb_gate_avgpos = args.cmb_gate_avgpos
            if args.cmb_gate_nonlinear:
                self.gate_linear = nn.Sequential(Linear(self.embed_dim, int(self.embed_dim/2)), nn.ReLU(),Linear(int(self.embed_dim/2),self.embed_dim))
            else:
                self.gate_linear = Linear(self.embed_dim, self.embed_dim)
        if self.cmb_gate == 2:
            self.gate_scalar_linear = Linear(self.embed_dim,1)
        else:
            self.gate_scalar_linear = None
        if self.cmb_gate == 3:
            self.gate_para = nn.Parameter(torch.Tensor(self.embed_dim))
            nn.init.uniform_(self.gate_para, -1/math.sqrt(self.embed_dim),1/math.sqrt(self.embed_dim))
        else:
            self.gate_para = None
        if self.cmb_gate == 4:  # only scalar
            self.point_gate = nn.Parameter(torch.Tensor([0.5]))
        else:
            self.point_gate = None
        self.info_linear = args.info_linear if 'info_linear' in args else 0
        if self.info_linear:
            self.enc_il_inc = args.enc_il_inc
            self.il_dropout = args.enc_il_dropout
            if not args.il_ratio:
                if args.enc_il_inc:
                    self.info_linear = Linear(3 * self.embed_dim, self.embed_dim, layer_id=layer_id,
                                              cur_linear='info_linear', args=args)
                else:
                    self.info_linear = Linear(2*self.embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='info_linear', args=args)
            else:
                if args.enc_il_inc:
                    self.info_linear = Linear(3 * self.embed_dim, int(self.embed_dim/args.il_ratio), layer_id=layer_id,
                                              cur_linear='info_linear', args=args)
                else:
                    self.info_linear = Linear(2*self.embed_dim, int(self.embed_dim/args.il_ratio), layer_id=layer_id, cur_linear='info_linear', args=args)
            self.il_relu = args.il_relu
            if self.il_relu > 0 and args.il_relu_linear:
                if not args.il_ratio:
                    self.il_relu_linear = Linear(self.embed_dim, self.embed_dim,args)
                else:
                    self.il_relu_linear = Linear(int(self.embed_dim/args.il_ratio), self.embed_dim, args)
            else:
                self.il_relu_linear = None
        else:
            self.info_linear = None
        self.linear_divide = args.linear_divide if 'linear_divide' in args else 0
        if self.linear_divide:
            self.linear_divide = Linear(self.embed_dim, 2*self.embed_dim, layer_id=layer_id,
                                         cur_linear='combine_linearr', args=args)
        else:
            self.linear_divide = None
        # self.self_attn_layer_norm = LayerNorm(self.embed_dim, args=args)
        if args.use_se:
            self.se = SE(args, self.embed_dim)
        else:
            self.se = None
        self.enc_tanh_list = args.enc_tanh_list if 'enc_tanh_list' in args else []
        self.input_dropout = args.input_dropout if 'input_dropout' in args else 0.0

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {
            '0': 'self_attn_layer_norm',
            '1': 'final_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = f'{name}.layer_norms.{old}.{m}'
                if k in state_dict:
                    state_dict[
                        f'{name}.{new}.{m}'
                    ] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if self.linear_divide:
            _, _, embed_dim = x.size()
            x1, x2 = torch.split(self.linear_divide(x), embed_dim, dim=2)
            x1, _ = self.self_attn(query=x1, key=x1, value=x1, key_padding_mask=encoder_padding_mask)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = F.dropout(
                self.fc2(F.dropout(self.activation_fn(self.fc1(x2)), p=self.activation_dropout, training=self.training)),
                p=self.dropout, training=self.training)
            x = x1 + x2 + residual
            x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
            return x
        x1 = F.dropout(x, p=self.input_dropout, training=self.training)
        x1, _ = self.self_attn(query=x1, key=x1, value=x1, key_padding_mask=encoder_padding_mask)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(
                self.fc2(F.dropout(self.activation_fn(self.fc1(x)), p=self.activation_dropout, training=self.training)),
                p=self.dropout, training=self.training)
        if self.cmb_2hffn:
            x2 = F.dropout(
                self.fc4(F.dropout(self.activation_fn(self.fc3(x2)), p=self.activation_dropout, training=self.training)),
                p=self.dropout, training=self.training)
        if self.se:
            x3 = F.dropout(self.se(x), p=self.dropout, training=self.training)
        else:
            x3 = None
        if self.combine_linear:
            x = self.combine_linear(torch.cat((x1, x2), -1))
        else:
            if self.cmb_gate:
                if self.point_gate is not None:
                    gate = self.point_gate.repeat([x.size(2)]).unsqueeze(0).repeat([x.size()[1], 1]).unsqueeze(0).repeat([x.size()[0], 1, 1])
                else:
                    if self.gate_para is not None:
                        gate = self.gate_para.unsqueeze(0).repeat([x.size()[1], 1]).unsqueeze(0).repeat([x.size()[0],1, 1])

                    else:
                        if self.cmb_gate_avgpos:
                            gate = self.gate_linear(torch.mean(x, dim=1))
                            if self.gate_scalar_linear:
                                gate = self.gate_scalar_linear(gate).squeeze()
                                gate = gate.unsqueeze(1).repeat([1, x.size()[1]]).unsqueeze(2).repeat([1, 1, x.size()[2]])
                            else:
                                gate = gate.unsqueeze(1).repeat([1, x.size()[1], 1])
                        else:
                            gate = self.gate_linear(x)
                            if self.gate_scalar_linear:
                                gate = self.gate_scalar_linear(gate).squeeze()
                                gate = gate.unsqueeze(2).repeat([1, 1, x.size()[2]])
                    gate = F.sigmoid(gate)  # added in 9/5
                cp_gate = torch.sub(torch.ones_like(x), gate)
                x = x1 * gate + x2 * cp_gate
            else:
                if self.enc_ratio > 0:
                    x = self.enc_ratio * (x1 + x2)
                else:
                    x = x1 + x2
            if self.info_linear:
                if self.enc_il_inc:
                    info = self.info_linear(torch.cat((residual, x1, x2), -1))
                else:
                    info = self.info_linear(torch.cat((x1, x2), -1))
                if self.il_relu > 0:
                    if self.il_relu == 1:
                        info = F.relu(info)
                    elif self.il_relu == 2:
                        info = F.leaky_relu(info)
                    elif self.il_relu == 3:
                        info = F.tanh(info)
                    if self.il_relu_linear:
                        info = self.il_relu_linear(info)
                info = F.dropout(info, p=self.il_dropout, training=self.training)
                x += info
        if x3 is not None:
            x += x3
        x += residual
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


class TransformerDense1EncoderLayer(TransformerCombineEncoderLayer):
    # 这里 norm所有层的concat 进行, 尽可能和densenet以及原始的combine都相似
    def __init__(self, layer_id, args,  in_planes):
        super(TransformerDense1EncoderLayer, self).__init__(layer_id=layer_id,args=args)
        self.reduction = nn.Linear(in_planes, args.encoder_embed_dim)
        self.self_attn_layer_norm_before = LayerNorm(in_planes, args=args)
        self.self_attn_layer_norm_after = LayerNorm(in_planes+args.encoder_embed_dim, args=args)
        self.self_attn_layer_norm = None

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm_before, x, before=True)
        x = self.reduction(x)
        x1, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        if 'self_mha' in self.enc_tanh_list:
            x1 = F.tanh(x1)
        # x2 我先不做tanh了，反正也没用
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(
            self.fc2(F.dropout(self.activation_fn(self.fc1(x)), p=self.activation_dropout, training=self.training)),
            p=self.dropout, training=self.training)
        x = x1 + x2
        x = torch.cat([residual, x], -1)
        x = self.maybe_layer_norm(self.self_attn_layer_norm_after, x, after=True)
        return x


class TransformerCacheEncoderLayer(TransformerCombineEncoderLayer):
    # 这里 将之前的 residual 都存在List 当做 Layer memory/cache  先做一个dense net 版本 在做一个attn版本
    # 将所有对残差连接和ln的操作都放在模型外面
    def __init__(self, layer_id, args, ):
        super(TransformerCacheEncoderLayer, self).__init__(layer_id=layer_id, args=args)

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            cache: residuals of past layers  type: list of tensor x
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        x1, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(
            self.fc2(F.dropout(self.activation_fn(self.fc1(x)), p=self.activation_dropout, training=self.training)),
            p=self.dropout, training=self.training)
        x = x1 + x2
        return x


class TransformerAttnEncoderLayer(TransformerCacheEncoderLayer):
    # 这里 将之前的 residual 都存在List 当做 Layer memory/cache  先做一个dense net 版本 在做一个attn版本
    # 将所有对残差连接和ln的操作都放在模型外面
    def __init__(self, layer_id, args, ):
        super(TransformerAttnEncoderLayer, self).__init__(layer_id=layer_id, args=args)
        self.fc1 = None
        self.fc2 = None

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            cache: residuals of past layers  type: list of tensor x
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        x1, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        return x1


class TransformerCombineDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
        final_norm (bool, optional): apply layer norm to the output of the
            final decoder layer (default: True).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, layer_id=0, args=args, cur_linear='in',bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        if args.reslayer == 'mixconv':
            self.layers.extend([
                TransformerCombineMixconvDecoderLayer(layer_id=i, args=args, no_encoder_attn=no_encoder_attn)
                for i in range(args.decoder_layers)
            ])
        else:
            self.layers.extend([
                TransformerCombineDecoderLayer(layer_id=i, args=args, no_encoder_attn=no_encoder_attn)
                for i in range(args.decoder_layers)
            ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, layer_id=args.decoder_layers-1, args=args,cur_linear='out',  bias=False) \
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
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim, args=args)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state)
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
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
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.normalize:
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
        if utils.item(state_dict.get('{}.version'.format(name), torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['{}.version'.format(name)] = torch.Tensor([1])

        return state_dict


class TransformerCombineDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, layer_id, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention820(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            layer_id=layer_id,
            args=args,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            cur_attn_type='ds'
        )
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, 'char_inputs', False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export, args=args)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention820(
                self.embed_dim, args.decoder_attention_heads,
                layer_id=layer_id,
                args=args,
                dropout=args.attention_dropout,
                cur_attn_type='dc',
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export, args=args)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim, layer_id=layer_id, args=args, cur_linear='fc1' )
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim, layer_id=layer_id, args=args, cur_linear='fc2')
        self.cmb_2hffn = args.cmb_2hffn if 'cmb_2hffn' in args else 0
        if self.cmb_2hffn:
            self.fc1 = Linear(self.embed_dim, int(args.encoder_ffn_embed_dim/2), layer_id=layer_id, cur_linear='fc1', args=args)
            self.fc2 = Linear(int(args.encoder_ffn_embed_dim / 2),self.embed_dim , layer_id=layer_id, cur_linear='fc2',
                              args=args)
            self.fc3 = Linear(self.embed_dim, int(args.encoder_ffn_embed_dim/2), layer_id=layer_id, cur_linear='fc3', args=args)
            self.fc4 = Linear(int(args.encoder_ffn_embed_dim / 2),self.embed_dim , layer_id=layer_id, cur_linear='fc4',
                              args=args)
        self.dec_info_linear = args.dec_info_linear if 'dec_info_linear' in args else 0
        if self.dec_info_linear:
            self.il_dropout = args.dec_il_dropout
            self.dec_il_list = args.dec_il_list
            self.info_linear = Linear(len(args.dec_il_list) * self.embed_dim, self.embed_dim, layer_id=layer_id,
                                      cur_linear='info_linear', args=args)
            self.il_relu = args.il_relu
            if self.il_relu and args.il_relu_linear:
                self.il_relu_linear = Linear(self.embed_dim,self.embed_dim,args)
            else:
                self.il_relu_linear = None
        else:
            self.info_linear = None
        self.dec_tanh_list = args.dec_tanh_list if 'dec_tanh_list' in args else []
        # self.decr = args.decr if 'decr' in args else 0
        self.dec_ratio = args.dec_ratio if 'dec_ratio' in args else 0
        self.attn_ratio= args.attn_ratio if 'attn_ratio' in args else 1
        self.dec_nffn = args.dec_nffn if 'dec_nffn' in args else 0
        self.need_attn = True
        self.onnx_trace = False

        self.se_list = args.se_list if 'se_list' in args else []
        self.se_para = args.se_para if 'se_para' in args else 0
        self.ffn_se = None
        self.self_mha_se = None
        self.cmb_se = None
        self.context_mha_se = None
        if self.se_para:
            if 'ffn' in self.se_list:
                self.ffn_se = ParaScore(args, self.embed_dim)
            if 'self_mha' in self.se_list:
                self.self_mha_se = ParaScore(args, self.embed_dim)
            if 'context_mha' in self.se_list:
                self.context_mha_se = ParaScore(args, self.embed_dim)
            if 'cmb' in self.se_list:
                self.cmb_se = ParaScore(args, self.embed_dim)
        else:
            if 'ffn' in self.se_list:
                self.ffn_se = SEScore(args, self.embed_dim)
            if 'self_mha' in self.se_list:
                self.self_mha_se = SEScore(args, self.embed_dim)
            if 'context_mha' in self.se_list:
                self.context_mha_se = SEScore(args, self.embed_dim)
            if 'cmb' in self.se_list:
                self.cmb_se = SEScore(args, self.embed_dim)
        self.se_on_x = args.se_on_x if 'se_on_x' in args else 0
        self.input_dropout = args.input_dropout if 'input_dropout' in args else 0


    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, x, encoder_out=None, encoder_padding_mask=None, incremental_state=None, prev_self_attn_state=None,
                prev_attn_state=None, self_attn_mask=None, self_attn_padding_mask=None,):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        # (1) dec_attn
        if prev_self_attn_state is not None:
            if incremental_state is None:
                    incremental_state = {}
            prev_key1, prev_value1 = prev_self_attn_state
            saved_state = {"prev_key": prev_key1, "prev_value": prev_value1}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x1 = F.dropout(x, p=self.input_dropout, training=self.training)
        x1, attn = self.self_attn(
                query=x1,
                key=x1,
                value=x1,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
        )
        if 'self_mha' in self.dec_tanh_list:
            x1 = F.tanh(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        # (2) enc_dec attn
        if self.encoder_attn is not None:
            if prev_attn_state is not None:
                if incremental_state is None:
                        incremental_state = {}
                prev_key2, prev_value2 = prev_attn_state
                saved_state = {"prev_key": prev_key2, "prev_value": prev_value2}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x2, attn = self.encoder_attn(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=(not self.training and self.need_attn),
            )
            if 'context_mha' in self.dec_tanh_list:
                x2 = F.tanh(x2)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
        # (3) ffn
        x3 = self.fc2(F.dropout(self.activation_fn(self.fc1(x)), p=self.activation_dropout, training=self.training))
        if 'ffn' in self.dec_tanh_list:
            x3 = F.tanh(x3)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

        if self.cmb_2hffn:
            x3 = F.dropout(
                self.fc4(F.dropout(self.activation_fn(self.fc3(x3)), p=self.activation_dropout, training=self.training)),
                p=self.dropout, training=self.training)
        if self.dec_nffn:
            x3 = 0
        if self.dec_ratio > 0:
            if self.dec_ratio > 1 and self.dec_ratio < 2 :
                x = (self.dec_ratio - 1 ) * (residual + self.attn_ratio*(x1 + x2) + x3)
            elif self.dec_ratio == 2:
                x = residual + 0.25 * x1 + 0.25 * x2 + 0.5 * x3
            else:
                x = residual + self.dec_ratio * (self.attn_ratio*(x1 + x2) +
                                                 x3)
        else:
            self_mha_se_score = 1
            context_mha_se_score = 1
            ffn_se_score = 1
            if self.self_mha_se:
                if self.se_on_x == 1:
                    self_mha_se_score = self.self_mha_se(x)
                else:
                    self_mha_se_score = self.self_mha_se(x1)
            if self.context_mha_se:
                if self.se_on_x:
                    context_mha_se_score = self.context_mha_se(x)
                else:
                    context_mha_se_score = self.context_mha_se(x2)
            if self.ffn_se:
                if self.se_on_x:
                    ffn_se_score = self.ffn_se(x)
                else:
                    ffn_se_score = self.ffn_se(x3)
            if self.cmb_se:
                if self.se_on_x:
                    ffn_se_score = self.cmb_se(x)
                    self_mha_se_score = ffn_se_score
                    context_mha_se_score = ffn_se_score
                else:
                    ffn_se_score = self.cmb_se(x1 + x2 + x3)
                    self_mha_se_score = ffn_se_score
                    context_mha_se_score = ffn_se_score
            x = residual + self_mha_se_score * x1 + context_mha_se_score * x2 + ffn_se_score * x3
        if self.dec_info_linear:
            info_list = []
            if 'residual' in self.dec_il_list:
                info_list.append(residual)
            if 'self_mha' in self.dec_il_list:
                info_list.append(x1)
            if 'context_mha' in self.dec_il_list:
                info_list.append(x2)
            if 'ffn' in self.dec_il_list:
                info_list.append(x3)
            info = self.info_linear(torch.cat(info_list, -1))
            if self.il_relu:
                info = F.relu(info)
                if self.il_relu_linear:
                    info = self.il_relu_linear(info)
            info = F.dropout(info, p=self.il_dropout, training=self.training)
            x += info
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


class SE(nn.Module):
    def __init__(self, args, embed_dim):
        super().__init__()
        self.se_ratio = args.se_ratio
        self.se_inner_dim = int(embed_dim/self.se_ratio)
        self.se_activation = args.se_activation
        self.se_res = args.se_res
        self.se_scale = args.se_scale
        self.se_head = args.se_head
        self.se_head_type = args.se_head_type
        self.se_avg_type = args.se_avg_type
        self.se_softmax_group = args.se_softmax_group
        # self.se_vl =args.se_vl

        if args.se_inner_ln:
            self.layer_norm = LayerNorm(self.se_inner_dim, args=args)
        else:
            self.layer_norm = None
        self.inner_relu = nn.ReLU()

        if self.se_activation == 'sigmoid':
            self.se_active = nn.Sigmoid()
        elif self.se_activation == 'softmax':
            if args.se_softmax_t != 0:
                self.se_active = softmax_t(args.se_softmax_t)
            else:
                self.se_active = nn.Softmax()
        else:
            self.se_active = None
        if self.se_active is None:
            self.se_head = 0
        self.se_head_dim = int(embed_dim / self.se_head) if self.se_head else None
        self.fc1 = nn.Linear(self.se_head, self.se_inner_dim) if self.se_head else nn.Linear(embed_dim, self.se_inner_dim)
        self.fc2 = nn.Linear(self.se_inner_dim, self.se_head) if self.se_head else nn.Linear(self.se_inner_dim, embed_dim)
        if args.se_vl:
            self.fc3 = nn.Linear(embed_dim, embed_dim)
        else:
            self.fc3 = None
        if self.se_avg_type == 1:
            self.context_attn_linear = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                                     nn.Linear(embed_dim, 1))
        elif self.se_avg_type == 3:
            self.context_attn_linear = nn.Sequential(nn.Linear(embed_dim, 1))
        else:
            self.context_attn_linear = None

    def forward(self, x):
        # tgt_len, bsz, embed_dim
        tgt_len, bsz, embed_dim = x.size()
        if self.se_avg_type == 4:
            mean_x = x
        elif self.se_avg_type == 2:  # average through time
            assert self.se_head == 0
            mean_x = torch.stack([torch.mean(x[:(i+1), :, :], dim=0) for i in range(tgt_len)])  # 写错了，不应该是i+1 而应该是i,还是应该是i+1
            # i+1就会影响teacher forcing,i则会nan, 虽然i+1 不可能影响teacher forcing啊
        elif self.se_avg_type in [1, 3]:
            mean_x = torch.mean(x * F.softmax(self.context_attn_linear(x).squeeze(2), dim=0).unsqueeze(2).expand_as(x), dim=0)  # 1 attn,3 simplified attn
        else:
            mean_x = torch.mean(x, dim=0)  # bsz, dim
        if self.se_head_type == 0 and self.se_head:
            mean_x = torch.mean(mean_x.reshape(bsz, self.se_head, self.se_head_dim), dim=2)  # bsz, se_head
        gc_x = mean_x
        if self.se_head_type == 1 and self.se_head:
            gc_x = torch.reshape(gc_x, (bsz, self.se_head, self.se_head_dim)).permute(0, 2, 1)  # bsz, se_head_dim, se_head
        gc_x = self.fc1(gc_x)
        if self.layer_norm:
            gc_x = self.layer_norm(gc_x)
        gc_x = self.fc2(self.inner_relu(gc_x))
        if self.se_head_type == 1 and self.se_head:
            gc_x = torch.mean(gc_x, dim=1)  # bsz,  se_head
        if self.se_active:
            if self.se_softmax_group:
                assert self.se_head == 0 and self.se_avg_type not in [2, 4]
                gc_x = self.se_active(torch.mean(torch.reshape(gc_x, [bsz,  torch.div(embed_dim, self.se_softmax_group), self.se_softmax_group]),dim=2))
            else:
                gc_x = self.se_active(gc_x)
        if self.se_head:
            gc_x = gc_x.unsqueeze(2).repeat([1, 1, self.se_head_dim]).reshape((bsz, embed_dim)).unsqueeze(0).repeat([tgt_len,1, 1, ])
        else:
            if self.se_avg_type in [2, 4]:
                gc_x = gc_x
            else:
                if self.se_softmax_group:
                    gc_x = gc_x.unsqueeze(2).repeat([1, 1, self.se_softmax_group]).reshape((bsz, embed_dim)).unsqueeze(0).repeat([tgt_len, 1, 1, ])
                    # print('to gcx unsqueeze gc_X shape',gc_x.size())
                else:
                    gc_x = torch.unsqueeze(gc_x, 0).expand_as(x)
        if self.se_active:
            if self.fc3:
                x = self.fc3(x)
            x = x*gc_x
        else:
            x = gc_x
        # 不乘而直接相加会很垃圾
        return x*self.se_scale



class TransformerCombineBaseEncoderLayer(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention820(
                    self.embed_dim, args.encoder_attention_heads, layer_id=layer_id, args=args,
                    dropout=args.attention_dropout, cur_attn_type='es'
                )

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, args=args)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
                activation=getattr(args, 'activation_fn', 'relu')
            )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim, layer_id=layer_id, cur_linear='fc1', args=args)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args)
        self.dropout = args.dropout

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {
            '0': 'self_attn_layer_norm',
            '1': 'final_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = f'{name}.layer_norms.{old}.{m}'
                if k in state_dict:
                    state_dict[
                        f'{name}.{new}.{m}'
                    ] = state_dict[k]
                    del state_dict[k]

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x1, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(
                self.fc2(F.dropout(self.activation_fn(self.fc1(x)), p=self.activation_dropout, training=self.training)),
                p=self.dropout, training=self.training)
        x = x1 + x2
        x += residual
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        return x


class TransformerCombine2LEncoderLayer(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention820(
                    self.embed_dim, args.encoder_attention_heads, layer_id=layer_id, args=args,
                    dropout=args.attention_dropout, cur_attn_type='es'
                )
        self.self_attn2 = MultiheadAttention820(
            self.embed_dim, args.encoder_attention_heads, layer_id=layer_id, args=args,
            dropout=args.attention_dropout, cur_attn_type='es'
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, args=args)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
                activation=getattr(args, 'activation_fn', 'relu')
            )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim, layer_id=layer_id, cur_linear='fc1', args=args)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args)
        self.fc3 = Linear(self.embed_dim, args.encoder_ffn_embed_dim, layer_id=layer_id, cur_linear='fc1', args=args)
        self.fc4 = Linear(args.encoder_ffn_embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args)
        self.dropout = args.dropout

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {
            '0': 'self_attn_layer_norm',
            '1': 'final_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = f'{name}.layer_norms.{old}.{m}'
                if k in state_dict:
                    state_dict[
                        f'{name}.{new}.{m}'
                    ] = state_dict[k]
                    del state_dict[k]

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x1, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(
                self.fc2(F.dropout(self.activation_fn(self.fc1(x)), p=self.activation_dropout, training=self.training)),
                p=self.dropout, training=self.training)
        x3, _ = self.self_attn2(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x4 = F.dropout(
                self.fc4(F.dropout(self.activation_fn(self.fc3(x)), p=self.activation_dropout, training=self.training)),
                p=self.dropout, training=self.training)
        x = x1 + x2 + x3 + x4
        x += residual
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        return x


class TransformerCombineMixconvEncoderLayer(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, args=args)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
                activation=getattr(args, 'activation_fn', 'relu')
            )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.encoder_normalize_before
        kernel_size = args.kernel_size if 'kernel_size' in args else 3
        self.dropout = args.dropout
        self.enc_ratio = args.enc_ratio if 'enc_ratio' in args else 0
        self.attn_ratio = args.attn_ratio if 'attn_ratio' in args else 1

        if args.attn1 == 'mha':
            self.attn1 = MHA(self.embed_dim, args.encoder_attention_heads, layer_id=layer_id, args=args,
                             dropout=args.attention_dropout, cur_attn_type='es')
        elif args.attn1 == 'origin':
            self.attn1 = MultiheadAttention820(self.embed_dim, args.encoder_attention_heads, layer_id=layer_id, args=args,
                                               dropout=args.attention_dropout, cur_attn_type='es')
        if args.attn2 == 'origin':
            self.attn2 = MultiheadAttention820(self.embed_dim, args.encoder_attention_heads, layer_id=layer_id, args=args,
                                               dropout=args.attention_dropout, cur_attn_type='es')
        elif args.attn2 == 'none':
            self.attn2 = MultiheadAttention820(self.embed_dim, args.encoder_attention_heads, layer_id=layer_id,
                                               args=args,
                                               dropout=args.attention_dropout, cur_attn_type='es')

        if args.conv2 == 'none':
            self.conv2 = None
        elif args.conv2 == 'fc':
            self.conv2 = Linear(in_features=self.embed_dim,out_features=self.embed_dim,args=args)
        elif args.conv2 == 'ffn':
            self.conv2 = nn.Sequential(Linear(self.embed_dim, args.encoder_ffn_embed_dim, layer_id=layer_id, cur_linear='fc1', args=args),
                                       nn.ReLU(),
                                       nn.Dropout(p=self.activation_dropout),
                                       Linear(args.encoder_ffn_embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args),
                                       nn.Dropout(p=self.dropout))
        elif args.conv2 == 'bigger_ffn':
            self.conv2 = nn.Sequential(Linear(self.embed_dim, int(1.5 * args.encoder_ffn_embed_dim), layer_id=layer_id, cur_linear='fc1', args=args),
                                       nn.ReLU(),
                                       nn.Dropout(p=self.activation_dropout),
                                       Linear(int(1.5 * args.encoder_ffn_embed_dim), self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args),
                                       nn.Dropout(p=self.dropout))
        elif args.conv2 == '2ffn':
            self.conv2 = nn.Sequential(
                Linear(self.embed_dim, int(2 * args.encoder_ffn_embed_dim), layer_id=layer_id, cur_linear='fc1',
                       args=args),
                nn.ReLU(),
                nn.Dropout(p=self.activation_dropout),
                Linear(int(2 * args.encoder_ffn_embed_dim), self.embed_dim, layer_id=layer_id, cur_linear='fc2',
                       args=args),
                nn.Dropout(p=self.dropout))
        elif args.conv2 == 'hffn':
            self.conv2 = nn.Sequential(Linear(self.embed_dim, int(args.encoder_ffn_embed_dim/2), layer_id=layer_id, cur_linear='fc1', args=args),
                                       nn.ReLU(),
                                       nn.Dropout(p=self.activation_dropout),
                                       Linear(int(args.encoder_ffn_embed_dim/2), self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args),
                                       nn.Dropout(p=self.dropout))
        elif args.conv2 == 'ffn_ln':
            self.conv2 = nn.Sequential(Linear(self.embed_dim, args.encoder_ffn_embed_dim, layer_id=layer_id, cur_linear= 'fc1', args=args),
                                       LayerNorm(args.encoder_ffn_embed_dim,args=args),
                                       nn.ReLU(),
                                       nn.Dropout(p=self.activation_dropout),
                                       Linear(args.encoder_ffn_embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args),
                                       nn.Dropout(p=self.dropout))
        elif args.conv2 == 'ffn_tanh':
            self.conv2 = nn.Sequential(
                Linear(self.embed_dim, args.encoder_ffn_embed_dim, layer_id=layer_id, cur_linear='fc1', args=args),
                nn.Tanh(),
                nn.Dropout(p=self.activation_dropout),
                Linear(args.encoder_ffn_embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args),
                nn.Dropout(p=self.dropout))
        elif args.conv2 == 'ffn_relu_tanh':
            self.conv2 = nn.Sequential(
                Linear(self.embed_dim, args.encoder_ffn_embed_dim, layer_id=layer_id, cur_linear='fc1', args=args),
                nn.ReLU(),
                nn.Dropout(p=self.activation_dropout),
                Linear(args.encoder_ffn_embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args),
                nn.Tanh(),
                nn.Dropout(p=self.dropout))
        elif args.conv2 == 'conv':
            self.conv2 = Conv(args, self.embed_dim, kernel_size)
        elif args.conv2 == 'conv_after':
            self.conv2= ConvAfter(args, self.embed_dim,kernel_size)
        elif args.conv2 == 'dynamic_conv':
            self.conv2 = DynamicConv(args, self.embed_dim, dynamic_type=0, cur_attn_type='es', layer_id=layer_id)
        elif args.conv2 == 'add_dynamic_conv':
            self.conv2 = DynamicConv(args, self.embed_dim, dynamic_type=1,cur_attn_type='es', layer_id=layer_id)
        elif args.conv2 == 'cat_dynamic_conv':
            self.conv2 = DynamicConv(args, self.embed_dim, dynamic_type=2, cur_attn_type='es', layer_id=layer_id)
        elif args.conv2 == 'cat_relu_dynamic_conv':
            self.conv2 = DynamicConv(args, self.embed_dim, dynamic_type=3, cur_attn_type='es', layer_id=layer_id)
        elif args.conv2 == 'ffn_small_noact':
            self.conv2 = nn.Sequential(
                Linear(self.embed_dim, int(args.encoder_ffn_embed_dim / 4), layer_id=layer_id, cur_linear='fc1',
                       args=args),
                nn.Dropout(p=self.activation_dropout),
                Linear(int(args.encoder_ffn_embed_dim / 4), self.embed_dim, layer_id=layer_id, cur_linear='fc2',
                       args=args),
                nn.Dropout(p=self.dropout))
        elif args.conv2 == 'ffn_small':
            self.conv2 = nn.Sequential(
                Linear(self.embed_dim, int(args.encoder_ffn_embed_dim / 4), layer_id=layer_id, cur_linear='fc1',
                       args=args),
                nn.ReLU(),
                nn.Dropout(p=self.activation_dropout),
                Linear(int(args.encoder_ffn_embed_dim / 4), self.embed_dim, layer_id=layer_id, cur_linear='fc2',
                       args=args),
                nn.Dropout(p=self.dropout))
        if args.conv3 == 'none':
            self.conv3 = None
        elif args.conv3 == 'ffn':
            self.conv3 = nn.Sequential(Linear(self.embed_dim, args.encoder_ffn_embed_dim, layer_id=layer_id, cur_linear='fc1', args=args),
                                       nn.ReLU(),
                                       nn.Dropout(p=self.activation_dropout),
                                       Linear(args.encoder_ffn_embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args),
                                       nn.Dropout(p=self.dropout))
        if args.info_linear:
            self.enc_il_inc = args.enc_il_inc
            self.il_dropout = args.enc_il_dropout
            if not args.il_ratio:
                if args.enc_il_inc:
                    self.info_linear = Linear(3 * self.embed_dim, self.embed_dim, layer_id=layer_id,
                                              cur_linear='info_linear', args=args)
                else:
                    self.info_linear = Linear(2*self.embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='info_linear', args=args)
            else:
                if args.enc_il_inc:
                    self.info_linear = Linear(3 * self.embed_dim, int(self.embed_dim/args.il_ratio), layer_id=layer_id,
                                              cur_linear='info_linear', args=args)
                else:
                    self.info_linear = Linear(2*self.embed_dim, int(self.embed_dim/args.il_ratio), layer_id=layer_id, cur_linear='info_linear', args=args)
            self.il_relu = args.il_relu
            if self.il_relu > 0 and args.il_relu_linear:
                self.il_2act = args.il_2act if 'il_2act' in args else 0
                if not args.il_ratio:
                    self.il_relu_linear = Linear(self.embed_dim, self.embed_dim,args)
                else:
                    self.il_relu_linear = Linear(int(self.embed_dim/args.il_ratio), self.embed_dim, args)
            else:
                self.il_relu_linear = None
        else:
            self.info_linear = None
        self.se_list = args.se_list if 'se_list' in args else []
        self.se_para = args.se_para if 'se_para' in args else 0
        self.ffn_se = None
        self.self_mha_se = None
        self.cmb_se = None
        if self.se_para:
            if 'ffn' in self.se_list:
                self.ffn_se = ParaScore(args, self.embed_dim)
            if 'self_mha' in self.se_list:
                self.self_mha_se = ParaScore(args, self.embed_dim)
            if 'cmb' in self.se_list:
                self.cmb_se = ParaScore(args, self.embed_dim)
        else:
            if 'ffn' in self.se_list:
                self.ffn_se = SEScore(args, self.embed_dim)
            if 'self_mha' in self.se_list:
                self.self_mha_se = SEScore(args, self.embed_dim)
            if 'cmb' in self.se_list:
                self.cmb_se = SEScore(args, self.embed_dim)
        self.se_on_x = args.se_on_x if 'se_on_x' in args else 0

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {
            '0': 'self_attn_layer_norm',
            '1': 'final_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = f'{name}.layer_norms.{old}.{m}'
                if k in state_dict:
                    state_dict[
                        f'{name}.{new}.{m}'
                    ] = state_dict[k]
                    del state_dict[k]

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x1, _ = self.attn1(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        if self.attn2:
            x2, _ = self.attn2(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
        else:
            x2 = 0
        if self.conv2 is not None:
            if self.conv2 in ['dynamic_conv', 'add_dynamic_conv', 'cat_dynamic_conv', 'cat_relu_dynamic_conv']:
                x3 = self.conv2(x, encoder_padding_mask=encoder_padding_mask, incremental_state=None)
            else:
                x3 = self.conv2(x)
        else:
            x3 = 0
        if self.conv3:
            x4 = self.conv3(x)
        else:
            x4 = 0
        if self.info_linear:
            if self.enc_il_inc:
                info = self.info_linear(torch.cat((residual, x1, x2), -1))
            else:
                info = self.info_linear(torch.cat((x1, x2), -1))
            if self.il_relu > 0:
                if self.il_relu == 1:
                    info = F.relu(info)
                elif self.il_relu == 2:
                    info = F.leaky_relu(info)
                elif self.il_relu == 3:
                    info = F.tanh(info)
                if self.il_relu_linear:
                    info = self.il_relu_linear(info)
                if self.il_relu_linear and self.il_2act:
                    if self.il_2act ==1:
                        info = F.relu(info)
                    elif self.il_2act == 3:
                        info = F.tanh(info)
            info = F.dropout(info, p=self.il_dropout, training=self.training)
        else:
            info = 0
        if self.enc_ratio > 1 and self.enc_ratio < 2:
            x = (self.enc_ratio - 1) * (residual + self.attn_ratio * x1 + x2 + x2 + x4)
        elif self.enc_ratio > 0:
            x = self.enc_ratio * (self.attn_ratio * x1 + x2 + x3 + x4 + info) + residual
        else:
            self_mha_se_score = 1
            ffn_se_score = 1
            if self.self_mha_se:
                if self.se_on_x:
                    self_mha_se_score = self.self_mha_se(x)
                else:
                    self_mha_se_score = self.self_mha_se(x1)
            if self.ffn_se:
                if self.se_on_x:
                    ffn_se_score = self.ffn_se(x)
                else:
                    ffn_se_score = self.ffn_se(x2)
            if self.cmb_se:
                if self.se_on_x:
                    ffn_se_score = self.cmb_se(x)
                    self_mha_se_score = ffn_se_score
                else:
                    ffn_se_score = self.cmb_se(x1+x2)
                    self_mha_se_score = ffn_se_score
            x = self_mha_se_score*x1 + x2 + ffn_se_score*x3 + x4 + residual + info
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        return x


class TransformerCombineMixconvDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, layer_id, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        if args.attn1 == 'mha':
            self.self_attn = MHA(
                embed_dim=self.embed_dim,
                num_heads=args.decoder_attention_heads,
                layer_id=layer_id,
                args=args,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                cur_attn_type='ds'
            )
        elif args.attn1 == 'origin':
            self.self_attn = MultiheadAttention820(
                embed_dim=self.embed_dim,
                num_heads=args.decoder_attention_heads,
                layer_id=layer_id,
                args=args,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                cur_attn_type='ds'
            )
        if args.attn2 == 'none':
            self.attn2 = None
        elif args.attn2 == 'origin':
            self.attn2 = MultiheadAttention820(
                embed_dim=self.embed_dim,
                num_heads=args.decoder_attention_heads,
                layer_id=layer_id,
                args=args,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                cur_attn_type='ds'
            )
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, 'char_inputs', False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export, args=args)

        if no_encoder_attn:
            self.encoder_attn = None
            # self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention820(
                self.embed_dim, args.decoder_attention_heads,
                layer_id=layer_id,
                args=args,
                dropout=args.attention_dropout,
                cur_attn_type='dc',
            )
            # self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export, args=args)

        if args.conv2 == 'none':
            self.conv2 = None
        elif args.conv2 == 'fc':
            self.conv2 = Linear(in_features=self.embed_dim,out_features=self.embed_dim)
        elif args.conv2 == 'ffn':
            self.conv2 = nn.Sequential(Linear(self.embed_dim, args.encoder_ffn_embed_dim, layer_id=layer_id, cur_linear='fc1', args=args),
                                       nn.ReLU(),
                                       nn.Dropout(p=self.activation_dropout),
                                       Linear(args.encoder_ffn_embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args),
                                       nn.Dropout(p=self.dropout))
        elif args.conv2 == 'bigger_ffn':
            self.conv2 = nn.Sequential(Linear(self.embed_dim, int(1.5 * args.encoder_ffn_embed_dim), layer_id=layer_id, cur_linear='fc1', args=args),
                                       nn.ReLU(),
                                       nn.Dropout(p=self.activation_dropout),
                                       Linear(int(1.5 * args.encoder_ffn_embed_dim), self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args),
                                       nn.Dropout(p=self.dropout))
        elif args.conv2 == '2ffn':
            self.conv2 = nn.Sequential(
                Linear(self.embed_dim, int(2 * args.encoder_ffn_embed_dim), layer_id=layer_id, cur_linear='fc1',
                       args=args),
                nn.ReLU(),
                nn.Dropout(p=self.activation_dropout),
                Linear(int(2 * args.encoder_ffn_embed_dim), self.embed_dim, layer_id=layer_id, cur_linear='fc2',
                       args=args),
                nn.Dropout(p=self.dropout))
        elif args.conv2 == 'hffn':
            self.conv2 = nn.Sequential(Linear(self.embed_dim, int(args.encoder_ffn_embed_dim/2), layer_id=layer_id, cur_linear='fc1', args=args),
                                       nn.ReLU(),
                                       nn.Dropout(p=self.activation_dropout),
                                       Linear(int(args.encoder_ffn_embed_dim/2), self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args),
                                       nn.Dropout(p=self.dropout))
        elif args.conv2 == 'ffn_relu_tanh':
            self.conv2 = nn.Sequential(
                Linear(self.embed_dim, args.encoder_ffn_embed_dim, layer_id=layer_id, cur_linear='fc1', args=args),
                nn.ReLU(),
                nn.Dropout(p=self.activation_dropout),
                Linear(args.encoder_ffn_embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args),
                nn.Tanh(),
                nn.Dropout(p=self.dropout))

        elif args.conv2 == 'dynamic_conv':
            self.conv2 = DynamicConv(args, self.embed_dim, dynamic_type=0, cur_attn_type='ds',layer_id=layer_id)
        elif args.conv2 == 'add_dynamic_conv':
            self.conv2 = DynamicConv(args, self.embed_dim, dynamic_type=1, cur_attn_type='ds',layer_id=layer_id)
        elif args.conv2 == 'cat_dynamic_conv':
            self.conv2 = DynamicConv(args, self.embed_dim, dynamic_type=2, cur_attn_type='ds',layer_id=layer_id)
        elif args.conv2 == 'cat_relu_dynamic_conv':
            self.conv2 = DynamicConv(args, self.embed_dim, dynamic_type=3, cur_attn_type='ds',layer_id=layer_id)
        elif args.conv2 == 'ffn_small':
            self.conv2 = nn.Sequential(
                Linear(self.embed_dim, int(args.encoder_ffn_embed_dim / 4), layer_id=layer_id, cur_linear='fc1',
                       args=args),
                nn.ReLU(),
                nn.Dropout(p=self.activation_dropout),
                Linear(int(args.encoder_ffn_embed_dim / 4), self.embed_dim, layer_id=layer_id, cur_linear='fc2',
                       args=args),
                nn.Dropout(p=self.dropout))
        if args.conv3 == 'ffn':
            self.conv3 = nn.Sequential(Linear(self.embed_dim, args.encoder_ffn_embed_dim, layer_id=layer_id, cur_linear='fc1', args=args),
                                       nn.ReLU(),
                                       nn.Dropout(p=self.activation_dropout),
                                       Linear(args.encoder_ffn_embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args),
                                       nn.Dropout(p=self.dropout))
        elif args.conv3 == 'none':
            self.conv3 = None
        # if args.dec_info_linear:
        #     self.il_dropout = args.dec_il_dropout
        #     self.dec_il_list = args.dec_il_list
        #     self.info_linear = Linear(len(args.dec_il_list) * self.embed_dim, self.embed_dim, layer_id=layer_id,
        #                                   cur_linear='info_linear', args=args)
        #     self.il_relu = args.il_relu
        #     if self.il_relu and args.il_relu_linear:
        #         self.il_relu_linear = Linear(self.embed_dim,self.embed_dim,args)
        #     else:
        #         self.il_relu_linear = None
        # else:
        #     self.info_linear = None
        self.dec_tanh_list = args.dec_tanh_list if 'dec_tanh_list' in args else []
        self.decr = args.decr if 'decr' in args else 0
        self.dec_nffn = args.dec_nffn if 'dec_nffn' in args else 0
        self.need_attn = True
        self.onnx_trace = False


    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, x, encoder_out=None, encoder_padding_mask=None, incremental_state=None, prev_self_attn_state=None,
                prev_attn_state=None, self_attn_mask=None, self_attn_padding_mask=None,):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        # (1) dec_attn
        if prev_self_attn_state is not None:
            if incremental_state is None:
                    incremental_state = {}
            prev_key1, prev_value1 = prev_self_attn_state
            saved_state = {"prev_key": prev_key1, "prev_value": prev_value1}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x1, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
        )
        if 'self_mha' in self.dec_tanh_list:
            x1 = F.tanh(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        if self.attn2:
            x2, _ = self.attn2(query=x,
                    key=x,
                    value=x,
                    key_padding_mask=self_attn_padding_mask,
                    incremental_state=incremental_state,
                    need_weights=False,
                    attn_mask=self_attn_mask,)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
        else:
            x2 = 0

        # (2) enc_dec attn
        if self.encoder_attn is not None:
            if prev_attn_state is not None:
                if incremental_state is None:
                        incremental_state = {}
                prev_key2, prev_value2 = prev_attn_state
                saved_state = {"prev_key": prev_key2, "prev_value": prev_value2}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x3, attn = self.encoder_attn(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=(not self.training and self.need_attn),
            )
            if 'context_mha' in self.dec_tanh_list:
                x3 = F.tanh(x3)
            x3 = F.dropout(x3, p=self.dropout, training=self.training)
        # (3) ffn
        # x3 = self.fc2(F.dropout(self.activation_fn(self.fc1(x)), p=self.activation_dropout, training=self.training))
        # if 'ffn' in self.dec_tanh_list:
        #     x3 = F.tanh(x3)
        # x3 = F.dropout(x3, p=self.dropout, training=self.training)
        if self.conv2 is not None:
            if self.conv2 in ['dynamic_conv','add_dynamic_conv','cat_dynamic_conv','cat_relu_dynamic_conv']:
                x4 = self.conv2(x, encoder_padding_mask=None,incremental_state=incremental_state)
            else:
                x4 = self.conv2(x)
        else:
            x4 = 0
        if self.conv3 is not None:
            x5 = self.conv3(x)
        else:
            x5 = 0
        # if self.decr:
        #     # x = residual + 0.25 * x1 + 0.25 * x2 + 0.5 * x3 + x4 # before9.9
        #     x = 0.5 * (residual + x1 + x2 + x3 + x4)
        # else:
        x = residual + x1 + x2 + x3 + x4 + x5
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


class DynamicConv(nn.Module):
    def __init__(self, args, embed_dim, dynamic_type=0, cur_attn_type='es', layer_id=0):
        super().__init__()
        dynamic_inner_dim = args.dynamic_inner_dim_ratio * embed_dim if 'dynamic_inner_dim_ratio' in args else embed_dim
        kernel_size = args.kernel_size if 'kernel_size' in args else 0
        self.dcb_relu = args.dcb_relu if 'dcb_relu' in args else 0
        attn_dynamic_type = args.attn_dynamic_type if 'attn_dynamic_type' in args else 0
        dynamic_depth_kernels = args.dynamic_depth_kernels if 'dynamic_depth_kernels' in args else []
        self.dynamic_padding = args.dynamic_padding if 'dynamic_padding' in args else 0
        if attn_dynamic_type == 1:
            if kernel_size == 0:
                kernel_size = dynamic_depth_kernels[layer_id]
        self.embed_dim = embed_dim
        self.dynamic_type = dynamic_type
        if dynamic_type == 3:
            assert self.dcb_relu == 1
        self.conv_list = nn.ModuleList([])
        self.conv_list.append(Linear(self.embed_dim, dynamic_inner_dim,cur_linear='fc1', args=args))
        if cur_attn_type == 'ds':
            padding_l= kernel_size - 1
            num_heads = args.encoder_attention_heads
        else:
            padding_l = kernel_size // 2 if kernel_size % 2 == 1 else ((kernel_size - 1) // 2, kernel_size // 2)
            num_heads = args.decoder_attention_heads
        if self.dcb_relu:
            self.conv_list.append(nn.ReLU())
            self.conv_list.append(nn.Dropout(p=getattr(args, 'activation_dropout', 0)))

        self.conv_list.append(DynamicConv1dTBC(dynamic_inner_dim, kernel_size,
                                               padding_l=padding_l,
                                               weight_softmax=True,
                                               num_heads=num_heads,
                                               weight_dropout=0.1, ))
        if args.dc_relu:
            self.conv_list.append(nn.ReLU())
            self.conv_list.append(nn.Dropout(p=getattr(args, 'activation_dropout', 0)))
        if self.dynamic_type in [2, 3]:
            self.conv_list.append(Linear(2*dynamic_inner_dim, self.embed_dim,cur_linear='fc2',args=args))
        else:
            self.conv_list.append(Linear(dynamic_inner_dim, self.embed_dim, cur_linear='fc2',args=args))
        self.conv_list.append(nn.Dropout(p=args.dropout))

    def forward(self, x, encoder_padding_mask=None, incremental_state=None):
        x_before = None
        for i in range(len(self.conv_list)):  # B T C
            layer_i = self.conv_list[i]
            if self.dynamic_type == 3 and i == 1:
                # print('layer%d| self.dynamic_type %d | i==1,self.dynamic_type==3 ' % (i,self.dynamic_type))
                x_before = x
            if (i == 1 and not self.dcb_relu) or (i == 3 and self.dcb_relu):
                if self.dynamic_type in [1, 2] and encoder_padding_mask is not None and self.dynamic_padding:
                    x = x.masked_fill(encoder_padding_mask.unsqueeze(2), 0)
                if self.dynamic_type == 1:
                    x = torch.transpose(layer_i(x.transpose(0, 1),incremental_state), 0, 1) + x # B T C  -> T B C
                elif self.dynamic_type == 2:
                    x = torch.cat((x, torch.transpose(layer_i(x.transpose(0, 1),incremental_state), 0, 1)), -1)
                elif self.dynamic_type == 3:
                    # print('layer%d| self.dynamic_type %d | i==3,self.dynamic_type==3 ' % (i,self.dynamic_type))
                    if encoder_padding_mask is not None and self.dynamic_padding:
                        x_before = x_before.masked_fill(encoder_padding_mask.unsqueeze(2), 0)
                    x = torch.cat((x, torch.transpose(layer_i(x_before.transpose(0, 1),incremental_state), 0, 1)), -1)
                else:
                    x = torch.transpose(layer_i(x.transpose(0, 1),incremental_state=incremental_state), 0, 1,)
            else:
                #RuntimeError: size mismatch, m1: [3648 x 768], m2: [1536 x 384] at /pytorch/aten/src/THC/generic/THCTensorMathBlas.cu:273
                x = layer_i(x)
        return x


class Conv(nn.Module):
    def __init__(self, args, embed_dim, kernel_size, layer_id=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2 if kernel_size % 2 == 1 else ((kernel_size - 1) // 2, kernel_size // 2)), nn.ReLU(),
            nn.Dropout(p=getattr(args, 'activation_dropout', 0)),
            Linear(self.embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args),
            nn.Dropout(p=args.dropout)])

    def forward(self, x):
        for i in range(len(self.conv_list)):
            # print('conv i', i, 'x size', x.size())
            layer_i = self.conv_list[i]
            if i == 0:
                    x = torch.transpose(layer_i(x.transpose(1, 2)), 1, 2)
            else:
                    x = layer_i(x)
        return x


class ConvAfter(nn.Module):
    def __init__(self, args, embed_dim, kernel_size, layer_id=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv_list = nn.ModuleList([
            Linear(self.embed_dim, self.embed_dim, layer_id=layer_id, cur_linear='fc2', args=args),
            nn.ReLU(),
            nn.Dropout(p=getattr(args, 'activation_dropout', 0)),
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2 if kernel_size % 2 == 1 else ((kernel_size - 1) // 2, kernel_size // 2)),
            nn.Dropout(p=args.dropout)])

    def forward(self, x):
        for i in range(len(self.conv_list)):
            # print('conv i', i, 'x size', x.size())
            layer_i = self.conv_list[i]
            if i == len(self.conv_list) - 2:
                    x = torch.transpose(layer_i(x.transpose(1, 2)), 1, 2)
            else:
                    x = layer_i(x)
        return x


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, layer_id=1, args=None,  kdim=None, vdim=None, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, cur_attn_type='es'):
        super().__init__()
        self.model_list = nn.ModuleList([])
        self.model_list.append(MultiheadAttention820(embed_dim, args.encoder_attention_heads, layer_id=layer_id, args=args,
                                                     dropout=args.attention_dropout, cur_attn_type=cur_attn_type))
        if args.mha_act == 'relu':
            self.model_list.append(nn.ReLU())
        elif args.mha_act == 'tanh':
            self.model_list.append(nn.Tanh())
        if args.mha_2fc:
            self.model_list.append(Linear(embed_dim, embed_dim, args=args))

    def forward(self, query, key, value, key_padding_mask=None,incremental_state=None,need_weights=False,
                attn_mask=None,):
        for i in range(len(self.model_list)):
            model_i = self.model_list[i]
            if i == 0:
                x, attn_weights = model_i(query=query, key=key, value=value, key_padding_mask=key_padding_mask,incremental_state=incremental_state,need_weights=need_weights,
                attn_mask=attn_mask,)
            else:
                x = model_i(x)
        return x, attn_weights


class SEScore(nn.Module):
    def __init__(self, args, embed_dim, ):
        super().__init__()
        self.embed_dim = embed_dim
        self.avg = (args.se_avg_type == 0)
        nonlinear = args.se_nonlinear if 'se_nonlinear' in args else 0
        self.se_softmax_t = args.se_softmax_t if 'se_softmax_t' in args else 0
        se_activation = args.se_activation
        if nonlinear:
            self.se_model = nn.Sequential(Linear(embed_dim, int(embed_dim/args.se_ratio),args=args),
                                          nn.ReLU(),
                                          Linear(int(embed_dim/args.se_ratio), embed_dim,args=args),)
        else:
            self.se_model = Linear(embed_dim, embed_dim,args=args)
        if se_activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
        else:
            self.act = nn.Sigmoid()
        self.se_scale = args.se_scale
        self.se_add = args.se_add

    def forward(self, x):
        # B T C
        if self.avg:
            mean_x = torch.mean(x, dim=1)
            if self.se_softmax_t:
                mean_x = mean_x/self.se_softmax_t
            scores = self.act(self.se_model(mean_x))
            scores = scores.unsqueeze(1).expand_as(x)
        else:
            if self.se_softmax_t:
                x = x/self.se_softmax_t
            scores = self.act(self.se_model(x))
        if self.se_add:
            scores = scores * self.se_scale + 1
        else:
            scores = scores * self.se_scale
        return scores * self.se_scale


class ParaScore(nn.Module):
    def __init__(self, args, embed_dim, ):
        super().__init__()
        self.embed_dim = embed_dim
        self.se_para = args.se_para if 'se_para' in args else 1
        self.weights = nn.Parameter(torch.ones([embed_dim]))
        # 1: ones   2: uniform
        if self.se_para == 2:
            nn.init.uniform_(self.weights, -0.1, 0.1)
        # if self.se_para in [1,2]:
        #     self.bias = nn.Parameter(nn.zeros([embed_dim]))
        # else:
        #     self.bias = None

    def forward(self, x):
        return self.weights