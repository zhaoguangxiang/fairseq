# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils
from .dynamic_convolution import DynamicConv1dTBC
bmm_fp16_support = tuple(int(x) for x in torch.version.cuda.split('.')) >= (9, 1, 0)


class MultiheadAttention820(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, layer_id=0, args=None,  kdim=None, vdim=None, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, cur_attn_type='es'):
        super().__init__()
        if cur_attn_type in args.use_att:  # es,ds,dc
            cur_san_active = True
        else:
            cur_san_active = False
        self.cur_attn_type = cur_attn_type
        self.args = args
        self.layer_id = layer_id

        self.cur_san_active = cur_san_active
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        # if self.qk_dropout == 1:
        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        if bias:
            # this should assert qkv_same_dim
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.qk_in = 0
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        # init
        self.add_zero_attn = add_zero_attn
        self.init_method = args.init_method
        self.qkv_init_scale = args.qkv_init_scale if 'qkv_init_scale' in args else 1
        self.attn_out_init_scale = args.attn_out_init_scale if 'attn_out_init_scale' in args else 1
        # cat dynamic
        self.kernel_size = args.kernel_size if 'kernel_size' in args else 0
        self.attn_dynamic_type = args.attn_dynamic_type if 'attn_dynamic_type' in args else 0
        self.dynamic_padding = args.dynamic_padding if 'dynamic_padding' in args else 0
        self.attn_dynamic_cat = args.attn_dynamic_cat if 'attn_dynamic_cat' in args else 0
        self.sep_out_proj = args.sep_out_proj if 'sep_out_proj' in args else 0
        self.gate_inc_attn = args.gate_inc_attn if 'gate_inc_attn' in args else 0  # for sep_out_proj of dynamics and self attn
        self.gate_before_proj = args.gate_before_proj if 'gate_before_proj' in args else 0
        self.conv_in_glu = args.conv_in_glu if 'conv_in_glu' in args else 0
        self.dynamic_gate = args.dynamic_gate if 'dynamic_gate' in args else 0
        attn_dynamic_indie_v = args.attn_dynamic_indie_v if 'attn_dynamic_indie_v' in args else 0
        attn_wide_kernels = args.attn_wide_kernels if 'attn_wide_kernels' in args else []
        dynamic_depth_kernels = args.dynamic_depth_kernels if 'dynamic_depth_kernels' in args else []
        self.attn_cat_relu = args.attn_cat_relu if 'attn_cat_relu' in args else 0
        self.entmax = args.entmax if 'entmax' in args else 0
        self.dynamics = nn.ModuleList([])
        self.k_list = []
        if self.conv_in_glu:
            self.conv_glu = nn.Linear(embed_dim, embed_dim)
        else:
            self.conv_glu = None
        if self.attn_dynamic_type == 1:
            if self.kernel_size == 0:
                self.kernel_size = dynamic_depth_kernels[layer_id]
            self.k_list.append(self.kernel_size)
        elif self.attn_dynamic_type == 2:
            self.k_list = attn_wide_kernels
        if self.attn_dynamic_type and cur_attn_type in ['es', 'ds']:
            for kernel_size in self.k_list:
                if cur_attn_type == 'ds':
                    padding_l = kernel_size-1
                    dynamic_num_heads = args.decoder_attention_heads
                else:
                    padding_l = kernel_size // 2 if kernel_size % 2 == 1 else ((self.kernel_size - 1) // 2, self.kernel_size // 2)
                    dynamic_num_heads = args.encoder_attention_heads
                dynamic = DynamicConv1dTBC(self.embed_dim, kernel_size, padding_l=padding_l, weight_softmax=True,
                                           num_heads=dynamic_num_heads, weight_dropout=0.1, )
                self.dynamics.append(dynamic)
        if attn_dynamic_indie_v:
            self.indie_vw = nn.Linear(embed_dim, embed_dim)
        else:
            self.indie_vw = None
        in_ratio = 1
        self.dynamic_gate_para = None
        self.cur_attn_type = cur_attn_type
        if self.attn_dynamic_type and cur_attn_type in ['es', 'ds']:
            if self.attn_dynamic_cat:
                if args.dynamic_gate:
                    if args.dynamic_gate == 1:  # [2] task based para
                        self.dynamic_gate_para = nn.Parameter(torch.Tensor([1.0/len(self.k_list) for _ in self.k_list]))
                    in_ratio = 2
                else:
                    in_ratio = len(self.k_list)+1

            elif self.sep_out_proj:
                self.out_projs = nn.ModuleList([])
                if args.dynamic_gate:
                    if args.dynamic_gate == 1:  # [2] task based para
                        if self.gate_before_proj:
                            self.dynamic_gate_para = nn.Parameter(torch.Tensor([1.0 / len(self.k_list) for _ in self.k_list]))
                        else:
                            self.dynamic_gate_para = nn.Parameter(torch.Tensor([1.0/len(self.k_list) for _ in self.k_list+1]))
                proj_num = 1 if self.gate_before_proj else len(self.k_list)
                for i in range(proj_num):
                    self.out_proji = nn.Linear(embed_dim,embed_dim,bias=bias)  # 先假设只有一个kernel
                    if self.args.big_km and 'attn_out' in self.args.big_km_list:  #10,10 检查到sep_out 结果差的原因了，这里写成kqv了,10.10 从qkv 改回attn_out
                        a = 1
                    else:
                        a = math.sqrt(args.attn_a) if 'attn_a' in args else math.sqrt(5)
                    nn.init.kaiming_uniform_(self.out_proji.weight, a=a)
                    self.out_projs.append(self.out_proji)
        if self.attn_cat_relu and cur_attn_type in ['es', 'ds']:
            in_ratio += 1
        self.out_proj = nn.Linear(in_ratio * embed_dim, embed_dim, bias=bias)
        # others
        self.reset_parameters()
        if args.init_topk_rho:  # 从正负分别clip掉超过某个比例的 为了简单只做标准的init clip
            k = max(int(args.init_topk_rho * 3 * embed_dim * embed_dim), 1)
            self.in_proj_weight = clip_init(tensor=self.in_proj_weight, k=k)
            k = max(int(args.init_topk_rho * embed_dim * embed_dim), 1)
            self.out_proj.weight = clip_init(tensor=self.out_proj.weight, k=k)

        self.onnx_trace = False
        self.div = args.div
        self.lb = args.lb


    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.init_method == 'xavier':
            if self.qkv_same_dim:
                nn.init.xavier_uniform_(self.in_proj_weight)
            else:
                nn.init.xavier_uniform_(self.k_proj_weight)
                nn.init.xavier_uniform_(self.v_proj_weight)
                nn.init.xavier_uniform_(self.q_proj_weight)
        elif self.init_method == 'xi':
            gain = (self.layer_id+1)**(-0.5)
            if self.qkv_same_dim:
                nn.init.xavier_uniform_(self.in_proj_weight, gain=gain)
            else:
                nn.init.xavier_uniform_(self.k_proj_weight, gain=gain)
                nn.init.xavier_uniform_(self.v_proj_weight, gain=gain)
                nn.init.xavier_uniform_(self.q_proj_weight, gain=gain)
        else:
            if self.qkv_same_dim:
                a = math.sqrt(self.args.qkv_a) if 'qkv_a' in self.args else math.sqrt(5)
                if self.args.big_km and 'qkv' in self.args.big_km_list:
                    a = 1
                nn.init.kaiming_uniform_(self.in_proj_weight, a=a)
                # print('km',  ' qkv w max', torch.max(self.in_proj_weight))
            else:
                if self.args.big_km and 'qkv' in self.args.big_km_list:
                    a = 1
                else:
                    a = math.sqrt(self.args.qkv_a) if 'qkv_a' in self.args else math.sqrt(5)
                nn.init.kaiming_uniform_(self.k_proj_weight, a=a)
                nn.init.kaiming_uniform_(self.v_proj_weight, a=a)
                nn.init.kaiming_uniform_(self.q_proj_weight, a=a)
        if self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.out_proj.weight)
        elif self.init_method == 'xi':
            gain = (self.layer_id + 1) ** (-0.5)
            nn.init.xavier_uniform_(self.out_proj.weight, gain=gain)
        else:
            if self.args.big_km and 'attn_out' in self.args.big_km_list:
                a = 1
            else:
                a = math.sqrt(self.args.attn_a) if 'attn_a' in self.args else math.sqrt(5)
            nn.init.kaiming_uniform_(self.out_proj.weight, a=a)
            # print('km', ' out_proj w max', torch.max(self.out_proj.weight))
            # print('km', 'out_proj_scale', self.attn_out_init_scale, ' attn_out_ max',
            #       torch.max(self.out_proj.weight))
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None
        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # print('attn_cat_relu %d |len dynamics size %d| tgt_len %d' % (self.attn_cat_relu, len(self.dynamics), tgt_len))
        if len(self.dynamics) > 0 or (self.attn_cat_relu and self.cur_attn_type in ['es','ds']):  # added in 9/11
            dynamic_x = v  # bsz*heads tgt_len head_dim
            if self.conv_glu:
                glu_v = self.conv_glu(value)
                glu_score = glu_v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
                dynamic_x = F.sigmoid(glu_score) * dynamic_x
        else:
            dynamic_x = None
        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            # print('key_padding_mask.size(0)',key_padding_mask.size(0),'bsz',bsz)
            # print('key_padding_mask.size(1)',key_padding_mask.size(1),'src_len',src_len)
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)
        if not bmm_fp16_support:
            q = q.float()
            k = k.float()
            v = v.float()

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if not bmm_fp16_support:
            attn_weights = attn_weights.type_as(query)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.float().masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        # 1
        if not self.cur_san_active:
            self.div = 0
        if self.div > 0:
            top_k = int(torch.ceil(torch.Tensor([src_len / self.div])))
            if top_k < self.lb:
                top_k = self.lb
                if top_k > src_len:
                    top_k = src_len
        else:
            top_k = -self.div
            if top_k > src_len:
                top_k = src_len
        # 2
        # print('attn_weights ', attn_weights.size())

        # if self.div:
        #     vk, _ = torch.topk(attn_weights, top_k)
        #     # print(value)
        #     tk = vk[:, :,  -1].unsqueeze(2).expand_as(attn_weights)
        #     mask_k = torch.lt(attn_weights, tk)
        #     attn_weights = attn_weights.masked_fill(mask_k, float('-inf')).type_as(attn_weights)
        # if not (self.args.no_softmax and self.cur_san_active):
        if self.entmax:
            from entmax import sparsemax, entmax15, entmax_bisect
            if self.entmax == 1:
                attn_weights = sparsemax(attn_weights.float(), dim=-1).type_as(attn_weights)
            elif self.entmax == 2:
                attn_weights = entmax15(attn_weights.float(), dim=-1).type_as(attn_weights)
            elif self.entmax == 3:
                attn_weights = entmax_bisect(attn_weights.float(), dim=-1).type_as(attn_weights)
        else:
            attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        if not bmm_fp16_support:
            attn_weights = attn_weights.float()  # bsz * self.num_heads, tgt_len, src_len
        attn = torch.bmm(attn_weights, v)  # bsz * self.num_heads, tgt_len, head_dim
        if not bmm_fp16_support:
            attn_weights = attn_weights.type_as(query)
            attn = attn.type_as(query)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # with open('log_dynmaic.txt','a+') as f:
        #     f.write('q: %s | k: %s |v: %s |query:%s |key:%s |value:%s |tgt_len:%s |bsz:%s |embed_dim:%s |attn_size: %s |weight size:%s |' %
        #             (q.size(),k.size(),v.size(),query.size(),key.size(),value.size(),tgt_len,bsz,embed_dim,attn.size(),attn_weights.size()))
        #     f.write('\n')
        if len(self.dynamics) > 0:
            # v bsz * self.num_heads, src_len, head_dim
            dynamic_results = []
            if self.indie_vw:
                dynamic_x = self.indie_vw(query)  # bsz, tgt_len,embed_dim
            # encoder_padding_mask bsz,src_len
            dynamic_x = dynamic_x.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            if key_padding_mask is not None and self.dynamic_padding and self.cur_attn_type == 'es':
                # print('dynamic x',dynamic_x.size(),'key padding mask',key_padding_mask.size())
                dynamic_x = dynamic_x.masked_fill(key_padding_mask.transpose(0, 1).unsqueeze(2), 0)
            # print('k_list',self.k_list)
            for i in range(len(self.k_list)):
                dynamic = self.dynamics[i]
                if self.cur_attn_type == 'es':
                    dynamic_res = dynamic(dynamic_x)  # tgt_len, bsz, embed_dim
                else:  # ds
                    dynamic_res = dynamic(dynamic_x, incremental_state)  # tgt_len, bsz, embed_dim
                dynamic_results.append(dynamic_res)
            proj_results = []
            if self.attn_dynamic_cat: # 将所有的dynamic cnn的结果和attn cat 然后out linear
                all_res = [attn]
                if self.dynamic_gate_para is not None: # 要么是加权的结果，要么是cat起来，最后都和attn 一起cat然后Out proj
                    dynamic_res = torch.sum(torch.stack(dynamic_results, dim=-1) * F.softmax(self.dynamic_gate_para), dim=-1)
                    # dynamic_res = 0
                    # for i in range(len(self.k_list)):
                    #     dynamic_res = dynamic_res + self.dynamic_gate_para[i]*dynamic_results[i]
                    all_res.append(dynamic_res)
                else:
                    all_res.extend(dynamic_results)
                # print('all_res[0]',all_res[0].size(),'all_res len',len(all_res))
                attn = torch.cat(all_res, -1)
            elif self.sep_out_proj:  # 不cat, 分开linear,
                if self.gate_before_proj:  # 在同时gate和 gate_before_proj 的时候
                    dynamic_res = torch.sum(torch.stack(dynamic_results, dim=-1) * F.softmax(self.dynamic_gate_para),
                                            dim=-1)
                    out_proj0 = self.out_projs[0]
                    proj_result = out_proj0(dynamic_res)

                else:
                    for i in range(len(self.k_list)):
                        out_proji = self.out_projs[i]
                        proj_results.append(out_proji(dynamic_results[i]))
            else:
                for dynamic_res in dynamic_results:
                    # dynamicx # bsz*heads tgt_len head_dim
                    # attn tgt_len, bsz, embed_dim
                    attn = attn + dynamic_res
        if self.attn_cat_relu and self.cur_attn_type in ['es','ds']:
            # print('attn_cat_relu %d |dynamic_x size %s| tgt_len %d' % (self.attn_cat_relu, dynamic_x.size(), tgt_len))
            dynamic_x = dynamic_x.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            attn = torch.cat((attn, F.relu(dynamic_x)), dim=-1)
        # print('len',len(self.dynamics),'attn',attn.size())
        attn = self.out_proj(attn)
        if self.sep_out_proj:
            if self.gate_before_proj:  # 如果gate before proj 直接相加就可以
                attn = attn + proj_result
            else:
                if self.dynamic_gate_para is not None:  # 如果  proj 后有 gate则和attn一起sum，否则相加
                    proj_results.append(attn)
                    attn = torch.sum(torch.stack(proj_results, dim=-1) * F.softmax(self.dynamic_gate_para), dim=-1)
                else:
                    for i in range(len(self.k_list)):
                        attn = attn + proj_results[i]
        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )

def clip_init(tensor, k):
    flat_tensor = tensor.view(-1)
    lowbound, _ = torch.kthvalue(flat_tensor, k)
    low_mask = torch.lt(tensor, lowbound)
    ne_upbound, _ = torch.kthvalue(torch.neg(flat_tensor), k)
    upbound = torch.neg(ne_upbound)
    upmask = torch.gt(tensor, upbound)
    tensor = tensor.masked_fill(low_mask, lowbound)
    tensor =tensor.masked_fill(upmask, upbound)
    return nn.Parameter(tensor)

class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, args=None, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and ' \
                                                             'value to be of the same size'

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False
        self.args = args
        use_attn_default = args.use_attn_default if 'use_attn_default' in args else 1
        if not use_attn_default:
            self.enable_torch_version = False
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1/math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1/math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1/math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query, key, value,
        key_padding_mask=None,
        incremental_state=None,
        need_weights=True,
        static_kv=False,
        attn_mask=None,
        before_softmax=False,
        need_head_weights=False,
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if self.enable_torch_version and not self.onnx_trace and incremental_state is None and not static_kv:
            return F.multi_head_attention_forward(query, key, value,
                                                  self.embed_dim, self.num_heads,
                                                  torch.empty([0]),
                                                  torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                                                  self.bias_k, self.bias_v,
                                                  self.add_zero_attn, self.dropout,
                                                  self.out_proj.weight, self.out_proj.bias,
                                                  self.training, key_padding_mask, need_weights,
                                                  attn_mask, use_separate_proj_weight=True,
                                                  q_proj_weight=self.q_proj.weight,
                                                  k_proj_weight=self.k_proj.weight,
                                                  v_proj_weight=self.v_proj.weight)

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            key_padding_mask = self._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=saved_state.get('prev_key_padding_mask', None),
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_padding_mask'] = key_padding_mask

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))  # bsz * self.num_heads, q_len, k_len
        key_len = attn_weights.size(2)
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)

        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        else:
            attn_weights = None

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask,
        prev_key_padding_mask,
        batch_size,
        src_len,
        static_kv,
    ):
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            key_padding_mask = torch.cat((prev_key_padding_mask, key_padding_mask), dim=1)
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(batch_size, src_len - prev_key_padding_mask.size(1)).bool()
            if prev_key_padding_mask.is_cuda:
                filler = filler.cuda()
            key_padding_mask = torch.cat((prev_key_padding_mask, filler), dim=1)
        elif key_padding_mask is not None:
            filler = torch.zeros(batch_size, src_len - key_padding_mask.size(1)).bool()
            if key_padding_mask.is_cuda:
                filler = filler.cuda()
            key_padding_mask = torch.cat((filler, key_padding_mask), dim=1)
        return key_padding_mask

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                if input_buffer[k] is not None:
                    input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + 'in_proj_weight'):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + 'q_proj.weight'] = state_dict[k][:dim]
                items_to_add[prefix + 'k_proj.weight'] = state_dict[k][dim:2*dim]
                items_to_add[prefix + 'v_proj.weight'] = state_dict[k][2*dim:]

                keys_to_remove.append(k)

                k_bias = prefix + 'in_proj_bias'
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + 'q_proj.bias'] = state_dict[k_bias][:dim]
                    items_to_add[prefix + 'k_proj.bias'] = state_dict[k_bias][dim:2*dim]
                    items_to_add[prefix + 'v_proj.bias'] = state_dict[k_bias][2*dim:]

                    keys_to_remove.append(prefix + 'in_proj_bias')

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
