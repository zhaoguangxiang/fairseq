import torch
import torch.nn as nn
import math


def clip_init(tensor, k):
    flat_tensor = tensor.view(-1)
    lowbound, _ = torch.kthvalue(flat_tensor, k)
    low_mask = torch.lt(tensor, lowbound)
    ne_upbound, _ = torch.kthvalue(torch.neg(flat_tensor), k)
    upbound = torch.neg(ne_upbound)
    upmask = torch.gt(tensor, upbound)
    tensor = tensor.masked_fill(low_mask, lowbound)
    tensor = tensor.masked_fill(upmask, upbound)
    return nn.Parameter(tensor)


def Linear(in_features, out_features, layer_id=0, args=None, cur_linear=None, bias=True, ):
    m = nn.Linear(in_features, out_features, bias)
    # print('kmfc:',cur_linear, ' max', torch.max(m.weight))
    if args is None:
        nn.init.xavier_uniform_(m.weight)
    else:
        if args.init_method == 'xavier':
            nn.init.xavier_uniform_(m.weight)
        elif args.init_method == 'fixup':
            nn.init.xavier_uniform_(m.weight,  gain=1/math.sqrt(6))
        elif args.init_method == 'xi':
            gain = (layer_id+1)**(-0.5)
            nn.init.xavier_uniform_(m.weight, gain=gain)
        if args.init_topk_rho:  # 从正负分别clip掉超过某个比例的
            k = max(int(args.init_topk_rho * in_features * out_features), 1)
            m.weight = clip_init(tensor=m.weight, k=k)
        if cur_linear == 'fc1' and 'fc1_a' in args:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(args.fc1_a))
        if cur_linear == 'fc2' and 'fc2_a' in args:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(args.fc2_a))
        if args.big_km and cur_linear in args.big_km_list:
            nn.init.kaiming_uniform_(m.weight, a=1)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m