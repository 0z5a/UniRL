import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import os
import random

from hymm.models.modules.moba_efficient import moba_attn_varlen
from hymm.utils.flash_attn_no_pad import flash_attn_no_pad, flash_attn_no_pad_v3
from hymm.utils.communications import all_gather, all_to_all_4D
from hymm.utils.parallel_states import get_sequence_parallel_state, nccl_info
from hymm.models.modules.flex_ssta import get_q_to_block_list, get_sliding_tile_attention_with_block_mask

from hymm.utils.block_attn_utils import (
    batch_slice_4d_tensor,
    batch_sliced_tensor_to_4d,
    qkv_to_flash_shape,
    flash_to_ori_shape,
    padding_text_seq_to_block_size
)

import hymm.utils.dynamic_ring_attention as DynRingAttn

try:
    from .flex_sta_ref import get_sliding_tile_attention_mask
    from torch.nn.attention.flex_attention import flex_attention
    flex_attention = torch.compile(flex_attention, dynamic=False)
    torch._dynamo.config.cache_size_limit = 192
    torch._dynamo.config.accumulated_cache_size_limit = 192
    flex_mask_cache = {}
except:
    print("Could not load Sliding Tile Attention of FlexAttn.")


try:
    from ptm_attn import ssta_3d_attention
except ImportError:
    print("Could not load PTM Sparse Attention. Please install the ptm_attn package.")

def attention(
        q,
        k,
        v,
        drop_rate=0,
        attn_mask=None,
        causal=False,
):

    qkv = torch.stack([q, k, v], dim=2)

    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.bool()

    x = flash_attn_no_pad(qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=None)

    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


@torch.compiler.disable
def parallel_attention(q, k, v, img_q_len, img_kv_len,
                       attn_mode=None, text_mask=None,
                       attn_param=None,
                       ):
    if nccl_info.use_dynamic_ring_attention:
        return dynamic_ring_attention(q, k, v, text_mask)
    else:
        return sequence_parallel_attention(q, k, v, img_q_len, img_kv_len, attn_mode, text_mask,
                                           attn_param=attn_param)


def sequence_parallel_attention(q, k, v,
                                img_q_len, img_kv_len,
                                attn_mode=None, text_mask=None,
                                attn_param=None,
                                ):
    assert attn_mode is not None
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v

    if get_sequence_parallel_state():
        # batch_size, seq_len, attn_heads, head_dim
        query = all_to_all_4D(query, nccl_info.sp_group, scatter_dim=2, gather_dim=1)
        key = all_to_all_4D(key, nccl_info.sp_group, scatter_dim=2, gather_dim=1)
        value = all_to_all_4D(value, nccl_info.sp_group, scatter_dim=2, gather_dim=1)

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // nccl_info.sp_size
            return encoder_state.narrow(
                dim, nccl_info.rank_within_spgroup * local_heads, local_heads
            )

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)

        # [b, s, h, d]
    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    # 短序列用flash3
    #if sequence_length < 20000:
    #    attn_mode = "flash3"

    if attn_mode == "torch":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        if text_mask is not None:
            attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        else:
            attn_mask = None
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(query.dtype)

        # transpose q,k,v dim to fit scaled_dot_product_attention
        query = query.transpose(1, 2)  # B * Head_num * length * dim
        key = key.transpose(1, 2)      # B * Head_num * length * dim
        value = value.transpose(1, 2)  # B * Head_num * length * dim
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(-1).unsqueeze(-1)
            attn_mask = attn_mask.transpose(1, 2) # B * 1 * length * 1
            attn_mask = attn_mask.expand(-1, -1, -1, attn_mask.size(-2)).contiguous()
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

        # transpose back
        hidden_states = hidden_states.transpose(1, 2)

    elif attn_mode == "flash":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)

        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)

    elif attn_mode == "flash3":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)
        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad_v3(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)

    elif attn_mode == "sta_flex":
        query = torch.cat([query, encoder_query], dim=1).permute(0,2,1,3)
        key = torch.cat([key, encoder_key], dim=1).permute(0,2,1,3)
        value = torch.cat([value, encoder_value], dim=1).permute(0,2,1,3)

        text_length = text_mask.sum(dim=-1)
        head_num = query.shape[1]

        if attn_param['win_type'] == 'fix':
            thw = attn_param['thw']
            win_size = attn_param['win_size'][0].copy()
            tile_size = attn_param['tile_size'] # (3, 4, 4)
            if thw[0] == 1:
                win_size[0] = 1
                tile_size = (1, tile_size[1], tile_size[2])
            mask_name = []
            for i in list(thw)+list(win_size):
                mask_name.append(str(i))
            mask_name = '-'.join(mask_name)
            if mask_name in attn_param['flex_mask_cache']:
                mask = attn_param['flex_mask_cache'][mask_name]
            elif mask_name in flex_mask_cache:
                mask = flex_mask_cache[mask_name]
            else:
                mask = get_sliding_tile_attention_mask(win_size, tile_size, thw, 256, query.device, text_max_len=256)
                flex_mask_cache[mask_name] = mask
                print(f'==> [rank-{torch.distributed.get_rank()}] STA generating new mask: {mask_name}')
            hidden_states = flex_attention(query, key, value, block_mask=mask).permute(0,2,1,3)
        elif attn_param['win_type'] == 'mix':
            thw = attn_param['thw']
            win_size = attn_param['win_size'].copy()
            tile_size = attn_param['tile_size'] # (3, 4, 4)
            if thw[0] == 1:
                _win_size = []
                for win in win_size:
                    win[0] = 1
                    _win_size.append(win)
                win_size = _win_size
                tile_size = (1, tile_size[1], tile_size[2])
            mask = attn_param['flex_mask_cache']['sta_mask']
            hidden_states = flex_attention(query, key, value, block_mask=mask).permute(0,2,1,3)
        elif attn_param['win_type'] == 'hybrid':
            thw = attn_param['thw']
            win_size = attn_param['win_size'][0].copy()
            tile_size = attn_param['tile_size'] # (3, 4, 4)
            if thw[0] == 1:
                win_size[0] = 1
                tile_size = (1, tile_size[1], tile_size[2])
            mask_name = []
            for i in list(thw)+list(win_size):
                mask_name.append(str(i))
            mask_name = '-'.join(mask_name)
            if mask_name in attn_param['flex_mask_cache']:
                mask = attn_param['flex_mask_cache'][mask_name]
            elif mask_name in flex_mask_cache:
                mask = flex_mask_cache[mask_name]
            else:
                mask = get_sliding_tile_attention_mask(win_size, tile_size, thw, 256, query.device, text_max_len=256)
                flex_mask_cache[mask_name] = mask
                print(f'==> [rank-{torch.distributed.get_rank()}] STA generating new mask: {mask_name}')
            hidden_states = flex_attention(query, key, value, block_mask=mask).permute(0,2,1,3)
        else:
            raise ValueError(f"Error Win Type {attn_param['win_type']}")
    elif attn_mode == "ptm_sparse_attn":
        sparse_type = attn_param["ptm_attn_sparse_type"] # sta/block_attn/ssta
        ssta_threshold = attn_param['ssta_threshold']
        ptm_attn_pad_type = attn_param['ptm_attn_pad_type'] # repeat/zero
        ptm_attn_use_text_mask = attn_param['ptm_attn_use_text_mask']
        ptm_attn_mask_share_within_head = attn_param['ptm_attn_mask_share_within_head']

        ssta_topk = attn_param['ssta_topk']
        thw = attn_param['thw']
        tile_size = attn_param['tile_size']
        win_size = attn_param['win_size'][0].copy()

        if thw[0] == 1:
            if np.prod(tile_size) == 128:
                tile_size = (1, 16, 8)
                win_size = [1, 1, 1]
            elif np.prod(tile_size) == 384:
                tile_size = (1, 16, 24)
                win_size = [1, 1, 1]
            elif np.prod(tile_size) == 64:
                tile_size = (1, 8, 8)
                win_size = [1, 1, 1]
            else:
                raise ValueError(f"Error tile_size {tile_size}, only support in [64, 128, 384]")

        query = torch.cat([query, encoder_query], dim=1).permute(0,2,1,3) # convert to (B, H, S, D)
        key = torch.cat([key, encoder_key], dim=1).permute(0,2,1,3) # convert to (B, H, S, D)
        value = torch.cat([value, encoder_value], dim=1).permute(0,2,1,3) # convert to (B, H, S, D)

        assert query.shape[-1] == 128, "The last dimension of query, key and value must be 128 for PTM Sparse Attention."

        hidden_states = ssta_3d_attention(query, key, value, thw,
                                          topk=ssta_topk,
                                          tile_thw=tile_size,
                                          kernel_thw=win_size,
                                          text_len=encoder_sequence_length,
                                          sparse_type=sparse_type,
                                          threshold=ssta_threshold,
                                          pad_type=ptm_attn_pad_type,
                                          text_mask=text_mask if ptm_attn_use_text_mask else None,
                                          sp_size=nccl_info.sp_size,
                                          sp_rank=nccl_info.rank_within_spgroup,
                                          sp_group=nccl_info.sp_group,
                                          mask_share_within_head=ptm_attn_mask_share_within_head).permute(0,2,1,3)
    elif attn_mode == "ssta_flex":
        ssta_topk = attn_param['ssta_topk']
        ssta_threshold = attn_param['ssta_threshold']
        enforce_share_topk_block=attn_param['ssta_enforce_share_topk_block']
        enforce_share_head = True # 不设置共享会出现TRITON_MAX_BLOCK超过错误(TODO)
        query = torch.cat([query, encoder_query], dim=1).permute(0,2,1,3)
        key = torch.cat([key, encoder_key], dim=1).permute(0,2,1,3)
        value = torch.cat([value, encoder_value], dim=1).permute(0,2,1,3)

        thw = attn_param['thw']
        win_size = attn_param['win_size'][0].copy()
        tile_size = attn_param['tile_size'] # (3, 4, 4)
        block_shape = [int(2*tile_x) for tile_x in tile_size] # 2倍的tile_size 大小(TODO, 设置为动态值)
        if thw[0] == 1:
            win_size[0] = 1
            tile_size = (1, tile_size[1], tile_size[2])
            block_shape = (1, block_shape[1], block_shape[2])
        #tube_t = max(1, int(np.around(thw[0]/3))) #1/3
        #tube_h = min(20, int(np.around(thw[1]/4))) #1/4
        #tube_w = min(20, int(np.around(thw[2]/4))) #1/4
        #block_shape=(tube_t, tube_h, tube_w)
        q_to_block_list = get_q_to_block_list(q=query, k=key,
                                              image_shape=thw,
                                              block_shape=block_shape,
                                              topk=ssta_topk,
                                              threshold=ssta_threshold,
                                              enforce_share_topk_block=enforce_share_topk_block,
                                              enforce_share_head=enforce_share_head,
                                              layer_name=attn_param['layer-name'])
        # print(f"""
        #        q_to_block_list.shape: {q_to_block_list.shape},
        #        q_to_block_list: {q_to_block_list[0,0,0:3,:,:]},
        #        text_length={text_mask.sum(dim=-1)},
        #        block_shape={block_shape},
        #        win_size={win_size},
        #        tile_size={tile_size},
        #        enforce_share_topk_block={enforce_share_topk_block},
        #        ssta_topk={ssta_topk},
        #        pad_length={pad_length},
        #        seq_len={seq_len},
        #        image_shape={thw},
        #        """)
        mask = get_sliding_tile_attention_with_block_mask(
            kernel_size=win_size, tile_size=tile_size, img_size=thw,
            head_size=None if enforce_share_head else query.shape[1],  # head_size设置为None，可在head间共享mask
            batch_size=query.shape[0],
            block_shape=block_shape,
            text_length=encoder_sequence_length,
            device='cuda',
            text_max_len=encoder_sequence_length, is_full=False, q_to_block_list=q_to_block_list)
        hidden_states = flex_attention(query, key, value, block_mask=mask).permute(0,2,1,3)

    elif attn_mode == "block_attn":
        image_shape=attn_param["thw"]
        ba_attn_chunk_size=attn_param['ba_attn_chunk_size']
        ba_attn_topk=attn_param['ba_attn_topk']
        ba_attn_causal=attn_param['ba_attn_causal']
        ba_attn_slice_method=attn_param['ba_attn_slice_method']
        if ba_attn_slice_method == "3D": # 对视频进行切分,按照3D方式进行切分
            #assert ba_attn_chunk_size in [1320]
            #e.g. ba_attn_chunk_size=384,  tube=(6,20,11) or (1, 20, 11) #244
            # 720p(129x704x1248---17x78x44)
            #17: 9,6,5,4;     1/2|1/3|1/4|1/5
            #78: 39,26,20,15; 1/2|1/3|1/4|1/5
            #44: 22,15,11,8;  1/2|1/3|1/4|1/5
            head, dim = query.shape[-2:]
            tube_t = max(1, int(np.around(image_shape[0]/3))) #1/3
            tube_h = min(20, int(np.around(image_shape[1]/4))) #1/4
            tube_w = min(20, int(np.around(image_shape[2]/4))) #1/4

            tube_shape=(tube_t, tube_h, tube_w)
            ba_attn_chunk_size=tube_t*tube_h*tube_w
            use_padding=True
            sparse = 1.0 - min(query.shape[1], ba_attn_topk * ba_attn_chunk_size)/query.shape[1]
            ori_qury_shape = query.shape
            query, pad_image_shape , pad_shape = batch_slice_4d_tensor(query, image_shape, tube_shape, head, dim, use_padding=use_padding)
            key, _, _ = batch_slice_4d_tensor(key, image_shape, tube_shape,  head, dim, use_padding=use_padding)
            value, _, _ = batch_slice_4d_tensor(value, image_shape, tube_shape,  head, dim, use_padding=use_padding)
            if int(os.environ["RANK"]) <= 0 and (random.random() < 0.01): # 偶尔打印一下
                print(f""" image_shape: {image_shape}
                            tube_shape: {tube_shape}
                            ori_query_shape: {ori_qury_shape}
                            pad_query_shape: {query.shape}
                            pad_image_shape: {pad_image_shape}
                            pad_shape: {pad_shape}
                            choose_total_kv_block_seq: {ba_attn_topk * ba_attn_chunk_size}
                            ba_attn_chunk_size: {ba_attn_chunk_size}
                            ba_attn_topk: {ba_attn_topk}
                            ba_attn_causal: {ba_attn_causal}
                            sparse: {sparse:.4f}
                        """)
            sequence_length_pad = query.shape[1]
        else:
            sequence_length_pad = sequence_length

        # 对文本进行padding, 保证文本长度为ba_attn_chunk_size的整数倍
        encoder_query,text_blcok_pad_size = padding_text_seq_to_block_size(encoder_query, ba_attn_chunk_size)
        encoder_key,_ = padding_text_seq_to_block_size(encoder_key, ba_attn_chunk_size)
        encoder_value,_ = padding_text_seq_to_block_size(encoder_value, ba_attn_chunk_size)

        # 将文本放在视觉Token前面
        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value  = torch.cat([encoder_value, value], dim=1)
        #attn_mask = F.pad(text_mask, (0, sequence_length), value=True)
        cu_seqlens_k = torch.cumsum(
            torch.tensor([0] + [key.shape[1]] * query.shape[0], device=query.device),
            dim=0,
            dtype=torch.int32,
        )
        hidden_states = moba_attn_varlen(
            q = qkv_to_flash_shape(query) if ba_attn_slice_method == "3D" else query,
            k = qkv_to_flash_shape(key) if ba_attn_slice_method == "3D" else key,
            v = qkv_to_flash_shape(value) if ba_attn_slice_method == "3D" else value,
            cu_seqlens=cu_seqlens_k,
            max_seqlen=key.shape[1],
            moba_chunk_size=ba_attn_chunk_size,
            moba_topk=ba_attn_topk,
            causal=ba_attn_causal,
            padding=False) # 前面已经做了padding, 这里不需要了
        if ba_attn_slice_method == "3D":
            hidden_states = flash_to_ori_shape(hidden_states, query.shape[0])
            total_len = hidden_states.shape[1]
            encoder_hidden_states, hidden_states  = hidden_states.split_with_sizes((total_len - sequence_length_pad,
                                                                                    sequence_length_pad), dim=1)
            if text_blcok_pad_size > 0: # 如果文本被padding了, 则去掉padding的部分
                encoder_hidden_states = encoder_hidden_states[:, :-text_blcok_pad_size]
            hidden_states = batch_sliced_tensor_to_4d(hidden_states, image_shape, tube_shape, pad_image_shape, pad_shape)
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
    else:
        raise NotImplementedError

    if get_sequence_parallel_state():
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes((sequence_length,
                                                                               encoder_sequence_length),
                                                                              dim=1)
        hidden_states = all_to_all_4D(hidden_states, nccl_info.sp_group, scatter_dim=1, gather_dim=2)
        encoder_hidden_states = all_gather(encoder_hidden_states, dim=2, group=nccl_info.sp_group).contiguous()
        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states = encoder_hidden_states.to(query.dtype)
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)

    return hidden_states


def dynamic_ring_attention(q, k, v, text_mask=None):
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v
    text_length = text_mask.sum()

    current_rank = nccl_info.rank_within_spgroup
    cp_group = nccl_info.sp_group
    cp_stream = nccl_info.sp_stream
    cp_rank_list = nccl_info.sp_rank_list

    if current_rank == nccl_info.sp_size - 1:
        part_txt_q = encoder_query[:, 0: text_length, ...]
        part_txt_k = encoder_key[:, 0: text_length, ...]
        part_txt_v = encoder_value[:, 0: text_length, ...]
        query = torch.cat([query, part_txt_q], dim=1)
        key = torch.cat([key, part_txt_k], dim=1)
        value = torch.cat([value, part_txt_v], dim=1)

    cu_seqlens_q = DynRingAttn.cache_cu_seqlens_q
    cu_seqlens_k = DynRingAttn.cache_cu_seqlens_k
    max_seqlen_q = DynRingAttn.cache_max_seqlen_q
    max_seqlen_k = DynRingAttn.cache_max_seqlen_k
    sub_attn = DynRingAttn.attn_forward_func(
        is_training=True,
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        cp_group=cp_group,
        cp_global_ranks=cp_rank_list,
        cp_stream=cp_stream
    )
    b, s, a, d = query.shape
    sub_attn = sub_attn.reshape(b, s, -1)
    if current_rank == nccl_info.sp_size - 1:
        # sub_seq_len = sub_attn.shape[1]
        img_attn, part_txt_attn = torch.split(sub_attn, [s - text_length, text_length], dim=1)
    else:
        img_attn = sub_attn
        part_txt_attn = torch.zeros([b, text_length, a*d], dtype=sub_attn.dtype, device=sub_attn.device)
    #broadcast
    torch.distributed.broadcast(part_txt_attn, src=cp_rank_list[nccl_info.sp_size - 1], group=cp_group)
    #b,s,h*d
    txt_attn = torch.zeros_like(encoder_query).reshape(b, encoder_query.shape[1], -1)
    txt_attn[:, 0:text_length, ...] = part_txt_attn
    attn = torch.cat([img_attn, txt_attn], dim=1)
    return attn


full_img_seq_len = -1
sub_seq_len = -1

def preprocess_dynamic_ring_attention(img, text_mask, freqs_cos, freqs_sin):
    global full_img_seq_len
    global sub_seq_len
    full_img_seq_len = img.shape[1]
    text_seq_len = text_mask.sum().item()
    seq_len = full_img_seq_len + text_seq_len
    sp_size = nccl_info.sp_size
    sp_rank = nccl_info.rank_within_spgroup
    sub_seq_len = last_sub_seq_len = int(seq_len / sp_size)
    if seq_len % sp_size != 0:
        last_sub_seq_len += seq_len % sp_size
    assert last_sub_seq_len > text_seq_len
    last_img_sub_seq_len = last_sub_seq_len - text_seq_len
    split_size = [sub_seq_len] * (sp_size - 1) + [last_img_sub_seq_len]
    img = torch.split(img, split_size, dim=1)[sp_rank]
    freqs_cos = torch.split(freqs_cos, split_size, dim=0)[sp_rank]
    freqs_sin = torch.split(freqs_sin, split_size, dim=0)[sp_rank]
    cu_seqlens = torch.Tensor([[0, sub_seq_len]] * (sp_size - 1) + [[0, last_sub_seq_len]]).to(img.device).to(
        torch.int32)
    max_seqlen = [sub_seq_len] * (sp_size - 1) + [last_sub_seq_len]
    DynRingAttn.cache_metadata(cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
    return img, freqs_cos, freqs_sin


def postprocess_dynamic_ring_attention(img):
    sp_size = nccl_info.sp_size
    sp_rank = nccl_info.rank_within_spgroup
    if sp_rank == sp_size - 1:
        last_img = torch.zeros([img.shape[0], sub_seq_len, img.shape[-1]], dtype=img.dtype, device=img.device)
        last_img[:, :img.shape[1], ...] = img
        img = last_img
    img = all_gather(img, dim=1, group=nccl_info.sp_group)
    img = img[:, :full_img_seq_len, ...]
    return img

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)

    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude

    return scaled_tensor.contiguous()

def flex_test(Q, K, V, kernel_size):
    mask = get_sliding_tile_attention_mask(kernel_size, (6, 8, 8), (36, 48, 48), 39, 'cuda', 0)
    output = flex_attention(Q, K, V, block_mask=mask)

    return output

if __name__ == '__main__':
    kernel_size_ls = [(6, 1, 6), (6, 6, 1)]
    b, h, d = 2, 24, 128
    causal = False
    mean = 1e-1
    std = 10
    import time
    from tqdm import tqdm
    t1 = time.time()
    for kernel_size in tqdm(kernel_size_ls):
        for _ in range(50):
            torch.manual_seed(0)
            Q = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
            K = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
            V = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
            pt_o = flex_test(Q, K, V, kernel_size)
    t2 = time.time()
    print(t2-t1)
