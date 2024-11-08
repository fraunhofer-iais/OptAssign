from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import anomaly_mode

from environment import Reset_State, Step_State
from exceptions import FunctionOrderError

anomaly_mode.set_detect_anomaly(True)


class Model(nn.Module):

    def __init__(self, num_resources, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = Encoder(**model_params)
        self.decoder = Decoder(num_resources, **model_params)
        self.encoded_nodes: Optional[Tensor] = None
        # shape: (batch, problem_size, EMBEDDING_DIM)

    def pre_forward(self, reset_state: Reset_State):
        self.encoded_nodes = self.encoder(reset_state.problems)
        assert self.encoded_nodes is not None
        # shape: (batch, problem_size, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state: Step_State):

        if self.encoded_nodes is None:
            raise FunctionOrderError(self.forward, self.pre_forward)

        batch_size = state.BATCH_IDX.size(0)
        problem_size = state.POMO_IDX.size(1)

        if state.current_node is None:
            selected = torch.arange(problem_size)[None, :].expand(
                batch_size, problem_size
            )  # list of node indices per batch
            prob = torch.ones(size=(batch_size, problem_size))

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, problem_size, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, problem_size, embedding)
            probs: Tensor = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
            # shape: (batch, problem_size, problem_size)

            if self.training or self.model_params["eval_type"] == "softmax":
                selected = (
                    probs.reshape(batch_size * problem_size, -1)
                    .multinomial(1)
                    .squeeze(dim=1)
                    .reshape(batch_size, problem_size)
                )
                # shape: (batch, problem_size)

                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(
                    batch_size, problem_size
                )
                # shape: (batch, problem_size)

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, problem_size)

                prob = None

        return selected, prob

    def get_norm_of_gradients(self) -> float:
        total_norm = 0
        parameters = [
            p for p in self.parameters() if p.grad is not None and p.requires_grad
        ]
        for p in parameters:
            assert p.grad is not None, f"Gradient of {p} is None"
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def get_norm_of_gradients_of_resource_times_related(self) -> float:
        total_norm = 0
        parameters = [
            p
            for p in self.decoder.times_encoder.parameters()
            if p.grad is not None and p.requires_grad
        ]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm


def _get_encoding(encoded_nodes: Tensor, node_index_to_pick: Tensor):
    # encoded_nodes.shape: (batch, problem_size, embedding)
    # node_index_to_pick.shape: (batch, problem_size)

    batch_size = node_index_to_pick.size(0)
    problem_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(
        batch_size, problem_size, embedding_dim
    )
    # shape: (batch, problem_size, embedding)

    return encoded_nodes.gather(dim=1, index=gathering_index)


########################################
# ENCODER
########################################


class Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        encoder_layer_num = self.model_params["encoder_layer_num"]

        self.embedding = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList(
            [EncoderLayer(**model_params) for _ in range(encoder_layer_num)]
        )

    def forward(self, data: Tensor):
        # data.shape: (batch, problem_size, 3)

        embedded_input = self.embedding(data)
        # shape: (batch, problem_size, embedding)

        out = embedded_input
        for layer in self.layers:
            out = layer(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1: Tensor):
        # input.shape: (batch, problem_size, EMBEDDING_DIM)
        head_num = self.model_params["head_num"]

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem_size, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem_size, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem_size, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        return self.addAndNormalization2(out1, out2)
        # shape: (batch, problem_size, EMBEDDING_DIM)


########################################
# DECODER
########################################


class Decoder(nn.Module):
    def __init__(self, num_resources, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_nodes: torch.Tensor):
        # encoded_nodes.shape: (batch, problem_size, embedding)
        head_num = self.model_params["head_num"]

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem_size, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem_size)

    def set_q1(self, encoded_q1: Tensor):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or problem_size
        head_num = self.model_params["head_num"]

        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node: Tensor, ninf_mask: Tensor):
        # encoded_last_node.shape: (batch, problem_size, embedding)
        # ninf_mask.shape: (batch, problem_size, problem_size)
        # current_resource_assignment_times.shape (batch, problem_size, resources)

        head_num = self.model_params["head_num"]

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (batch, head_num, problem_size, qkv_dim)

        q = self.q_first + q_last
        # shape: (batch, head_num, problem_size, qkv_dim)

        if self.k is None or self.v is None:
            raise FunctionOrderError(
                self.forward,
                self.set_kv,
                f"'{self.set_kv.__qualname__}' is usually called by '{Model.pre_forward.__qualname__}' which should be called before '{self.forward.__qualname__}' is called.",
            )
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, problem_size, head_num*qkv_dim)

        multi_head_attention_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem_size, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        if self.single_head_key is None:
            raise FunctionOrderError(
                self.forward,
                self.set_kv,
                f"'{self.set_kv.__qualname__}' is usually called by '{Model.pre_forward.__qualname__}' which should be called before '{self.forward.__qualname__}' is called.",
            )
        score = torch.matmul(multi_head_attention_out, self.single_head_key)
        # shape: (batch, problem_size, problem_size)

        sqrt_embedding_dim = self.model_params["embedding_dim"] ** (1 / 2)
        logit_clipping = self.model_params["logit_clipping"]

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, problem_size, problem_size)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        # shape: (batch, problem_size, problem_size)

        return F.softmax(score_masked, dim=2)


########################################
# NN SUB CLASS / FUNCTIONS
########################################


def reshape_by_heads(qkv: Tensor, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, head_num, n, key_dim)

    return q_reshaped.transpose(1, 2)


def multi_head_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    rank2_ninf_mask: Optional[Tensor] = None,
    rank3_ninf_mask: Optional[Tensor] = None,
):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem_size, key_dim)
    # rank2_ninf_mask.shape: (batch, problem_size)
    # rank3_ninf_mask.shape: (batch, group, problem_size)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem_size)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(
            batch_s, head_num, n, input_s
        )
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(
            batch_s, head_num, n, input_s
        )

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem_size)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num*key_dim)

    return out_transposed.reshape(batch_s, n, head_num * key_dim)


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.norm = nn.InstanceNorm1d(
            embedding_dim, affine=True, track_running_stats=False
        )

    def forward(self, input1: Tensor, input2: Tensor):
        # input.shape: (batch, problem_size, embedding)

        added = input1 + input2
        # shape: (batch, problem_size, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem_size)

        normalized = self.norm(transposed)
        # shape: (batch, problem_size, embedding)

        return normalized.transpose(1, 2)


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1: Tensor):
        # input.shape: (batch, problem_size, embedding)

        return self.W2(F.relu(self.W1(input1)))
