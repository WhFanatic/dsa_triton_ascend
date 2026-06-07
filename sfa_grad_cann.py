# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Custom operator wrapper for aclnnSparseFlashAttentionGrad.

Provides a MindSpore Cell for static graph (GRAPH_MODE) execution via
ops.Custom + CustomRegOp.  No C++ compilation needed.

Usage (BSND, MLA-absorb):
    ms.set_context(mode=ms.GRAPH_MODE)

    cann_grad = SparseFlashAttentionGradCANN(scale_value=scale, sparse_mode=3)
    d_query, d_key, d_value, d_query_rope, d_key_rope = cann_grad(
        query, key, value, sparse_indices,
        d_out, out, softmax_max, softmax_sum,
        query_rope=qr, key_rope=kr,
        actual_seq_lengths_query=act_q, actual_seq_lengths_kv=act_k)

Requirements:
    - MindSpore == master / 2.9.0
    - Ascend 910B with CANN (aclnnSparseFlashAttentionGrad available)
    - GRAPH_MODE
"""
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops import DataType, CustomRegOp
import mindspore.common.dtype as mstype

_ACLNN_WORKSPACE_SIGNATURE = (
    "const aclTensor* query, const aclTensor* key, "
    "const aclTensor* value, const aclTensor* sparseIndices, "
    "const aclTensor* dOut, const aclTensor* out, "
    "const aclTensor* softmaxMax, const aclTensor* softmaxSum, "
    "const aclTensor* actualSeqLengthsQueryOptional, const aclTensor* actualSeqLengthskvOptional, "
    "const aclTensor* queryRopeOptional, const aclTensor* keyRopeOptional, "
    "double scaleValue, int64_t sparseBlockSize, "
    "char* layoutOptional, int64_t sparseMode, "
    "int64_t preTokens, int64_t nextTokens, "
    "bool deterministic, "
    "aclTensor* dQueryOut, aclTensor* dKeyOut, "
    "aclTensor* dValueOut, aclTensor* dQueryRopeOutOptional, "
    "aclTensor* dKeyRopeOutOptional, "
    "uint64_t* workspaceSize, aclOpExecutor** executor"
)


def _build_reg_info():
    return CustomRegOp("aclnnSparseFlashAttentionGrad") \
        .input(0, "query", "required") \
        .input(1, "key", "required") \
        .input(2, "value", "required") \
        .input(3, "sparse_indices", "required") \
        .input(4, "d_out", "required") \
        .input(5, "out", "required") \
        .input(6, "softmax_max", "required") \
        .input(7, "softmax_sum", "required") \
        .input(8, "actual_seq_lengths_query", "required") \
        .input(9, "actual_seq_lengths_kv", "required") \
        .input(10, "query_rope", "required") \
        .input(11, "key_rope", "required") \
        .attr("scale_value", "required", "float") \
        .attr("sparse_block_size", "required", "int") \
        .attr("layout", "required", "str") \
        .attr("sparse_mode", "required", "int") \
        .attr("pre_tokens", "required", "int") \
        .attr("next_tokens", "required", "int") \
        .attr("deterministic", "required", "bool") \
        .output(0, "d_query", "required") \
        .output(1, "d_key", "required") \
        .output(2, "d_value", "required") \
        .output(3, "d_query_rope", "required") \
        .output(4, "d_key_rope", "required") \
        .dtype_format(
            DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.I32_Default,
            DataType.F16_Default, DataType.F16_Default, DataType.F32_Default, DataType.F32_Default,
            DataType.I32_Default, DataType.I32_Default, DataType.F16_Default, DataType.F16_Default,
            DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
        ) \
        .dtype_format(
            DataType.BF16_Default, DataType.BF16_Default, DataType.BF16_Default, DataType.I32_Default,
            DataType.BF16_Default, DataType.BF16_Default, DataType.F32_Default, DataType.F32_Default,
            DataType.I32_Default, DataType.I32_Default, DataType.BF16_Default, DataType.BF16_Default,
            DataType.BF16_Default, DataType.BF16_Default, DataType.BF16_Default, DataType.BF16_Default, DataType.BF16_Default,
        ) \
        .target("Ascend") \
        .get_op_info()


def _infer_shape(*args):
    return [args[0], args[1], args[2], args[10], args[11]]


def _infer_dtype(*args):
    return [args[0], args[1], args[2], args[10], args[11]]


class SparseFlashAttentionGradCANN(Cell):
    """aclnnSparseFlashAttentionGrad wrapped as a MindSpore Cell.

    Returns (d_query, d_key, d_value, d_query_rope, d_key_rope), matching
    the triton SparseFlashAttentionGradTriton output tuple.

    construct args:
        query, key, value, sparse_indices,
        d_out, out, softmax_max, softmax_sum,
        query_rope, key_rope,                    -- required (ops.Custom no None)
        actual_seq_lengths_query, actual_seq_lengths_kv,  -- required (aclTensor*)
    Init attrs (passed to CANN as kernel attrs):
        scale_value, sparse_block_size, layout, sparse_mode,
        pre_tokens, next_tokens, deterministic
    """
    def __init__(self, scale_value=1.0, sparse_block_size=1, layout="BSND",
                 sparse_mode=3, pre_tokens=2147483647, next_tokens=2147483647,
                 deterministic=False):
        super().__init__()
        self.scale_value = scale_value
        self.sparse_block_size = sparse_block_size
        self.layout = layout
        self.sparse_mode = sparse_mode
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.deterministic = deterministic
        reg_info = _build_reg_info()
        self._custom_op = ops.Custom(
            "aclnnSparseFlashAttentionGrad",
            out_shape=_infer_shape,
            out_dtype=_infer_dtype,
            func_type="aot",
            bprop=None,
            reg_info=reg_info,
        ).add_prim_attr("value_depend", [8, 9])
        self._custom_op._generate_get_worspace_size_func_by_types(_ACLNN_WORKSPACE_SIGNATURE)

    def construct(self, query, key, value, sparse_indices,
                  d_out, out, softmax_max, softmax_sum,
                  query_rope=None, key_rope=None,
                  actual_seq_lengths_query=None, actual_seq_lengths_kv=None):
        if query_rope is None or key_rope is None:
            raise ValueError(
                "query_rope and key_rope are required and cannot be None. "
                "MindSpore ops.Custom does not support None tensor inputs.")
        if actual_seq_lengths_query is None or actual_seq_lengths_kv is None:
            raise ValueError(
                "actual_seq_lengths_query and actual_seq_lengths_kv are required "
                "and cannot be None. MindSpore ops.Custom does not support None "
                "tensor inputs.")
        return self._custom_op(
            query, key, value, sparse_indices,
            d_out, out, softmax_max, softmax_sum,
            actual_seq_lengths_query, actual_seq_lengths_kv,
            query_rope, key_rope,
            self.scale_value, self.sparse_block_size,
            self.layout, self.sparse_mode,
            self.pre_tokens, self.next_tokens,
            self.deterministic)