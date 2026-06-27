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
"""
Custom operator wrappers for dense LightningIndexer CANN baselines.

Provides ready-to-use MindSpore Cells for static graph (GRAPH_MODE) execution
via ops.Custom + CustomRegOp:
  - aclnnDenseLightningIndexerSoftmaxLse
  - aclnnDenseLightningIndexerGradKLLoss
"""
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops import DataType, CustomRegOp
import mindspore.common.dtype as mstype

INT64_MAX = 9223372036854775807

_LSE_ACLNN_WORKSPACE_SIGNATURE = (
    "const aclTensor* queryIndex, const aclTensor* keyIndex, "
    "const aclTensor* weight, "
    "const aclIntArray* actualSeqLengthsQueryOptional, "
    "const aclIntArray* actualSeqLengthsKeyOptional, "
    "const char* layoutOptional, int64_t sparseMode, "
    "int64_t preTokens, int64_t nextTokens, "
    "aclTensor* softmaxMaxOut, aclTensor* softmaxSumOut, "
    "uint64_t* workspaceSize, aclOpExecutor** executor"
)

_GRAD_ACLNN_WORKSPACE_SIGNATURE = (
    "const aclTensor* query, const aclTensor* key, "
    "const aclTensor* queryIndex, const aclTensor* keyIndex, "
    "const aclTensor* weights, const aclTensor* softmaxMax, "
    "const aclTensor* softmaxSum, const aclTensor* softmaxMaxIndex, "
    "const aclTensor* softmaxSumIndex, const aclTensor* queryRope, "
    "const aclTensor* keyRope, const aclIntArray* actualSeqQlen, "
    "const aclIntArray* actualSeqKlen, double scaleValue, "
    "const char* layout, int64_t sparseMode, int64_t preTokens, "
    "int64_t nextTokens, "
    "aclTensor* dQueryIndex, aclTensor* dKeyIndex, "
    "aclTensor* dWeights, aclTensor* loss, "
    "uint64_t* workspaceSize, aclOpExecutor** executor"
)


def _build_lse_reg_info():
    """Build CustomRegOp registration info for aclnnDenseLightningIndexerSoftmaxLse."""
    return CustomRegOp("aclnnDenseLightningIndexerSoftmaxLse") \
        .input(0, "query_index", "required") \
        .input(1, "key_index", "required") \
        .input(2, "weight", "required") \
        .attr("actual_seq_qlen", "optional", "listInt") \
        .attr("actual_seq_klen", "optional", "listInt") \
        .attr("layout", "required", "str") \
        .attr("sparse_mode", "required", "int") \
        .attr("pre_tokens", "required", "int") \
        .attr("next_tokens", "required", "int") \
        .output(0, "softmax_max", "required") \
        .output(1, "softmax_sum", "required") \
        .dtype_format(
            DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
            DataType.F32_Default, DataType.F32_Default,
        ) \
        .dtype_format(
            DataType.BF16_Default, DataType.BF16_Default, DataType.BF16_Default,
            DataType.F32_Default, DataType.F32_Default,
        ) \
        .target("Ascend") \
        .get_op_info()


def _infer_lse_shape(*args):
    """Infer output shapes for dense index softmax LSE."""
    qi_shape = args[0]
    ki_shape = args[1]
    if len(qi_shape) == 3:
        t1 = qi_shape[0]
        nidx2 = ki_shape[1] if len(ki_shape) > 1 else 1
        out_shape = [nidx2, t1]
    else:
        b = qi_shape[0]
        s1 = qi_shape[1]
        nidx2 = ki_shape[2] if len(ki_shape) > 2 else 1
        out_shape = [b, nidx2, s1]
    return [out_shape, out_shape]


def _infer_lse_dtype(*args):
    """Infer output dtypes for dense index softmax LSE."""
    _ = args
    return [mstype.float32, mstype.float32]


class DenseLightningIndexerSoftmaxLse(Cell):
    """aclnnDenseLightningIndexerSoftmaxLse wrapped as a MindSpore Cell."""

    def __init__(self):
        super().__init__()
        reg_info = _build_lse_reg_info()
        self._custom_op = ops.Custom(
            "aclnnDenseLightningIndexerSoftmaxLse",
            out_shape=_infer_lse_shape,
            out_dtype=_infer_lse_dtype,
            func_type="aot",
            bprop=None,
            reg_info=reg_info,
        ).add_prim_attr("value_depend", [3, 4])
        self._custom_op._generate_get_worspace_size_func_by_types(
            _LSE_ACLNN_WORKSPACE_SIGNATURE)

    def construct(self, query_index, key_index, weight,
                  actual_seq_qlen=None, actual_seq_klen=None,
                  layout="BSND", sparse_mode=3,
                  pre_tokens=INT64_MAX, next_tokens=INT64_MAX):
        """Forward pass. See class docstring for argument details."""
        if actual_seq_qlen is None and len(query_index.shape) == 4:
            actual_seq_qlen = [query_index.shape[1]] * query_index.shape[0]
        if actual_seq_klen is None and len(key_index.shape) == 4:
            actual_seq_klen = [key_index.shape[1]] * key_index.shape[0]
        return self._custom_op(
            query_index, key_index, weight,
            actual_seq_qlen, actual_seq_klen,
            layout, sparse_mode,
            pre_tokens, next_tokens)


def _build_grad_reg_info():
    """Build CustomRegOp registration info for aclnnDenseLightningIndexerGradKLLoss."""
    return CustomRegOp("aclnnDenseLightningIndexerGradKLLoss") \
        .input(0, "query", "required") \
        .input(1, "key", "required") \
        .input(2, "query_index", "required") \
        .input(3, "key_index", "required") \
        .input(4, "weights", "required") \
        .input(5, "softmax_max", "required") \
        .input(6, "softmax_sum", "required") \
        .input(7, "softmax_max_index", "required") \
        .input(8, "softmax_sum_index", "required") \
        .input(9, "query_rope", "required") \
        .input(10, "key_rope", "required") \
        .attr("actual_seq_qlen", "optional", "listInt") \
        .attr("actual_seq_klen", "optional", "listInt") \
        .attr("scale_value", "required", "float") \
        .attr("layout", "required", "str") \
        .attr("sparse_mode", "required", "int") \
        .attr("pre_tokens", "required", "int") \
        .attr("next_tokens", "required", "int") \
        .output(0, "d_query_index", "required") \
        .output(1, "d_key_index", "required") \
        .output(2, "d_weights", "required") \
        .output(3, "loss", "required") \
        .dtype_format(
            DataType.F16_Default, DataType.F16_Default,
            DataType.F16_Default, DataType.F16_Default,
            DataType.F16_Default, DataType.F32_Default,
            DataType.F32_Default, DataType.F32_Default,
            DataType.F32_Default, DataType.F16_Default,
            DataType.F16_Default, DataType.F16_Default,
            DataType.F16_Default, DataType.F16_Default,
            DataType.F32_Default,
        ) \
        .dtype_format(
            DataType.BF16_Default, DataType.BF16_Default,
            DataType.BF16_Default, DataType.BF16_Default,
            DataType.BF16_Default, DataType.F32_Default,
            DataType.F32_Default, DataType.F32_Default,
            DataType.F32_Default, DataType.BF16_Default,
            DataType.BF16_Default, DataType.BF16_Default,
            DataType.BF16_Default, DataType.BF16_Default,
            DataType.F32_Default,
        ) \
        .target("Ascend") \
        .get_op_info()


def _infer_grad_shape(*args):
    """Infer output shapes for dense grad KL loss."""
    qi_s, ki_s, w_s = args[2], args[3], args[4]
    return [qi_s, ki_s, w_s, [1]]


def _infer_grad_dtype(*args):
    """Infer output dtypes for dense grad KL loss."""
    qi_t, ki_t, w_t = args[2], args[3], args[4]
    return [qi_t, ki_t, w_t, mstype.float32]


class DenseLightningIndexerGradKLLoss(Cell):
    """aclnnDenseLightningIndexerGradKLLoss wrapped as a MindSpore Cell."""

    def __init__(self):
        super().__init__()
        reg_info = _build_grad_reg_info()
        self._custom_op = ops.Custom(
            "aclnnDenseLightningIndexerGradKLLoss",
            out_shape=_infer_grad_shape,
            out_dtype=_infer_grad_dtype,
            func_type="aot",
            bprop=None,
            reg_info=reg_info,
        ).add_prim_attr("value_depend", [11, 12])
        self._custom_op._generate_get_worspace_size_func_by_types(
            _GRAD_ACLNN_WORKSPACE_SIGNATURE)

    def construct(self, query, key, query_index, key_index, weights,
                  softmax_max, softmax_sum,
                  softmax_max_index, softmax_sum_index,
                  scale_value=1.0,
                  query_rope=None, key_rope=None,
                  actual_seq_qlen=None, actual_seq_klen=None,
                  layout="BSND", sparse_mode=3,
                  pre_tokens=INT64_MAX, next_tokens=INT64_MAX):
        """Forward pass. See class docstring for argument details."""
        if query_rope is None or key_rope is None:
            raise ValueError(
                "query_rope and key_rope are required and cannot be None. "
                "MindSpore ops.Custom does not support None tensor inputs.")
        if actual_seq_qlen is None and len(query_index.shape) == 4:
            actual_seq_qlen = [query_index.shape[1]] * query_index.shape[0]
        if actual_seq_klen is None and len(key_index.shape) == 4:
            actual_seq_klen = [key_index.shape[1]] * key_index.shape[0]
        return self._custom_op(
            query, key, query_index, key_index, weights,
            softmax_max, softmax_sum,
            softmax_max_index, softmax_sum_index,
            query_rope, key_rope,
            actual_seq_qlen, actual_seq_klen,
            scale_value, layout, sparse_mode,
            pre_tokens, next_tokens)
