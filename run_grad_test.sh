#!/bin/bash
# SparseFlashAttentionGrad 算子测试驱动。
# 用法:  ./run_grad_test.sh <子命令> [额外 pytest 参数]
#
# 子命令(按"每步只新增一个失败来源"的顺序设计):
#   host    第0步 纯 numpy 算法门禁(无需 Ascend):有限差分 + kernel 逻辑 vs golden
#   fwd     第1步 前向依赖算子(反向 golden 依赖它生成 out/softmax 统计)
#   smoke   第2步 反向冒烟:__main__ 单 shape,几秒确认能编译/能跑/不 NaN  ← 日常首选
#   basic   第3步 反向大 shape 功能自检(grid/UB/atomic 稳定性,不比对参考)
#   golden  第4步 反向精度全量 vs numpy golden(D×dtype×mode×block 全矩阵)
#   guards  反向接口守卫:不支持的入参必须正确抛错(无需 NPU 计算,秒级)
#   cann    第5步 对齐 CANN ms.grad(ops.sparse_flash_attention)(仅 D=512,验收门禁)
#   all     顺序跑 fwd→smoke→basic→golden→cann(任一步失败即停)
#
# 何时跑哪步(取决于"改了什么",不必每次全跑):
#   只改 grad host 包装/autotune/布局 → smoke→golden(host/fwd 可跳)
#   改了 grad kernel 计算公式         → host(算法) + smoke→golden
#   改了前向 sparse_flash_attention   → fwd 必跑 + smoke→cann
#   改了 _numpy.py golden             → host 必跑 + smoke→golden
#   日常复测:smoke 绿了再上 golden/cann;smoke 挂了回头查 fwd
#
# 额外参数透传给 pytest,例如:
#   ./run_grad_test.sh golden -k "512 and float16"
#   ./run_grad_test.sh cann   -x

set -u

export ASCEND_RT_VISIBLE_DEVICES=8
export TRITON_END=mindspore
export TRITON_BACKEND=mindspore
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

GRAD_TEST=test_sfa_grad_triton.py
FWD_TEST=test_sfa_triton.py

cmd=${1:-}
shift 2>/dev/null || true

run_host() {
    # 第0步:host 算法校验,不依赖 Ascend。失败说明数学/golden/kernel 公式有错。
    echo ">>> [0] host 算法门禁 (numpy, 无需 NPU)"
    python verify_grad_algo.py
}

run_fwd() {
    echo ">>> [1] 前向依赖算子 (test_sfa_triton.py)"
    pytest --forked "$FWD_TEST" -v "$@"
}

run_smoke() {
    echo ">>> [2] 反向冒烟 (__main__ 单 shape)"
    python "$GRAD_TEST"
}

run_basic() {
    echo ">>> [3] 反向功能自检 (大 shape, 无参考对比)"
    pytest --forked "$GRAD_TEST" -v -k test_basic "$@"
}

run_golden() {
    echo ">>> [4] 反向精度全量 vs numpy golden"
    pytest --forked "$GRAD_TEST" -v -k test_golden "$@"
}

run_guards() {
    echo ">>> 反向接口守卫 (不支持入参须抛错, 无需 NPU)"
    pytest --forked "$GRAD_TEST" -v -k test_guards "$@"
}

run_cann() {
    echo ">>> [5] 对齐 CANN ms.grad(ops.sparse_flash_attention)"
    pytest --forked "$GRAD_TEST" -v -k test_accuracy "$@"
}

case "$cmd" in
    host)   run_host ;;
    fwd)    run_fwd "$@" ;;
    smoke)  run_smoke ;;
    basic)  run_basic "$@" ;;
    golden) run_golden "$@" ;;
    guards) run_guards "$@" ;;
    cann)   run_cann "$@" ;;
    all)
        run_fwd && run_smoke && run_basic && run_golden && run_cann
        ;;
    *)
        echo "用法: $0 {host|fwd|smoke|basic|golden|guards|cann|all} [pytest 参数]"
        echo "  host   numpy 算法门禁(无需 NPU)"
        echo "  fwd    前向依赖算子"
        echo "  smoke  反向冒烟(日常首选, 单 shape)"
        echo "  basic  反向大 shape 功能自检"
        echo "  golden 反向精度全量 vs numpy golden"
        echo "  guards 接口守卫(不支持入参须抛错, 无需 NPU)"
        echo "  cann   对齐 CANN 反向(D=512 验收)"
        echo "  all    fwd→smoke→basic→golden→cann 顺序跑"
        exit 1
        ;;
esac
