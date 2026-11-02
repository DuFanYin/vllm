# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level parity: ReplaySSM standard decode vs the baseline SSM kernel."""

import pytest

from vllm.platforms import current_platform
from vllm.v1.metrics.reader import Counter

from ...models.utils import check_logprobs_close
from ...utils import large_gpu_mark, multi_gpu_test

# FlashInfer 0.7.0's standard SSU cannot compile Nano's DIM=80 configuration.
MAMBA2_FLASHINFER_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
requires_flashinfer = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="FlashInfer Mamba SSU is CUDA-only"
)
FLASHINFER_MODELS = [
    pytest.param(
        MAMBA2_FLASHINFER_MODEL, marks=[large_gpu_mark(min_gb=40), requires_flashinfer]
    ),
]

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
]
# The NVFP4 model is not reproducible across engine instances; these prompts
# avoid near-ties and decode confidently past the replay window.
FLASHINFER_PROMPTS = [
    "The capital of France is",
    "It was the best of times, it was the worst of times,",
]


@pytest.mark.parametrize("model_name", FLASHINFER_MODELS)
def test_replayssm_flashinfer_decode_matches_baseline(vllm_runner, model_name):
    # Compare logprobs, not greedy ids: ReplaySSM's fp arithmetic can flip a
    # near-tie. Baseline and ReplaySSM use the same backend so only ReplaySSM
    # behavior varies.
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        mamba_backend="flashinfer",
    )
    with vllm_runner(model_name, **common) as llm:
        baseline = llm.generate_greedy_logprobs(
            FLASHINFER_PROMPTS, max_tokens=32, num_logprobs=5
        )
    with vllm_runner(
        model_name,
        use_replayssm=True,
        replayssm_buffer_len=16,
        **common,
    ) as llm:
        replay = llm.generate_greedy_logprobs(
            FLASHINFER_PROMPTS, max_tokens=32, num_logprobs=5
        )

    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=replay,
        name_0="baseline",
        name_1="replayssm_flashinfer",
    )


@pytest.mark.parametrize("model_name", FLASHINFER_MODELS)
def test_replayssm_flashinfer_spec_decode_matches_baseline(vllm_runner, model_name):
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 3,
            "prompt_lookup_max": 3,
        },
    )
    with vllm_runner(model_name, mamba_backend="flashinfer", **common) as llm:
        baseline = llm.generate_greedy_logprobs(
            FLASHINFER_PROMPTS, max_tokens=32, num_logprobs=5
        )
    with vllm_runner(
        model_name,
        use_replayssm=True,
        replayssm_buffer_len=16,
        mamba_backend="flashinfer",
        **common,
    ) as llm:
        replay = llm.generate_greedy_logprobs(
            FLASHINFER_PROMPTS, max_tokens=32, num_logprobs=5
        )

    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=replay,
        name_0="baseline_spec",
        name_1="replayssm_flashinfer_spec",
    )


@multi_gpu_test(num_gpus=2)
@large_gpu_mark(min_gb=40)
@requires_flashinfer
def test_replayssm_flashinfer_mtp(vllm_runner):
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        mamba_backend="flashinfer",
        tensor_parallel_size=2,
        disable_log_stats=False,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
    )
    with vllm_runner(
        MAMBA2_FLASHINFER_MODEL,
        use_replayssm=True,
        replayssm_buffer_len=16,
        **common,
    ) as llm:
        outputs = llm.generate_greedy(PROMPTS, max_tokens=32)
        draft_count = sum(
            metric.value
            for metric in llm.llm.get_metrics()
            if isinstance(metric, Counter)
            and metric.name == "vllm:spec_decode_num_drafts"
        )

    # At least one request must run past the 16-token replay window; another
    # may legitimately stop early on EOS.
    assert any(len(token_ids) > 16 for token_ids, _ in outputs)
    assert draft_count > 0
