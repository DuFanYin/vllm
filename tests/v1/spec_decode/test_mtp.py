# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest import mock

import pytest
import torch.nn as nn

from vllm.config import SpeculativeConfig
from vllm.models.deepseek_v4.common.spec_decode import needs_mtp_hidden_buffer


def test_glm_mtp_defers_lm_head(default_vllm_config):
    from vllm.model_executor.models import deepseek_mtp
    from vllm.models.glm5next.common import mtp

    config = mock.MagicMock(
        hidden_size=16,
        rms_norm_eps=1e-5,
        index_topk=8,
        index_kpool=4,
    )
    vllm_config = mock.MagicMock()
    vllm_config.speculative_config.draft_model_config.hf_text_config = config
    vllm_config.scheduler_config.max_num_batched_tokens = 4

    with (
        mock.patch.object(mtp, "Glm5NextDecoderLayer", return_value=nn.Identity()),
        mock.patch.object(deepseek_mtp, "ParallelLMHead") as parallel_lm_head,
        mock.patch.object(mtp.current_platform, "device_type", "cpu"),
    ):
        layer = mtp.Glm5NextMultiTokenPredictorLayer(vllm_config, "model.layers.1")

    parallel_lm_head.assert_not_called()
    assert layer.shared_head.head is None


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (None, False),
        ("ngram", False),
        ("mtp", True),
        ("eagle3", True),
        ("draft_model", True),
        ("dspark", False),
    ],
)
def test_deepseek_v4_mtp_hidden_buffer_skips_dspark(method, expected):
    """Only drafters that read DeepSeek V4's pre-hc_head residual get the
    buffer; DSpark drafts from the target's aux hidden states."""
    spec_config = None
    if method is not None:
        spec_config = SpeculativeConfig.__new__(SpeculativeConfig)
        object.__setattr__(spec_config, "method", method)
    assert needs_mtp_hidden_buffer(spec_config) is expected
