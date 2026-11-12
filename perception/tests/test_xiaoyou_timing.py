"""The probe must reject working audio that misses the board's timing gate."""
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[2] / 'tools/test_xiaoyou_mcp.py'
spec = importlib.util.spec_from_file_location('xiaoyou_mcp_probe', path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize('ttft,rtf,passed', [
    (.7, .6, True), (1., .6, True), (.7, 1., False),
    (1.001, .5, False), (.5, 1.001, False),
    (-.1, .5, False), (.5, 0., False), (float('nan'), .5, False),
])
def test_board_timing_gate(ttft, rtf, passed):
    result = probe.timing_verdict([dict(first_pcm_seconds=ttft, rtf_to_last_pcm=rtf)])
    assert result['passed'] is passed
    assert result['ttft_limit_seconds'] == 1.0


def test_empty_run_cannot_qualify():
    assert not probe.timing_verdict([])['passed']
