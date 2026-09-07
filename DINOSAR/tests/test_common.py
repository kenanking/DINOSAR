import math

import torch

from dinosar.common import get_peak_flops, log_message


def test_get_peak_flops_cpu_returns_inf():
    assert math.isinf(get_peak_flops(torch.device("cpu")))


def test_log_message_prints_on_rank_zero(capsys, monkeypatch):
    monkeypatch.delenv("RANK", raising=False)

    log_message("hello")

    assert capsys.readouterr().out == "hello\n"


def test_log_message_suppresses_nonzero_rank(capsys, monkeypatch):
    monkeypatch.setenv("RANK", "1")

    log_message("hidden")

    assert capsys.readouterr().out == ""


def test_log_message_can_print_on_all_ranks(capsys, monkeypatch):
    monkeypatch.setenv("RANK", "1")

    log_message("shown", all_ranks=True)

    assert capsys.readouterr().out == "shown\n"
