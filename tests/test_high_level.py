from torch_logs.imports import *
import torch_logs.high_level as tl

from .fixtures import *


def test_log_progress(tmp_dir, losses):
    tl.log_progress(losses)

    assert os.path.exists("losses.csv")
    assert os.path.exists("plot.png")
    assert os.path.exists("plot.html")


def test_log_checkpoint(tmp_dir, model):
    tl.log_checkpoint(model)

    assert os.path.exists("model.pt")
    assert os.path.exists("weights.pt")


def test_experiment(
    tmp_dir,
):
    with tl.experiment("test_experiment"):
        pass
    
    assert os.path.exists("test_experiment/pid.txt")
    assert os.path.exists("test_experiment/config.py")
    assert os.path.exists("test_experiment/")

    with tl.experiment("test_experiment"):
        assert os.path.exists("pid.txt")
        assert os.path.exists("config.py")

    assert os.path.exists("test_experiment/pid.txt")
    assert os.path.exists("test_experiment/config.py")
    assert os.path.exists("test_experiment/")


def test_simple_logs_logging(
    tmp_dir, model, losses, progress_freq=5, checkpoint_freq=10
):
    tl.simple_logs(
        "test_experiment",
        0,
        model,
        losses,
        progress_freq=progress_freq,
        checkpoint_freq=checkpoint_freq,
    )
    
    assert os.path.exists("test_experiment/")
    assert os.path.exists("test_experiment/losses.csv")
    assert os.path.exists("test_experiment/plot.png")
    assert os.path.exists("test_experiment/plot.html")
    assert os.path.exists("test_experiment/config.py")
    assert os.path.exists("test_experiment/pid.txt")
    assert os.path.exists("test_experiment/000000/model.pt")
    assert os.path.exists("test_experiment/000000/weights.pt")


def test_simple_logs_events(
    tmp_dir, model, losses, progress_freq=5, checkpoint_freq=10
):
    def simple_logs_for_iter(i: int):
        return tl.simple_logs(
            "test_experiment",
            i,
            model,
            losses,
            progress_freq=progress_freq,
            checkpoint_freq=checkpoint_freq,
        )

    events = simple_logs_for_iter(0)
    assert events == set([tl.Event.INIT, tl.Event.PROGRESS, tl.Event.CHECKPOINT])

    events = simple_logs_for_iter(1)
    assert events == set([])

    events = simple_logs_for_iter(progress_freq)
    assert events == set([tl.Event.PROGRESS])

    events = simple_logs_for_iter(checkpoint_freq)
    assert events == set([tl.Event.PROGRESS, tl.Event.CHECKPOINT])
