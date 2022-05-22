from torch_logs.imports import *
import torch_logs.low_level as tl

from .fixtures import *


def test_directory(tmp_dir) -> None:
    with tl.directory("test_dir"):
        open("test_file.txt", "w").close()
        assert os.path.exists("test_file.txt")

    # Leaving the with block puts us back in the original directory
    assert os.path.exists("test_dir/")
    assert os.path.exists("test_dir/test_file.txt")

    # The directory is not deleted when we call `directory` again
    with tl.directory("test_dir"):
        assert os.path.exists("test_file.txt")

    # The directory is restored even with exceptions
    try:
        with tl.directory("test_dir"):
            raise Exception
    except:
        pass
    assert os.path.exists("test_dir/")


def test_log_pid(tmp_dir):
    tl.log_pid()
    
    assert os.path.exists("pid.txt")
    assert int(open("pid.txt", "r").read())


def test_log_config(tmp_dir) -> None:
    tl.log_config()

    assert os.path.exists("config.py")


def test_log_losses(tmp_dir, losses) -> None:
    tl.log_losses(losses)

    lines = iter(open("losses.csv", "r"))
    assert next(lines) == "a,b\n"
    assert next(lines) == "1.0,3.0\n"
    assert next(lines) == "2.0,4.0\n"

    with pytest.raises(StopIteration):
        next(lines)


def test_log_weights(tmp_dir, model) -> None:
    tl.log_weights(model)

    assert os.path.exists("weights.pt")


def test_log_model(tmp_dir, model) -> None:
    torch.save(model, "model.pt")

    assert os.path.exists("model.pt")


def test_plot_losses(tmp_dir, losses) -> None:
    tl.plot_losses(losses)

    assert os.path.exists("plot.png")
    assert os.path.exists("plot.html")
