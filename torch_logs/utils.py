from .imports import *

import logging, sys, atexit
from pathlib import Path


def capture_text_output(unbuffered=True):
    if unbuffered:
        os.environ["PYTHONUNBUFFERED"] = "1"

    configure_text_logging()
    tee_output()
    print_on_quit()


def configure_text_logging():
    py_logger = logging.getLogger("torch_loops")

    handler = logging.FileHandler("loops.log")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
        )
    )
    py_logger.addHandler(handler)
    py_logger.setLevel(logging.INFO)


def tee_output():
    tee = subprocess.Popen(["tee", "stdout.log"], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())  # type: ignore

    tee = subprocess.Popen(["tee", "stderr.log"], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())  # type: ignore


def print_on_quit():
    atexit.register(lambda: print("QUIT", file=sys.stderr))


def latest_numbered_directory(path: str):
    nums = set()

    if Path(path).exists():
        for subpath in Path(path).iterdir():
            if subpath.is_dir():
                try:
                    nums.add(int(subpath.name))
                except ValueError:
                    pass

    return -1 if len(nums) == 0 else max(nums)
