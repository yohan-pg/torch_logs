import pytest 

import warnings
warnings.simplefilter("ignore")

import pandas as pd 
import torch.nn as nn 
import os 
import shutil


@pytest.fixture
def tmp_dir():
    cwd = os.getcwd()
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
    os.mkdir("tmp")
    os.chdir("tmp")
    yield
    os.chdir(cwd)


@pytest.fixture()
def losses():
    return pd.DataFrame(dict(a=[1.0, 2.0], b=[3.0, 4.0]))


@pytest.fixture()
def model():
    return nn.Linear(2, 2)

