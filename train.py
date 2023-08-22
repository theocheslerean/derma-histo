from pytorch_lightning.cli import LightningCLI

from model import HistoModel
from dataset import WrapperDataset

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    cli = LightningCLI(model_class=HistoModel, datamodule_class=WrapperDataset)
