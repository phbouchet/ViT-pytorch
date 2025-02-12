# coding=utf-8
import json
import argparse
import logging

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from models.vit import ViT
from utils.data_utils import get_loader

logger = logging.getLogger(__name__)

def load_clargs():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--expt", required=True, help="Path to JSON experiment config file. Found in experiments/.")
    parser.add_argument("--mode", required=True, help="Specify training or evaluation. Options: train, eval.")
    parser.add_argument("--ckpt_path", required=False, help="Path to checkpoint for evaluation.")

    clargs = parser.parse_args()

    return clargs

def load_args(config_path : str = "experiments/hymenoptera_pretrain.json"):
    with open(config_path, "r") as fp:
        args = json.load(fp)

    return args

def load_model(args):

    model = ViT(config=args)

    if ("pretrained_dir" in args):
        model.load_from(np.load(args["pretrained_dir"]))

    return model

def train(args, model):

    train_loader, test_loader = get_loader(args)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1,
                                                       mode="max",
                                                       monitor="accuracy",
                                                       filename='{epoch}-{val_loss:.2f}-{accuracy:.2f}')

    tb_logger = pl_loggers.TensorBoardLogger(name=args["name"], save_dir=args["logdir"])

    trainer = pl.Trainer(max_epochs = args["max_epochs"],
                         callbacks  = [checkpoint_callback],
                         logger     = tb_logger,
                         check_val_every_n_epoch=1)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

def eval(args, model):

    _, test_loader = get_loader(args)

    trainer = pl.Trainer()
    trainer.validate(model, test_loader)

def main():

    clargs = load_clargs()
    args   = load_args(clargs.expt)

    model  = load_model(args)

    if (clargs.mode.lower() == "train"):
        train(args, model)

    elif (clargs.mode.lower() == "eval"):
        ckpt_path = clargs.ckpt_path
        model = ViT.load_from_checkpoint(ckpt_path, config=args)

        eval(args, model)

if __name__ == "__main__":
    main()
