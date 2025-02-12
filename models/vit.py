# coding=utf-8
import logging

import pytorch_lightning as pl

import torch.nn as nn
import torch.optim as optim

import torchmetrics.classification as cls
import torchvision.models as models
from torchvision import transforms
from torchvision.models.vision_transformer import VisionTransformer

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from PIL import Image

from utils.scheduler import WarmupCosineSchedule

logger = logging.getLogger(__name__)

class ViT(pl.LightningModule):
    def __init__(self, config):
        super(ViT, self).__init__()

        if (config["pretrained"]):
            self.vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit_model = VisionTransformer(image_size  = config["img_size"],
                                               patch_size  = config["patch_size"],
                                               num_layers  = config["num_layers"],
                                               num_heads   = config["num_heads"],
                                               hidden_dim  = config["hidden_dim"],
                                               mlp_dim     = config["mlp_dim"],
                                               num_classes = config["num_classes"])

        self.num_classes     = config["num_classes"]
        self.vit_model.heads = nn.Identity()
        self.head    = nn.Linear(config["hidden_size"], config["num_classes"])
        self.loss    = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=self.num_classes)

        self.lr = config["learning_rate"]
        self.weight_decay = config["weight_decay"]

        self.metrics = {
            "accuracy" : cls.MulticlassAccuracy        (num_classes=self.num_classes).to("cuda"),
            "roc_auc"  : cls.MulticlassAUROC           (num_classes=self.num_classes).to("cuda"),
            "f1_score" : cls.MulticlassF1Score         (num_classes=self.num_classes).to("cuda"),
            "roc"      : cls.MulticlassROC             (num_classes=self.num_classes).to("cuda"),
            "cm"       : cls.MulticlassConfusionMatrix (num_classes=self.num_classes).to("cuda"),
            }

    def forward(self, x):
        x      = self.vit_model(x)
        logits = self.head(x)

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)

        self.log("train_loss", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        val_loss = self.loss(logits, y)

        for _, metric in self.metrics.items():
            metric.update(logits, y.int())

        self.log("val_loss", val_loss, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):

        if self.trainer.sanity_checking:
            return

        for metric_name, metric in self.metrics.items():
            if metric_name == "roc" or metric_name == "cm":
                continue

            self.log(metric_name, metric.compute(), prog_bar=True)

        self._plot_confusion_matrix()
        self._plot_roc()

        for metric_name, metric in self.metrics.items():
            metric.reset()

    def _plot_roc(self):
        fig, ax = self.metrics["roc"].plot(score=True)

        fig.set_size_inches(10, 10)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        fig.savefig("img/roc_curve.png")
        plt.close(fig)

        img = Image.open("img/roc_curve.png")
        img_tensor = transforms.ToTensor()(img)
        self.logger.experiment.add_image("ROC Curve", img_tensor, self.current_epoch)

    def _plot_confusion_matrix(self):
        fig, ax = self.metrics["cm"].plot()
        fig.set_size_inches(10, 10)

        ax.set_title("Confusion Matrix")

        fig.savefig("img/confusion_matrix.png")
        plt.close(fig)

        img = Image.open("img/confusion_matrix.png")
        img_tensor = transforms.ToTensor()(img)
        self.logger.experiment.add_image("Confusion Matrix", img_tensor, self.current_epoch)

    def configure_optimizers(self):
        optimizer    = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        lr_scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=1000)
        return [optimizer], [lr_scheduler]
