'''
    Author: Theodor Cheslerean-Boghiu
    Date: August 22nd 2023
    Version 1.0
'''
from collections import OrderedDict

import copy

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import pytorch_lightning as pl
import timm

import torchmetrics as tm
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassAUROC

from torchvision import models


class HistoModel(pl.LightningModule):
    def __init__(self,
              learning_rate: float =1e-3,
              weight_decay: float =1e-3,
              num_classes: int =4):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        
        self.patch_extraction = None
        
        self.patch_feature_extraction = models.resnet50(pretrained=True)
        self.patch_feature_extraction.fc = torch.nn.Linear(in_features=2048, out_features=384, bias=True)
        
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 384))
        
        self.fusion_layer = torch.nn.Sequential(*[
            timm.models.vision_transformer.Block(
                dim=384,
                num_heads=8,
            )
            for i in range(4)])
        
        self.fc_norm = torch.nn.LayerNorm(384)

        self.classifier = torch.nn.Linear(384,4)
        
        self.loss = torch.nn.CrossEntropyLoss()

        macro_metrics = tm.MetricCollection({
                "acc": MulticlassAccuracy(num_classes=num_classes, average='macro'),
                "auc": MulticlassAUROC(num_classes=num_classes, average='macro'),
            })
        per_class_metrics = tm.MetricCollection({
                "per_class_acc": MulticlassAccuracy(num_classes=num_classes, average='none'),
                "per_class_auc": MulticlassAUROC(num_classes=num_classes, average='none'),
            })
        self.macro_train_metrics = macro_metrics.clone(prefix="train/")
        self.macro_val_metrics = macro_metrics.clone(prefix="val/")
        
        self.per_class_train_metrics = per_class_metrics.clone(prefix="train/")
        self.per_class_val_metrics = per_class_metrics.clone(prefix="val/")
    

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def forward(self, x):
        
        outputs = []
        for i in range(x.shape[0]):
            outputs.append(self.patch_feature_extraction(x[i]))
        x = torch.stack(outputs, dim=0)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.fusion_layer(x)
        x = self.fc_norm(x[:, 0])
        x = self.classifier(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss(output, y)
        self.macro_train_metrics(output, y)
        self.per_class_train_metrics(output, y)
        
        self.log("train/loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def on_train_epoch_end(self):
        per_class_metrics = self.per_class_train_metrics.compute()

        for key, value in per_class_metrics.items():
            metrics_dict = {}
            for i, metric in enumerate(value):
                metrics_dict[i] = metric 
            self.logger.experiment.add_scalars(str(key),
                        metrics_dict,
                        global_step=self.current_epoch)
        
        self.log_dict(self.macro_train_metrics.compute(),
            prog_bar=False,
            logger=True)
        self.per_class_train_metrics.reset()
        self.macro_train_metrics.reset()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss(output, y)
        self.macro_val_metrics(output, y)
        self.per_class_val_metrics(output, y)
        
        self.log("val/loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        
        return loss

    def on_validation_epoch_end(self):
        per_class_metrics = self.per_class_val_metrics.compute()

        for key, value in per_class_metrics.items():
            metrics_dict = {}
            for i, metric in enumerate(value):
                metrics_dict[i] = metric 
            self.logger.experiment.add_scalars(str(key),
                        metrics_dict,
                        global_step=self.current_epoch)
            
        self.log_dict(self.macro_val_metrics.compute(),
            prog_bar=True,
            logger=True)
        self.per_class_val_metrics.reset()
        self.macro_val_metrics.reset()
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x), y

# class DINO_SL_Model(pl.LightningModule):
#     def __init__(
#             self,
#             batch_size: int = 256,
#             logits_size: int = 128,
#             vit_type: str = 'vit_small_patch16_224',
#             learning_rate: float = 1e-4,
#             labels: list = None,
#             class_weights: list = None,
#             ssl_pretrained: bool=False,
#             ckpt_path: str = None
#         ):
#         super().__init__()
        
#         self.learning_rate = learning_rate
#         self.labels = labels

#         self.feature_extractor = timm.create_model('vit_small_patch16_224', pretrained=True, global_pool='token')
#         self.n_latent_features = self.feature_extractor.embed_dim
#         self.feature_extractor.head = torch.nn.Identity()
        
#         self.encoder = torch.nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=6)
#         self.fusion = torch.nn.TransformerEncoder(self.encoder, 8)

#         self.classifier = torch.nn.Sequential(OrderedDict([
#             ('linear1', torch.nn.Linear(in_features=384, out_features=logits_size))]))
            
#         def init_weights(m):
#             if isinstance(m, nn.Linear):
#                 torch.nn.init.xavier_uniform(m.weight)
#                 m.bias.data.fill_(0.01)

#         self.loss = nn.CrossEntropyLoss(torch.tensor(class_weights))
        
#         macro_metrics = tm.MetricCollection({
#                 "acc": MulticlassAccuracy(num_classes=logits_size, average='macro'),
#                 "auc": MulticlassAUROC(num_classes=logits_size, average='macro'),
#             })
#         per_class_metrics = tm.MetricCollection({
#                 "per_class_acc": MulticlassAccuracy(num_classes=logits_size, average='none'),
#                 "per_class_auc": MulticlassAUROC(num_classes=logits_size, average='none'),
#             })
#         self.macro_train_metrics = macro_metrics.clone(prefix="train/")
#         self.macro_val_metrics = macro_metrics.clone(prefix="val/")
        
#         self.per_class_train_metrics = per_class_metrics.clone(prefix="train/")
#         self.per_class_val_metrics = per_class_metrics.clone(prefix="val/")

#     def forward(self, x):
        
        
        
#         embedding = self.encoder(x).flatten(start_dim=1)
#         return self.classifier(embedding)

#     def training_step(self, batch, batch_idx):
#         _, image, label = batch
               
#         output = self(image)
        
#         loss = self.loss(output, label)
        
#         self.macro_train_metrics(output, label)
#         self.per_class_train_metrics(output, label)
        
#         self.log("train/loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        
#         return loss

#     def on_train_epoch_end(self):
#         per_class_metrics = self.per_class_train_metrics.compute()

#         for key, value in per_class_metrics.items():
#             metrics_dict = {}
#             for i, metric in enumerate(value):
#                 metrics_dict[str(self.labels[i])] = metric 
#             self.logger.experiment.add_scalars(str(key),
#                         metrics_dict,
#                         global_step=self.current_epoch)
        
#         self.log_dict(self.macro_train_metrics.compute(),
#             prog_bar=False,
#             logger=True)
#         self.per_class_train_metrics.reset()
#         self.macro_train_metrics.reset()
    
#     def validation_step(self, batch, batch_idx):
#         _, image, label = batch
        
#         output = self(image)
        
#         loss = self.loss(output, label)
        
#         self.macro_val_metrics(output, label)
#         self.per_class_val_metrics(output, label)
        
#         self.log("val/loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        
#         return loss

#     def on_validation_epoch_end(self):
#         per_class_metrics = self.per_class_val_metrics.compute()

#         for key, value in per_class_metrics.items():
#             metrics_dict = {}
#             for i, metric in enumerate(value):
#                 metrics_dict[str(self.labels[i])] = metric 
#             self.logger.experiment.add_scalars(str(key),
#                         metrics_dict,
#                         global_step=self.current_epoch)
            
#         self.log_dict(self.macro_val_metrics.compute(),
#             prog_bar=True,
#             logger=True)
#         self.per_class_val_metrics.reset()
#         self.macro_val_metrics.reset()