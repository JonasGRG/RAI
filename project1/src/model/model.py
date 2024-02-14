import timm
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy

class DermatologyModel(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        # Initialize ResNet-18 from timm
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)
        self.accuracy = Accuracy(task="binary", num_classes=2)

    def forward(self, x):
        # Forward pass through the model
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Training step
        images, labels = batch
        outputs = self(images)
        labels = labels.to(torch.int64)
        loss = F.cross_entropy(outputs, labels)
        labels = F.one_hot(labels)
        acc = self.accuracy(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        images, labels = batch
        outputs = self(images)
        labels = labels.to(torch.int64)
        loss = F.cross_entropy(outputs, labels)
        labels = F.one_hot(labels)
        acc = self.accuracy(outputs, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        return optimizer
