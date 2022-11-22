
import pkgutil
from .group import CyclicGroup
from .gcnn import Cake_GroupEquivariantCNN, GroupEquivariantCNN
# We demonstrate our models on the MNIST dataset.
import torchvision
import torch
import pytorch_lightning as pl

import os
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.nn as nn 
import torch.optim as optim
from pytorch_lightning import loggers as pl_loggers

class DataModule(pl.LightningModule):

    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_name, model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
        optimizer = optim.AdamW(
            self.parameters(), **self.hparams.optimizer_hparams)
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log('test_acc', acc, prog_bar=True)

model_dict = {
    'Cake_GCNN': Cake_GroupEquivariantCNN,
    'GCNN': GroupEquivariantCNN
}

def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f"Unknown model name \"{model_name}\". Available models are: {str(model_dict.keys())}"

CHECKPOINT_PATH = './checkpoints'

def train_model(train_loader, test_loader, model_name, save_name=None, device='cpu',**kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    if save_name is None:
        save_name = model_name

    comet_logger = pl_loggers.CometLogger(save_dir="logs/")

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),                          # Where to save models
                         gpus=1 if str(device)=="cuda:0" else 0,                                             # We run on a single GPU (if possible)
                         max_epochs=100,                                                                      # How many epochs to train for if no patience is set
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                     LearningRateMonitor("epoch")],
                         logger=comet_logger,
                        )
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = DataModule.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(12) # To be reproducable
        model = DataModule(model_name=model_name, **kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = DataModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on test set
    val_result = trainer.test(model.to(device), test_loader, verbose=False)
    result = {"val": val_result[0]["test_acc"]}

    return model, result


def train():
    #TODO: Add command line arguments/config file
    DATASET_PATH = './data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # We normalize the training data.
    train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                    ])

    # To demonstrate the generalization capabilities our rotation equivariant layers bring, we apply a random
    # rotation between 0 and 360 deg to the test set.
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.RandomRotation(
                                                        [0, 360],
                                                        torchvision.transforms.InterpolationMode.BILINEAR,
                                                        fill=0),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                    ])

    train_ds = torchvision.datasets.MNIST(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    test_ds = torchvision.datasets.MNIST(root=DATASET_PATH, train=False, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=8)
    
    # net = train_model(  train_loader,
    #                     test_loader,
    #                     model_name="Cake_GCNN",
    #                     model_hparams={'group':CyclicGroup(4), 
    #                                 'in_channels':1, 
    #                                 'out_channels':10, 
    #                                 'kernel_size':3, 
    #                                 'hidden_dims':[32,16,16,16],  
    #                                 'resolution':(28,28), 
    #                                 'wavelet_type':'b_spline', 
    #                                 'slices':4},
    #                     optimizer_name="Adam",
    #                     optimizer_hparams={"lr": 1e-2,
    #                                         "weight_decay": 1e-4},
    #                     save_name='cnn-pretrained')

    net = train_model(  train_loader,
                        test_loader,
                        model_name="GCNN",
                        model_hparams={'group':CyclicGroup(4), 
                                    'in_channels':1, 
                                    'out_channels':10, 
                                    'kernel_size':3, 
                                    'hidden_dims':[32,16,16,16]},
                        optimizer_name="Adam",
                        optimizer_hparams={"lr": 1e-2,
                                            "weight_decay": 1e-4},
                        save_name='cnn-pretrained')


if __name__ == "__main__":
    #TODO Add command line arguments
    train()