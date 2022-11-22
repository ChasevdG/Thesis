import torch
from torch.nn import AdaptiveAvgPool3d
from GCNN.kernels import InterpolativeGroupKernel, InterpolativeLiftingKernel
from GCNN.group import CyclicGroup
import os

import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torchvision
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

CHECKPOINT_PATH = './checkpoints'

class LiftingConvolution(torch.nn.Module):

    def __init__(self, group, in_channels, out_channels, kernel_size):
        super().__init__()

        self.kernel = InterpolativeLiftingKernel(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        """ Perform lifting convolution

        @param x: Input sample [batch_dim, in_channels, spatial_dim_1, 
            spatial_dim_2]
        @return: Function on a homogeneous space of the group
            [batch_dim, out_channels, num_group_elements, spatial_dim_1, 
            spatial_dim_2]
        """

        # Obtain convolution kernels transformed under the group.
        
        ## YOUR CODE STARTS HERE ##
        conv_kernels = self.kernel.sample()

        ## AND ENDS HERE ##

        # Apply lifting convolution. Note that using a reshape we can fold the
        # group dimension of the kernel into the output channel dimension. We 
        # treat every transformed kernel as an additional output channel. This
        # way we can use pytorch's conv2d function!

        # Question: Do you see why we (can) do this?

        ## YOUR CODE STARTS HERE ##
        x = torch.nn.functional.conv2d(
            input=x,
            weight=conv_kernels.reshape(
                self.kernel.out_channels * self.kernel.group.elements().numel(),
                self.kernel.in_channels,
                self.kernel.kernel_size,
                self.kernel.kernel_size
            ),
        )
        ## AND ENDS HERE ##

        # Reshape [batch_dim, in_channels * num_group_elements, spatial_dim_1, 
        # spatial_dim_2] into [batch_dim, in_channels, num_group_elements, 
        # spatial_dim_1, spatial_dim_2], separating channel and group 
        # dimensions.
        x = x.view(
            -1,
            self.kernel.out_channels,
            self.kernel.group.elements().numel(),
            x.shape[-1],
            x.shape[-2]
        )

        return x

class Model(pl.LightningModule):
    
        def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
            
            super().__init__()
            self.save_hyperparameters()

            self.kernel_size = model_hparams['kernel_size']
            group = model_hparams['group']
            in_channels = model_hparams['in_channels']
            
            self.lift = LiftingConvolution(group, in_channels, 1, self.kernel_size)
            self.optimizer_hparams = optimizer_hparams

        def forward(self, x):
            x_ = self.lift(x)
            return x_

        def entropy_loss(self, x_):
            '''
            Calulates the entropy across the group actions from the output of the network

            x_ is the output of lifting; BxCxGxHxW
            '''
            x_ = x_/torch.sum(x_, dim=2, keepdim=True) # Normalize the output to be probabilities
            return -torch.sum(x_*torch.log(x_), dim=2) # Calculate the entropy
        
        def heisenberg_loss(self, x_, reduction='mean'):
            '''
                We want to maximimize our knowledge of direction and location, thus we
                minimize Var(G)*Var(X)
            '''
            x_loss = torch.var(x_, dim=(3,4), keepdim=True)
            g_loss = torch.var(x_, dim=(2), keepdim=True)
            loss = g_loss*x_loss # Var(G)*Var(X)
            if reduction == 'mean':
                loss = torch.mean(loss)
            return loss

        def reconstruction_loss(self, x, x_):
            B, C, G, H, W = x.shape
            # The ammount the image would need to be padded
            reduction = self.kernel_size//2

            # Make the image size the same as the output of the network
            x = x[reduction:-reduction, reduction:-reduction]
            return torch.mean(torch.abs(x - x_))
        
        def training_step(self, batch, batch_idx):
            x, _ = batch
            x_ = self(x)
            loss = self.reconstruction_loss(x, x_) + self.entropy_loss(x_)
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, _ = batch
            x_ = self(x)
            loss = self.reconstruction_loss(x, x_) + self.entropy_loss(x_)
            self.log('val_loss', loss)
            self.log('kernel', self.conv.kernel.kernel)
            return loss

        def configure_optimizers(self):
            # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(
                self.parameters(), **self.optimizer_hparams)
            return [optimizer], []

def train_model(train_loader, test_loader, model_name ,model_hparams, optimizer_name, optimizer_hparams, save_name=None, device='cpu',**kwargs):
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
        model = Model.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(12) # To be reproducable
        group = CyclicGroup(8)
        model = Model( model_name, model_hparams, optimizer_name, optimizer_hparams)
        trainer.fit(model, train_loader, test_loader)
        model = Model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on test set
    val_result = trainer.test(model.to(device), test_loader, verbose=False)
    result = {"val": val_result[0]["test_acc"]}

    return model, result
    
def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f"Unknown model name \"{model_name}\". Available models are: {str(model_dict.keys())}"

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
    
    net = train_model(  train_loader,
                        test_loader,
                        model_name="Cake_GCNN",
                        save_name="Cake_GCNN",
                        model_hparams={'group':CyclicGroup(4), 
                                    'in_channels':1,
                                    'kernel_size':14, 
                                    },
                        optimizer_name="Adam",
                        optimizer_hparams={"lr": 1e-2,
                                        "weight_decay": 1e-4},)