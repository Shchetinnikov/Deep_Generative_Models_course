import numpy as np
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import transforms, datasets
from model import Generator, Discriminator
import torchvision.utils as vutils


class GAN(pl.LightningModule):
    """
    GAN for homework https://github.com/Skyfallk/2024_deep_gen_models/tree/main/HW_2.GAN_train
    """

    def __init__(self, kwargs):
        super().__init__()


        self.nz = kwargs['nz']
        self.beta1 = kwargs['beta1']
        self.lr_d = kwargs['lr_d']
        self.lr_g = kwargs['lr_g']

        self.netG = Generator(in_channels=1024)
        self.netD = Discriminator()
        self.criterion = nn.BCELoss()

        self.save_hyperparameters()
        self.automatic_optimization = False

    def forward(self, z):
        return self.netG(z)

    def adversarial_loss(self, y_hat, y):
        return self.criterion(y_hat, y)

    def training_step(self, batch, batch_idx):
        optimizerD, optimizerG = self.optimizers()

        real_images, _ = batch
        b_size = real_images.size(0)
        real_label = torch.full((b_size,), 1., dtype=torch.float, device=self.device)
        fake_label = torch.full((b_size,), 0., dtype=torch.float, device=self.device)

        # Train Discriminator

        noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
        fake_images = self.netG(noise)
        real_pred = self.netD(real_images).view(-1)
        fake_pred = self.netD(fake_images.detach()).view(-1)
        d_loss_real = self.adversarial_loss(real_pred, real_label)
        d_loss_fake = self.adversarial_loss(fake_pred, fake_label)
        d_loss = d_loss_real + d_loss_fake
        self.log('train_loss_d', d_loss)

        optimizerD.zero_grad()
        self.manual_backward(d_loss)
        optimizerD.step()

        # Train Generator

        noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
        fake_images = self.netG(noise)
        fake_pred = self.netD(fake_images).view(-1)
        g_loss = self.adversarial_loss(fake_pred, real_label)
        self.log('train_loss_g', g_loss)

        optimizerG.zero_grad()
        self.manual_backward(g_loss)
        optimizerG.step()

        # tensorboard = self.logger.experiment
        # tensorboard.add_image(tag=f"generated_images epoch {self.current_epoch}", img_tensor=fake_images)

        self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar=True)

    def configure_optimizers(self):
        optimizerD = Adam(self.netD.parameters(), lr=self.lr_d, betas=(self.beta1, 0.999))
        optimizerG = Adam(self.netG.parameters(), lr=self.lr_g, betas=(self.beta1, 0.999))
        return [optimizerD, optimizerG], []


if __name__ == '__main__':
    # TODO: get params from hydra (general yaml)
    hparams = {
        'dataroot': 'dataset/celeba',
        'workers': 16,
        'batch_size': 128 * 2,
        'image_size': 128,
        'nc': 3,
        'nz': 100,
        'ngf': 64,
        'ndf': 64,
        'num_epochs': 10,
        'lr_g': 0.00008,
        'lr_d': 0.00008,
        'beta1': 0.5,
        'gpus': 1
    }
    # TODO: write summary transforms
    transform = transforms.Compose([
        transforms.Resize(hparams['image_size']),
        transforms.CenterCrop(hparams['image_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(root=hparams['dataroot'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=hparams['workers'])

    model = GAN(hparams)
    trainer = pl.Trainer(
        max_epochs=hparams['num_epochs'],
        devices="auto",
        log_every_n_steps=1,
        logger=pl.loggers.TensorBoardLogger('logs/', name='GAN', log_graph=True)
    )
    trainer.fit(model, train_dataloaders=dataloader)

    # load
    checkpoint = "./logs/GAN/version_0/checkpoints/epoch=2-step=4752.ckpt"
    autoencoder = GAN.load_from_checkpoint(kwargs=hparams, checkpoint_path=checkpoint)

    generator = autoencoder.netG
    generator.eval()

    # plot imgs
    img_list = []
    fixed_noise = torch.randn(hparams['batch_size'], hparams['nz'], 1, 1, device='cuda')
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=1, normalize=True))

    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()
