from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from docutils.nodes import image

from generator import Generator
from discriminator import Discriminator
import datetime


def setup_seed():
    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


class DCGAN_Model:
    def __init__(self, dataset_name, image_size, num_epochs, batch_size=128):
        setup_seed()

        self.dataset_name = dataset_name

        # Root directory for dataset
        self.dataroot = f"C:\\Users\\ankit\\Workspaces\\CS7150\\data\\{dataset_name}"

        # Number of workers for dataloader
        self.workers = 4

        # Batch size during training
        self.batch_size = batch_size

        # Spatial size of training images. All images will be resized to this
        #   size using a transformer.
        self.image_size = image_size

        # Number of channels in the training images. For color images this is 3
        self.nc = 3

        # Size of z latent vector (i.e. size of generator input)
        self.nz = 100

        # Size of feature maps in generator
        self.ngf = 64

        # Size of feature maps in discriminator
        self.ndf = 64

        # Number of training epochs
        self.num_epochs = num_epochs

        # Learning rate for optimizers
        self.lr = 0.0002

        # Beta1 hyperparam for Adam optimizers
        self.beta1 = 0.5

        # Number of GPUs available. Use 0 for CPU mode.
        self.ngpu = 1

        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")

        # create dataset, dataloader objects
        self.setup_data()

    def setup_data(self):
        self.dataset = dset.ImageFolder(root=self.dataroot,
                                        transform=transforms.Compose([
                                            transforms.Resize(self.image_size),
                                            transforms.CenterCrop(self.image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]))
        # Create the dataloader
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                      shuffle=True, num_workers=self.workers)

    def plot_training_data(self):
        # Plot some training images
        real_batch = next(iter(self.dataloader))
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(
            np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=2, normalize=True).cpu(),
                         (1, 2, 0)))
        plt.show()

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Initialize discriminator and generator
    def init_gen_disc(self):
        self.netG = Generator(self.ngpu, self.nc, self.ngf, self.nz, output_size=self.image_size).to(self.device)

        # Handle multi-gpu if desired
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            netG = nn.DataParallel(self.netG, list(range(self.ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.netG.apply(self.weights_init)

        # Create the Discriminator
        self.netD = Discriminator(self.ngpu, self.nc, self.ndf, input_size=self.image_size).to(self.device)

        # Handle multi-gpu if desired
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            self.netD = nn.DataParallel(self.netD, list(range(self.ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.netD.apply(self.weights_init)

        # Print the model
        print(self.netD)

        # Print the model
        print(self.netG)

    def train_setup(self):
        self.criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

    def train(self):
        self.train_setup()

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                output = self.netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.netG(noise)
                label.fill_(self.fake_label)
                # Classify all fake batch with D
                output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, self.num_epochs, i, len(self.dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.num_epochs - 1) and (i == len(self.dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

            self.save_models(epoch, final=False)

        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        self.save_models(epoch=-1, final=True)

        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        HTML(ani.to_jshtml())

    def save_models(self, epoch, final):
        now = datetime.datetime.now()
        if final:
            name = f'trained_model_{now.strftime("%a_%H_%M")}.pth'
        else:
            name = f'epoch_{epoch}.pth'

        torch.save({
            'generator': self.netG.state_dict(),
            'discriminator': self.netD.state_dict(),
            'optimizerG': self.optimizerG.state_dict(),
            'optimizerD': self.optimizerD.state_dict(),
        }, f'models/{self.dataset_name}/' + name)


if __name__ == "__main__":
    image_size = 32
    model = DCGAN_Model('celeb', image_size=32, num_epochs=5, batch_size=128)
    # model.plot_training_data()
    model.init_gen_disc()
    model.train()
