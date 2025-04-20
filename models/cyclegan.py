import torch
import torch.nn as nn
import itertools

############################################
# Resnet-style Generator
############################################

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(
        self,
        input_channels = 1,
        output_channels = 1,
        n_res_blocks = 12,
        n_filters = 64
    ):
        super().__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                in_channels = input_channels,
                out_channels = n_filters,
                kernel_size = 7
            ),
            nn.InstanceNorm2d(n_filters),
            nn.ReLU(True)
        ]

        # Downsampling
        in_channels = n_filters
        for _ in range(2):
            out_channels = min(in_channels * 2, 256)
            model += [
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(True)
            ]
            in_channels = out_channels

        # Residual blocks
        for _ in range(n_res_blocks):
            model.append(ResidualBlock(in_channels))

        # Upsample
        out_channels = in_channels // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(True)
            ]
            in_channels = out_channels
            out_channels = in_channels // 2

        # Output
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


############################################
# PatchGAN Discriminator
############################################

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()

        model = [
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


############################################
# Cycle-GAN Model Definition
############################################

class CycleGAN(nn.Module):
    def __init__(self, generator_type='resnet', device="cuda"):
        super().__init__()
        self.device = device

        # Initalize Generator
        if generator_type == 'unet':
            raise NotImplementedError
        else:
            self.G_XtoY = Generator(1, 1)
            self.G_YtoX = Generator(1, 1)

        # Initailize Discriminator
        self.D_X = Discriminator(1)
        self.D_Y = Discriminator(1)

        # Move models to the target device
        self.to(device)

        # Optimizers
        self.opt_G = torch.optim.Adam(
            itertools.chain(self.G_XtoY.parameters(), self.G_YtoX.parameters()),
            lr=2e-4, betas=(0.5, 0.999)
        )
        self.opt_D_X = torch.optim.Adam(self.D_X.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.opt_D_Y = torch.optim.Adam(self.D_Y.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # Losses
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # Weights for losses
        self.lambda_cycle = 10.0
        self.lambda_id = 0.1

        # Add learning rate scheduler
        self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_G, T_max=200, eta_min=1e-5
        )
        self.scheduler_D_X = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_D_X, T_max=200, eta_min=1e-5
        )
        self.scheduler_D_Y = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_D_Y, T_max=200, eta_min=1e-5
        )
        
        # Track training losses, validation losses & metrics
        self.history = {
            'G_losses': [], 'D_X_losses': [], 'D_Y_losses': [],
            'cycle_losses': [], 'identity_losses': [],
            'val_psnr': [], 'val_ssim': [],
            'val_t1_to_t2_psnr': [], 'val_t1_to_t2_ssim': [], 
            'val_t2_to_t1_psnr': [], 'val_t2_to_t1_ssim': [],
            'val_G_losses': [], 'val_D_X_losses': [], 'val_D_Y_losses': [],
            'val_cycle_losses': [], 'val_identity_losses': []
        }

    def forward(self, real_X, real_Y):
        """
        Forward pass for CycleGAN.
        Args:
            real_X: T1 images
            real_Y: T2 images
        Returns:
            Tuple containing:
              - fake_Y: Generated T2 from T1
              - rec_X: Reconstructed T1 (after translating fake_Y back)
              - fake_X: Generated T1 from T2
              - rec_Y: Reconstructed T2 (after translating fake_X back)
        """
        # X->Y
        fake_Y = self.G_XtoY(real_X)
        rec_X = self.G_YtoX(fake_Y)
        # Y->X
        fake_X = self.G_YtoX(real_Y)
        rec_Y = self.G_XtoY(fake_X)

        return fake_Y, rec_X, fake_X, rec_Y

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))

        pred_fake = netD(fake.detach())
        loss_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        return (loss_real + loss_fake) * 0.5

    def backward_D_X(self, real_X, fake_X):
        self.loss_D_X = self.backward_D_basic(self.D_X, real_X, fake_X)
        self.loss_D_X.backward()

    def backward_D_Y(self, real_Y, fake_Y):
        self.loss_D_Y = self.backward_D_basic(self.D_Y, real_Y, fake_Y)
        self.loss_D_Y.backward()

    def backward_G(self, real_X, real_Y, fake_Y, rec_X, fake_X, rec_Y):
        # Adversarial losses
        pred_fake_Y = self.D_Y(fake_Y)
        self.loss_G_XtoY = self.criterion_GAN(pred_fake_Y, torch.ones_like(pred_fake_Y))
        pred_fake_X = self.D_X(fake_X)
        self.loss_G_YtoX = self.criterion_GAN(pred_fake_X, torch.ones_like(pred_fake_X))
        
        # Cycle-consistency losses
        self.loss_cycle_X = self.criterion_cycle(rec_X, real_X) * self.lambda_cycle
        self.loss_cycle_Y = self.criterion_cycle(rec_Y, real_Y) * self.lambda_cycle
        
        # Identity losses
        idt_X = self.G_YtoX(real_Y)
        idt_Y = self.G_XtoY(real_X)
        self.loss_idt_X = self.criterion_identity(idt_X, real_Y) * self.lambda_cycle * self.lambda_id
        self.loss_idt_Y = self.criterion_identity(idt_Y, real_X) * self.lambda_cycle * self.lambda_id
        
        # Total generator loss
        self.loss_G = (self.loss_G_XtoY + self.loss_G_YtoX +
                       self.loss_cycle_X + self.loss_cycle_Y +
                       self.loss_idt_X + self.loss_idt_Y)
        self.loss_G.backward()

    def optimize(self, real_X, real_Y):
        # Forward pass
        fake_Y, rec_X, fake_X, rec_Y = self.forward(real_X, real_Y)
        
        # Update generators
        self.opt_G.zero_grad()
        self.backward_G(real_X, real_Y, fake_Y, rec_X, fake_X, rec_Y)
        self.opt_G.step()
        
        # Update discriminator for domain X
        self.opt_D_X.zero_grad()
        self.backward_D_X(real_X, fake_X)
        self.opt_D_X.step()
        
        # Update discriminator for domain Y
        self.opt_D_Y.zero_grad()
        self.backward_D_Y(real_Y, fake_Y)
        self.opt_D_Y.step()
        
        # Return outputs for logging/visualization if needed
        return fake_Y, rec_X, fake_X, rec_Y
        