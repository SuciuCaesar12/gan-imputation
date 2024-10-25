import torch
import torch.nn as nn

from losses import *
from config import settings


C, H, W = INPUT_SHAPE = settings.mnist.input_shape
NUM_CLASSES = settings.mnist.num_classes
N_FEATURES = C * H * W

HIDDEN_DIM = settings.hexa.hidden_dim
LR = settings.hexa.lr


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),  # Input channels, output channels, kernel size
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))  # Reduces dimension by a factor of 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)  # Fully connected layer
        
        self._init_losses()

    def _init_losses(self):
        self.loss_ce = LossCE()
    
    def configure_optimizers(self):
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # Flatten the tensor
        out = self.fc(out)
        return out


class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=C + 1, out_channels=HIDDEN_DIM, kernel_size=5, stride=2),
            nn.Conv2d(in_channels=HIDDEN_DIM, out_channels=2 * HIDDEN_DIM, kernel_size=5, stride=2),
            nn.Conv2d(in_channels=2 * HIDDEN_DIM, out_channels=4 * HIDDEN_DIM, kernel_size=3, stride=2)
        ])
        
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(HIDDEN_DIM),
            nn.BatchNorm2d(2 * HIDDEN_DIM),
            nn.BatchNorm2d(4 * HIDDEN_DIM)
        ])
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.adapter = nn.Linear(in_features=self._calculate_num_features(), out_features=HIDDEN_DIM)
    
    @torch.no_grad()
    def _calculate_num_features(self):
        self.eval()
        dummy_x, dummy_m = torch.zeros(1, *INPUT_SHAPE), torch.zeros(1, *INPUT_SHAPE)
        x = torch.cat([dummy_x, dummy_m], dim=1)
        
        for conv, bn in zip(self.convs, self.bns):
            x = self.leaky_relu(bn(conv(x)))
        
        self.train()
        return x.numel()
    
    def forward(self, x, m):
        bs = x.shape[0]
        x = x.view(bs, *INPUT_SHAPE)
        m = m.view(bs, *INPUT_SHAPE)
        x = torch.cat([x, m], dim=1)
        
        for conv, bn in zip(self.convs, self.bns):
            x = self.leaky_relu(bn(conv(x)))
        
        return self.leaky_relu(self.adapter(x.view(bs, -1)))


class GeneratorMI(nn.Module):
    
    def __init__(self):
        super(GeneratorMI, self).__init__()
        self.fc1 = nn.Linear(in_features=HIDDEN_DIM, out_features=4 * 4 * HIDDEN_DIM)
        
        self.deconvs = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=HIDDEN_DIM, out_channels=2 * HIDDEN_DIM, kernel_size=4, stride=2, padding=1),  # (4,4) -> (7,7)
            nn.ConvTranspose2d(in_channels=2 * HIDDEN_DIM, out_channels=HIDDEN_DIM, kernel_size=4, stride=2, padding=1),  # (7,7) -> (14,14)
            nn.ConvTranspose2d(in_channels=HIDDEN_DIM, out_channels=C, kernel_size=4, stride=2, padding=1)  # (14,14) -> (28,28)
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm2d(2 * HIDDEN_DIM),
            nn.BatchNorm2d(HIDDEN_DIM),
            nn.BatchNorm2d(C)
        ])

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()
    
    @torch.no_grad()
    def _calculate_in_features(self):
        self.eval()
        dummy_h = torch.zeros(1, HIDDEN_DIM)
        batch_size = dummy_h.shape[0]
        dummy_h = self.leaky_relu(self.fc1(dummy_h))
        dummy_h = dummy_h.view(batch_size, HIDDEN_DIM, 4, 4)
        
        for deconv, bn in zip(self.deconvs, self.bns):
            dummy_h = bn(deconv(dummy_h))
        
        self.train()
        return dummy_h.numel()
    
    def forward(self, h):
        bs = h.shape[0]
        HIDDEN_DIM = h.shape[1]
        h = self.leaky_relu(self.fc1(h))
        h = h.view(bs, HIDDEN_DIM, 4, 4)
        
        for step, (deconv, bn) in enumerate(zip(self.deconvs, self.bns)):
            h = bn(deconv(h))
            if step < len(self.deconvs) - 1:
                h = self.leaky_relu(h)
        
        return self.sigmoid(h[:, :, :H, :W].reshape(bs, -1))


class DiscriminatorMI(nn.Module):

    def __init__(self):
        super(DiscriminatorMI, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.utils.spectral_norm(nn.Conv2d(in_channels=C, out_channels=HIDDEN_DIM, kernel_size=5, stride=2)),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=HIDDEN_DIM, out_channels=2 * HIDDEN_DIM, kernel_size=5, stride=2)),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=2 * HIDDEN_DIM, out_channels=4 * HIDDEN_DIM, kernel_size=3, stride=2))
        ])
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        self.embedding = nn.Embedding(NUM_CLASSES, self._calculate_num_features())
        self.fc = nn.Linear(in_features=self._calculate_num_features(), out_features=N_FEATURES + 1)
        self.sigmoid = nn.Sigmoid()
    
    @torch.no_grad()
    def _calculate_num_features(self):
        self.eval()
        dummy_x_hat = torch.zeros(1, *INPUT_SHAPE)
        
        for conv in self.convs:
            dummy_x_hat = self.leaky_relu(conv(dummy_x_hat))
        
        self.train()
        return dummy_x_hat.numel()
        
    def forward(self, x_hat, y):
        bs = x_hat.shape[0]
        x_hat = x_hat.view(bs, *INPUT_SHAPE)
        
        for conv in self.convs:
            x_hat = self.leaky_relu(conv(x_hat))
            
        x_hat = x_hat.view(bs, -1)
        
        y_emb = self.embedding(torch.argmax(y, dim=1))
        projection = torch.sum(x_hat * y_emb, dim=1, keepdim=True)
        
        logits = self.fc(x_hat) + projection
        return self.sigmoid(logits)


class HexaGAN(nn.Module):
    
    def __init__(self):
        super(HexaGAN, self).__init__()
        
        self.encoder = Encoder()
        self.generator_mi = GeneratorMI()
        self.discriminator_mi = DiscriminatorMI()
        self._init_losses()
        
        self.summary()
    
    def encode(self, x, m, z = None):
        z = torch.rand_like(x) if z is None else z
        x_tilde = m * x + (1 - m) * z
        return x_tilde, self.encoder(x_tilde, m)
    
    def impute(self, x, m):
        x_tilde, h = self.encode(x, m)
        x_bar = self.generator_mi(h)
        x_hat = m * x_tilde + (1 - m) * x_bar
        return x_hat, h, x_tilde
    
    def discriminate(self, x_hat, y):
        return self.discriminator_mi(x_hat, y)
    
    def _init_losses(self):
        self.loss_recon = LossReconstruct()
        self.loss_gen_mi = LossGenMI()
        self.loss_disc_data_mi = LossDiscDataMI()
    
    def losses(self):
        return [self.loss_recon, self.loss_gen_mi, self.loss_disc_data_mi]
    
    def configure_optimizers(self):
        self.encoder_opt = torch.optim.RMSprop(self.encoder.parameters(), lr=LR)
        self.generator_mi_opt = torch.optim.RMSprop(self.generator_mi.parameters(), lr=LR)
        self.discriminator_mi_opt = torch.optim.RMSprop(self.discriminator_mi.parameters(), lr=LR)

    def summary(self):
        print("=" * 40)
        print("Summary of HEXAGAN Model".center(40))
        print("=" * 40)
        
        print(f"{'# Parameters Encoder:':<30} {sum(p.numel() for p in self.encoder.parameters()):,}")
        print(f"{'# Parameters Generator_mi:':<30} {sum(p.numel() for p in self.generator_mi.parameters()):,}")
        print(f"{'# Parameters Discriminator_mi:':<30} {sum(p.numel() for p in self.discriminator_mi.parameters()):,}")
        print(f"{'# Parameters Total:':<30} {sum(p.numel() for p in self.parameters()):,}")
        
        print("=" * 40)
        print()