import torch.nn as nn
from torchvision import models

class FCN_ResNet34(nn.Module):
    def __init__(self, n_class):
        super(FCN_ResNet34, self).__init__()
        self.n_class = n_class

        # Load pre-trained ResNet34
        resnet34 = models.resnet34(pretrained=True)

        # Freeze the early layers or the entire encoder as needed
        for param in resnet34.parameters():
            param.requires_grad = False

        self.encoder = nn.Sequential(*list(resnet34.children())[:-2])
        self.relu = nn.ReLU(inplace=True)

        # Decoder (as per the original FCN structure, adjusted for ResNet34 output channels)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        # Classifier
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)  # Adjust the input channel as per the last deconv layer

    def forward(self, x):
        # Encoder
        x5 = self.encoder(x)

        # Decoder
        x = self.bn1(self.relu(self.deconv1(x5)))
        x = self.bn2(self.relu(self.deconv2(x)))
        x = self.bn3(self.relu(self.deconv3(x)))
        x = self.bn4(self.relu(self.deconv4(x)))
        x = self.bn5(self.relu(self.deconv5(x)))

        # Classifier
        score = self.classifier(x)

        return score
