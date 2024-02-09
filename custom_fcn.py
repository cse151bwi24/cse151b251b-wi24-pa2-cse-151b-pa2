import torch.nn as nn
import torch

#ToDO Fill in the __ values
class custom_FCN(nn.Module):

    def __init__(self, n_class):
        # TODO: Skeleton code given for default FCN network. Fill in the blanks with the shapes
        super().__init__()
        self.n_class = n_class
        # Encoder
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1_1 = nn.BatchNorm2d(32)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2_1 = nn.BatchNorm2d(64)
        self.conv3_1 = nn.Conv2d(64,128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3_1 = nn.BatchNorm2d(128)
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4_1 = nn.BatchNorm2d(256)
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5_1 = nn.BatchNorm2d(512)

        self.conv1_2 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1_2 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2_2 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64,128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3_2 = nn.BatchNorm2d(128)
        self.conv4_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4_2 = nn.BatchNorm2d(256)
        self.conv5_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5_2 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        # Classifier
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)

    #TODO Complete the forward pass
    def forward(self, x):
        # Encoder
        x1 = self.bnd1_1(self.relu(self.conv1_1(x)))
        x1 = self.bnd2_1(self.relu(self.conv2_1(x1)))
        x1 = self.bnd3_1(self.relu(self.conv3_1(x1)))
        x1 = self.bnd4_1(self.relu(self.conv4_1(x1)))
        x1 = self.bnd5_1(self.relu(self.conv5_1(x1)))

        x2 = self.bnd1_2(self.relu(self.conv1_2(x)))
        x2 = self.bnd2_2(self.relu(self.conv2_2(x2)))
        x2 = self.bnd3_2(self.relu(self.conv3_2(x2)))
        x2 = self.bnd4_2(self.relu(self.conv4_2(x2)))
        x2 = self.bnd5_2(self.relu(self.conv5_2(x2)))

        # Decoder
        # print('x1.shape: ' + str(x1.shape) + ' x2,shape: ' + str(x2.shape))
        x = self.bn1(self.relu(self.deconv1(torch.concatenate((x1, x2), dim=1))))
        x = self.bn2(self.relu(self.deconv2(x)))
        x = self.bn3(self.relu(self.deconv3(x)))
        x = self.bn4(self.relu(self.deconv4(x)))
        x = self.bn5(self.relu(self.deconv5(x)))

        # Classifier
        score = self.classifier(x)

        return score  # size=(N, n_class, H, W)