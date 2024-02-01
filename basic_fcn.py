import torch.nn as nn

#ToDO Fill in the __ values
class FCN(nn.Module):

    def __init__(self, n_class):
        # TODO: Skeleton code given for default FCN network. Fill in the blanks with the shapes
        super().__init__()
        self.n_class = n_class
        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(16)

        # Classifier
        self.classifier = nn.Conv2d(16, self.n_class, kernel_size=1)

    #TODO Complete the forward pass
    def forward(self, x):
        # Encoder
        x = self.relu(self.bnd1(self.conv1(x)))
        x = self.relu(self.bnd2(self.conv2(x)))
        x = self.relu(self.bnd3(self.conv3(x)))
        x = self.relu(self.bnd4(self.conv4(x)))
        x5 = self.relu(self.bnd5(self.conv5(x)))

        # Decoder
        x = self.relu(self.bn1(self.deconv1(x5)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.relu(self.bn5(self.deconv5(x)))

        # Classifier
        score = self.classifier(x)

        return score  # size=(N, n_class, H, W)
