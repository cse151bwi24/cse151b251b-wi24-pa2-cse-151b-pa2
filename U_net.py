import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

#ToDO Fill in the __ values
class U_net(nn.Module):

    def __init__(self, n_class):
        # TODO: Skeleton code given for default FCN network. Fill in the blanks with the shapes
        super().__init__()
        self.n_class = n_class
        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3)
        self.bn8 = nn.BatchNorm2d(512)

        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3)
        self.bn9 = nn.BatchNorm2d(1024)
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3)
        self.bn10 = nn.BatchNorm2d(1024)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(1024, 512, kernel_size=3)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3)
        self.bn12 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(512, 256, kernel_size=3)
        self.bn13 = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(256, 256, kernel_size=3)
        self.bn14 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv15 = nn.Conv2d(256, 128, kernel_size=3)
        self.bn15 = nn.BatchNorm2d(128)
        self.conv16 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn16 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv17 = nn.Conv2d(128, 64, kernel_size=3)
        self.bn17 = nn.BatchNorm2d(64)
        self.conv18 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn18 = nn.BatchNorm2d(64)

        # Classifier
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)

    #TODO Complete the forward pass
    def forward(self, x):

        original_inputs_size = x.shape

        x = self.data_extrapolation(x, 94)

        # Encoder
        x1 = self.bn1(self.relu(self.conv1(x)))
        x1 = self.bn2(self.relu(self.conv2(x1)))
        x2 = self.bn3(self.relu(self.conv3(self.maxpool(x1))))
        x2 = self.bn4(self.relu(self.conv4(x2)))
        x3 = self.bn5(self.relu(self.conv5(self.maxpool(x2))))
        x3 = self.bn6(self.relu(self.conv6(x3)))
        x4 = self.bn7(self.relu(self.conv7(self.maxpool(x3))))
        x4 = self.bn8(self.relu(self.conv8(x4)))
        x5 = self.bn9(self.relu(self.conv9(self.maxpool(x4))))
        x5 = self.bn10(self.relu(self.conv10(x5)))

        # Decoder
        x6 = self.deconv1(x5)
        crop1 = transforms.CenterCrop((x6.shape[2], x6.shape[3]))
        x6 = self.bn11(self.relu(self.conv11(torch.concatenate((crop1(x4), x6), dim=1))))
        x6 = self.bn12(self.relu(self.conv12(x6)))

        x7 = self.deconv2(x6)
        crop2 = transforms.CenterCrop((x7.shape[2], x7.shape[3]))
        x7 = self.bn13(self.relu(self.conv13(torch.concatenate((crop2(x3), x7), dim=1))))
        x7 = self.bn14(self.relu(self.conv14(x7)))

        x8 = self.deconv3(x7)
        crop3 = transforms.CenterCrop((x8.shape[2], x8.shape[3]))
        x8 = self.bn15(self.relu(self.conv15(torch.concatenate((crop3(x2), x8), dim=1))))
        x8 = self.bn16(self.relu(self.conv16(x8)))

        x9 = self.deconv4(x8)
        crop4 = transforms.CenterCrop((x9.shape[2], x9.shape[3]))
        x9 = self.bn17(self.conv17(torch.concatenate((crop4(x1), x9), dim=1)))
        x9 = self.bn18(self.relu(self.conv18(self.relu(x9))))

        # Classifier
        score = self.classifier(x9)
        crop5 = transforms.CenterCrop((original_inputs_size[2], original_inputs_size[3]))
        score = crop5(score)

        return score  # size=(N, n_class, H, W)
    
    def data_extrapolation(self, data, width):
        
        horizontal_fliped_data = torch.flip(data, dims=[3])
        vertical_fliped_data = torch.flip(data, dims=[2])
        diagonal_fliped_data = torch.flip(data, dims=[2, 3])

        # print(str(diagonal_fliped_data[:, :, -width:, -width:].shape) + ' ' + str(vertical_fliped_data[:, :, -width:, :].shape) + ' ' + str(diagonal_fliped_data[:, :, -width:, :width].shape))
        extrapolated_data1 = torch.concatenate((diagonal_fliped_data[:, :, -width:, -width:], vertical_fliped_data[:, :, -width:, :], diagonal_fliped_data[:, :, -width:, :width]), axis=3) 
        extrapolated_data2 = torch.concatenate((horizontal_fliped_data[:, :, :, -width:], data, horizontal_fliped_data[:, :, :, :width]), axis=3)
        extrapolated_data3 = torch.concatenate((diagonal_fliped_data[:, :, :width, -width:], vertical_fliped_data[:, :, :width, :], diagonal_fliped_data[:, :, :width, :width]), axis=3)
        extrapolated_data = torch.concatenate((extrapolated_data1, extrapolated_data2, extrapolated_data3), axis=2)

        return extrapolated_data
