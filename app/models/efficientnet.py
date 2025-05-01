import torch.nn as nn
import torchvision.models as models  # type: ignore


class EfficientNet(nn.Module):

    # pre-trained EfficientNetB0 is used for evaluation
    # define model setup with individual settings like Dropout and a linear layer depending on the number of classes to predict
    def __init__(self, num_classes: int, grayscale=False):
        super(EfficientNet, self).__init__()
        self.weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights=self.weights)
        self.req_input_shape = self.weights.transforms().crop_size[0]

        if grayscale:
            out_channels = self.model.features[0][0].out_channels
            kernel_size = self.model.features[0][0].kernel_size
            stride = self.model.features[0][0].stride
            padding = self.model.features[0][0].padding
            self.model.features[0][0] = nn.Conv2d(1, out_channels, kernel_size=kernel_size, stride=stride,
                                                  padding=padding, bias=False)

        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
