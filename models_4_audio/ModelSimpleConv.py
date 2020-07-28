
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                padding=1,
                stride=1,
                kernel_size=3
            ),

            torch.nn.ReLU(),

            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                padding=1,
                stride=1,
                kernel_size = 3
            ),

            torch.nn.ReLU(),

            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                padding=1,
                stride=1,
                kernel_size = 3
            ),
            torch.nn.ReLU(),

            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                padding=1,
                stride=1,
                kernel_size = 3
            ),
            torch.nn.ReLU()
        )

        self.fc = nn.Linear(
            in_features=128,
            out_features=args.classes_amount)


    def forward(self, x):
        out = self.encoder.forward(x)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc.forward(out)
        out = F.softmax(out, dim=1)

        return out