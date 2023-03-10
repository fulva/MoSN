import torch as th
import torch.nn as nn
th.manual_seed(10)

class ImageToState(nn.Module):
    """
    IMage Encoder
    """
    def __init__(self, num, size):
        super(ImageToState, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1), # 32*32*3 -> 32*32*8
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32*32*8 -> 16*16*8 
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8)
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), # 16*16*8 -> 16*16*16
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16*16*16 -> 8*8*16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 8*8*16 -> 8*8*32
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8*8*32 -> 4*4*32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 32, num),
            nn.Sigmoid()
        )
    def forward(self, x):
        e1 = self.encoder_conv(x)
        e2 = self.encoder_conv2(e1)
        e3 = self.encoder_conv3(e2)
        e3 = e3.contiguous().view(e3.size(0), -1)
        encoding = self.fc(e3)
        return encoding

def initialize_model(num, w_size):
    F_b = ImageToState(num, w_size)
    F_m = ImageToState(num, w_size)
    if th.cuda.is_available():
        F_b.cuda()
        F_m.cuda()
    for param_b, param_m in zip(F_b.parameters(), F_m.parameters()):
        param_m.data.copy_(param_b.data)  # initialize
        param_m.requires_grad = False  # not update by gradient
    return F_b, F_m
