"""
Model for RGB dataset
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.baseConv1 = nn.Conv2d(3, 32, 5, 2)
        self.baseConv2 = nn.Conv2d(32, 64, 5, 2)
        self.baseConv3 = nn.Conv2d(64, 128, 3, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.classFc1 = nn.Linear(8064, 300)
        self.classFc2 = nn.Linear(300, 1)
                
        self.regFc1 = nn.Linear(8064, 1000)
        self.regFc2 = nn.Linear(1000, 100)
        self.regFc3 = nn.Linear(100, 7) 
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.baseConv1(x)))
        x = self.pool(F.relu(self.baseConv2(x)))
        x = self.pool(F.relu(self.baseConv3(x)))
        x = x.view(x.size(0), -1)

        x1 = F.relu(self.classFc1(x))
        x1 = self.classFc2(x1)
        x1 = F.sigmoid(x1)
        x2 = F.relu(self.regFc1(x))
        x2 = F.relu(self.regFc2(x2))
        x2 = self.regFc3(x2)

        return x1, x2


