import torch.nn.functional as F
import torch.nn as nn

class ChatBotModel(nn.Module):

    def __init__(self, x_size, y_size):
        super(ChatBotModel, self).__init__()

        self.linear_one = nn.Linear(in_features= x_size, out_features=80)
        self.linear_two = nn.Linear(in_features=80, out_features=45)
        self.linear_three = nn.Linear(in_features=45, out_features= y_size)

    def forward(self, x):

        x = self.linear_one(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.linear_two(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.linear_three(x)

        return F.softmax(x)