import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import torch.nn.functional as F


class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.backbone = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2").transformer
        self.fc = nn.Linear(768, 1)

    def forward_one(self, x):
        x = self.backbone(**x).last_hidden_state
        x = x.mean(dim=2)
        # import pdb;pdb.set_trace()
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.fc(dis)
        return out


# for test
if __name__ == '__main__':
    net = Siamese()
    print(net)
    print(list(net.parameters()))