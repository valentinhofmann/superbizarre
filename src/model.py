import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


class BertClassifier(nn.Module):

    def __init__(self):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear_1 = nn.Linear(768, 100)
        self.linear_2 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, derivatives, masks, segs):

        n_tokens_pad = (derivatives >= 106).float().sum(dim=-1)

        output_bert = self.dropout(self.bert(derivatives, attention_mask=masks, token_type_ids=segs)[0])
        output_bert_pad = output_bert * (derivatives >= 106).float().unsqueeze(-1).expand(-1, -1, 768)
        h = output_bert_pad.sum(dim=1) / n_tokens_pad.unsqueeze(-1).expand(-1, 768)
        h = self.dropout(F.relu(self.linear_1(h)))
        output = torch.sigmoid(self.linear_2(h).squeeze(-1))

        return output
