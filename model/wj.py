from torch import nn


class VitSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(VitSelfAttention, self).__init__()
        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_head_size = hidden_size // num_heads
        self.num_heads = num_heads
        assert self.attention_head_size * num_heads == hidden_size, "hidden size must be divisible by num_heads"

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size, num_patches, hidden_size = x.shape
        key_layer = self.transpose_for_scores(self.key(x))
        query_layer = self.transpose_for_scores(self.query(x))
        value_layer = self.transpose_for_scores(self.value(x))

        import torch
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (hidden_size ** 0.5)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).reshape(batch_size, num_patches, hidden_size)
        return context_layer

class VitAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(VitAttention, self).__init__()
        self.self_attention = VitSelfAttention(hidden_size, num_heads)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        attention_output = self.self_attention(x)
        attention_output = self.dense(attention_output)
        return attention_output
