from torch.nn import Module, MultiheadAttention, Linear


class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, device='cpu'):
        super(MultiHeadAttention, self).__init__()
        self.msa = MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True, device=device)
        self.query = Linear(embed_dim, embed_dim, device=device)
        self.key = Linear(embed_dim, embed_dim, device=device)
        self.value = Linear(embed_dim, embed_dim, device=device)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attn_output, attention = self.msa(query, key, value)
        return attn_output, attention
