from torch.nn import Module, LayerNorm, Sequential, Linear, GELU, Dropout

from model.msa import MultiHeadAttention


class TransformerEncoderLayer(Module):
    def __init__(self, embed_dim, num_head, hidden_size, dropout=0.1, eps=1e-6, device='cpu'):
        super(TransformerEncoderLayer, self).__init__()
        self.LayerNorm1 = LayerNorm(embed_dim, eps=eps, device=device)
        self.LayerNorm2 = LayerNorm(embed_dim, eps=eps, device=device)
        self.msa = MultiHeadAttention(embed_dim, num_head, dropout, device=device)
        self.mlp = Sequential(
            Linear(embed_dim, hidden_size, device=device),
            GELU(),
            Dropout(dropout),
            Linear(hidden_size, embed_dim, device=device),
        )

    def forward(self, embed):
        bot = self.LayerNorm1(embed)
        bot, att = self.msa(bot)
        mid = bot + embed
        top = self.LayerNorm2(mid)
        top = self.mlp(top)
        return top + mid