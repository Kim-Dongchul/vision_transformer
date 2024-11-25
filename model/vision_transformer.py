from torch.nn import Module, Linear

from model.patch_embedding import PatchEmbedding
from model.transformer_encoder_layer import TransformerEncoderLayer


class VisionTransformer(Module):
    def __init__(self,
                 img_shape,
                 patch_size: int,
                 num_trans,
                 num_cls,
                 embed_dim,
                 num_head,
                 dim_feedforward,
                 dropout,
                 device):
        super(VisionTransformer, self).__init__()
        self.embedding = PatchEmbedding(img_shape, patch_size, device=device)
        self.trans_enc_layers = [TransformerEncoderLayer(
            embed_dim,
            num_head,
            dim_feedforward,
            dropout,
            device=device,
        ) for _ in range(num_trans-1)]
        self.linear1 = Linear(embed_dim, dim_feedforward, device=device)
        self.linear2 = Linear(dim_feedforward, num_cls, device=device)

    def forward(self, x):
        x = self.embedding(x)
        for trans_enc_layer in self.trans_enc_layers:
            x = trans_enc_layer(x)
        x = x[:, 0, :]
        x = self.linear1(x)
        x = self.linear2(x)
        return x
