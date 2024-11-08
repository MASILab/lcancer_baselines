import math
import torch
from torch import nn
from torch.autograd import Variable
from einops import rearrange, repeat

class PositionalTimeEncoding(nn.Module):
    # Fixed time-invariant positional encoding
    def __init__(self, dim, dropout=0.1, seq_len=2, num_patches=1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(seq_len*num_patches, dim)
        position = torch.arange(0, seq_len).repeat_interleave(num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, t):
        x = x + Variable(self.pe, requires_grad=False)
        return self.dropout(x)

class AbsTimeEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, seq_len=2, num_patches=1):
        super().__init__()
        self.num_patches = num_patches
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        div_term = torch.exp(torch.arange(0, dim, 2) *-(math.log(10000.0) / dim))
        self.register_buffer('div_term', div_term)
        self.dim = dim
        
    def forward(self, x, t):
        device, dtype = x.device, x.dtype
        pe = torch.zeros(x.shape, device=device, dtype=dtype)
        
        # repeat times into shape [b, t, dim]
        time_position = repeat(t, 'b t -> b t d', d=int(self.dim/2))
        time_position = time_position.repeat_interleave(self.num_patches, dim=1)
        pe[:, :, 0::2] = torch.sin(time_position * self.div_term.expand_as(time_position))
        pe[:, :, 1::2] = torch.cos(time_position * self.div_term.expand_as(time_position))
        x = x + Variable(pe, requires_grad=False)
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, qkv_bias=False, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=qkv_bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TimeAwareV1(nn.Module):
    """
    V1 models time distance as a flipped sigmoid function
    Learnable parameters for each attention head:
        self.a describes slope of decay
        self.c describes position of decay
    """

    def __init__(self, heads=8):
        super().__init__()
        self.heads = heads
        # initialize from [0,1], which fits decay to scale of fractional months
        self.a = nn.Parameter(torch.rand(heads), requires_grad=True)
        # initialize from [0,12], which fits position to the scale of fractional months
        self.c = nn.Parameter(12 * torch.rand(heads), requires_grad=True)

    def forward(self, x, R):
        *_, n = x.shape
        b, _, t = R.shape
        num_patches = int(n/t)
        # repeat R for each head
        R = repeat(R, 'b t1 t2 -> b h t1 t2', h=self.heads)
        # repeat parameters
        a = repeat(self.a, 'h -> b h t1 t2', b=b, t1=t, t2=t)
        c = repeat(self.c, 'h -> b h t1 t2', b=b, t1=t, t2=t)
        # flipped sigmoid with learnable parameters
        R = 1 / (1 + torch.exp(torch.abs(a) * R - torch.abs(c)))
        # repeat values along last two dimensions according to number of patches
        R = R.repeat_interleave(num_patches, dim=2)
        R = R.repeat_interleave(num_patches, dim=3)
        return x * R


class TDMaskedAttention(nn.Module):
    """
    Implements distance aware attention weight scaling from https://arxiv.org/pdf/2010.06925.pdf#cite.yan2019tener
    """
    def __init__(self, dim, heads=8, dim_head=64, qkv_bias=False, dropout=0.1, time_aware=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.time_aware = time_aware
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        self.time_dist = TimeAwareV1(heads=heads)
        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=qkv_bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, m, R):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        m = repeat(m, 'b d1 d2 -> b h d1 d2', h=self.heads)
        qk = torch.matmul(q, k.transpose(-1, -2))
        dots = (self.relu(qk) + m) * self.scale # ((Q*K^T) + M )/sqrt(d) 

        if self.time_aware:
            dots = self.time_dist(dots, R)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return self.dropout(out)


class MaskTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout=0.1, time_aware=False):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TDMaskedAttention(dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias, dropout=dropout, time_aware=time_aware),
                FeedForward(dim, mlp_dim, dropout),
            ]))

    def forward(self, x, m, R):
        for attn, ff in self.layers:
            x = attn(x, m, R) + x
            x = ff(x) + x
        return x


class FeatViT(nn.Module):
    def __init__(self, *, num_feat, feat_dim, num_classes, dim, depth, heads, mlp_dim, qkv_bias=False,
        dim_head=64, dropout=0.1, time_embedding="AbsTimeEncoding"):
        """
        num_feat: number of input features
        feat_dim: dimension of input features
        dim: dimension after linear projection of input
        depth: number of Transformer blocks
        heads: number of heads in multi-headed attention
        mlp_dim: dimension of MLP layer in each Transformer block
        qkv_bias: enable bias for qkv if True
        seq_length: number of items in the sequence
        dim_head: dimension of q,k,v,z
        """
        super().__init__()
        self.feat_embedding = nn.Sequential(
            # CHANGE: added time dimension
            nn.Linear(feat_dim, dim),
        )

        # different types of positional embeddings
        # 1. PositionalEncoding: Fixed alternating sin cos with position
        # 2. AbsTimeEmb: Fixed alternating sin cos with time
        # 3. Learnable: self.time_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        time_emb_dict = {
            "PositionalEncoding": PositionalTimeEncoding,
            "AbsTimeEncoding": AbsTimeEncoding,
            # "LearnableEmb": LearnableEmb,
        }
        self.time_embedding = time_emb_dict[time_embedding](dim, 0.1, seq_len=2, num_patches=num_feat)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, feat, times, padding):
        b, t, n, d = feat.shape

        x = self.feat_embedding(feat)
        x = rearrange(x, 'b t ... d -> b (t ...) d')

        # mask 
        # padding = repeat(padding, 'b t -> b (t n)', n=n)
        # mask = torch.einsum('bi, bj -> bij', (padding, padding))
        # mask = torch.where(mask==1, 0, -9e15)

        x = self.time_embedding(x, times)
        
        x = self.transformer(x)
        x = x.mean(dim=1)

        latent = self.to_latent(x)
        return self.linear_head(latent), latent

class MaskedFeatViT(nn.Module):
    def __init__(self, *, num_feat, feat_dim, num_classes, dim, depth, heads, mlp_dim, qkv_bias=False, 
        dim_head=64, dropout=0.1, time_embedding=None, time_aware=True,):
        """
        num_feat: number of input features
        feat_dim: dimension of input features
        dim: dimension after linear projection of input
        depth: number of Transformer blocks
        heads: number of heads in multi-headed attention
        mlp_dim: dimension of MLP layer in each Transformer block
        qkv_bias: enable bias for qkv if True
        seq_length: number of items in the sequence
        dim_head: dimension of q,k,v,z
        time_aware: whether to use time-aware attention
        """
        super().__init__()
        self.feat_embedding = nn.Sequential(
            nn.Linear(feat_dim, dim),
        )
        time_emb_dict = {
            "PositionalEncoding": PositionalTimeEncoding,
            "AbsTimeEncoding": AbsTimeEncoding,
            # "LearnableEmb": LearnableEmb,
        }
        self.time_embedding = time_emb_dict[time_embedding](dim, 0.1, seq_len=2, num_patches=num_feat) if time_embedding else None

        self.transformer = MaskTransformer(dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout, time_aware=time_aware)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, feat, times, padding):
        b, t, n, d = feat.shape

        x = self.feat_embedding(feat)
        x = rearrange(x, 'b t ... d -> b (t ...) d')

        # mask 
        padding = repeat(padding, 'b t -> b (t n)', n=n)
        mask = torch.einsum('bi, bj -> bij', (padding, padding))
        mask = torch.where(mask==1, 0, -9e15)

        # Create distance matrix from timesf
        # rows i maps to key token i, column j maps to query token j
        R = torch.zeros(b, t, t, device=x.device, dtype=torch.float32)
        for n in range(b):
            for i in range(t):
                for j in range(t):
                    R[n, i, j] = torch.abs(times[n, 0] - times[n, i]) 
        
        if self.time_embedding is not None:
            x = self.time_embedding(x, times)

        x = self.transformer(x, mask, R)
        x = x.mean(dim=1)

        latent = self.to_latent(x)
        return self.linear_head(latent), latent