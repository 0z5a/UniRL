import einops
import torch
import torch.nn as nn
from collections import namedtuple


class Codebook(nn.Module):
    def __init__(self, num_codebook_vectors, latent_dim, beta=0.25):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = latent_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * (torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss


LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'codebook_entropy', 'commitment', 'avg_probs', 'extra_weighted_loss'])

def vid_to_image(x, src_shape, return_handle=False):
    import einops
    x_shape = einops.parse_shape(x, src_shape)
    x = einops.rearrange(x, f'{src_shape} -> (b t) c h w')
    if return_handle:
        return x, lambda tensor: einops.rearrange(tensor, f'(b t) c h w -> {src_shape}', **x_shape)
    else:
        return x
class VQGANCodebookWrapper(Codebook):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero = torch.tensor(0.)
        self.embed_dim = 14

    def forward(self, h, return_loss_breakdown=True):
        h, handle = vid_to_image(h, 'b c t h w', True)
        codebook_mapping, codebook_indices, q_loss = super().forward(h)
        ret = (codebook_mapping, self.zero, codebook_indices)
        if not return_loss_breakdown:
            return ret
        return ret, LossBreakdown(self.zero, self.zero, self.zero, self.zero, q_loss)


from torch.nn import functional as F
def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss
class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage, alpha=0.25, simVQ=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.alpha = alpha
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.simVQ = simVQ

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

        if self.simVQ:
            self.embedding_proj = nn.Linear(self.e_dim, self.e_dim)


    def forward(self, z, codebook_mask=None):
        if codebook_mask is None:
            codebook_mask = 1
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        if self.simVQ:
            embedding = self.embedding_proj(embedding)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = embedding[min_encoding_indices].view(z.shape)

        soft = self.training
        soft = False

        if soft:
            affinity = -d
            flat_affinity = affinity.reshape(-1, affinity.shape[-1])
            flat_affinity /= 0.01
            probs = F.softmax(flat_affinity, dim=-1)
            z_q = torch.einsum('nm,mc->nc', probs, embedding).view(z.shape)

        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # compute loss for embedding
        if self.training and not soft:
            vq_loss = torch.mean((z_q - z.detach()) ** 2) * self.alpha * codebook_mask
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) * codebook_mask
            if self.entropy_loss_ratio:
                entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)
            else:
                entropy_loss = torch.tensor(0., device=z.device)
        else:
            vq_loss = torch.tensor(0., device=z.device)
            commit_loss = torch.tensor(0., device=z.device)
            entropy_loss = torch.tensor(0., device=z.device)

        # preserve gradients
        if not soft:
            z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        if self.simVQ:
            embedding = self.embedding_proj(embedding)

        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q

class VQGANCodebookWrapper2(VectorQuantizer):

    def __init__(self, num_codebook_vectors, latent_dim, beta=1., alpha=0.25, l2_norm=True, entropy_loss_ratio=1., simVQ=False):
        super().__init__(
            n_e=num_codebook_vectors,
            e_dim=latent_dim,
            beta=beta,
            alpha=alpha,
            entropy_loss_ratio=entropy_loss_ratio,
            l2_norm=l2_norm,
            show_usage=False,
            simVQ=simVQ,
        )
        self.zero = torch.tensor(0.)
        import math
        self.embed_dim = math.log(num_codebook_vectors, 2)

    def forward(self, h, return_loss_breakdown=True):

        ndim = h.ndim
        if ndim == 5:
            t = h.shape[2]
            h = einops.rearrange(h, 'b c t h w -> (b t) c h w')


        z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices) = super().forward(h)

        if ndim == 5:
            z_q = einops.rearrange(z_q, '(b t) c h w -> b c t h w', t=t)

        if self.training:
            q_loss = vq_loss + commit_loss + entropy_loss
        else:
            q_loss = self.zero

        ret = (z_q, self.zero, min_encoding_indices)
        if not return_loss_breakdown:
            return ret
        return ret, LossBreakdown(self.zero, self.zero, self.zero, self.zero, q_loss)
