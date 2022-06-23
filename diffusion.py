from numpy import dtype
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Callable, Union, Any, TypeVar, Tuple
from inspect import isfunction
import math
from einops import rearrange
import tqdm
from functools import partial
Tensor = TypeVar('torch.tensor')

class base_diffusion(pl.LightningModule):
    def __init__(self, num_timesteps: int, **kwargs: Any) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        print(self.num_timesteps)
        self.image_size = 64
        #create model here
        self.net = Unet(dim = 56, dim_mults = (1, 2, 4, 8))
        self.gd = GaussianDiffusion(self.net, image_size = self.image_size,
                            timesteps = self.num_timesteps,   # number of steps
                            loss_type = 'l1'    # L1 or L2 - doesn't actually do anything right now
        )

    def forward(self, input: Tensor, **kwargs) -> Any:
        input = input.cuda()
        b, c, h, w = input.shape
        noise = torch.randn_like(input)
        t = torch.randint(0, self.num_timesteps, (b,)).long().cuda()
        x = self.gd.q_sample(x_start=input, t=t, noise=noise).cuda()
        model_out = self.net(x, t) #predicting the noise atm
        return model_out, input, noise, t, x

    def loss_function(self,
                      *args,
                      batch=True,
                      **kwargs) -> dict:
        output = args[0]
        noise = args[2]
        if batch:
            return {'loss': F.l1_loss(output, noise)}
        else:
            return {'loss': F.l1_loss(output, noise, reduction='none').mean(dim=[1,2,3])}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        return self.gd.sample(num_samples)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        # return self.gd.sample(144)
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.decode(self.encode(x))

    def to_latent(self, x: Tensor): 
        l = self.encode(x)
        batchsize = l.shape[0]
        size = l.shape[1]*l.shape[2]*l.shape[3]
        return l.reshape([batchsize, size]) #since clustering needs each point to be vecotr-shaped

    def encode(self, x:Tensor, **kwargs) -> Tensor:
        b, c, h, w = x.shape
        noise = torch.randn_like(x)
        t = torch.Tensor(b*[self.gd.num_timesteps-1]).long().cuda()
        noised = self.gd.q_sample(x_start=x, t=t, noise=noise) #image with added noise
        return noised

    def decode(self, x, **kwargs) -> Tensor: #x is what's returned by encode method above
        res = self.gd.p_sample_loop(x)
        return res


    def configure_optimizers(self, params):
        return self.net.configure_optimizers(params)


# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_time_emb = True,
        resnet_block_groups = 8,
        learned_variance = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out * 2, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_conv = nn.Sequential(
            block_klass(dim, dim),
            nn.Conv2d(dim, self.out_dim, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)

    def configure_optimizers(self, params):
        optimizer = optim.Adam(self.parameters(), lr=params['LR'])
        return [optimizer]

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine'
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and denoise_fn.channels != denoise_fn.out_dim)

        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.objective = objective

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_output = self.denoise_fn(x, t)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, img):
        device = self.betas.device
        b = img.shape[0]

        for i in reversed(range(0, self.num_timesteps)): #img updates at ea timestep
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        image_size = self.image_size
        channels = self.channels
        shape = (batch_size, channels, image_size, image_size)
        device = self.betas.device
        img = torch.randn(shape, device=device)
        return self.p_sample_loop(img)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.denoise_fn(x, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target)
        return loss

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)


# def default(val, d):
#     if val is not None:
#         return val
#     return d() if isfunction(d) else d

# class Residual(pl.LightningModule):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x, *args, **kwargs):
#         return self.fn(x, *args, **kwargs) + x

# class SinusoidalPosEmb(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, x):
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim) * -emb).cuda()
#         # emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = x[:, None] * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb

# def Upsample(dim):
#     return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

# def Downsample(dim):
#     return nn.Conv2d(dim, dim, 4, 2, 1)

# class LayerNorm(pl.LightningModule):
#     def __init__(self, dim, eps = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
#         self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

#     def forward(self, x):
#         var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
#         mean = torch.mean(x, dim = 1, keepdim = True)
#         return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# class PreNorm(pl.LightningModule):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = LayerNorm(dim)

#     def forward(self, x):
#         x = self.norm(x)
#         return self.fn(x)

# class Block(pl.LightningModule):
#     def __init__(self, dim, dim_out, groups = 8):
#         super().__init__()
#         self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
#         self.norm = nn.GroupNorm(groups, dim_out)
#         self.act = nn.SiLU()

#     def forward(self, x, scale_shift = None):
#         x = self.proj(x)
#         x = self.norm(x)

#         if scale_shift is not None:
#             scale, shift = scale_shift
#             x = x * (scale + 1) + shift

#         x = self.act(x)
#         return x

# class ResnetBlock(pl.LightningModule):
#     def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(time_emb_dim, dim_out * 2)
#         ) if (time_emb_dim is not None) else None

#         self.block1 = Block(dim, dim_out, groups = groups)
#         self.block2 = Block(dim_out, dim_out, groups = groups)
#         self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

#     def forward(self, x, time_emb = None):

#         scale_shift = None
#         if self.mlp is not None and time_emb is not None:
#             time_emb = self.mlp(time_emb)
#             time_emb = rearrange(time_emb, 'b c -> b c 1 1')
#             scale_shift = time_emb.chunk(2, dim = 1)

#         h = self.block1(x, scale_shift = scale_shift)

#         h = self.block2(h)
#         return h + self.res_conv(x)

# class LinearAttention(pl.LightningModule):
#     def __init__(self, dim, heads = 4, dim_head = 32):
#         super().__init__()
#         self.scale = dim_head ** -0.5
#         self.heads = heads
#         hidden_dim = dim_head * heads
#         self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Conv2d(hidden_dim, dim, 1),
#             LayerNorm(dim)
#         )

#     def forward(self, x):
#         b, c, h, w = x.shape
#         qkv = self.to_qkv(x).chunk(3, dim = 1)
#         q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

#         q = q.softmax(dim = -2)
#         k = k.softmax(dim = -1)

#         q = q * self.scale
#         context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

#         out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
#         out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
#         return self.to_out(out)

# class Attention(nn.Module):
#     def __init__(self, dim, heads = 4, dim_head = 32):
#         super().__init__()
#         self.scale = dim_head ** -0.5
#         self.heads = heads
#         hidden_dim = dim_head * heads
#         self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
#         self.to_out = nn.Conv2d(hidden_dim, dim, 1)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         qkv = self.to_qkv(x).chunk(3, dim = 1)
#         q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
#         q = q * self.scale

#         sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
#         sim = sim - sim.amax(dim = -1, keepdim = True).detach()
#         attn = sim.softmax(dim = -1)

#         out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
#         out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
#         return self.to_out(out)

# class Unet(pl.LightningModule):
#     def __init__(
#         self,
#         dim,
#         init_dim = None,
#         out_dim = None,
#         dim_mults=(1, 2, 4, 8),
#         channels = 3,
#         with_time_emb = True,
#         resnet_block_groups = 8,
#         learned_variance = False
#     ):
#         super().__init__()
#         # determine dimensions

#         self.channels = channels

#         init_dim = default(init_dim, dim // 3 * 2)
#         self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

#         dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))

#         # time embeddings

#         if with_time_emb:
#             time_dim = dim * 4
#             self.time_mlp = nn.Sequential(
#                 SinusoidalPosEmb(dim),
#                 nn.Linear(dim, time_dim),
#                 nn.GELU(),
#                 nn.Linear(time_dim, time_dim)
#             )
#         else:
#             time_dim = None
#             self.time_mlp = None

#         # layers

#         self.downs = nn.ModuleList([])
#         self.ups = nn.ModuleList([])
#         num_resolutions = len(in_out)

#         for ind, (dim_in, dim_out) in enumerate(in_out):
#             is_last = ind >= (num_resolutions - 1)

#             self.downs.append(nn.ModuleList([
#                 ResnetBlock(dim_in, dim_out, time_emb_dim = time_dim, groups = resnet_block_groups),
#                 ResnetBlock(dim_out, dim_out, time_emb_dim = time_dim, groups = resnet_block_groups),
#                 Residual(PreNorm(dim_out, LinearAttention(dim_out))),
#                 Downsample(dim_out) if not is_last else nn.Identity()
#             ]))

#         mid_dim = dims[-1]
#         self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, groups = resnet_block_groups)
#         self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
#         self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, groups = resnet_block_groups)

#         for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
#             is_last = ind >= (num_resolutions - 1)

#             self.ups.append(nn.ModuleList([
#                 ResnetBlock(dim_out * 2, dim_in, time_emb_dim = time_dim, groups = resnet_block_groups),
#                 ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, groups = resnet_block_groups),
#                 Residual(PreNorm(dim_in, LinearAttention(dim_in))),
#                 Upsample(dim_in) if not is_last else nn.Identity()
#             ]))

#         default_out_dim = channels * (1 if not learned_variance else 2)
#         self.out_dim = default(out_dim, default_out_dim)

#         self.final_conv = nn.Sequential(
#             ResnetBlock(dim, dim, groups = resnet_block_groups),
#             nn.Conv2d(dim, self.out_dim, 1)
#         )

#     def forward(self, x, time):
#         x = self.init_conv(x)
#         t = self.time_mlp(time).cuda() if (self.time_mlp is not None) else None
#         h = []

#         for block1, block2, attn, downsample in self.downs:
#             x = block1(x, t)
#             x = block2(x, t)
#             x = attn(x)
#             h.append(x)
#             x = downsample(x)

#         x = self.mid_block1(x, t)
#         x = self.mid_attn(x)
#         x = self.mid_block2(x, t)

#         for block1, block2, attn, upsample in self.ups:
#             x = torch.cat((x, h.pop()), dim=1)
#             x = block1(x, t)
#             x = block2(x, t)
#             x = attn(x)
#             x = upsample(x)

#         return self.final_conv(x)

#     def configure_optimizers(self, params):
#         optimizer = optim.Adam(self.parameters(), lr=params['LR'])
#         return [optimizer]

# def extract(a, t, x_shape):
#     b, *_ = t.shape
#     out = a.gather(-1, t.cuda())
#     return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# def noise_like(shape, device, repeat=False):
#     repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
#     noise = lambda: torch.randn(shape, device=device)
#     return repeat_noise() if repeat else noise()

# def cosine_beta_schedule(timesteps, s = 0.008):
#     """
#     cosine schedule
#     as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
#     """
#     steps = timesteps + 1
#     x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
#     alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     return torch.clip(betas, 0, 0.999)

# def linear_beta_schedule(timesteps):
#     beta_start = 0.0001
#     beta_end = 0.02
#     return torch.linspace(beta_start, beta_end, timesteps)


# class GaussianDiffusion(pl.LightningModule):
#     def __init__(
#         self,
#         denoise_fn,
#         *,
#         image_size,
#         channels = 3,
#         timesteps = 1000,
#         loss_type = 'l1',
#         objective = 'pred_noise',
#         beta_schedule = 'cosine'
#     ):
#         super().__init__()
#         assert not (type(self) == GaussianDiffusion and denoise_fn.channels != denoise_fn.out_dim)

#         self.channels = channels
#         self.image_size = image_size
#         self.denoise_fn = denoise_fn
#         self.objective = objective

#         betas = linear_beta_schedule(timesteps)

#         alphas = 1. - betas
#         alphas_cumprod = torch.cumprod(alphas, axis=0)
#         alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

#         timesteps, = betas.shape
#         self.num_timesteps = int(timesteps)
#         self.loss_type = loss_type

#         # helper function to register buffer from float64 to float32
#         register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

#         register_buffer('betas', betas)
#         register_buffer('alphas_cumprod', alphas_cumprod)
#         register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

#         # calculations for diffusion q(x_t | x_{t-1}) and others

#         register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
#         register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
#         register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
#         register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
#         register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

#         # calculations for posterior q(x_{t-1} | x_t, x_0)

#         posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

#         # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

#         register_buffer('posterior_variance', posterior_variance)

#         # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

#         register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
#         register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
#         register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

#     def predict_start_from_noise(self, x_t, t, noise):
#         return (
#             extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
#             extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
#         )

#     def q_posterior(self, x_start, x_t, t):
#         posterior_mean = (
#             extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
#             extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
#         )
#         posterior_variance = extract(self.posterior_variance, t, x_t.shape)
#         posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
#         return posterior_mean, posterior_variance, posterior_log_variance_clipped

#     def p_mean_variance(self, x, t, clip_denoised: bool):
#         model_output = self.denoise_fn(x, t)

#         if self.objective == 'pred_noise':
#             x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
#         elif self.objective == 'pred_x0':
#             x_start = model_output
#         else:
#             raise ValueError(f'unknown objective {self.objective}')

#         if clip_denoised:
#             x_start.clamp_(-1., 1.)

#         model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
#         return model_mean, posterior_variance, posterior_log_variance

#     @torch.no_grad()
#     def p_sample(self, x, t, clip_denoised=True):
#         b, *_, device = *x.shape, x.device
#         model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
#         noise = torch.randn_like(x)
#         # no noise when t == 0
#         nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
#         return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

#     @torch.no_grad()
#     def p_sample_loop(self, input): #input noise from layer T
#         device = self.betas.device
#         b = input.shape[0]

#         for i in reversed(range(0, self.num_timesteps)):
#             img = self.p_sample(input, torch.full((b,), i, device=device, dtype=torch.long))

#         img = (img + 1) * 0.5
#         return img

#     @torch.no_grad()
#     def sample(self, batch_size = 16):
#         shape = (batch_size, self.channels, self.image_size, self.image_size)
#         input = torch.randn(shape, device=self.betas.device)
#         return self.p_sample_loop(input)

#     def q_sample(self, x_start, t, noise=None):
#         noise = default(noise, lambda: torch.randn_like(x_start))

#         return (
#             extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
#             extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
#         )

#     @property
#     def loss_fn(self):
#         if self.loss_type == 'l1':
#             return F.l1_loss
#         elif self.loss_type == 'l2':
#             return F.mse_loss
#         else:
#             raise ValueError(f'invalid loss type {self.loss_type}')

#     def p_losses(self, x_start, t, noise = None):
#         b, c, h, w = x_start.shape
#         noise = default(noise, lambda: torch.randn_like(x_start))

#         x = self.q_sample(x_start=x_start, t=t, noise=noise)
#         model_out = self.denoise_fn(x, t)

#         if self.objective == 'pred_noise':
#             target = noise
#         elif self.objective == 'pred_x0':
#             target = x_start
#         else:
#             raise ValueError(f'unknown objective {self.objective}')

#         loss = self.loss_fn(model_out, target)
#         return loss

#     def forward(self, img, *args, **kwargs):
#         b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
#         assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
#         t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

#         img = img*2 - 1
#         return self.p_losses(img, t, *args, **kwargs)

