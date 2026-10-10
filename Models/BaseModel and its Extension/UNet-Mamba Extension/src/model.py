import math
from typing import Optional, Union, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# ---------------------------------------------------------------------------
# 1. Optimized Checkpointed Selective Scan Operator (Pure PyTorch, Low VRAM)
# ---------------------------------------------------------------------------
@torch.amp.autocast('cuda', enabled=False)
def _scan_chunk_fn(u_chunk, delta_chunk, A_exp, B_chunk, C_chunk, x_state):
    u_chunk = u_chunk.float()
    delta_chunk = delta_chunk.float()
    A_exp = A_exp.float()
    B_chunk = B_chunk.float()
    C_chunk = C_chunk.float()
    x_state = x_state.float()
    
    log_deltaA = delta_chunk.unsqueeze(-1) * A_exp.unsqueeze(0).unsqueeze(2)
    log_deltaA = torch.clamp(log_deltaA, min=-10.0, max=0.0)
    log_cum_A = torch.cumsum(log_deltaA, dim=2)
    log_cum_A = torch.clamp(log_cum_A, min=-15.0, max=0.0)
    
    cum_A = torch.exp(log_cum_A)
    inv_cum_A = torch.exp(-log_cum_A)
    
    deltaB_u = (delta_chunk * u_chunk).unsqueeze(-1) * B_chunk
    scaled_b = deltaB_u * inv_cum_A
    cum_scaled_b = torch.cumsum(scaled_b, dim=2) + x_state.unsqueeze(2)
    
    x_chunk = cum_A * cum_scaled_b
    new_state = x_chunk[:, :, -1, :]
    
    y_chunk = (x_chunk * C_chunk).sum(dim=-1)
    return y_chunk, new_state


def chunked_checkpoint_selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False, chunk_size=2048):
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    
    K_D = delta.shape[1]
    if A.dim() == 2 and K_D != A.shape[0]:
        K = K_D // dim
        A_exp = A.repeat(K, 1)
    else:
        A_exp = A
        
    L = u.shape[2]
    
    if is_variable_B and B.dim() == 4 and B.shape[1] != K_D:
        B = torch.repeat_interleave(B, K_D // B.shape[1], dim=1)
    if is_variable_C and C.dim() == 4 and C.shape[1] != K_D:
        C = torch.repeat_interleave(C, K_D // C.shape[1], dim=1)

    ys = []
    x_state = torch.zeros((batch, K_D, dstate), dtype=torch.float32, device=u.device)

    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        
        u_chunk = u[:, :, start:end]
        delta_chunk = delta[:, :, start:end]
        
        if not is_variable_B:
            B_chunk = B.unsqueeze(0).unsqueeze(2).expand(batch, K_D, end-start, dstate)
        else:
            if B.dim() == 3:
                B_chunk = B[:, :, start:end].permute(0, 2, 1).unsqueeze(1).expand(batch, K_D, end-start, dstate)
            else:
                B_chunk = B[:, :, :, start:end].permute(0, 1, 3, 2)
                
        if not is_variable_C:
            C_chunk = C.unsqueeze(0).unsqueeze(2).expand(batch, K_D, end-start, dstate)
        else:
            if C.dim() == 3:
                C_chunk = C[:, :, start:end].permute(0, 2, 1).unsqueeze(1).expand(batch, K_D, end-start, dstate)
            else:
                C_chunk = C[:, :, :, start:end].permute(0, 1, 3, 2)
                
        if u.requires_grad:
            y_chunk, x_state = checkpoint.checkpoint(_scan_chunk_fn, u_chunk, delta_chunk, A_exp, B_chunk, C_chunk, x_state, use_reentrant=False)
        else:
            y_chunk, x_state = _scan_chunk_fn(u_chunk, delta_chunk, A_exp, B_chunk, C_chunk, x_state)
            
        ys.append(y_chunk)
        
    y = torch.cat(ys, dim=2)
    out = y if D is None else y + u * D.view(1, -1, 1)
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    
    return out if not return_last_state else (out, x_state)


# ---------------------------------------------------------------------------
# 2. 2D State-Space Module (SS2D) & VSSBlock
# ---------------------------------------------------------------------------
class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2
        )
        self.act = nn.SiLU()
        
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj
        
        self.dt_projs = (
            nn.Linear(self.dt_rank, self.d_inner, bias=True),
            nn.Linear(self.dt_rank, self.d_inner, bias=True),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        # S4D init A
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner)
        self.A_logs = nn.Parameter(torch.log(A).repeat(2, 1, 1))
        self.D = nn.Parameter(torch.ones(2, self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        L = H * W
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (B, d_inner, H, W)
        
        # 2 directions: H-first, W-first
        x_hw = x.reshape(B, self.d_inner, L)
        x_wh = x.permute(0, 1, 3, 2).reshape(B, self.d_inner, L)
        
        xs = torch.stack([x_hw, x_wh], dim=1) # (B, 2, d_inner, L)
        
        # Projection for dt, B, C
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        dts = dts + self.dt_projs_bias.unsqueeze(0).unsqueeze(-1)
        
        As = -torch.exp(torch.clamp(self.A_logs, max=4.0)) # (2, d_inner, d_state)
        
        # Reshape for selective scan
        u = xs.reshape(B, 2 * self.d_inner, L)
        delta = dts.reshape(B, 2 * self.d_inner, L)
        A = As.reshape(2 * self.d_inner, self.d_state)
        B_scan = Bs.reshape(B, 2, self.d_state, L)
        C_scan = Cs.reshape(B, 2, self.d_state, L)
        D_scan = self.D.reshape(2 * self.d_inner)
        z_scan = z.permute(0, 3, 1, 2).reshape(B, self.d_inner, L).repeat(1, 2, 1)
        
        out_scan = chunked_checkpoint_selective_scan_ref(
            u, delta, A, B_scan, C_scan, D=D_scan, z=z_scan, delta_softplus=True, chunk_size=2048
        )
        
        # Reverse directions and merge
        out_scan = out_scan.reshape(B, 2, self.d_inner, L)
        y_hw = out_scan[:, 0]
        y_wh = out_scan[:, 1].reshape(B, self.d_inner, W, H).permute(0, 1, 3, 2).reshape(B, self.d_inner, L)
        
        y = y_hw + y_wh
        y = y.reshape(B, self.d_inner, H, W).permute(0, 2, 3, 1).contiguous()
        
        out = self.out_proj(y)
        return out

class VSSBlock(nn.Module):
    def __init__(self, hidden_dim=0, drop_path=0., norm_layer=nn.LayerNorm, d_state=16, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attn = SS2D(d_model=hidden_dim, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.self_attn(self.ln_1(x)))
        return x

class VSSLayer(nn.Module):
    def __init__(self, dim, depth, drop_path=0., norm_layer=nn.LayerNorm, d_state=16, use_checkpoint=False, **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                d_state=d_state
            )
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint and x.requires_grad:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x


# ---------------------------------------------------------------------------
# 3. ResT Backbone Encoder (4-channel stem)
# ---------------------------------------------------------------------------
class ResTStem(nn.Module):
    def __init__(self, in_chans=4, embed_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(embed_dim // 2)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResTEncoder(nn.Module):
    def __init__(self, in_chans=4, embed_dims=[64, 128, 256, 512], depths=[2, 2, 2, 2]):
        super().__init__()
        self.stem = ResTStem(in_chans=in_chans, embed_dim=embed_dims[0])
        
        self.stage1 = nn.Sequential(*[BasicBlock(embed_dims[0], embed_dims[0]) for _ in range(depths[0])])
        self.stage2 = nn.Sequential(
            BasicBlock(embed_dims[0], embed_dims[1], stride=2),
            *[BasicBlock(embed_dims[1], embed_dims[1]) for _ in range(depths[1]-1)]
        )
        self.stage3 = nn.Sequential(
            BasicBlock(embed_dims[1], embed_dims[2], stride=2),
            *[BasicBlock(embed_dims[2], embed_dims[2]) for _ in range(depths[2]-1)]
        )
        self.stage4 = nn.Sequential(
            BasicBlock(embed_dims[2], embed_dims[3], stride=2),
            *[BasicBlock(embed_dims[3], embed_dims[3]) for _ in range(depths[3]-1)]
        )

    def forward(self, x):
        x = self.stem(x)   # (B, 64, H/4, W/4)
        c1 = self.stage1(x)
        c2 = self.stage2(c1) # (B, 128, H/8, W/8)
        c3 = self.stage3(c2) # (B, 256, H/16, W/16)
        c4 = self.stage4(c3) # (B, 512, H/32, W/32)
        return [c1, c2, c3, c4]


# ---------------------------------------------------------------------------
# 4. Mamba Decoder, Local Supervision (LSM) & UNetMamba Wrapper
# ---------------------------------------------------------------------------
class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        # input x is (B, C, H, W)
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = self.norm(x)
        return x # (B, 2H, 2W, C/2)

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = self.norm(x)
        return x # (B, 4H, 4W, dim)

class LocalSupervision(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False)

    def forward(self, x, h, w):
        l1 = self.conv3(x)
        l2 = self.conv1(x)
        out = self.drop(l1 + l2)
        out = self.conv_out(out)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return out

class MambaSegDecoder(nn.Module):
    def __init__(self, num_classes=2, encoder_channels=[64, 128, 256, 512], decode_channels=64, d_state=16):
        super().__init__()
        self.num_classes = num_classes
        n_stages = len(encoder_channels)
        
        stages = []
        expand_layers = []
        lsm_layers = []
        concat_back_dim = []
        
        for s in range(1, n_stages):
            in_below = encoder_channels[-s]
            in_skip = encoder_channels[-(s + 1)]
            
            expand_layers.append(PatchExpand(dim=in_below, dim_scale=2))
            stages.append(VSSLayer(dim=in_skip, depth=2, d_state=d_state, use_checkpoint=True))
            concat_back_dim.append(nn.Linear(2 * in_skip, in_skip))
            lsm_layers.append(LocalSupervision(in_skip, num_classes))
            
        expand_layers.append(FinalPatchExpand_X4(dim=encoder_channels[0], dim_scale=4))
        stages.append(nn.Identity())
        
        self.stages = nn.ModuleList(stages)
        self.expand_layers = nn.ModuleList(expand_layers)
        self.concat_back_dim = nn.ModuleList(concat_back_dim)
        self.lsm = nn.ModuleList(lsm_layers)
        self.seg = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1, bias=True)

    def forward(self, skips, h, w, is_training=True):
        lres_input = skips[-1]
        ls = []
        
        for s in range(len(self.stages)):
            x = self.expand_layers[s](lres_input)
            
            if s < (len(self.stages) - 1):
                x = torch.cat((x, skips[-(s + 2)].permute(0, 2, 3, 1)), dim=-1)
                x = self.concat_back_dim[s](x)
                
            x = self.stages[s](x).permute(0, 3, 1, 2)
            
            if s == (len(self.stages) - 1):
                seg_out = self.seg(x)
            else:
                if is_training:
                    ls.append(self.lsm[s](x, h, w))
            lres_input = x
            
        if is_training:
            return seg_out, sum(ls)
        else:
            return seg_out

class UNetMamba(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_classes=2,
        embed_dim=64,
        decode_channels=64,
        pretrained=False,
        d_state=16,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        encoder_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        
        self.encoder = ResTEncoder(in_chans=in_channels, embed_dims=encoder_channels, depths=[2, 2, 2, 2])
        self.decoder = MambaSegDecoder(
            num_classes=num_classes,
            encoder_channels=encoder_channels,
            decode_channels=decode_channels,
            d_state=d_state
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        skips = self.encoder(x)
        if self.training:
            main_out, aux_out = self.decoder(skips, h, w, is_training=True)
            return main_out, aux_out
        else:
            main_out = self.decoder(skips, h, w, is_training=False)
            return main_out

def build_model(config=None):
    if config is None:
        from config import load_config
        config = load_config()
    m_cfg = config['model']
    model = UNetMamba(
        in_channels=m_cfg.get('in_channels', 4),
        num_classes=m_cfg.get('num_classes', 2),
        embed_dim=m_cfg.get('embed_dim', 64),
        decode_channels=m_cfg.get('decode_channels', 64),
        pretrained=m_cfg.get('pretrained', False)
    )
    return model
