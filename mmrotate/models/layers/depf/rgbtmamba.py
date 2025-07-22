import math
from functools import partial
from typing import Callable, Any

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath

from .ssm_utils import CrossMerge_rgbt_k4, CrossScan_rgbt_k4, CrossScanning_Mechanism, CrossRecover_Mechanism,Slice_Scanning,Feature_return

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmrotate.models.backbones.vmamba.csms6s import SelectiveScanOflex
from mmrotate.models.backbones.vmamba.vmamba import Mlp, mamba_init
from mmrotate.registry import MODELS
from pytorch_wavelets import DWTForward,DWTInverse


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        out = out.permute(0, 2, 3, 1)
        return out
        

class MultiScaleSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(2, 1, 5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x) + self.conv2(x) + self.conv3(x)
        return self.sigmoid(x)


class MultiScaleAttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = ConvModule(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()
        self.msab = MultiScaleSpatialAttention()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        attn = self.msab(x)
        return attn


@MODELS.register_module()
class MTAttentionBlock(BaseModule):
    def __init__(
        self,
        in_channels,
        init_cfg=dict(type="Xavier", layer="Conv2d", distribution="uniform"),
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_ins = len(in_channels)
        self.in_channels = in_channels
        self.cab = nn.ModuleList()

        for i in range(self.num_ins):
            self.cab.append(MultiScaleAttentionLayer(in_channels[i]))

    def forward(self, x):
        outs = []
        for i in range(self.num_ins):
            attn = self.cab[i](x[i])
            out = x[i] * attn
            outs.append(out)
        return outs


class DSSM(nn.Module, mamba_init):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=4,
        ssm_ratio=2,
        dt_rank="auto",
        # dwconv ===============
        d_conv=3,  # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        disable_z=False,
        k_group=4,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()

        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.disable_z = disable_z
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj_vi = nn.Linear(self.d_model, d_proj, bias=bias, **factory_kwargs)
        self.in_proj_ir = nn.Linear(self.d_model, d_proj, bias=bias, **factory_kwargs)
        self.in_proj_sub = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        self.out_norm_vi = nn.LayerNorm(self.d_inner)
        self.out_norm_ir = nn.LayerNorm(self.d_inner)

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d_vi = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_ir = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_sub = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # out proj =======================================
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # x proj ============================
        #k_group = 4
        self.k_group = k_group
        self.x_proj = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) for _ in range(self.k_group)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for _ in range(self.k_group)]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.k_group, merge=True)  # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.k_group, merge=True)  # (K * D)

        # z attn =======================================
        self.z_vi_ca = ChannelAttention(self.d_inner)
        self.z_ir_ca = ChannelAttention(self.d_inner)

    def forward_core(
        self,
        x_sub: torch.Tensor,
        x_vi: torch.Tensor,
        x_ir: torch.Tensor,
        x_proj_bias: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        delta_softplus=True,
        nrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        SelectiveScan=SelectiveScanOflex,
        CrossScan=CrossScan_rgbt_k4,
        CrossMerge=CrossMerge_rgbt_k4,
    ):
        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds

        B, D, H, W = x_vi.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows == 0:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        if backnrows == 0:
            if D % 4 == 0:
                backnrows = 4
            elif D % 3 == 0:
                backnrows = 3
            elif D % 2 == 0:
                backnrows = 2
            else:
                backnrows = 1

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

        xs = CrossScan.apply(x_sub, x_vi, x_ir)  # B, C, H, W -> B, 4, C, 2 * H * W

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        xs: torch.Tensor = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus).view(B, K, -1, 2 * H * W)

        _, y_vi, y_ir = CrossMerge.apply(xs)

        y_vi = y_vi.view(B, -1, H, W)
        y_vi = y_vi.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y_vi = self.out_norm_vi(y_vi).view(B, H, W, -1)

        y_ir = y_ir.view(B, -1, H, W)
        y_ir = y_ir.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y_ir = self.out_norm_ir(y_ir).view(B, H, W, -1)

        y_vi = y_vi.to(x_vi.dtype)
        y_ir = y_ir.to(x_ir.dtype)
        return y_vi, y_ir

    def forward(self, x_vi: torch.Tensor, x_ir: torch.Tensor, before=True):

        if before:
            x_sub = x_vi - x_ir
            x_sub = self.in_proj_sub(x_sub)
            x_sub_trans = x_sub.permute(0, 3, 1, 2).contiguous()

        x_vi = self.in_proj_vi(x_vi)
        x_ir = self.in_proj_ir(x_ir)

        if not self.disable_z:
            x_vi, z_vi = x_vi.chunk(2, dim=-1)  # (b, h, w, d)
            z_vi = self.act(z_vi)
            z_vi = self.z_vi_ca(z_vi) * z_vi + z_vi
            x_ir, z_ir = x_ir.chunk(2, dim=-1)  # (b, h, w, d)
            z_ir = self.act(z_ir)
            z_ir = self.z_ir_ca(z_ir) * z_ir + z_ir

        x_vi_trans = x_vi.permute(0, 3, 1, 2).contiguous()
        x_ir_trans = x_ir.permute(0, 3, 1, 2).contiguous()

        if self.d_conv > 1:
            x_vi_conv = self.act(self.conv2d_vi(x_vi_trans))
            x_ir_conv = self.act(self.conv2d_ir(x_ir_trans))

            if before:
                x_sub_conv = self.act(self.conv2d_sub(x_sub_trans))  # (b, d, h, w)
            else:
                x_sub_conv = x_vi_conv - x_ir_conv

        y_vi, y_ir = self.forward_core(x_sub_conv, x_vi_conv, x_ir_conv)  # b, d, h, w -> b, h, w, d

        if not self.disable_z:
            y_vi = y_vi * z_vi
            y_ir = y_ir * z_ir

        y = self.dropout(self.out_proj(y_vi + y_ir))
        return y


###############################My Block##########################################
from .ssm_utils import CrossScanning_Mechanism, CrossRecover_Mechanism,Slice_Scanning,Feature_return

class ChannelAttention_mini(nn.Module):
     def __init__(self, in_channel,out_channel,whole=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 7, padding=3, bias=False,groups=in_channel)
        self.conv2 = nn.Conv2d(in_channel, out_channel, 5, padding=2, bias=False,groups=in_channel)
        self.conv3 = nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False,groups=in_channel)
        self.whole = whole
     def forward(self,x):
         x1 = self.conv1(x)
         x2 = self.conv2(x) 
         x3 = self.conv3(x)
        # x = x1+x2+x3
         return x1+x2+x3 if self.whole else (x1,x2,x3)
         
class SS2D(DSSM):
    def forward_core(
        self,
        x_vi: torch.Tensor,
        x_ir: torch.Tensor,
        x_proj_bias: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        delta_softplus=True,
        nrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        SelectiveScan=SelectiveScanOflex,
        CrossScan=Slice_Scanning,
        CrossMerge=Feature_return
    ):
        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds

        B, D, H, W = x_vi.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows == 0:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        if backnrows == 0:
            if D % 4 == 0:
                backnrows = 4
            elif D % 3 == 0:
                backnrows = 3
            elif D % 2 == 0:
                backnrows = 2
            else:
                backnrows = 1

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

        x1,_ = torch.sort(x_vi,descending=False)
        x2,_ = torch.sort(x_ir,descending=False)
        xs = CrossScan.apply(x1, x2)  # B, C, H, W -> B, 4, C, 2 * H * W

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        xs: torch.Tensor = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus).view(B, K, -1, 2 * H * W)

        y_vi, y_ir = CrossMerge.apply(xs)
       # print(y_vi.shape)
        y_vi = y_vi.view(B, -1, H, W)
        y_vi = y_vi.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y_vi = self.out_norm_vi(y_vi).view(B, H, W, -1)

        y_ir = y_ir.view(B, -1, H, W)
        y_ir = y_ir.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y_ir = self.out_norm_ir(y_ir).view(B, H, W, -1)

        y_vi = y_vi.to(x_vi.dtype)
        y_ir = y_ir.to(x_ir.dtype)
        return y_vi, y_ir

    def forward(self, x_vi: torch.Tensor, x_ir: torch.Tensor):
        x_vi = self.in_proj_vi(x_vi)
        x_ir = self.in_proj_ir(x_ir)

        if not self.disable_z:
            x_vi, z_vi = x_vi.chunk(2, dim=-1)  # (b, h, w, d)
            z_vi = self.act(z_vi)
            z_vi = self.z_vi_ca(z_vi) * z_vi + z_vi
            x_ir, z_ir = x_ir.chunk(2, dim=-1)  # (b, h, w, d)
            z_ir = self.act(z_ir)
            z_ir = self.z_ir_ca(z_ir) * z_ir + z_ir

        x_vi_trans = x_vi.permute(0, 3, 1, 2).contiguous()
        x_ir_trans = x_ir.permute(0, 3, 1, 2).contiguous()

        if self.d_conv > 1:
            x_vi_conv = self.act(self.conv2d_vi(x_vi_trans))
            x_ir_conv = self.act(self.conv2d_ir(x_ir_trans))

        y_vi, y_ir = self.forward_core(x_vi_conv, x_ir_conv)  # b, d, h, w -> b, h, w, d

        if not self.disable_z:
            y_vi = y_vi * z_vi
            y_ir = y_ir * z_ir
        '''y_vi = self.out_proj(y_vi)
        y_ir = self.out_proj(y_ir)'''
        y = self.dropout(self.out_proj(y_vi + y_ir))
        return y

class Priority_Mamba(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        mlp_ratio=0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 4,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        use_checkpoint: bool = False,
        act_layer=nn.GELU,
        mlp_drop: float = 0.1,
        disable_z: bool = False,
        k_group: int = 2,
        off_dim:int=128,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1_vi = norm_layer(hidden_dim)
        self.norm1_ir = norm_layer(hidden_dim)
        self.offset = nn.Sequential(
            nn.Conv2d(off_dim,2*off_dim,kernel_size = 3, padding=1, groups=off_dim),
            nn.SiLU(),
            nn.Conv2d(off_dim*2,2*off_dim,kernel_size = 3, padding=1, groups=off_dim),
            nn.SiLU(),
            nn.Conv2d(off_dim*2,off_dim,kernel_size = 3, padding=1, groups=off_dim),
            nn.SiLU(),
            norm_layer(hidden_dim),
            nn.Conv2d(off_dim,off_dim,kernel_size = 1)
        )

        self.op = SS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            disable_z=disable_z,
            k_group = k_group,
            **kwargs,
        )
        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=mlp_drop,
                channels_first=False,
            )

    def _forward(self, x_vi: torch.Tensor, x_ir: torch.Tensor):
        x_vi1,x_ir1 = self.norm1_vi(x_vi), self.norm1_ir(x_ir)  # B,H,W,C
        de = x_vi1-x_ir1 #2,128,160,96
        off = self.offset(de)
        x_vi2,x_ir2 = x_vi1+off,x_ir1+off
        x = self.op(x_vi2,x_ir2)
        x = x_vi + x_ir + self.drop_path(x)

        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, x_vi: torch.Tensor, x_ir: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_vi, x_ir)
        else:
            return self._forward(x_vi, x_ir)

@MODELS.register_module()
class PModule(BaseModule):
    def __init__(
        self,
        in_channels,
        drop_path: float = 0.1,
        mlp_ratio=4.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 1,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        act_layer=nn.GELU,
        mlp_drop: float = 0.1,
        sub_before: bool = True,
        disable_z: bool = False,
        init_cfg: dict | checkpoint.List[dict] | None = None,
    ):
        super().__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.num_ins = len(in_channels)
        self.crossModalFusBlock = nn.ModuleList()

        for i in range(self.num_ins):
            self.crossModalFusBlock.append(
                Priority_Mamba(
                    hidden_dim=in_channels[i],
                    drop_path=drop_path,
                    mlp_ratio=mlp_ratio,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop_rate,
                    d_state=d_state,
                    dt_rank=dt_rank,
                    ssm_ratio=ssm_ratio,
                    act_layer=act_layer,
                    mlp_drop=mlp_drop,
                    sub_before=sub_before,
                    disable_z=disable_z,
                    off_dim = 2**(7-i),
                )
            )

    def forward(self, x_vi: torch.Tensor, x_ir: torch.Tensor):
        outs = []
        for i in range(self.num_ins):
            x_vi_ = x_vi[i].permute(0, 2, 3, 1)
            x_ir_ = x_ir[i].permute(0, 2, 3, 1)
            out = self.crossModalFusBlock[i](x_vi_, x_ir_)
            outs.append(out.permute(0, 3, 1, 2))
        return tuple(outs)

############################## Enhancement #########################################
class FEB(nn.Module):
    def __init__(self, nc):
        super(FEB, self).__init__()
        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0,groups=nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0,groups=nc))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0,groups=nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0,groups=nc))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        return x_out + x

class SSW(nn.Module, mamba_init):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=4,
        ssm_ratio=2,
        dt_rank="auto",
        # dwconv ===============
        d_conv=3,  # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        disable_z=False,
        ms_in = 3,
        ms_out = 192,
        wave = 'haar',
        mode = 'symmetric',
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()

        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.disable_z = disable_z
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj_vi = nn.Linear(self.d_model, d_proj, bias=bias, **factory_kwargs)
        self.in_proj_ir = nn.Linear(self.d_model, d_proj, bias=bias, **factory_kwargs)
        self.in_proj_sub = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        self.out_norm_h = nn.LayerNorm(self.d_inner)
        self.out_norm_v = nn.LayerNorm(self.d_inner)
        self.out_norm_d = nn.LayerNorm(self.d_inner)

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d_vi = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_ir = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_sub = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # out proj =======================================
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # x proj ============================
        k_group = 6
        self.x_proj = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) for _ in range(k_group)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for _ in range(k_group)]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=k_group, merge=True)  # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=k_group, merge=True)  # (K * D)

        # z attn =======================================
        #Please note that the channel changing
        self.z_cah = ChannelAttention_mini(ms_in,ms_out,True)
        self.z_cav = ChannelAttention_mini(ms_in,ms_out,True)
        self.z_cad = ChannelAttention_mini(ms_in,ms_out,True)
        self.z_cal = ChannelAttention_mini(ms_in,ms_out,False)
        self.idwt = DWTInverse(wave=wave, mode=mode)
        self.changeh = nn.Sequential(nn.Linear(ms_out,3),nn.SiLU())
        self.changev = nn.Sequential(nn.Linear(ms_out,3),nn.SiLU())
        self.changed = nn.Sequential(nn.Linear(ms_out,3),nn.SiLU())
        self.changehl = nn.Sequential(nn.Linear(ms_out,3),nn.SiLU())
        self.changelh = nn.Sequential(nn.Linear(ms_out,3),nn.SiLU())
        self.changehh = nn.Sequential(nn.Linear(ms_out,3),nn.SiLU())
        self.fft = FEB(nc=3)

    def forward_core(
        self,
        x_h: torch.Tensor,
        x_v: torch.Tensor,
        x_d: torch.Tensor,
        x_proj_bias: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        delta_softplus=True,
        nrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        SelectiveScan=SelectiveScanOflex,
        CrossScan=CrossScanning_Mechanism,
        CrossMerge=CrossRecover_Mechanism,
    ):
        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds

        #print(x_h.shape)
        B, D, H, W = x_h.shape
       
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows == 0:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        if backnrows == 0:
            if D % 4 == 0:
                backnrows = 4
            elif D % 3 == 0:
                backnrows = 3
            elif D % 2 == 0:
                backnrows = 2
            else:
                backnrows = 1

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

        #x_hm,x_vm,x_dm = self.z_cah(x_h),self.z_cav(x_v),self.z_cad(x_d),
        xs = CrossScan.apply(x_h, x_v, x_d)  # B, C, H, W -> B, 4, C, 2 * H * W
        '''print(xs.shape) #B,6,3,2*h/2*w/2
        print(x_proj_weight.shape) #6,14,192'''
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        xs: torch.Tensor = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus).view(B, K, -1, 2 * H * W)

        y_h, y_v, y_d = CrossMerge.apply(xs)

        y_h = y_h.view(B, -1, H, W)
        y_h = y_h.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y_h = self.out_norm_h(y_h).view(B, H, W, -1)
        
        y_v = y_v.view(B, -1, H, W)
        y_v = y_v.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y_v = self.out_norm_v(y_v).view(B, H, W, -1)

        y_d = y_d.view(B, -1, H, W)
        y_d = y_d.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y_d = self.out_norm_d(y_d).view(B, H, W, -1)

        y_h = y_h.to(x_h.dtype)
        y_v = y_v.to(x_v.dtype)
        y_d = y_d.to(x_d.dtype)
        return y_h,y_v,y_d

    '''def forward(self,low,high):
        hl,lh,hh = high[0],high[1],high[2]
        y_h,y_v,y_d = self.forward_core(hl,lh,hh)
        y_h,y_v,y_d = self.changeh(y_h),self.changev(y_v),self.changed(y_d)
        y_h,y_v,y_d = y_h.permute(0,3,1,2),y_v.permute(0,3,1,2),y_d.permute(0,3,1,2)
        h = hl*y_h
        v = lh*y_v
        d = hh*y_d
        h = torch.unsqueeze(h,dim=2)
        v = torch.unsqueeze(v,dim=2)
        d = torch.unsqueeze(d,dim=2)
        high = torch.cat([h,v,d],dim=2)
       #print(high.shape)
        x = self.idwt((low,[high]))
        return x'''
    
    '''def forward(self,low,high):
        hl,lh,hh = high[:,:,0,:,:],high[:,:,1,:,:],high[:,:,2,:,:]
        hlx,lhx,hhx = self.z_cah(hl),self.z_cav(lh),self.z_cad(hh)
        #print(hlx.shape)
        hlx,lhx,hhx = self.changehl(hlx.permute(0,2,3,1)),self.changelh(lhx.permute(0,2,3,1)),self.changehh(hhx.permute(0,2,3,1))
        hlx,lhx,hhx = hlx.permute(0,3,1,2),lhx.permute(0,3,1,2),hhx.permute(0,3,1,2)
        #print(hlx.shape)
        h = torch.unsqueeze(hlx,dim=2)
        v = torch.unsqueeze(lhx,dim=2)
        d = torch.unsqueeze(hhx,dim=2)
        highs = torch.cat([h,v,d],dim=2)
        x1,x2,x3 = self.z_cal(low)
        y_h,y_v,y_d = self.forward_core(x1,x2,x3)
        y_h,y_v,y_d = self.changeh(y_h),self.changev(y_v),self.changed(y_d)
        y_h,y_v,y_d = y_h.permute(0,3,1,2),y_v.permute(0,3,1,2),y_d.permute(0,3,1,2)
        lower = low*(y_h+y_v+y_d)
        x = self.idwt((lower,[highs]))
        return x'''
    
    def forward(self,low,high):
        x1,x2,x3 = self.z_cal(low)
        y_h,y_v,y_d = self.forward_core(x1,x2,x3)
        y_h,y_v,y_d = self.changeh(y_h),self.changev(y_v),self.changed(y_d)
        y_h,y_v,y_d = y_h.permute(0,3,1,2),y_v.permute(0,3,1,2),y_d.permute(0,3,1,2)
        lower = low*(y_h+y_v+y_d)
        lower = self.fft(lower)
        x = self.idwt((lower,[high]))
        return x


        
@MODELS.register_module()
class SPE(nn.Module):
    def __init__(self,
                 ms,
                 wave= 'haar',
                 mode = 'symmetric'):
       super().__init__()
       self.SSW1 = SSW(ms = ms,wave=wave, mode=mode) 
       self.SSW2 = SSW(ms = ms,wave=wave, mode=mode)
       self.dwt = DWTForward(J=2,wave=wave, mode=mode)
       '''self.fft1 = FEB(nc = 3)
       self.fft2 = FEB(nc = 3)'''
       self.fcn = nn.SiLU()

    def forward(self,x):
        #print(x.shape) 2,3,512 640
        low2,high = self.dwt(x)
        #print(low2.shape) 2,3 128,160
        high1,high2 = high[0],high[1] #2,3,3,128,160
        '''hl1,hl2 = high1[:,:,0,:,:],high2[:,:,0,:,:]
        lh1,lh2 = high1[:,:,1,:,:],high2[:,:,1,:,:]
        hh1,hh2 = high1[:,:,2,:,:],high2[:,:,2,:,:]
        high11 = [hl1,lh1,hh1]
        high22 = [hl2,lh2,hh2]'''
        x1 = self.SSW2(low2,high2)
        #x1 = self.fft1(x1)
        x2 = self.fcn(x1)
        x3 = self.SSW1(x2,high1)
       # x3 = self.fft2(x3)
        x4 = x3+x
        return x4