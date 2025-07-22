import torch

class CrossScanning_Mechanism(torch.autograd.Function):
     @staticmethod
     def forward(ctx, x_h: torch.Tensor, x_v: torch.Tensor, x_d: torch.Tensor):
        # B, C, H, W -> B, 2, C, 2 * H * W
        B, C, H, W = x_h.shape
        ctx.shape = (B, C, H, W)
        x_f = (x_h+x_v+x_d)/3
        x_fus = x_v.new_empty((B, 6, C, 2 * H * W))
        x_fus[:, 0] = torch.concat([x_h.flatten(2, 3), x_f.flatten(2, 3)], dim=2)
        x_fus[:, 1] = torch.flip(x_fus[:, 0], dims=[-1])
        x_fus[:, 2] = torch.concat([x_v.flatten(2, 3), x_f.flatten(2, 3)], dim=2)
        x_fus[:, 3] = torch.flip(x_fus[:, 2], dims=[-1])
        x_fus[:, 4] = torch.concat([x_d.flatten(2, 3), x_f.flatten(2, 3)], dim=2)
        x_fus[:, 5] = torch.flip(x_fus[:, 4], dims=[-1])
        return x_fus

     @staticmethod
     def backward(ctx, x_fus: torch.Tensor):
        # out: (b, 2, d, l)
        B, C, H, W = ctx.shape
        # L = 2 * H * W
        x_fus_1 = x_fus[:, 0] + x_fus[:, 1].flip(dims=[-1])  # B, d, 2 * H * W
        x_fus_2 = x_fus[:, 2] + x_fus[:, 3].flip(dims=[-1])
        x_fus_3 = x_fus[:, 4] + x_fus[:, 5].flip(dims=[-1])

        # get B, d, H*W
        return (
            x_fus_1[:, :, 0 : H * W].view(B, -1, H, W),
            x_fus_2[:, :, 0 : H * W].view(B, -1, H, W),
            x_fus_3[:, :, 0 : H * W].view(B, -1, H, W)
        )
#
class CrossRecover_Mechanism(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_fus: torch.Tensor):
        B, K, D, L = x_fus.shape
        # ctx.shape = (H, W)
        # ys = ys.view(B, K, D, -1)
        x_fus_1 = x_fus[:, 0] + x_fus[:, 1].flip(dims=[-1])  # B, d, 2 * H * W, broadcast
        x_fus_2 = x_fus[:, 2] + x_fus[:, 3].flip(dims=[-1])
        x_fus_3 = x_fus[:, 4] + x_fus[:, 5].flip(dims=[-1])
        # y = ys[:, :, 0:L//2] + ys[:, :, L//2:L]
        return (
            x_fus_1[:, :, 0 : L // 2],
            x_fus_2[:, :, 0 : L // 2],
            x_fus_3[:, :, 0 : L // 2]
        )

    @staticmethod
    def backward(ctx, x_h: torch.Tensor, x_v: torch.Tensor, x_d: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        # H, W = ctx.shape
        B, C, L = x_v.shape
        x_fus = x_v.new_empty((B, 6, C, 2 * L))
        x_f = (x_h+x_v+x_d)/3
        x_fus[:, 0] = torch.cat([x_h, x_f], dim=2)
        x_fus[:, 1] = torch.flip(x_fus[:, 0], dims=[-1])
        x_fus[:, 2] = torch.cat([x_v, x_f], dim=2)
        x_fus[:, 3] = torch.flip(x_fus[:, 2], dims=[-1])
        x_fus[:, 4] = torch.cat([x_d, x_f], dim=2)
        x_fus[:, 5] = torch.flip(x_fus[:, 4], dims=[-1])
        x_fus = x_fus.view(B, 6, C, 2 * L)
        return x_fus

class Slice_Scanning(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1: torch.Tensor, x2: torch.Tensor):
        # B, C, H, W -> B, 2, C, 2 * H * W
        B, C, H,W= x1.shape
        ctx.shape = (B, C, H, W)
        x_fus = x1.new_empty((B, 2, C, 2*H*W))
        x_fus[:, 0] = torch.concat([x1.flatten(2, 3), x2.flatten(2, 3)], dim=2)
        x_fus[:, 1] = torch.flip(x_fus[:, 0], dims=[-1])
        return x_fus

    @staticmethod
    def backward(ctx, x_fus: torch.Tensor):
        # out: (b, 2, d, l)
        #print(x_fus.shape)
        B, C, H, W = ctx.shape
        x_fus_1 = x_fus[:, 0] + x_fus[:, 1].flip(dims=[-1])  # B, d, 2 * H * W
        return (
            x_fus_1[:, :, 0 : H*W].view(B, -1, H,W),
            x_fus_1[:, :, H*W: 2 *H*W].view(B, -1, H,W)
        )
        
class Feature_return(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_fus: torch.Tensor):
        B, K, D, L = x_fus.shape
        # ctx.shape = (H, W)
        # ys = ys.view(B, K, D, -1)
        x_fus_1 = x_fus[:, 0] + x_fus[:, 1].flip(dims=[-1])  # B, d, 2 * H * W, broadcast
        return (
            x_fus_1[:, :, 0 : L // 2],
            x_fus_1[:, :, L // 2 : L],
        )

    @staticmethod
    def backward(ctx, x1: torch.Tensor, x2: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        # H, W = ctx.shape
        B, C, L = x1.shape
        x_fus = x1.new_empty((B, 2, C, 2 * L))

        x_fus[:, 0] = torch.cat([x1,x2], dim=2)
        x_fus[:, 1] = torch.flip(x_fus[:, 0], dims=[-1])
        return x_fus