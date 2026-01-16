import torch
import torch.nn as nn
import torch.nn.functional as F

def get_xy_grid(h, w, device):
    y = torch.linspace(0, h-1, h, device=device) / h
    x = torch.linspace(0, w-1, w, device=device) / w
    mesh_y, mesh_x = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((mesh_x, mesh_y), dim=0).unsqueeze(0)

class DifferentiableSSN(nn.Module):
    def __init__(self, num_spixels, n_iter=3, temperature=1.0):
        super().__init__()
        self.num_spixels = num_spixels
        self.n_iter = n_iter
        self.temperature = temperature

    def initial_centers(self, features, xy):
        B, C, H, W = features.shape
        s = int(self.num_spixels**0.5)
        s = max(1, s)
        feat_pooled = F.adaptive_avg_pool2d(features, (s, s)).view(B, C, -1)
        xy_pooled = F.adaptive_avg_pool2d(xy, (s, s)).view(B, 2, -1)
        return torch.cat([feat_pooled, xy_pooled], dim=1)

    def forward(self, features):
        B, C, H, W = features.shape
        device = features.device
        xy = get_xy_grid(H, W, device).repeat(B, 1, 1, 1)
        pixel_data = torch.cat([features, xy], dim=1)
        pixel_flat = pixel_data.view(B, C+2, -1)
        
        centers = self.initial_centers(features, xy)
        Q = None
        for i in range(self.n_iter):
            p = pixel_flat.transpose(1, 2)
            c = centers.transpose(1, 2)
            dist = torch.cdist(p, c)
            Q = F.softmax(-dist * self.temperature, dim=-1)
            
            if i < self.n_iter - 1:
                Q_t = Q.transpose(1, 2)
                normalization = torch.sum(Q_t, dim=2, keepdim=True) + 1e-6
                centers_new = torch.bmm(Q_t, p) / normalization
                centers = centers_new.transpose(1, 2)
                
        return Q, centers[:, :C, :]

class HighFreqExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        high_freq = x - self.pool(x)
        return self.conv(high_freq)

class AmplitudeRegulator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale_map = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        self.conv_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        mask = self.scale_map(x)
        x_regulated = x * (0.5 + 0.5 * mask) 
        return self.conv_out(x_regulated)

class NodeFusionGAT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=2, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, nodes_ir, nodes_vi):
        x_ir = nodes_ir.transpose(1, 2)
        x_vi = nodes_vi.transpose(1, 2)
        x_concat = torch.cat([x_ir, x_vi], dim=1)
        
        attn_out, _ = self.cross_attn(x_concat, x_concat, x_concat)
        x_fused = self.norm1(x_concat + attn_out)
        x_fused = self.norm2(x_fused + self.ffn(x_fused))
        
        k = nodes_ir.shape[2]
        out_ir, out_vi = torch.split(x_fused, k, dim=1)
        return (out_ir + out_vi).transpose(1, 2) / 2.0

class SingleScaleFusion(nn.Module):
    def __init__(self, dim, num_spixels):
        super().__init__()
        self.ssn = DifferentiableSSN(num_spixels)
        self.gat = NodeFusionGAT(dim)
        
        self.local_cnn = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        self.hf_extractor = HighFreqExtractor(dim)
        
        self.regulator = AmplitudeRegulator(dim)
        
        self.fusion_conv = nn.Conv2d(dim * 2, dim, 3, 1, 1)

    def forward(self, feat_ir, feat_vi):
        B, C, H, W = feat_ir.shape
        
        # 1. Local Base Features
        local_feat = self.local_cnn(torch.cat([feat_ir, feat_vi], dim=1))
        
        # 2. Global Graph Features
        feat_mean = (feat_ir + feat_vi) / 2.0
        Q, _ = self.ssn(feat_mean)
        
        flat_ir = feat_ir.view(B, C, -1)
        flat_vi = feat_vi.view(B, C, -1)
        Q_t = Q.transpose(1, 2)
        norm = Q_t.sum(dim=2, keepdim=True) + 1e-6
        
        nodes_ir = torch.bmm(Q_t, flat_ir.transpose(1, 2)) / norm
        nodes_vi = torch.bmm(Q_t, flat_vi.transpose(1, 2)) / norm
        
        nodes_fused = self.gat(nodes_ir.transpose(1, 2), nodes_vi.transpose(1, 2))
        
        global_feat = torch.bmm(Q, nodes_fused.transpose(1, 2))
        global_feat = global_feat.transpose(1, 2).view(B, C, H, W)
        
        base_fused = self.fusion_conv(torch.cat([local_feat, global_feat], dim=1))
        
        hf_ir = self.hf_extractor(feat_ir)
        hf_vi = self.hf_extractor(feat_vi)
        hf_max = torch.max(hf_ir, hf_vi) 
        
        features_with_detail = base_fused + torch.tanh(hf_max)
        out = self.regulator(features_with_detail)
        
        return out

class MS_GAT_Fusion(nn.Module):
    def __init__(self, dim=64, scales=[64, 128, 256]): 
        super().__init__()
        self.scales = scales
        
        self.branches = nn.ModuleList([
            SingleScaleFusion(dim, s) for s in scales
        ])
        
        self.gate_net = nn.Sequential(
            nn.Conv2d(dim * len(scales), dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(dim // 2, len(scales), kernel_size=1),
            nn.Softmax(dim=1) 
        )
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
            nn.BatchNorm2d(dim)
        )
        
        self.final_conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, feat_ir, feat_vi):
        scale_outs = []
        for branch in self.branches:
            scale_outs.append(branch(feat_ir, feat_vi))
        
        cat_feats = torch.cat(scale_outs, dim=1)
        weights = self.gate_net(cat_feats)
        
        moe_out = 0
        for i, branch_out in enumerate(scale_outs):
            w = weights[:, i:i+1, :, :] 
            moe_out += w * branch_out
            
        base_out = self.shortcut(torch.cat([feat_ir, feat_vi], dim=1))
        
        out = base_out + moe_out
        
        return self.final_conv(out)