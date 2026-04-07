import torch
import torch.nn as nn
import torch.nn.functional as F

class OSRA(nn.Module):
    def __init__(self, in_channels=256, num_parts=5, init_margin=0.15):
        super(OSRA, self).__init__()
        self.in_channels = in_channels
        self.num_parts = num_parts
        
        # 1. 可学习的部位查询向量 (Part Queries)
        self.part_queries = nn.Parameter(torch.randn(1, num_parts, in_channels))
        nn.init.normal_(self.part_queries, std=0.02)
        
        # 2. 可学习的温度参数 (初始化为 0.1)
        self.temperature = nn.Parameter(torch.tensor([0.1]))
        
        # 3. 可学习的正交容忍度 (Learnable Margin)
        self.margin = nn.Parameter(torch.tensor([init_margin]))
        
        # 4. 空间反哺的 Cross-Attention 组件
        # [采纳建议]: 在 q_proj 中增加 BatchNorm2d，提升空间特征映射的稳定性
        self.q_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 5. 残差门控 (Zero-initialization)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, supp_feat, supp_mask):
        """
        输入: 
            supp_feat: [B, C, H, W]
            supp_mask: [B, 1, H, W] (0-1 mask)
        输出: 
            enhanced_supp: [B, C, H, W]
            div_loss: 标量 tensor, 软正交损失
            part_tokens: [B, K, C] 提取的部位特征，供 CADH 蒸馏使用
        """
        B, C, H, W = supp_feat.shape
        flat_mask = supp_mask.view(B, -1) 
        
        # ==========================================
        # 【健壮性补丁】防爆机制：处理无前景的极端情况
        # ==========================================
        if flat_mask.sum() == 0:
            zero_loss = torch.tensor(0.0, device=supp_feat.device, requires_grad=True)
            zero_tokens = torch.zeros(B, self.num_parts, C, device=supp_feat.device)
            return supp_feat, zero_loss, zero_tokens
            
        # ==========================================
        # 阶段一：基于可学习 Query 的 Part Token 提取
        # ==========================================
        flat_feat = supp_feat.view(B, C, -1).transpose(1, 2) # [B, H*W, C]
        
        queries_norm = F.normalize(self.part_queries.expand(B, -1, -1), p=2, dim=-1) # [B, K, C]
        flat_feat_norm = F.normalize(flat_feat, p=2, dim=-1) # [B, H*W, C]
        
        temp = torch.clamp(self.temperature, min=0.01)
        sim_matrix = torch.bmm(queries_norm, flat_feat_norm.transpose(1, 2)) / temp # [B, K, H*W]
        
        bg_mask = (1.0 - flat_mask).bool() # True 表示背景
        sim_matrix = sim_matrix.masked_fill(bg_mask.unsqueeze(1), -float('inf'))
        
        attn_weights = F.softmax(sim_matrix, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0) 
        
        part_tokens = torch.bmm(attn_weights, flat_feat) # [B, K, C]
        
        # ==========================================
        # 阶段二：计算自适应软正交多样性损失 (Adaptive Soft Orthogonal Loss)
        # ==========================================
        norm_tokens = F.normalize(part_tokens, p=2, dim=-1) # [B, K, C]
        cos_sim = torch.bmm(norm_tokens, norm_tokens.transpose(1, 2)) # [B, K, K]
        
        diag_mask = 1.0 - torch.eye(self.num_parts, device=supp_feat.device).unsqueeze(0) # [1, K, K]
        cos_sim = cos_sim * diag_mask
        
        adaptive_margin = torch.clamp(self.margin, min=0.0, max=0.5)
        div_loss = torch.clamp(cos_sim - adaptive_margin, min=0.0).sum() / (B * self.num_parts * (self.num_parts - 1) + 1e-5)
        
        # ==========================================
        # 阶段三：Cross-Attention 空间反哺与残差融合
        # ==========================================
        Q = self.q_proj(supp_feat).view(B, C, -1).transpose(1, 2) # [B, H*W, C]
        K = self.k_proj(part_tokens) # [B, K, C]
        V = self.v_proj(part_tokens) # [B, K, C]
        
        refine_attn = torch.bmm(Q, K.transpose(1, 2)) / (C ** 0.5) # [B, H*W, K]
        refine_attn = F.softmax(refine_attn, dim=-1)
        
        attn_out = torch.bmm(refine_attn, V) # [B, H*W, C]
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W) # [B, C, H, W]
        attn_out = self.out_proj(attn_out)
        
        enhanced_supp = supp_feat + self.gamma * attn_out
        
        # 【接口升级】：完整返回三大要素
        return enhanced_supp, div_loss, part_tokens