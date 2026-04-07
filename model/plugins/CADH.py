import torch
import torch.nn as nn
import torch.nn.functional as F

class CADH(nn.Module):
    def __init__(self, stu_channels=256, tea_channels=512, token_channels=256, embed_dim=256, 
                 lambda_bg=0.5, bg_margin=0.1, stu_temp=0.1, tea_temp=0.2):
        """
        CADH: Teacher-Guided Decoupled Distillation (教师指导的解耦蒸馏)
        
        参数:
        - stu_channels: 中层 Student 特征的维度 (如 256)
        - tea_channels: 深层 Teacher 特征的维度 (如 ResNet layer4 的 512 或 2048)
        - token_channels: 来自 OSRA 的 part_tokens 的维度
        - embed_dim: 统一映射的对比空间维度
        - lambda_bg: 背景排斥损失的权重
        - bg_margin: 允许 Student 的适度探索边界 (Teacher-Bounded Margin)
        - stu_temp, tea_temp: 解耦温度，tea_temp 提供适度平滑的监督信号
        """
        super(CADH, self).__init__()
        self.lambda_bg = lambda_bg
        self.bg_margin = bg_margin
        self.stu_temp = stu_temp
        self.tea_temp = tea_temp
        
        # 降维投影：将 Student, Teacher 和 Part Tokens 映射到统一的语义空间
        self.stu_proj = nn.Sequential(
            nn.Conv2d(stu_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.tea_proj = nn.Sequential(
            nn.Conv2d(tea_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.token_proj = nn.Linear(token_channels, embed_dim)

    def forward(self, student_feat, teacher_feat, part_tokens, query_mask_gt):
        """
        输入:
            student_feat: [B, C_stu, H_s, W_s] (中层特征)
            teacher_feat: [B, C_tea, H_t, W_t] (深层特征)
            part_tokens: [B, K, C_tok] (来自 OSRA 的正锚点)
            query_mask_gt: [B, 1, H, W] (Query 的真实标签 Mask)
        输出:
            cadh_loss: 标量 tensor, 包含前景部位分配蒸馏与背景排斥蒸馏
        """
        B, _, H_s, W_s = student_feat.shape
        
        # 【极其关键的补丁】：切断梯度回传，保护 OSRA 的部位正交性！
        part_tokens = part_tokens.detach()
        
        # 1. 空间与 Mask 健壮对齐
        if teacher_feat.shape[-2:] != (H_s, W_s):
            teacher_feat = F.interpolate(teacher_feat, size=(H_s, W_s), mode='bilinear', align_corners=True)
            
        flat_mask = F.interpolate(query_mask_gt.float(), size=(H_s, W_s), mode='nearest').view(B, H_s * W_s)
        fg_mask = (flat_mask == 1).float()
        bg_mask = (flat_mask == 0).float()
        
        # 2. 映射与 L2 归一化 (防梯度爆炸)
        stu_embed = self.stu_proj(student_feat).view(B, -1, H_s * W_s).transpose(1, 2)
        stu_norm = F.normalize(stu_embed, p=2, dim=-1)
        
        tea_embed = self.tea_proj(teacher_feat).view(B, -1, H_s * W_s).transpose(1, 2)
        tea_norm = F.normalize(tea_embed, p=2, dim=-1)
        
        tok_embed = self.token_proj(part_tokens)
        tok_norm = F.normalize(tok_embed, p=2, dim=-1)
        
        # 3. 相似度计算 (解耦温度)
        sim_stu = torch.bmm(stu_norm, tok_norm.transpose(1, 2)) / self.stu_temp
        with torch.no_grad():
            sim_tea = torch.bmm(tea_norm, tok_norm.transpose(1, 2)) / self.tea_temp
            
        # ==========================================
        # 核心一：Foreground Part-Assignment KL Distillation
        # 加入 Teacher Confidence Threshold (教师置信度门控)
        # ==========================================
        log_prob_stu = F.log_softmax(sim_stu, dim=-1)
        prob_tea = F.softmax(sim_tea, dim=-1)
        
        # Teacher 置信度掩码：只在 Teacher 确信度 > 0.5 的像素上进行蒸馏
        tea_conf = prob_tea.max(dim=-1)[0] # [B, HW]
        conf_mask = (tea_conf > 0.5).float()
        
        kl_div = F.kl_div(log_prob_stu, prob_tea, reduction='none').sum(dim=-1)
        
        # 只有真实前景 且 Teacher 足够自信 的像素才计算 Loss
        valid_fg_mask = fg_mask * conf_mask
        loss_fg = (kl_div * valid_fg_mask).sum() / (valid_fg_mask.sum() + 1e-5)
        
        # ==========================================
        # 核心二：Teacher-Bounded Background Repulsion 
        # 加入 clamp 保护防止负值过度惩罚
        # ==========================================
        max_sim_stu = sim_stu.max(dim=-1)[0]
        # 限制 Teacher 的底线为 0，防止负值相似度导致过度惩罚
        max_sim_tea = sim_tea.max(dim=-1)[0].clamp(min=0.0) 
        
        # 边界自适应惩罚：只在 Student 越界时施加惩罚
        bg_penalty = F.relu(max_sim_stu - max_sim_tea - self.bg_margin) 
        loss_bg = (bg_penalty * bg_mask).sum() / (bg_mask.sum() + 1e-5)
        
        return loss_fg + self.lambda_bg * loss_bg