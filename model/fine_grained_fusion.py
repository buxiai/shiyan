import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

class FineGrainedAttrFusion(nn.Module):
    def __init__(self, attr_dict, prompt_template="A photo of a {}", visual_dim=256, text_dim=512, alpha=0.22):
        """
        attr_dict: dict, 外部传入的类别属性字典 (如 {'airplane': ['wing', 'tail']})
        prompt_template: str, 提示词模板。
                         PASCAL 用 "A photo of a {}"
                         iSAID 用 "An aerial view of a {}"
        """
        super(FineGrainedAttrFusion, self).__init__()
        self.alpha = alpha
        self.attr_dict = attr_dict
        self.prompt_template = prompt_template
        
        # 1. 加载冻结的轻量级 CLIP
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # 2. 视觉特征映射层
        self.vis_proj = nn.Conv2d(visual_dim, text_dim, kernel_size=1)
        
        # 3. 特征融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(visual_dim + text_dim, visual_dim, kernel_size=1),
            nn.BatchNorm2d(visual_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, supp_feat, supp_mask, class_names):
        B, C, H, W = supp_feat.shape
        device = supp_feat.device
        
        proj_supp_feat = self.vis_proj(supp_feat)
        proj_supp_feat_norm = F.normalize(proj_supp_feat, p=2, dim=1)
        
        batch_fused_text_feats = []

        for b in range(B):
            c_name = class_names[b]
            # 获取当前类别的细粒度属性，如果没有则退化为类名本身
            attributes = self.attr_dict.get(c_name, [c_name])
            valid_text_feats = []
            
            # --- A. 动态套用 Prompt 模板 ---
            prompts = [self.prompt_template.format(attr) for attr in attributes]
            
            with torch.no_grad():
                inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(device)
                text_outputs = self.text_encoder(**inputs)
                text_feats = text_outputs.pooler_output
                text_feats_norm = F.normalize(text_feats, p=2, dim=1)

            # --- B. 计算余弦相似度并动态过滤 ---
            mask_b = supp_mask[b].view(-1)
            feat_b = proj_supp_feat_norm[b].view(proj_supp_feat_norm.shape[1], -1)
            sim_map = torch.mm(text_feats_norm, feat_b)

            for i in range(len(attributes)):
                fg_sim = sim_map[i][mask_b == 1]
                max_sim = fg_sim.max() if len(fg_sim) > 0 else torch.tensor(0.0).to(device)
                
                if max_sim >= self.alpha:
                    valid_text_feats.append(text_feats[i])
            
            # --- C. 聚合有效特征 ---
            if len(valid_text_feats) > 0:
                agg_text_feat = torch.stack(valid_text_feats).mean(dim=0)
            else:
                agg_text_feat = torch.zeros_like(text_feats[0])
            
            agg_text_spatial = agg_text_feat.view(text_dim, 1, 1).expand(text_dim, H, W)
            batch_fused_text_feats.append(agg_text_spatial)

        batch_text_spatial = torch.stack(batch_fused_text_feats).to(device)
        concat_feat = torch.cat([supp_feat, batch_text_spatial], dim=1)
        out_feat = self.fusion_conv(concat_feat)
        
        return out_feat