import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class GroundingModel(nn.Module):
    def __init__(self, region_dim=2048, text_dim=1024, num_heads=8, temperature=1.0):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained("roberta-large")
        self.region_proj = nn.Linear(region_dim, text_dim)
        self.cross_attn = nn.MultiheadAttention(text_dim, num_heads, batch_first=True)
        self.region_cls = nn.Linear(text_dim, 1)
        self.sit_gate = nn.Sequential(
            nn.Linear(text_dim, text_dim), nn.ReLU(), nn.Linear(text_dim, 1)
        )
        self.temperature = temperature
        self.anchor_scorer = nn.Sequential(
            nn.Linear(text_dim * 3, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, 1)
        )

    def forward(self, input_ids, attn_masks, region_feats):
        B, E, L = input_ids.shape
        reg_proj = self.region_proj(region_feats)

        # Text encoding
        txt = self.text_encoder(
            input_ids=input_ids.view(B*E, L),
            attention_mask=attn_masks.view(B*E, L)
        )
        txt_enc = txt.last_hidden_state.view(B, E, L, -1)
        txt_pooled = (txt_enc * attn_masks.unsqueeze(-1)).sum(dim=2) / \
                    attn_masks.sum(dim=2, keepdim=True).clamp(min=1)

        logits_list = []
        for e in range(E):
            out_r2t, _ = self.cross_attn(
                reg_proj, txt_enc[:, e], txt_enc[:, e]
            )  # (B, R, D)

            out_t2r, _ = self.cross_attn(
                txt_enc[:, e], reg_proj, reg_proj
            )  # (B, L, D)

            out_t2r_pooled = out_t2r.mean(dim=1).unsqueeze(1).expand(-1, reg_proj.size(1), -1)

            out = out_r2t + out_t2r_pooled 

            logits_list.append(self.region_cls(out).squeeze(-1))

        logits = torch.stack(logits_list, dim=1)  # (B, E, R)

        # Joint softmax
        logits_flat = logits.view(B, -1) / self.temperature
        probs_flat = torch.softmax(logits_flat, dim=1)
        probs = probs_flat.view(B, E, -1)

        gate = torch.sigmoid(self.sit_gate(txt_pooled)).squeeze(-1)
        return logits, probs, reg_proj, txt_pooled, gate


# ---------------- Losses ----------------
def joint_softmax_loss(probs, labels, gate=None):
    B, E, R = probs.shape
    losses = []
    for b in range(B):
        pos_mask = labels[b] > 0
        if pos_mask.sum() == 0:
            continue
        pos_prob_sum = probs[b][pos_mask].sum()
        sample_loss = -torch.log(pos_prob_sum + 1e-8)
        if gate is not None:
            sample_loss = gate[b].mean() * sample_loss
        losses.append(sample_loss)
    if len(losses) == 0:
        return torch.tensor(0.0, device=probs.device, requires_grad=True)
    return torch.stack(losses).mean()

def select_anchor_per_sample_with_mlp(model, reg_feats, txt_pool, labels):
    """
    MLP-based Anchor Selection
    - score = f([txt; region; txt*region])
    """
    B, E, D = txt_pool.shape
    anchors = []

    for b in range(B):
        pos_mask = labels[b] > 0  # (E, R)

        best_score, best_e, best_r = -1e9, None, None
        for e in range(E):
            pos_idx = pos_mask[e].nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue

            txt_vec = txt_pool[b, e].unsqueeze(0).expand(len(pos_idx), -1)  # (num_pos, D)
            reg_vec = reg_feats[b, pos_idx]                                 # (num_pos, D)

            pair_feat = torch.cat([txt_vec, reg_vec, txt_vec * reg_vec], dim=-1)  # (num_pos, 3D)

            scores = model.anchor_scorer(pair_feat).squeeze(-1)  # (num_pos,)

            max_score, idx = scores.max(dim=0)
            if max_score > best_score:
                best_score = max_score.item()
                best_e = e
                best_r = pos_idx[idx].item()

        anchors.append((best_e, best_r))

    return anchors


def anchor_exclusion_loss_shared_feat(probs, reg_feats, anchors, margin=0.5):
    """
    Distance-aware exclusion 
    """
    B, E, R, D = reg_feats.shape[0], probs.shape[1], probs.shape[2], reg_feats.shape[2]
    losses = []

    for b in range(B):
        anchor_e, anchor_r = anchors[b]
        if anchor_e is None:
            continue

        anchor_feat = reg_feats[b, anchor_r]  # (D,)
        sims = F.cosine_similarity(anchor_feat.unsqueeze(0), reg_feats[b], dim=1)  # (R,)

        for e2 in range(E):
            if e2 == anchor_e:
                continue
            confs = probs[b, e2]  # (R,)
            losses.append((sims * F.relu(confs - margin)).mean())

    if not losses:
        return torch.tensor(0.0, device=probs.device, requires_grad=True)
    return torch.stack(losses).mean()

def region_exclusion_loss(logits, labels, margin=0.5, top_k=5):
    B, E, R = logits.shape
    losses = []
    for b in range(B):
        for e in range(E):
            pos_idx = (labels[b, e] > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            for anchor in pos_idx:
                anchor_score = logits[b, e, anchor]
                neg_idx = (labels[b, e] == 0).nonzero(as_tuple=True)[0]
                if len(neg_idx) == 0:
                    continue
                diffs = anchor_score - logits[b, e, neg_idx]
                hard_idx = torch.topk(-diffs, k=min(top_k, len(diffs))).indices
                hard_neg = neg_idx[hard_idx]
                for r in hard_neg:
                    losses.append(F.relu(margin - (anchor_score - logits[b, e, r])))
    if len(losses) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return torch.stack(losses).mean()
def anchor_consistency_loss_shared_adaptive(
    reg_feats, txt_pool, ent_types, labels, anchors, alpha=0.7, beta=0.5
):
    """
    Adaptive Consistency Loss (variance-based)
    - 동일 타입 엔티티 grounding 간 일관성 유지
    - type별 시각적 다양성(variance)에 따라 가중치 조절
    """
    B, E, D = txt_pool.shape
    losses = []

    for b in range(B):
        anchor_e, anchor_r = anchors[b]
        if anchor_e is None:
            continue
        anchor_feat = reg_feats[b, anchor_r]

        # 🔹 (1) batch 내 type별 region feature variance 계산
        type_var = {}
        for e in range(E):
            t = ent_types[b][e]
            feats = reg_feats[b]  # (R, D)
            # 해당 엔티티의 positive region features만 선택
            pos_idx = (labels[b, e] > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            type_var.setdefault(t, []).append(reg_feats[b, pos_idx].mean(dim=0))

        # 평균적 type variance
        for t, feat_list in type_var.items():
            stack = torch.stack(feat_list)
            var = torch.var(F.normalize(stack, dim=-1))
            # exp(-var / beta): variance 작을수록 weight ↑
            type_var[t] = torch.exp(-var / beta)

        # 🔹 (2) consistency loss 계산 (type별 weight 반영)
        for e2 in range(E):
            if e2 == anchor_e:
                continue
            t = ent_types[b][e2]
            if ent_types[b][anchor_e] != t:
                continue
            w = type_var.get(t, torch.tensor(1.0, device=reg_feats.device))

            sims = F.cosine_similarity(anchor_feat.unsqueeze(0), reg_feats[b], dim=1)
            sims[anchor_r] = -1.0
            best_idx = sims.argmax().item()

            target_feat = reg_feats[b, best_idx]
            sim = F.cosine_similarity(
                anchor_feat.unsqueeze(0), target_feat.unsqueeze(0), dim=1
            ).squeeze()

            losses.append(w * F.relu(alpha - sim))

    if len(losses) == 0:
        return torch.tensor(0.0, device=reg_feats.device, requires_grad=True)
    return torch.stack(losses).mean()


# def anchor_consistency_loss_with_prototype(
#     reg_feats, txt_pool, ent_types, labels, anchors, alpha=0.7
# ):
#     """
#     reg_feats: (B, R, D)
#     txt_pool:  (B, E, D)
#     ent_types: (B, E)
#     labels:    (B, E, R)
#     anchors:   list of (anchor_e, anchor_r)
#     """

#     B, E, D = txt_pool.shape
#     device = reg_feats.device
#     losses = []

#     for b in range(B):
#         anchor_e, anchor_r = anchors[b]
#         if anchor_e is None:
#             continue

#         anchor_type = ent_types[b][anchor_e]
#         anchor_feat = reg_feats[b, anchor_r]  # (D,)

#         # 🔥 prototype용 feature 수집
#         proto_feats = []

#         for b2 in range(B):
#             if b2 == b:
#                 continue

#             for e2 in range(E):
#                 if ent_types[b2][e2] != anchor_type:
#                     continue

#                 pos_idx = (labels[b2, e2] > 0).nonzero(as_tuple=True)[0]
#                 if len(pos_idx) == 0:
#                     continue

#                 # 해당 엔티티의 대표 feature
#                 ent_feat = reg_feats[b2, pos_idx].mean(dim=0)
#                 proto_feats.append(ent_feat)

#         if len(proto_feats) == 0:
#             continue

#         # 🔥 type prototype
#         prototype = torch.stack(proto_feats).mean(dim=0)  # (D,)

#         sim = F.cosine_similarity(
#             anchor_feat.unsqueeze(0),
#             prototype.unsqueeze(0)
#         )

#         losses.append(F.relu(alpha - sim))

#     if not losses:
#         return torch.tensor(0.0, device=device)

#     return torch.stack(losses).mean()

def anchor_consistency_loss_cross_sample(
    reg_feats, txt_pool, ent_types, labels, anchors, alpha=0.7
):
    B, E, D = txt_pool.shape
    losses = []

    # 🔥 anchor는 sample b에서 선택됨
    for b in range(B):
        anchor_e, anchor_r = anchors[b]
        if anchor_e is None:
            continue
        
        anchor_type = ent_types[b][anchor_e]
        anchor_feat = reg_feats[b, anchor_r]  # (D,)

        for b2 in range(B):
            if b2 == b:
                continue  # 중요!!

            for e2 in range(E):
                if ent_types[b2][e2] != anchor_type:
                    continue

                pos_idx = (labels[b2, e2] > 0).nonzero(as_tuple=True)[0]
                if len(pos_idx) == 0:
                    continue

                tgt_feat = reg_feats[b2, pos_idx].mean(dim=0)

                sim = F.cosine_similarity(anchor_feat.unsqueeze(0), tgt_feat.unsqueeze(0))

                losses.append(F.relu(alpha - sim))

    if not losses:
        return torch.tensor(0.0, device=reg_feats.device)
    return torch.stack(losses).mean()

# ---------------- Train / Eval ----------------
def train_one_epoch(model, device, loader, opt):
    model.train()
    total_loss = 0
    for batch in loader:
        tb = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits, probs, reg_proj, txt_pool, gate = model(tb["input_ids"], tb["attn_masks"], tb["region_feats"])

        anchors = select_anchor_per_sample_with_mlp(model, reg_proj, txt_pool, tb["labels"])
        loss_ce   = joint_softmax_loss(probs, tb["labels"], gate=gate)

        # loss_cons = anchor_consistency_loss_shared_adaptive(
        #     reg_proj, txt_pool, tb["ent_types"], tb["labels"], anchors
        # )
        loss_cons = anchor_consistency_loss_cross_sample(reg_proj, txt_pool, tb["ent_types"], tb["labels"], anchors)
        # loss_cons = anchor_consistency_loss_with_prototype(reg_proj, txt_pool, tb["ent_types"], tb["labels"], anchors)
        loss_excl = anchor_exclusion_loss_shared_feat(probs, reg_proj, anchors)
        loss_region = region_exclusion_loss(logits, tb["labels"], margin=0.5)

        loss = (
            1.0 * loss_ce +
            0.3 * loss_cons +
            0.5 * loss_excl + 
            1.0 * loss_region
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()

    return total_loss / len(loader)
