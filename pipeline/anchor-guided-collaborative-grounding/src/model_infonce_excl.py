import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class GroundingModel(nn.Module):
    def __init__(self, region_dim=2048, text_dim=None, num_heads=8, temperature=1.0, model_name="roberta-large"):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(model_name)
        text_dim = int(text_dim or self.text_encoder.config.hidden_size)
        self.region_proj = nn.Linear(region_dim, text_dim)
        self.cross_attn = nn.MultiheadAttention(text_dim, num_heads, batch_first=True)
        self.region_cls = nn.Linear(text_dim, 1)
        self.sit_gate = nn.Sequential(
            nn.Linear(text_dim, text_dim), nn.ReLU(), nn.Linear(text_dim, 1)
        )
        self.temperature = temperature
        # Force sim-head probabilities for stable prediction behavior.
        self.use_sim_head_for_probs = True

    def forward(self, input_ids, attn_masks, region_feats, **kwargs):
        del kwargs
        bsz, ent_n, seq_len = input_ids.shape
        reg_proj = self.region_proj(region_feats)

        txt = self.text_encoder(
            input_ids=input_ids.view(bsz * ent_n, seq_len),
            attention_mask=attn_masks.view(bsz * ent_n, seq_len),
        )
        txt_enc = txt.last_hidden_state.view(bsz, ent_n, seq_len, -1)
        txt_pooled = (txt_enc * attn_masks.unsqueeze(-1)).sum(dim=2) / attn_masks.sum(dim=2, keepdim=True).clamp(min=1)

        logits_list = []
        for e in range(ent_n):
            out_r2t, _ = self.cross_attn(reg_proj, txt_enc[:, e], txt_enc[:, e])
            out_t2r, _ = self.cross_attn(txt_enc[:, e], reg_proj, reg_proj)
            out_t2r_pooled = out_t2r.mean(dim=1).unsqueeze(1).expand(-1, reg_proj.size(1), -1)
            out = out_r2t + out_t2r_pooled
            logits_list.append(self.region_cls(out).squeeze(-1))

        logits = torch.stack(logits_list, dim=1)  # [B, E, R]

        if self.use_sim_head_for_probs:
            sim_logits = torch.einsum(
                "bed,brd->ber",
                F.normalize(txt_pooled, dim=-1),
                F.normalize(reg_proj, dim=-1),
            )
            logits_for_probs = sim_logits
        else:
            logits_for_probs = logits

        probs = torch.softmax(logits_for_probs.view(bsz, -1) / self.temperature, dim=1).view(bsz, ent_n, -1)
        gate = torch.sigmoid(self.sit_gate(txt_pooled)).squeeze(-1)
        return logits, probs, reg_proj, txt_pooled, gate


def anchor_exclusion_loss(
    probs,
    reg_feats,
    margin=0.3,
    mode="sim_prob",
    variant="pairwise",
    labels=None,
    txt_pool=None,
    ent_types=None,
    region_boxes=None,
    pair_alpha=0.5,
    pair_beta=0.3,
    pair_gamma=None,
    pair_region_sim="iou",
    pair_type_mode="neutral",
    pair_type_factor=1.5,
    pair_name_thr=0.5,
    pair_region_thr=0.5,
    pair_easy_scale=0.5,
    pair_hard_scale=1.5,
):
    bsz, ent_n = probs.shape[:2]
    losses = []
    mode_l = str(mode).lower().strip()
    variant_l = str(variant).lower().strip()

    if variant_l != "pairwise":
        return torch.tensor(0.0, device=probs.device, requires_grad=True)
    if labels is None or txt_pool is None or ent_types is None:
        return torch.tensor(0.0, device=probs.device, requires_grad=True)

    alpha = max(0.0, float(pair_alpha))
    beta = max(0.0, float(pair_beta))
    if pair_gamma is None:
        gamma = max(0.0, 1.0 - alpha - beta)
    else:
        gamma = max(0.0, float(pair_gamma))
        s = alpha + beta + gamma
        if s > 0:
            alpha, beta, gamma = alpha / s, beta / s, gamma / s
        else:
            alpha, beta, gamma = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    # Keep argument for backward compatibility, but region similarity is fixed to IoU.
    _ = pair_region_sim
    pair_type_mode_l = str(pair_type_mode).lower().strip()
    type_factor = max(0.0, float(pair_type_factor))
    name_thr = min(1.0, max(0.0, float(pair_name_thr)))
    region_thr = min(1.0, max(0.0, float(pair_region_thr)))
    easy_scale = max(0.0, float(pair_easy_scale))
    hard_scale = max(0.0, float(pair_hard_scale))

    def _pairwise_iou_sim(boxes, pos_idx_a, pos_idx_b):
        if boxes is None or len(pos_idx_a) == 0 or len(pos_idx_b) == 0:
            return torch.tensor(0.0, device=probs.device)
        a = boxes[pos_idx_a]
        b = boxes[pos_idx_b]
        xx1 = torch.maximum(a[:, None, 0], b[None, :, 0])
        yy1 = torch.maximum(a[:, None, 1], b[None, :, 1])
        xx2 = torch.minimum(a[:, None, 2], b[None, :, 2])
        yy2 = torch.minimum(a[:, None, 3], b[None, :, 3])
        inter = torch.clamp(xx2 - xx1, min=0.0) * torch.clamp(yy2 - yy1, min=0.0)
        area_a = torch.clamp(a[:, 2] - a[:, 0], min=0.0) * torch.clamp(a[:, 3] - a[:, 1], min=0.0)
        area_b = torch.clamp(b[:, 2] - b[:, 0], min=0.0) * torch.clamp(b[:, 3] - b[:, 1], min=0.0)
        iou = inter / (area_a[:, None] + area_b[None, :] - inter + 1e-8)
        return iou.max()

    for b in range(bsz):
        prototypes, pos_indices, valid = [], [], []
        for e in range(ent_n):
            pos_idx = (labels[b, e] > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                prototypes.append(None)
                pos_indices.append(None)
                valid.append(False)
                continue
            prototypes.append(reg_feats[b, pos_idx].mean(dim=0))
            pos_indices.append(pos_idx)
            valid.append(True)
        if sum(valid) <= 1:
            continue

        for e in range(ent_n):
            if not valid[e]:
                continue
            proto_e = prototypes[e]
            txt_e = txt_pool[b, e]
            type_e = ent_types[b][e]
            sims_region = F.cosine_similarity(proto_e.unsqueeze(0), reg_feats[b], dim=1)
            for e2 in range(ent_n):
                if e2 == e or (not valid[e2]):
                    continue
                txt_e2 = txt_pool[b, e2]
                type_e2 = ent_types[b][e2]
                name_sim = F.cosine_similarity(txt_e.unsqueeze(0), txt_e2.unsqueeze(0)).squeeze(0)
                name_sim = (name_sim + 1.0) * 0.5
                type_sim = 1.0 if (type_e != "" and type_e == type_e2) else 0.0
                if type_sim > 0.0:
                    if pair_type_mode_l == "sharp":
                        type_sim = type_sim * type_factor
                    elif pair_type_mode_l == "smooth":
                        type_sim = type_sim / max(type_factor, 1e-6)

                boxes_b = None if region_boxes is None else region_boxes[b]
                region_sim = _pairwise_iou_sim(boxes_b, pos_indices[e], pos_indices[e2])

                pair_dis = (alpha * (1.0 - name_sim)) + (beta * (1.0 - region_sim)) + (gamma * (1.0 - type_sim))
                similar_pair = bool((name_sim >= name_thr).item()) or bool((region_sim >= region_thr).item())
                hard_negative_pair = bool((name_sim < name_thr).item()) and bool((region_sim < region_thr).item())
                if similar_pair:
                    sep_scale = easy_scale
                elif hard_negative_pair:
                    sep_scale = hard_scale
                else:
                    sep_scale = 1.0

                confs = probs[b, e2]
                if mode_l == "sim_prob":
                    base = (sims_region * confs).mean()
                else:
                    base = (sims_region * F.relu(confs - margin)).mean()
                losses.append(pair_dis * sep_scale * base)

    if not losses:
        return torch.tensor(0.0, device=probs.device, requires_grad=True)
    return torch.stack(losses).mean()


def region_text_infonce_loss(reg_feats, txt_pool, labels, tau=0.07, topk_neg=0):
    reg_n = F.normalize(reg_feats, dim=-1)
    txt_n = F.normalize(txt_pool, dim=-1)
    scores = torch.einsum("bed,brd->ber", txt_n, reg_n)
    losses = []
    tau = max(1e-6, float(tau))
    k_neg = int(topk_neg)

    bsz, ent_n = txt_pool.shape[:2]
    for b in range(bsz):
        for e in range(ent_n):
            pos_idx = (labels[b, e] > 0).nonzero(as_tuple=True)[0]
            neg_idx = (labels[b, e] == 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            if k_neg > 0 and k_neg < len(neg_idx):
                neg_scores = scores[b, e, neg_idx]
                hard_local = torch.topk(neg_scores, k=k_neg).indices
                neg_idx = neg_idx[hard_local]

            pos_logits = scores[b, e, pos_idx] / tau
            neg_logits = scores[b, e, neg_idx] / tau
            all_logits = torch.cat([pos_logits, neg_logits], dim=0)
            losses.append(-(torch.logsumexp(pos_logits, dim=0) - torch.logsumexp(all_logits, dim=0)))

    if len(losses) == 0:
        return torch.tensor(0.0, device=reg_feats.device, requires_grad=True)
    return torch.stack(losses).mean()


def train_one_epoch(
    model,
    device,
    loader,
    opt,
    excl_w=0.6,
    info_nce_w=1.0,
    **kwargs,
):
    model.train()
    total_loss = 0.0
    total_excl = 0.0
    total_info_nce = 0.0

    for batch in loader:
        tb = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        _, probs, reg_proj, txt_pool, _ = model(tb["input_ids"], tb["attn_masks"], tb["region_feats"])

        loss_excl = anchor_exclusion_loss(
            probs,
            reg_proj,
            margin=float(kwargs.get("excl_margin", 0.3)),
            mode=str(kwargs.get("excl_mode", "sim_prob")),
            variant=str(kwargs.get("excl_variant", "pairwise")),
            labels=tb["labels"],
            txt_pool=txt_pool,
            ent_types=tb["ent_types"],
            region_boxes=tb.get("region_boxes", None),
            pair_alpha=float(kwargs.get("excl_pair_alpha", 0.5)),
            pair_beta=float(kwargs.get("excl_pair_beta", 0.3)),
            pair_gamma=(
                float(kwargs["excl_pair_gamma"])
                if kwargs.get("excl_pair_gamma") is not None
                else None
            ),
            pair_region_sim=str(kwargs.get("excl_pair_region_sim", "iou")),
            pair_type_mode=str(kwargs.get("excl_pair_type_mode", "neutral")),
            pair_type_factor=float(kwargs.get("excl_pair_type_factor", 1.5)),
            pair_name_thr=float(kwargs.get("excl_pair_name_thr", 0.5)),
            pair_region_thr=float(kwargs.get("excl_pair_region_thr", 0.5)),
            pair_easy_scale=float(kwargs.get("excl_pair_easy_scale", 0.5)),
            pair_hard_scale=float(kwargs.get("excl_pair_hard_scale", 1.5)),
        )
        loss_info_nce = region_text_infonce_loss(
            reg_proj,
            txt_pool,
            tb["labels"],
            tau=float(kwargs.get("info_nce_tau", 0.07)),
            topk_neg=int(kwargs.get("info_nce_topk_neg", 0)),
        )

        loss = float(excl_w) * loss_excl + float(info_nce_w) * loss_info_nce

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += float(loss.item())
        total_excl += float(loss_excl.item())
        total_info_nce += float(loss_info_nce.item())

    denom = max(1, len(loader))
    return {
        "total": total_loss / denom,
        "excl": total_excl / denom,
        "info_nce": total_info_nce / denom,
    }

