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
        self.anchor_scorer = nn.Sequential(
            nn.Linear(text_dim * 3, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, 1)
        )

    def forward(self, input_ids, attn_masks, region_feats, **kwargs):
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

        logits_flat = logits.view(B, -1) / self.temperature
        probs_flat = torch.softmax(logits_flat, dim=1)
        probs = probs_flat.view(B, E, -1)

        gate = torch.sigmoid(self.sit_gate(txt_pooled)).squeeze(-1)
        return logits, probs, reg_proj, txt_pooled, gate

# ---------------- Anchor Selection ----------------
def select_anchor_per_sample_with_mlp(model, reg_feats, txt_pool, labels, return_reliability=False, gap_temp=0.10):
    """
    MLP-based Anchor Selection
    - score = f([txt; region; txt*region])
    """
    B, E, D = txt_pool.shape
    anchors = []
    reliabilities = []

    for b in range(B):
        pos_mask = labels[b] > 0  # (E, R)

        best_score, best_e, best_r = -1e9, None, None
        cand_scores = []
        for e in range(E):
            pos_idx = pos_mask[e].nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue

            txt_vec = txt_pool[b, e].unsqueeze(0).expand(len(pos_idx), -1)  # (num_pos, D)
            reg_vec = reg_feats[b, pos_idx]                                 # (num_pos, D)

            pair_feat = torch.cat([txt_vec, reg_vec, txt_vec * reg_vec], dim=-1)  # (num_pos, 3D)

            scores = model.anchor_scorer(pair_feat).squeeze(-1)  # (num_pos,)
            cand_scores.append(scores)

            max_score, idx = scores.max(dim=0)
            if max_score > best_score:
                best_score = max_score.item()
                best_e = e
                best_r = pos_idx[idx].item()

        anchors.append((best_e, best_r))
        if return_reliability:
            if len(cand_scores) == 0:
                reliabilities.append(torch.tensor(0.0, device=reg_feats.device))
            else:
                all_scores = torch.cat(cand_scores, dim=0)
                if all_scores.numel() >= 2:
                    top2 = torch.topk(all_scores, k=2).values
                    gap = top2[0] - top2[1]
                else:
                    gap = torch.tensor(1.0, device=all_scores.device)
                t = max(1e-6, float(gap_temp))
                reliabilities.append(torch.sigmoid(gap / t))

    if return_reliability:
        if len(reliabilities) == 0:
            rel = torch.tensor([], device=reg_feats.device)
        else:
            rel = torch.stack(reliabilities, dim=0)
        return anchors, rel
    return anchors

# ---------------- Losses ----------------
def cross_entropy_loss(probs, labels, gate=None):
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


def anchor_exclusion_loss(
    probs,
    reg_feats,
    anchors,
    margin=0.5,
    mode="relu_margin",
    sample_weights=None,
    variant="anchor",
    labels=None,
    txt_pool=None,
    ent_types=None,
    region_boxes=None,
    ent_name_w=0.7,
    ent_type_w=0.3,
    pair_alpha=0.5,
    pair_beta=0.3,
    pair_region_sim="none",
    pair_type_mode="neutral",
    pair_type_factor=1.5,
    pair_name_thr=0.5,
    pair_region_thr=0.5,
    pair_easy_scale=0.5,
    pair_hard_scale=1.5,
):
    B, E, R, D = reg_feats.shape[0], probs.shape[1], probs.shape[2], reg_feats.shape[2]
    losses = []
    variant_l = str(variant).lower().strip()
    mode_l = str(mode).lower().strip()

    def _pairwise_iou_sim(boxes, pos_idx_a, pos_idx_b):
        if boxes is None or len(pos_idx_a) == 0 or len(pos_idx_b) == 0:
            return torch.tensor(0.0, device=probs.device)
        a = boxes[pos_idx_a]  # (Na, 4)
        b = boxes[pos_idx_b]  # (Nb, 4)
        if a.numel() == 0 or b.numel() == 0:
            return torch.tensor(0.0, device=probs.device)
        xx1 = torch.maximum(a[:, None, 0], b[None, :, 0])
        yy1 = torch.maximum(a[:, None, 1], b[None, :, 1])
        xx2 = torch.minimum(a[:, None, 2], b[None, :, 2])
        yy2 = torch.minimum(a[:, None, 3], b[None, :, 3])
        inter_w = torch.clamp(xx2 - xx1, min=0.0)
        inter_h = torch.clamp(yy2 - yy1, min=0.0)
        inter = inter_w * inter_h
        area_a = torch.clamp(a[:, 2] - a[:, 0], min=0.0) * torch.clamp(a[:, 3] - a[:, 1], min=0.0)
        area_b = torch.clamp(b[:, 2] - b[:, 0], min=0.0) * torch.clamp(b[:, 3] - b[:, 1], min=0.0)
        union = area_a[:, None] + area_b[None, :] - inter
        iou = inter / (union + 1e-8)
        return iou.max()

    # New variant: within-sample pairwise entity exclusion
    # - not anchor-only
    # - repulsion strength is proportional to entity pair similarity
    #   w_pair = alpha*s_name + beta*s_region + (1-alpha-beta)*s_type
    if variant_l == "pairwise":
        if labels is None or txt_pool is None or ent_types is None:
            return torch.tensor(0.0, device=probs.device, requires_grad=True)
        alpha = max(0.0, float(pair_alpha))
        beta = max(0.0, float(pair_beta))
        gamma = max(0.0, 1.0 - alpha - beta)
        pair_region_sim_l = str(pair_region_sim).lower().strip()
        pair_type_mode_l = str(pair_type_mode).lower().strip()
        type_factor = max(0.0, float(pair_type_factor))
        name_thr = min(1.0, max(0.0, float(pair_name_thr)))
        region_thr = min(1.0, max(0.0, float(pair_region_thr)))
        easy_scale = max(0.0, float(pair_easy_scale))
        hard_scale = max(0.0, float(pair_hard_scale))
        for b in range(B):
            sample_w = 1.0 if sample_weights is None else sample_weights[b]
            # Build per-entity positive region prototype
            prototypes = []
            pos_indices = []
            valid = []
            for e in range(E):
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

            for e in range(E):
                if not valid[e]:
                    continue
                proto_e = prototypes[e]
                txt_e = txt_pool[b, e]
                type_e = ent_types[b][e]
                sims_region = F.cosine_similarity(proto_e.unsqueeze(0), reg_feats[b], dim=1)  # (R,)
                for e2 in range(E):
                    if e2 == e or (not valid[e2]):
                        continue
                    txt_e2 = txt_pool[b, e2]
                    type_e2 = ent_types[b][e2]
                    name_sim = F.cosine_similarity(txt_e.unsqueeze(0), txt_e2.unsqueeze(0)).squeeze(0)
                    name_sim = (name_sim + 1.0) * 0.5  # [-1,1] -> [0,1]
                    type_sim = 1.0 if (type_e != "" and type_e == type_e2) else 0.0
                    if type_sim > 0.0:
                        if pair_type_mode_l == "sharp":
                            type_sim = type_sim * type_factor
                        elif pair_type_mode_l == "smooth":
                            type_sim = type_sim / max(type_factor, 1e-6)

                    if pair_region_sim_l == "feat":
                        region_sim = F.cosine_similarity(
                            proto_e.unsqueeze(0), prototypes[e2].unsqueeze(0)
                        ).squeeze(0)
                        region_sim = (region_sim + 1.0) * 0.5  # [-1,1] -> [0,1]
                    elif pair_region_sim_l == "iou":
                        boxes_b = None if region_boxes is None else region_boxes[b]
                        region_sim = _pairwise_iou_sim(boxes_b, pos_indices[e], pos_indices[e2])
                    else:
                        # "none" means unavailable signal; keep neutral instead of forcing dissimilar.
                        region_sim = torch.tensor(0.5, device=probs.device)

                    # Dissimilarity-driven exclusion:
                    # similar names/high IoU => weaker push, both low => stronger push.
                    name_dis = 1.0 - name_sim
                    region_dis = 1.0 - region_sim
                    type_dis = 1.0 - type_sim
                    pair_dis = (alpha * name_dis) + (beta * region_dis) + (gamma * type_dis)

                    similar_pair = bool((name_sim >= name_thr).item()) or bool((region_sim >= region_thr).item())
                    hard_negative_pair = bool((name_sim < name_thr).item()) and bool((region_sim < region_thr).item())
                    if similar_pair:
                        sep_scale = easy_scale
                    elif hard_negative_pair:
                        sep_scale = hard_scale
                    else:
                        sep_scale = 1.0

                    confs = probs[b, e2]  # (R,)
                    if mode_l == "sim_prob":
                        base = (sims_region * confs).mean()
                    else:
                        base = (sims_region * F.relu(confs - margin)).mean()
                    losses.append(sample_w * pair_dis * sep_scale * base)

        if not losses:
            return torch.tensor(0.0, device=probs.device, requires_grad=True)
        return torch.stack(losses).mean()

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
            if mode_l == "sim_prob":
                base = (sims * confs).mean()
            else:
                base = (sims * F.relu(confs - margin)).mean()
            if sample_weights is not None:
                base = base * sample_weights[b]
            losses.append(base)

    if not losses:
        return torch.tensor(0.0, device=probs.device, requires_grad=True)
    return torch.stack(losses).mean()

def region_margin_loss(logits, labels, margin=0.5, top_k=5):
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


def ungroundable_all_region_suppression_loss(logits, labels, margin=0.0):
    """
    For ungroundable entities (no positive region in labels),
    suppress scores over all regions so text-vs-region scores stay low.
    """
    B, E, R = logits.shape
    losses = []
    m = float(margin)
    for b in range(B):
        for e in range(E):
            if (labels[b, e] > 0).any():
                continue
            # Penalize high score on any region for ungroundable entities.
            losses.append(F.softplus(logits[b, e] - m).mean())
    if len(losses) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return torch.stack(losses).mean()


def anchor_consistency_loss(
    reg_feats,
    txt_pool,
    ent_types,
    labels,
    anchors,
    alpha=0.7,
    sim_thr=-1.0,
    weighted_sim=False,
    sim_tau=0.2,
    sim_min=-1.0,
    sim_weight_cap=5.0,
    sample_weights=None,
):
    B, E, D = txt_pool.shape
    losses = []

    for b in range(B):
        anchor_e, anchor_r = anchors[b]
        if anchor_e is None:
            continue

        anchor_type = ent_types[b][anchor_e]
        if anchor_type == "":
            continue

        # InfoNCE-style text-guided consistency:
        # maximize alignment between anchor region and same-type texts
        # against all candidate texts in the batch.
        tau = max(1e-6, float(sim_tau))
        anchor_feat = reg_feats[b, anchor_r]  # (D,)
        pos_txt, neg_txt = [], []

        for b2 in range(B):
            for e2 in range(E):
                t = ent_types[b2][e2]
                if t == "":
                    continue
                if t == anchor_type:
                    pos_txt.append(txt_pool[b2, e2])
                else:
                    neg_txt.append(txt_pool[b2, e2])

        if len(pos_txt) == 0 or len(neg_txt) == 0:
            continue

        pos_txt = torch.stack(pos_txt, dim=0)  # (N_pos, D)
        neg_txt = torch.stack(neg_txt, dim=0)  # (N_neg, D)
        all_txt = torch.cat([pos_txt, neg_txt], dim=0)  # (N_all, D)

        # cosine logits with temperature
        anchor_norm = F.normalize(anchor_feat.unsqueeze(0), dim=1)  # (1, D)
        all_txt_norm = F.normalize(all_txt, dim=1)                  # (N_all, D)
        logits = torch.matmul(all_txt_norm, anchor_norm.squeeze(0)) / tau  # (N_all,)
        n_pos = pos_txt.size(0)

        # Optional similarity filtering to reduce noisy positives
        if float(sim_thr) > -1.0:
            pos_logits = logits[:n_pos]
            pos_mask = pos_logits >= float(sim_thr)
            if pos_mask.any():
                pos_logits = pos_logits[pos_mask]
            else:
                continue
        else:
            pos_logits = logits[:n_pos]

        # L = -log( sum exp(pos) / sum exp(all) )
        log_num = torch.logsumexp(pos_logits, dim=0)
        log_den = torch.logsumexp(logits, dim=0)
        base = -(log_num - log_den)
        if sample_weights is not None:
            base = base * sample_weights[b]
        losses.append(base)

    if not losses:
        return torch.tensor(0.0, device=reg_feats.device)
    return torch.stack(losses).mean()

# ---------------- Train ----------------
def train_one_epoch(
    model,
    device,
    loader,
    opt,
    ce_w=1.0,
    cons_w=0.3,
    excl_w=0.5,
    margin_w=1.0,
    **kwargs,
):
    model.train()
    total_loss = 0
    total_ce = 0
    total_cons = 0
    total_excl = 0
    total_margin = 0
    total_ung_push = 0
    for batch in loader:
        tb = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits, probs, reg_proj, txt_pool, gate = model(tb["input_ids"], tb["attn_masks"], tb["region_feats"])

        use_anchor_reliability = bool(kwargs.get("anchor_use_reliability", False))
        anchor_rel_gap_temp = float(kwargs.get("anchor_rel_gap_temp", 0.10))
        if use_anchor_reliability:
            anchors, anchor_reliability = select_anchor_per_sample_with_mlp(
                model, reg_proj, txt_pool, tb["labels"],
                return_reliability=True,
                gap_temp=anchor_rel_gap_temp,
            )
        else:
            anchors = select_anchor_per_sample_with_mlp(model, reg_proj, txt_pool, tb["labels"])
            anchor_reliability = None

        cons_alpha = float(kwargs.get("cons_alpha", 0.7))
        cons_sim_thr = float(kwargs.get("cons_sim_thr", -1.0))
        cons_weighted_sim = bool(kwargs.get("cons_weighted_sim", False))
        cons_sim_tau = float(kwargs.get("cons_sim_tau", 0.2))
        cons_sim_min = float(kwargs.get("cons_sim_min", -1.0))
        cons_sim_weight_cap = float(kwargs.get("cons_sim_weight_cap", 5.0))
        excl_margin = float(kwargs.get("excl_margin", 0.5))
        excl_mode = str(kwargs.get("excl_mode", "relu_margin"))
        excl_variant = str(kwargs.get("excl_variant", "anchor"))
        excl_ent_name_w = float(kwargs.get("excl_ent_name_w", 0.7))
        excl_ent_type_w = float(kwargs.get("excl_ent_type_w", 0.3))
        excl_pair_alpha = float(kwargs.get("excl_pair_alpha", 0.5))
        excl_pair_beta = float(kwargs.get("excl_pair_beta", 0.3))
        excl_pair_region_sim = str(kwargs.get("excl_pair_region_sim", "none"))
        excl_pair_type_mode = str(kwargs.get("excl_pair_type_mode", "neutral"))
        excl_pair_type_factor = float(kwargs.get("excl_pair_type_factor", 1.5))
        excl_pair_name_thr = float(kwargs.get("excl_pair_name_thr", 0.5))
        excl_pair_region_thr = float(kwargs.get("excl_pair_region_thr", 0.5))
        excl_pair_easy_scale = float(kwargs.get("excl_pair_easy_scale", 0.5))
        excl_pair_hard_scale = float(kwargs.get("excl_pair_hard_scale", 1.5))
        rank_margin = float(kwargs.get("rank_margin", 0.5))
        rank_topk = int(kwargs.get("rank_topk", 5))
        ung_push_margin = float(kwargs.get("ung_push_margin", 0.0))

        loss_ce = cross_entropy_loss(probs, tb["labels"], gate=gate)
        loss_cons = anchor_consistency_loss(
            reg_proj,
            txt_pool,
            tb["ent_types"],
            tb["labels"],
            anchors,
            alpha=cons_alpha,
            sim_thr=cons_sim_thr,
            weighted_sim=cons_weighted_sim,
            sim_tau=cons_sim_tau,
            sim_min=cons_sim_min,
            sim_weight_cap=cons_sim_weight_cap,
            sample_weights=anchor_reliability,
        )
        loss_excl = anchor_exclusion_loss(
            probs,
            reg_proj,
            anchors,
            margin=excl_margin,
            mode=excl_mode,
            sample_weights=anchor_reliability,
            variant=excl_variant,
            labels=tb["labels"],
            txt_pool=txt_pool,
            ent_types=tb["ent_types"],
            region_boxes=tb.get("region_boxes", None),
            ent_name_w=excl_ent_name_w,
            ent_type_w=excl_ent_type_w,
            pair_alpha=excl_pair_alpha,
            pair_beta=excl_pair_beta,
            pair_region_sim=excl_pair_region_sim,
            pair_type_mode=excl_pair_type_mode,
            pair_type_factor=excl_pair_type_factor,
            pair_name_thr=excl_pair_name_thr,
            pair_region_thr=excl_pair_region_thr,
            pair_easy_scale=excl_pair_easy_scale,
            pair_hard_scale=excl_pair_hard_scale,
        )
        loss_margin = region_margin_loss(logits, tb["labels"], margin=rank_margin, top_k=rank_topk)
        loss_ung_push = ungroundable_all_region_suppression_loss(
            logits,
            tb["labels"],
            margin=ung_push_margin,
        )

        loss = (
            float(ce_w) * loss_ce +
            float(cons_w) * loss_cons +
            float(excl_w) * loss_excl +
            float(margin_w) * loss_margin +
            float(kwargs.get("ung_push_w", 0.0)) * loss_ung_push
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_cons += loss_cons.item()
        total_excl += loss_excl.item()
        total_margin += loss_margin.item()
        total_ung_push += loss_ung_push.item()

    denom = max(1, len(loader))
    return {
        "total": total_loss / denom,
        "ce": total_ce / denom,
        "cons": total_cons / denom,
        "excl": total_excl / denom,
        "margin": total_margin / denom,
        "ung_push": total_ung_push / denom,
        "rt_pull": 0.0,
        "rt_push": 0.0,
        "rt_mid_pull": 0.0,
        "anchor_align": 0.0,
        "proto_cons": 0.0,
        "mod_proto": 0.0,
        "tripod": 0.0,
        "multi_proto_cons": 0.0,
        "hybrid_proto_cons": 0.0,
    }