import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class GroundingModel(nn.Module):
    def __init__(
        self,
        region_dim=2048,
        text_dim=None,
        num_heads=8,
        temperature=1.0,
        model_name="roberta-large",
    ):
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
        # View-specific region heads (ent/type/know/ctx).
        self.region_view_ent = nn.Linear(text_dim, text_dim)
        self.region_view_type = nn.Linear(text_dim, text_dim)
        self.region_view_know = nn.Linear(text_dim, text_dim)
        self.region_view_ctx = nn.Linear(text_dim, text_dim)
        self._last_views = {}

    def _encode_pooled(self, input_ids, attn_masks):
        B, E, L = input_ids.shape
        txt = self.text_encoder(
            input_ids=input_ids.view(B * E, L),
            attention_mask=attn_masks.view(B * E, L),
        )
        txt_enc = txt.last_hidden_state.view(B, E, L, -1)
        txt_pool = (txt_enc * attn_masks.unsqueeze(-1)).sum(dim=2) / attn_masks.sum(dim=2, keepdim=True).clamp(min=1)
        return txt_enc, txt_pool

    def forward(
        self,
        input_ids,
        attn_masks,
        region_feats,
        ent_input_ids=None,
        ent_attn_masks=None,
        type_input_ids=None,
        type_attn_masks=None,
        know_input_ids=None,
        know_attn_masks=None,
        ctx_input_ids=None,
        ctx_attn_masks=None,
    ):
        B, E, L = input_ids.shape
        reg_proj = self.region_proj(region_feats)

        # Text encoding
        txt_enc, txt_pooled = self._encode_pooled(input_ids, attn_masks)

        _, txt_ent = self._encode_pooled(ent_input_ids, ent_attn_masks) if ent_input_ids is not None else (None, txt_pooled)
        _, txt_type = self._encode_pooled(type_input_ids, type_attn_masks) if type_input_ids is not None else (None, txt_pooled)
        _, txt_know = self._encode_pooled(know_input_ids, know_attn_masks) if know_input_ids is not None else (None, txt_pooled)
        _, txt_ctx = self._encode_pooled(ctx_input_ids, ctx_attn_masks) if ctx_input_ids is not None else (None, txt_pooled)

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
        self._last_views = {
            "txt_main": txt_pooled,
            "txt_ent": txt_ent,
            "txt_type": txt_type,
            "txt_know": txt_know,
            "txt_ctx": txt_ctx,
            "reg_main": reg_proj,
            "reg_ent": self.region_view_ent(reg_proj),
            "reg_type": self.region_view_type(reg_proj),
            "reg_know": self.region_view_know(reg_proj),
            "reg_ctx": self.region_view_ctx(reg_proj),
        }
        return logits, probs, reg_proj, txt_pooled, gate

# ---------------- Anchor Selection ----------------
def _collect_anchor_candidates_with_mlp(model, reg_feats, txt_pool, labels):
    """Collect all positive (entity, region) candidates with MLP score."""
    B, E, _ = txt_pool.shape
    candidates = []
    for b in range(B):
        sample_cands = []
        pos_mask = labels[b] > 0
        for e in range(E):
            pos_idx = pos_mask[e].nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            txt_vec = txt_pool[b, e].unsqueeze(0).expand(len(pos_idx), -1)
            reg_vec = reg_feats[b, pos_idx]
            pair_feat = torch.cat([txt_vec, reg_vec, txt_vec * reg_vec], dim=-1)
            scores = model.anchor_scorer(pair_feat).squeeze(-1)
            for i, r in enumerate(pos_idx):
                sample_cands.append((e, int(r.item()), scores[i]))
        candidates.append(sample_cands)
    return candidates


def _select_anchor_with_entropy(probs, labels, eps=1e-8):
    """
    Entropy-based anchor entity selection:
    1) choose entity with lowest region entropy (most confident)
    2) choose highest-probability positive region for that entity.
    """
    B, E, R = probs.shape
    anchors = []

    for b in range(B):
        best_entropy = None
        best_e = None
        best_r = None

        for e in range(E):
            pos_idx = (labels[b, e] > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue

            p = probs[b, e]
            p = p / p.sum().clamp(min=eps)
            entropy = -(p * torch.log(p.clamp(min=eps))).sum() / torch.log(
                torch.tensor(float(max(2, R)), device=p.device)
            )

            pos_scores = p[pos_idx]
            r = pos_idx[pos_scores.argmax()].item()
            ent_val = float(entropy.item())
            if (best_entropy is None) or (ent_val < best_entropy):
                best_entropy = ent_val
                best_e = e
                best_r = r

        anchors.append((best_e, best_r))

    return anchors


def _build_anchor_bundle(
    sample_cands,
    probs_b,
    topk=1,
    softmax_temp=1.0,
    use_reliability=False,
    rel_gap_temp=0.10,
):
    """
    Build soft top-k anchors:
    - pairs: selected (entity, region)
    - weights: softmax over candidate scores
    - reliability: confidence scalar from top gap + anchor probability
    """
    device = probs_b.device
    if len(sample_cands) == 0:
        return {"pairs": [], "weights": torch.zeros(0, device=device), "reliability": torch.tensor(0.0, device=device)}

    sample_cands = sorted(sample_cands, key=lambda x: float(x[2].detach().item()), reverse=True)
    k = max(1, min(int(topk), len(sample_cands)))
    chosen = sample_cands[:k]
    pairs = [(int(e), int(r)) for e, r, _ in chosen]
    raw_scores = torch.stack([s for _, _, s in chosen], dim=0)
    weights = torch.softmax(raw_scores / max(1e-6, float(softmax_temp)), dim=0)

    reliability = torch.tensor(1.0, device=device)
    if bool(use_reliability):
        if len(sample_cands) >= 2:
            gap = sample_cands[0][2] - sample_cands[1][2]
            gap_rel = torch.sigmoid(gap / max(1e-6, float(rel_gap_temp)))
        else:
            gap_rel = torch.tensor(1.0, device=device)
        top_e, top_r = pairs[0]
        anchor_conf = probs_b[top_e, top_r].clamp(0.0, 1.0)
        reliability = gap_rel * anchor_conf

    return {"pairs": pairs, "weights": weights, "reliability": reliability}


def select_anchor_per_sample(
    model,
    reg_feats,
    txt_pool,
    labels,
    probs=None,
    mode="mlp",
    topk=1,
    softmax_temp=1.0,
    use_reliability=False,
    rel_gap_temp=0.10,
):
    mode = str(mode).lower()
    if mode == "entropy":
        if probs is None:
            raise ValueError("probs is required when anchor selection mode is 'entropy'")
        hard_anchors = _select_anchor_with_entropy(probs, labels)
        bundles = []
        for b, (e, r) in enumerate(hard_anchors):
            if e is None:
                bundles.append({"pairs": [], "weights": torch.zeros(0, device=probs.device), "reliability": torch.tensor(0.0, device=probs.device)})
            else:
                bundles.append(_build_anchor_bundle(
                    [(int(e), int(r), torch.tensor(0.0, device=probs.device))],
                    probs[b],
                    topk=1,
                    softmax_temp=1.0,
                    use_reliability=use_reliability,
                    rel_gap_temp=rel_gap_temp,
                ))
        return bundles

    all_cands = _collect_anchor_candidates_with_mlp(model, reg_feats, txt_pool, labels)
    bundles = []
    for b in range(len(all_cands)):
        bundles.append(
            _build_anchor_bundle(
                all_cands[b],
                probs[b] if probs is not None else torch.zeros_like(labels[b]),
                topk=topk,
                softmax_temp=softmax_temp,
                use_reliability=use_reliability,
                rel_gap_temp=rel_gap_temp,
            )
        )
    return bundles

# ---------------- Losses ----------------
def _combine_views(*pairs):
    """
    pairs: (tensor, weight) where tensor shape is [B, E, D] or [B, R, D].
    Returns normalized weighted sum.
    """
    total = None
    wsum = 0.0
    for t, w in pairs:
        if t is None:
            continue
        wf = float(w)
        if wf == 0.0:
            continue
        total = t * wf if total is None else total + t * wf
        wsum += wf
    if total is None:
        return None
    total = total / max(1e-8, wsum)
    return F.normalize(total, dim=-1)


def _scores_from_query_region(query, region):
    # query: [B, E, D], region: [B, R, D] -> [B, E, R]
    return torch.einsum("bed,brd->ber", F.normalize(query, dim=-1), F.normalize(region, dim=-1))


def region_text_infonce_loss(reg_feats, txt_pool, labels, tau=0.07, topk_neg=0):
    """
    Supervised InfoNCE over entity-text (anchor) and regions:
    - positives: regions with gold label==1 for each (b, e)
    - negatives: regions with label==0
    """
    scores = _scores_from_query_region(txt_pool, reg_feats)  # [B, E, R]
    B, E, _ = scores.shape
    losses = []
    tau = max(1e-6, float(tau))
    k_neg = int(topk_neg)

    for b in range(B):
        for e in range(E):
            pos_idx = (labels[b, e] > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            neg_idx = (labels[b, e] == 0).nonzero(as_tuple=True)[0]
            if len(neg_idx) == 0:
                continue

            if k_neg > 0 and k_neg < len(neg_idx):
                neg_scores = scores[b, e, neg_idx]
                hard_neg_local = torch.topk(neg_scores, k=k_neg).indices
                neg_idx = neg_idx[hard_neg_local]

            pos_logits = scores[b, e, pos_idx] / tau
            neg_logits = scores[b, e, neg_idx] / tau
            all_logits = torch.cat([pos_logits, neg_logits], dim=0)

            # Multiple positives: use log-sum-exp numerator.
            log_num = torch.logsumexp(pos_logits, dim=0)
            log_den = torch.logsumexp(all_logits, dim=0)
            losses.append(-(log_num - log_den))

    if not losses:
        return torch.tensor(0.0, device=reg_feats.device, requires_grad=True)
    return torch.stack(losses).mean()


def region_margin_loss_from_scores(scores, labels, margin=0.5, top_k=5):
    B, E, R = scores.shape
    losses = []
    for b in range(B):
        for e in range(E):
            pos_idx = (labels[b, e] > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            neg_idx = (labels[b, e] == 0).nonzero(as_tuple=True)[0]
            if len(neg_idx) == 0:
                continue
            for anchor in pos_idx:
                anchor_score = scores[b, e, anchor]
                diffs = anchor_score - scores[b, e, neg_idx]
                hard_idx = torch.topk(-diffs, k=min(top_k, len(diffs))).indices
                for r in neg_idx[hard_idx]:
                    losses.append(F.relu(margin - (anchor_score - scores[b, e, r])))
    if not losses:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    return torch.stack(losses).mean()


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
    ent_types=None,
    margin=0.5,
    same_type_w=1.0,
    diff_type_w=1.0,
    topk=0,
    mode="relu_margin",
):
    B, E, R, D = reg_feats.shape[0], probs.shape[1], probs.shape[2], reg_feats.shape[2]
    losses = []

    for b in range(B):
        info = anchors[b]
        if isinstance(info, tuple):
            anchor_e, anchor_r = info
            if anchor_e is None:
                continue
            pairs = [(anchor_e, anchor_r)]
            weights = torch.tensor([1.0], device=probs.device)
            reliability = torch.tensor(1.0, device=probs.device)
        else:
            pairs = info.get("pairs", [])
            weights = info.get("weights", torch.zeros(0, device=probs.device))
            reliability = info.get("reliability", torch.tensor(1.0, device=probs.device))
            if len(pairs) == 0:
                continue

        sample_losses = []
        for i, (anchor_e, anchor_r) in enumerate(pairs):
            anchor_feat = reg_feats[b, anchor_r]
            sims = F.cosine_similarity(anchor_feat.unsqueeze(0), reg_feats[b], dim=1)
            pair_losses = []
            for e2 in range(E):
                if e2 == anchor_e:
                    continue
                confs = probs[b, e2]
                pair_w = 1.0
                if ent_types is not None:
                    pair_w = float(same_type_w) if ent_types[b][e2] == ent_types[b][anchor_e] else float(diff_type_w)
                mode_l = str(mode).lower().strip()
                if mode_l == "sim_prob":
                    push_term = sims * confs
                else:
                    push_term = sims * F.relu(confs - margin)
                if int(topk) > 0 and int(topk) < push_term.numel():
                    hard_idx = torch.topk(push_term, k=int(topk)).indices
                    push_term = push_term[hard_idx]
                pair_losses.append(pair_w * push_term.mean())
            if pair_losses:
                sample_losses.append(weights[i] * torch.stack(pair_losses).mean())
        if sample_losses:
            losses.append(reliability * torch.stack(sample_losses).sum())

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


def anchor_consistency_loss(
    reg_feats,
    txt_pool,
    ent_types,
    labels,
    anchors,
    alpha=0.7,
    sim_thr=-1.0,
    target_pool="mean",
    hard_topk=0,
    include_intra=False,
    weighted_sim=False,
    sim_tau=0.2,
    sim_min=-1.0,
    sim_weight_cap=5.0,
):
    B, E, D = txt_pool.shape
    losses = []

    for b in range(B):
        info = anchors[b]
        if isinstance(info, tuple):
            anchor_e, anchor_r = info
            if anchor_e is None:
                continue
            pairs = [(anchor_e, anchor_r)]
            weights = torch.tensor([1.0], device=reg_feats.device)
            reliability = torch.tensor(1.0, device=reg_feats.device)
        else:
            pairs = info.get("pairs", [])
            weights = info.get("weights", torch.zeros(0, device=reg_feats.device))
            reliability = info.get("reliability", torch.tensor(1.0, device=reg_feats.device))
            if len(pairs) == 0:
                continue

        pair_losses = []
        for i, (anchor_e, anchor_r) in enumerate(pairs):
            anchor_type = ent_types[b][anchor_e]
            anchor_feat = reg_feats[b, anchor_r]

            inner_losses = []
            for b2 in range(B):
                if b2 == b and not bool(include_intra):
                    continue
                for e2 in range(E):
                    if b2 == b and e2 == anchor_e:
                        continue
                    if ent_types[b2][e2] != anchor_type:
                        continue
                    pos_idx = (labels[b2, e2] > 0).nonzero(as_tuple=True)[0]
                    if len(pos_idx) == 0:
                        continue
                    tgt_regs = reg_feats[b2, pos_idx]
                    if str(target_pool).lower() == "max":
                        sims_to_anchor = F.cosine_similarity(
                            anchor_feat.unsqueeze(0), F.normalize(tgt_regs, dim=-1), dim=1
                        )
                        tgt_feat = tgt_regs[sims_to_anchor.argmax()]
                    else:
                        tgt_feat = tgt_regs.mean(dim=0)
                    sim = F.cosine_similarity(anchor_feat.unsqueeze(0), tgt_feat.unsqueeze(0))
                    sim_scalar = float(sim.item())
                    if sim_thr > -1.0 and sim_scalar < sim_thr:
                        continue
                    if float(sim_min) > -1.0 and sim_scalar < float(sim_min):
                        continue
                    base = F.relu(alpha - sim)
                    if bool(weighted_sim):
                        tau = max(1e-6, float(sim_tau))
                        logw = torch.clamp(sim / tau, min=-float(sim_weight_cap), max=float(sim_weight_cap))
                        w = torch.exp(logw)
                        inner_losses.append(w * base)
                    else:
                        inner_losses.append(base)
            if int(hard_topk) > 0 and len(inner_losses) > int(hard_topk):
                stacked = torch.stack(inner_losses).view(-1)
                hard_idx = torch.topk(stacked, k=min(int(hard_topk), int(stacked.numel())), dim=0).indices
                inner_losses = [stacked[i] for i in hard_idx]
            if inner_losses:
                pair_losses.append(weights[i] * torch.stack(inner_losses).mean())
        if pair_losses:
            losses.append(reliability * torch.stack(pair_losses).sum())

    if not losses:
        return torch.tensor(0.0, device=reg_feats.device)
    return torch.stack(losses).mean()


def anchor_referenced_alignment_loss(
    scores,
    labels,
    anchors,
    margin=0.0,
    target_pool="max",
    detach_target=True,
    include_anchor_entity=False,
):
    """
    Anchor-referenced relative alignment:
    use anchor (entity, region) score as a target level and pull
    other entities' gold text-region scores up to that level.
    """
    B, E, _ = labels.shape
    losses = []

    for b in range(B):
        info = anchors[b]
        if isinstance(info, tuple):
            anchor_e, anchor_r = info
            if anchor_e is None:
                continue
            pairs = [(anchor_e, anchor_r)]
            weights = torch.tensor([1.0], device=scores.device)
            reliability = torch.tensor(1.0, device=scores.device)
        else:
            pairs = info.get("pairs", [])
            weights = info.get("weights", torch.zeros(0, device=scores.device))
            reliability = info.get("reliability", torch.tensor(1.0, device=scores.device))
            if len(pairs) == 0:
                continue

        sample_losses = []
        for i, (anchor_e, anchor_r) in enumerate(pairs):
            anchor_score = scores[b, anchor_e, anchor_r]
            if bool(detach_target):
                anchor_score = anchor_score.detach()
            target = anchor_score - float(margin)

            pair_losses = []
            for e2 in range(E):
                if not bool(include_anchor_entity) and e2 == anchor_e:
                    continue
                pos_idx = (labels[b, e2] > 0).nonzero(as_tuple=True)[0]
                if len(pos_idx) == 0:
                    continue
                pos_scores = scores[b, e2, pos_idx]
                if str(target_pool).lower() == "mean":
                    cur = pos_scores.mean()
                else:
                    cur = pos_scores.max()
                pair_losses.append(F.relu(target - cur))

            if pair_losses:
                sample_losses.append(weights[i] * torch.stack(pair_losses).mean())

        if sample_losses:
            losses.append(reliability * torch.stack(sample_losses).sum())

    if not losses:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    return torch.stack(losses).mean()


def anchor_tripod_loss(
    reg_feats,
    labels,
    anchors,
    probs=None,
    margin=0.2,
    neg_topk=3,
    score_weighted_neg=True,
):
    """
    Intra-sample tripod loss using selected anchors:
      anchor should be close to same-entity positives and far from hard negatives.
    """
    B, E, R = labels.shape
    losses = []
    for b in range(B):
        info = anchors[b]
        if isinstance(info, tuple):
            anchor_e, anchor_r = info
            if anchor_e is None:
                continue
            pairs = [(anchor_e, anchor_r)]
            weights = torch.tensor([1.0], device=reg_feats.device)
            reliability = torch.tensor(1.0, device=reg_feats.device)
        else:
            pairs = info.get("pairs", [])
            weights = info.get("weights", torch.zeros(0, device=reg_feats.device))
            reliability = info.get("reliability", torch.tensor(1.0, device=reg_feats.device))
            if len(pairs) == 0:
                continue

        sample_losses = []
        for i, (anchor_e, anchor_r) in enumerate(pairs):
            anchor_feat = F.normalize(reg_feats[b, anchor_r].unsqueeze(0), dim=-1)  # [1, D]
            pos_idx = (labels[b, anchor_e] > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            pos_feats = F.normalize(reg_feats[b, pos_idx], dim=-1)
            pos_sim = (pos_feats * anchor_feat).sum(dim=-1).max()

            neg_mask = labels[b, anchor_e] == 0
            neg_idx = neg_mask.nonzero(as_tuple=True)[0]
            if len(neg_idx) == 0:
                continue
            neg_feats = F.normalize(reg_feats[b, neg_idx], dim=-1)
            neg_sim = (neg_feats * anchor_feat).sum(dim=-1)

            if probs is not None and bool(score_weighted_neg):
                if E > 1:
                    other_mask = torch.ones(E, dtype=torch.bool, device=probs.device)
                    other_mask[anchor_e] = False
                    neg_conf = probs[b, other_mask][:, neg_idx].max(dim=0).values
                else:
                    neg_conf = torch.ones_like(neg_sim)
                hard_score = neg_sim + neg_conf
            else:
                hard_score = neg_sim

            k = min(max(1, int(neg_topk)), int(hard_score.numel()))
            hard_idx = torch.topk(hard_score, k=k).indices
            neg_sim_hard = neg_sim[hard_idx]
            tripod = F.relu(float(margin) - pos_sim + neg_sim_hard).mean()
            sample_losses.append(weights[i] * tripod)
        if sample_losses:
            losses.append(reliability * torch.stack(sample_losses).sum())

    if not losses:
        return torch.tensor(0.0, device=reg_feats.device, requires_grad=True)
    return torch.stack(losses).mean()


def region_text_contrastive_loss(
    reg_feats,
    txt_pool,
    labels,
    probs=None,
    neg_margin=0.2,
    detach_text=True,
    push_sim_weighted=False,
    push_topk=0,
    push_weight_power=1.0,
    push_detach_weights=True,
    groundability_weighted=False,
    groundability_power=1.0,
    groundability_min=0.0,
    groundability_detach=True,
):
    """
    Region-text contrastive objective (intra-image):
    1) Pull regions toward their matched entity text.
    2) Push regions away from other entities' text in the same image.
    """
    B, E, _ = txt_pool.shape
    pull_losses = []
    push_losses = []

    for b in range(B):
        txt_b = txt_pool[b].detach() if detach_text else txt_pool[b]
        txt_b = F.normalize(txt_b, dim=-1)  # [E, D]

        for e in range(E):
            pos_idx = (labels[b, e] > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            gw = torch.tensor(1.0, device=reg_feats.device)
            if bool(groundability_weighted) and probs is not None:
                gw = probs[b, e].max().clamp(0.0, 1.0)
                gw = torch.clamp(gw.pow(float(groundability_power)), min=float(groundability_min), max=1.0)
                if bool(groundability_detach):
                    gw = gw.detach()

            pos_regs = reg_feats[b, pos_idx]  # [K, D]
            pos_regs = F.normalize(pos_regs, dim=-1)

            # Pull: matched text should be close to positive regions.
            pos_txt = txt_b[e].unsqueeze(0)  # [1, D]
            pos_sims = (pos_regs * pos_txt).sum(dim=-1)
            pull_losses.append(gw * (1.0 - pos_sims).mean())

            # Push: other entity texts in same image should be farther.
            if E > 1:
                other_mask = torch.ones(E, dtype=torch.bool, device=txt_b.device)
                other_mask[e] = False
                neg_txt = txt_b[other_mask]  # [E-1, D]
                if neg_txt.numel() > 0:
                    neg_sims = pos_regs @ neg_txt.t()  # [K, E-1]
                    txt_neg_sims = (txt_b[e].unsqueeze(0) @ neg_txt.t()).squeeze(0)  # [E-1]

                    if int(push_topk) > 0 and txt_neg_sims.numel() > int(push_topk):
                        k = min(int(push_topk), int(txt_neg_sims.numel()))
                        top_idx = torch.topk(txt_neg_sims, k=k).indices
                        neg_sims = neg_sims[:, top_idx]
                        txt_neg_sims = txt_neg_sims[top_idx]

                    push_term = F.relu(neg_sims - neg_margin)
                    if bool(push_sim_weighted):
                        weights = torch.clamp(txt_neg_sims, min=0.0)
                        weights = weights.pow(float(push_weight_power))
                        weights = weights / (weights.mean() + 1e-8)
                        if bool(push_detach_weights):
                            weights = weights.detach()
                        push_term = push_term * weights.unsqueeze(0)

                    push_losses.append(gw * push_term.mean())

    zero = torch.tensor(0.0, device=reg_feats.device, requires_grad=True)
    pull = torch.stack(pull_losses).mean() if pull_losses else zero
    push = torch.stack(push_losses).mean() if push_losses else zero
    return pull, push


def region_text_mid_pull_loss(
    reg_feats,
    txt_pool,
    labels,
    alpha=0.5,
):
    """
    Middle-representation pull:
    text and positive region are aligned via an intermediate prototype.
    """
    B, E, _ = txt_pool.shape
    losses = []
    a = float(alpha)
    a = max(0.0, min(1.0, a))

    for b in range(B):
        txt_b = F.normalize(txt_pool[b], dim=-1)
        for e in range(E):
            pos_idx = (labels[b, e] > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            pos_regs = F.normalize(reg_feats[b, pos_idx], dim=-1)
            reg_pos_mean = F.normalize(pos_regs.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
            txt_e = txt_b[e]
            proto = F.normalize((a * txt_e + (1.0 - a) * reg_pos_mean).unsqueeze(0), dim=-1).squeeze(0)
            sim_txt = (txt_e * proto).sum()
            sim_reg = (reg_pos_mean * proto).sum()
            losses.append((1.0 - sim_txt) + (1.0 - sim_reg))

    if not losses:
        return torch.tensor(0.0, device=reg_feats.device, requires_grad=True)
    return torch.stack(losses).mean()


def prototype_consistency_loss(
    reg_feats,
    ent_types,
    labels,
    alpha=0.7,
    min_support=2,
):
    """
    Prototype-mediated cross-image consistency:
    each entity's positive-region representation aligns to the
    leave-one-out type prototype built from other images/entities.
    """
    type_to_feats = {}
    B, E, _ = labels.shape

    for b in range(B):
        for e in range(E):
            ent_type = ent_types[b][e]
            if not ent_type:
                continue
            pos_idx = (labels[b, e] > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            ent_feat = reg_feats[b, pos_idx].mean(dim=0)
            type_to_feats.setdefault(ent_type, []).append(ent_feat)

    losses = []
    for _, feats in type_to_feats.items():
        n = len(feats)
        if n < max(2, int(min_support)):
            continue
        stack = torch.stack(feats, dim=0)   # [N, D]
        stack = F.normalize(stack, dim=-1)
        summed = stack.sum(dim=0, keepdim=True)
        for i in range(n):
            loo_proto = (summed - stack[i : i + 1]) / float(n - 1)
            loo_proto = F.normalize(loo_proto, dim=-1)
            sim = (stack[i : i + 1] * loo_proto).sum(dim=-1)
            losses.append(F.relu(alpha - sim))

    if not losses:
        return torch.tensor(0.0, device=reg_feats.device, requires_grad=True)
    return torch.stack(losses).mean()


def modality_prototype_loss(
    reg_feats,
    txt_pool,
    ent_types,
    labels,
    alpha=0.7,
    min_support=2,
):
    """
    Modality prototype alignment:
    For each type, build a shared text-region prototype and align both
    region positives and text representations to that middle prototype.
    """
    type_to_regs = {}
    type_to_txts = {}
    B, E, _ = labels.shape

    for b in range(B):
        for e in range(E):
            t = ent_types[b][e]
            if not t:
                continue
            pos_idx = (labels[b, e] > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            reg_e = reg_feats[b, pos_idx].mean(dim=0)
            txt_e = txt_pool[b, e]
            type_to_regs.setdefault(t, []).append(reg_e)
            type_to_txts.setdefault(t, []).append(txt_e)

    losses = []
    for t in type_to_regs.keys():
        regs = type_to_regs[t]
        txts = type_to_txts[t]
        n = min(len(regs), len(txts))
        if n < max(2, int(min_support)):
            continue
        regs = F.normalize(torch.stack(regs[:n], dim=0), dim=-1)
        txts = F.normalize(torch.stack(txts[:n], dim=0), dim=-1)
        proto = F.normalize((regs.mean(dim=0) + txts.mean(dim=0)).unsqueeze(0), dim=-1)
        sim_r = (regs * proto).sum(dim=-1)
        sim_t = (txts * proto).sum(dim=-1)
        losses.append(F.relu(alpha - sim_r).mean())
        losses.append(F.relu(alpha - sim_t).mean())

    if not losses:
        return torch.tensor(0.0, device=reg_feats.device, requires_grad=True)
    return torch.stack(losses).mean()


def multi_prototype_consistency_loss(
    reg_feats,
    ent_types,
    labels,
    alpha=0.7,
    num_prototypes=3,
    min_support=2,
    assign_iters=2,
):
    """
    Multi-visual prototype consistency:
    type-specific positives are clustered into K prototypes (hard assignment),
    then each feature is aligned to its nearest prototype.
    """
    type_to_feats = {}
    B, E, _ = labels.shape

    for b in range(B):
        for e in range(E):
            ent_type = ent_types[b][e]
            if not ent_type:
                continue
            pos_idx = (labels[b, e] > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            ent_feat = reg_feats[b, pos_idx].mean(dim=0)
            type_to_feats.setdefault(ent_type, []).append(ent_feat)

    losses = []
    for _, feats in type_to_feats.items():
        n = len(feats)
        if n < max(2, int(min_support)):
            continue

        feats = F.normalize(torch.stack(feats, dim=0), dim=-1)  # [N, D]
        k = max(1, min(int(num_prototypes), n))

        # Deterministic initialization to avoid run-to-run randomness.
        init_idx = torch.linspace(0, n - 1, steps=k, device=feats.device).long()
        protos = feats[init_idx].clone()  # [K, D]

        for _ in range(max(1, int(assign_iters))):
            sims = feats @ protos.t()       # [N, K]
            assign = sims.argmax(dim=-1)    # [N]
            new_protos = []
            for j in range(k):
                mask = assign == j
                if bool(mask.any()):
                    pj = feats[mask].mean(dim=0)
                else:
                    pj = protos[j]
                new_protos.append(F.normalize(pj.unsqueeze(0), dim=-1).squeeze(0))
            protos = torch.stack(new_protos, dim=0)

        final_sims = feats @ protos.t()      # [N, K]
        nearest = final_sims.max(dim=-1).values
        losses.append(F.relu(alpha - nearest).mean())

    if not losses:
        return torch.tensor(0.0, device=reg_feats.device, requires_grad=True)
    return torch.stack(losses).mean()


def hybrid_multi_prototype_consistency_loss(
    reg_feats,
    ent_types,
    labels,
    alpha=0.7,
    num_prototypes=3,
    min_support=2,
    assign_iters=2,
    beta=0.5,
):
    """
    Unified hybrid consistency (single loss):
    combines (a) nearest multi-prototype alignment and
    (b) leave-one-out class prototype alignment.
    """
    type_to_feats = {}
    B, E, _ = labels.shape

    for b in range(B):
        for e in range(E):
            ent_type = ent_types[b][e]
            if not ent_type:
                continue
            pos_idx = (labels[b, e] > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            ent_feat = reg_feats[b, pos_idx].mean(dim=0)
            type_to_feats.setdefault(ent_type, []).append(ent_feat)

    losses = []
    for _, feats in type_to_feats.items():
        n = len(feats)
        if n < max(2, int(min_support)):
            continue

        feats = F.normalize(torch.stack(feats, dim=0), dim=-1)  # [N, D]
        k = max(1, min(int(num_prototypes), n))

        init_idx = torch.linspace(0, n - 1, steps=k, device=feats.device).long()
        protos = feats[init_idx].clone()

        for _ in range(max(1, int(assign_iters))):
            sims = feats @ protos.t()
            assign = sims.argmax(dim=-1)
            new_protos = []
            for j in range(k):
                mask = assign == j
                if bool(mask.any()):
                    pj = feats[mask].mean(dim=0)
                else:
                    pj = protos[j]
                new_protos.append(F.normalize(pj.unsqueeze(0), dim=-1).squeeze(0))
            protos = torch.stack(new_protos, dim=0)

        nearest_sim = (feats @ protos.t()).max(dim=-1).values  # [N]

        summed = feats.sum(dim=0, keepdim=True)
        loo_proto = (summed - feats) / float(max(1, n - 1))
        loo_proto = F.normalize(loo_proto, dim=-1)
        loo_sim = (feats * loo_proto).sum(dim=-1)  # [N]

        hybrid_sim = float(beta) * nearest_sim + (1.0 - float(beta)) * loo_sim
        losses.append(F.relu(alpha - hybrid_sim).mean())

    if not losses:
        return torch.tensor(0.0, device=reg_feats.device, requires_grad=True)
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
    info_nce_w=0.0,
    info_nce_tau=0.07,
    info_nce_topk_neg=0,
    cons_alpha=0.7,
    cons_sim_thr=-1.0,
    cons_target_pool="mean",
    cons_hard_topk=0,
    cons_include_intra=False,
    cons_weighted_sim=False,
    cons_sim_tau=0.2,
    cons_sim_min=-1.0,
    cons_sim_weight_cap=5.0,
    excl_margin=0.5,
    excl_same_type_w=1.0,
    excl_diff_type_w=1.0,
    excl_topk=0,
    excl_mode="relu_margin",
    rank_margin=0.5,
    rank_topk=5,
    rt_pull_w=0.0,
    rt_push_w=0.0,
    rt_neg_margin=0.2,
    rt_push_sim_weighted=False,
    rt_push_topk=0,
    rt_push_weight_power=1.0,
    rt_push_detach_weights=True,
    rt_groundability_weighted=False,
    rt_groundability_power=1.0,
    rt_groundability_min=0.0,
    rt_groundability_detach=True,
    rt_mid_pull_w=0.0,
    rt_mid_alpha=0.5,
    anchor_align_w=0.0,
    anchor_align_margin=0.0,
    anchor_align_target_pool="max",
    anchor_align_detach_target=True,
    anchor_align_include_anchor_entity=False,
    proto_cons_w=0.0,
    proto_cons_alpha=0.7,
    proto_min_support=2,
    mod_proto_w=0.0,
    mod_proto_alpha=0.7,
    mod_proto_min_support=2,
    tripod_w=0.0,
    tripod_margin=0.2,
    tripod_neg_topk=3,
    tripod_score_weighted_neg=True,
    multi_proto_cons_w=0.0,
    multi_proto_alpha=0.7,
    multi_proto_k=3,
    multi_proto_min_support=2,
    multi_proto_assign_iters=2,
    hybrid_proto_cons_w=0.0,
    hybrid_proto_alpha=0.7,
    hybrid_proto_k=3,
    hybrid_proto_min_support=2,
    hybrid_proto_assign_iters=2,
    hybrid_proto_beta=0.5,
    anchor_select_mode="mlp",
    anchor_topk=1,
    anchor_softmax_temp=1.0,
    anchor_use_reliability=False,
    anchor_rel_gap_temp=0.10,
    use_multi_view=False,
    mv_w_ent=0.65,
    mv_w_know=0.20,
    mv_w_type=0.15,
    mv_w_ctx=0.0,
):
    model.train()
    total_loss = 0
    total_ce = 0
    total_cons = 0
    total_excl = 0
    total_margin = 0
    total_info_nce = 0
    total_rt_pull = 0
    total_rt_push = 0
    total_rt_mid_pull = 0
    total_anchor_align = 0
    total_proto_cons = 0
    total_mod_proto = 0
    total_tripod = 0
    total_multi_proto_cons = 0
    total_hybrid_proto_cons = 0
    for batch in loader:
        tb = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits, probs, reg_proj, txt_pool, gate = model(
            tb["input_ids"],
            tb["attn_masks"],
            tb["region_feats"],
            ent_input_ids=tb.get("ent_input_ids"),
            ent_attn_masks=tb.get("ent_attn_masks"),
            type_input_ids=tb.get("type_input_ids"),
            type_attn_masks=tb.get("type_attn_masks"),
            know_input_ids=tb.get("know_input_ids"),
            know_attn_masks=tb.get("know_attn_masks"),
            ctx_input_ids=tb.get("ctx_input_ids"),
            ctx_attn_masks=tb.get("ctx_attn_masks"),
        )
        views = getattr(model, "_last_views", {})
        txt_ent = views.get("txt_ent", txt_pool)
        txt_type = views.get("txt_type", txt_pool)
        txt_know = views.get("txt_know", txt_pool)
        txt_ctx = views.get("txt_ctx", txt_pool)
        reg_ent = views.get("reg_ent", reg_proj)
        reg_type = views.get("reg_type", reg_proj)
        reg_know = views.get("reg_know", reg_proj)
        reg_ctx = views.get("reg_ctx", reg_proj)

        q_margin = _combine_views(
            (txt_ent, mv_w_ent),
            (txt_know, mv_w_know),
            (txt_type, mv_w_type),
            (txt_ctx, mv_w_ctx),
        )
        r_margin = _combine_views(
            (reg_ent, mv_w_ent),
            (reg_know, mv_w_know),
            (reg_type, mv_w_type),
            (reg_ctx, mv_w_ctx),
        )
        q_excl = _combine_views((txt_ent, 1.0), (txt_know, 1.0))
        r_excl = _combine_views((reg_ent, 1.0), (reg_know, 1.0))

        zero = torch.tensor(0.0, device=logits.device)

        need_anchor = any(
            float(w) > 0.0 for w in (cons_w, excl_w, tripod_w, anchor_align_w)
        )
        anchors = None
        if need_anchor:
            anchors = select_anchor_per_sample(
                model,
                r_excl if bool(use_multi_view) and r_excl is not None else reg_proj,
                q_excl if bool(use_multi_view) and q_excl is not None else txt_pool,
                tb["labels"],
                probs=probs,
                mode=anchor_select_mode,
                topk=anchor_topk,
                softmax_temp=anchor_softmax_temp,
                use_reliability=anchor_use_reliability,
                rel_gap_temp=anchor_rel_gap_temp,
            )

        loss_ce = cross_entropy_loss(probs, tb["labels"], gate=gate) if float(ce_w) > 0.0 else zero

        if float(cons_w) > 0.0:
            loss_cons = anchor_consistency_loss(
                reg_type if bool(use_multi_view) and reg_type is not None else reg_proj,
                txt_pool,
                tb["ent_types"],
                tb["labels"],
                anchors,
                alpha=cons_alpha,
                sim_thr=cons_sim_thr,
                target_pool=cons_target_pool,
                hard_topk=cons_hard_topk,
                include_intra=cons_include_intra,
                weighted_sim=cons_weighted_sim,
                sim_tau=cons_sim_tau,
                sim_min=cons_sim_min,
                sim_weight_cap=cons_sim_weight_cap,
            )
        else:
            loss_cons = zero

        if float(excl_w) > 0.0:
            loss_excl = anchor_exclusion_loss(
                (
                    torch.softmax(
                        (_scores_from_query_region(q_excl, r_excl) / max(1e-6, float(model.temperature))).view(probs.shape[0], -1),
                        dim=1,
                    ).view_as(probs)
                    if bool(use_multi_view) and q_excl is not None and r_excl is not None
                    else probs
                ),
                r_excl if bool(use_multi_view) and r_excl is not None else reg_proj,
                anchors,
                ent_types=tb["ent_types"],
                margin=excl_margin,
                same_type_w=excl_same_type_w,
                diff_type_w=excl_diff_type_w,
                topk=excl_topk,
                mode=excl_mode,
            )
        else:
            loss_excl = zero

        if float(margin_w) > 0.0:
            if bool(use_multi_view) and q_margin is not None and r_margin is not None:
                margin_scores = _scores_from_query_region(q_margin, r_margin)
                loss_margin = region_margin_loss_from_scores(
                    margin_scores, tb["labels"], margin=rank_margin, top_k=rank_topk
                )
            else:
                loss_margin = region_margin_loss(logits, tb["labels"], margin=rank_margin, top_k=rank_topk)
        else:
            loss_margin = zero

        if float(info_nce_w) > 0.0:
            loss_info_nce = region_text_infonce_loss(
                r_margin if bool(use_multi_view) and r_margin is not None else reg_proj,
                q_margin if bool(use_multi_view) and q_margin is not None else txt_pool,
                tb["labels"],
                tau=info_nce_tau,
                topk_neg=info_nce_topk_neg,
            )
        else:
            loss_info_nce = zero

        if float(rt_pull_w) > 0.0 or float(rt_push_w) > 0.0:
            loss_rt_pull, loss_rt_push = region_text_contrastive_loss(
                r_excl if bool(use_multi_view) and r_excl is not None else reg_proj,
                q_excl if bool(use_multi_view) and q_excl is not None else txt_pool,
                tb["labels"],
                probs=probs,
                neg_margin=rt_neg_margin,
                detach_text=True,
                push_sim_weighted=rt_push_sim_weighted,
                push_topk=rt_push_topk,
                push_weight_power=rt_push_weight_power,
                push_detach_weights=rt_push_detach_weights,
                groundability_weighted=rt_groundability_weighted,
                groundability_power=rt_groundability_power,
                groundability_min=rt_groundability_min,
                groundability_detach=rt_groundability_detach,
            )
        else:
            loss_rt_pull, loss_rt_push = zero, zero

        loss_rt_mid_pull = (
            region_text_mid_pull_loss(
                r_excl if bool(use_multi_view) and r_excl is not None else reg_proj,
                q_excl if bool(use_multi_view) and q_excl is not None else txt_pool,
                tb["labels"],
                alpha=rt_mid_alpha,
            )
            if float(rt_mid_pull_w) > 0.0
            else zero
        )

        if float(anchor_align_w) > 0.0:
            if bool(use_multi_view) and q_excl is not None and r_excl is not None:
                align_scores = _scores_from_query_region(q_excl, r_excl)
            else:
                align_scores = logits
            loss_anchor_align = anchor_referenced_alignment_loss(
                align_scores,
                tb["labels"],
                anchors,
                margin=anchor_align_margin,
                target_pool=anchor_align_target_pool,
                detach_target=anchor_align_detach_target,
                include_anchor_entity=anchor_align_include_anchor_entity,
            )
        else:
            loss_anchor_align = zero

        loss_proto_cons = (
            prototype_consistency_loss(
                reg_type if bool(use_multi_view) and reg_type is not None else reg_proj,
                tb["ent_types"],
                tb["labels"],
                alpha=proto_cons_alpha,
                min_support=proto_min_support,
            )
            if float(proto_cons_w) > 0.0
            else zero
        )
        loss_mod_proto = (
            modality_prototype_loss(
                reg_proj,
                txt_pool,
                tb["ent_types"],
                tb["labels"],
                alpha=mod_proto_alpha,
                min_support=mod_proto_min_support,
            )
            if float(mod_proto_w) > 0.0
            else zero
        )
        loss_tripod = (
            anchor_tripod_loss(
                r_excl if bool(use_multi_view) and r_excl is not None else reg_proj,
                tb["labels"],
                anchors,
                probs=probs,
                margin=tripod_margin,
                neg_topk=tripod_neg_topk,
                score_weighted_neg=tripod_score_weighted_neg,
            )
            if float(tripod_w) > 0.0
            else zero
        )
        loss_multi_proto_cons = (
            multi_prototype_consistency_loss(
                reg_type if bool(use_multi_view) and reg_type is not None else reg_proj,
                tb["ent_types"],
                tb["labels"],
                alpha=multi_proto_alpha,
                num_prototypes=multi_proto_k,
                min_support=multi_proto_min_support,
                assign_iters=multi_proto_assign_iters,
            )
            if float(multi_proto_cons_w) > 0.0
            else zero
        )
        loss_hybrid_proto_cons = (
            hybrid_multi_prototype_consistency_loss(
                reg_type if bool(use_multi_view) and reg_type is not None else reg_proj,
                tb["ent_types"],
                tb["labels"],
                alpha=hybrid_proto_alpha,
                num_prototypes=hybrid_proto_k,
                min_support=hybrid_proto_min_support,
                assign_iters=hybrid_proto_assign_iters,
                beta=hybrid_proto_beta,
            )
            if float(hybrid_proto_cons_w) > 0.0
            else zero
        )

        loss = (
            ce_w * loss_ce +
            cons_w * loss_cons +
            excl_w * loss_excl +
            margin_w * loss_margin +
            info_nce_w * loss_info_nce +
            rt_pull_w * loss_rt_pull +
            rt_push_w * loss_rt_push +
            rt_mid_pull_w * loss_rt_mid_pull +
            anchor_align_w * loss_anchor_align +
            proto_cons_w * loss_proto_cons +
            mod_proto_w * loss_mod_proto +
            tripod_w * loss_tripod +
            multi_proto_cons_w * loss_multi_proto_cons +
            hybrid_proto_cons_w * loss_hybrid_proto_cons
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
        total_info_nce += loss_info_nce.item()
        total_rt_pull += loss_rt_pull.item()
        total_rt_push += loss_rt_push.item()
        total_rt_mid_pull += loss_rt_mid_pull.item()
        total_anchor_align += loss_anchor_align.item()
        total_proto_cons += loss_proto_cons.item()
        total_mod_proto += loss_mod_proto.item()
        total_tripod += loss_tripod.item()
        total_multi_proto_cons += loss_multi_proto_cons.item()
        total_hybrid_proto_cons += loss_hybrid_proto_cons.item()

    denom = max(1, len(loader))
    return {
        "total": total_loss / denom,
        "ce": total_ce / denom,
        "cons": total_cons / denom,
        "excl": total_excl / denom,
        "margin": total_margin / denom,
        "info_nce": total_info_nce / denom,
        "rt_pull": total_rt_pull / denom,
        "rt_push": total_rt_push / denom,
        "rt_mid_pull": total_rt_mid_pull / denom,
        "anchor_align": total_anchor_align / denom,
        "proto_cons": total_proto_cons / denom,
        "mod_proto": total_mod_proto / denom,
        "tripod": total_tripod / denom,
        "multi_proto_cons": total_multi_proto_cons / denom,
        "hybrid_proto_cons": total_hybrid_proto_cons / denom,
    }
