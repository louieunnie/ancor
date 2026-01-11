import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def sigmoid_focal_with_logits(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    logits = logits.float()
    targets = targets.float().to(logits.device)
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t).pow(gamma))
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean() if reduction == "mean" else loss.sum()

# ---------------------------------------------------------
# Region Cross Attention
# ---------------------------------------------------------
class SpanRegionCrossAttn(nn.Module):
    def __init__(self, span_dim, vis_dim, attn_dim=256, num_heads=4):
        super().__init__()
        self.q_proj = nn.Linear(span_dim, attn_dim)
        self.k_proj = nn.Linear(vis_dim, attn_dim)
        self.v_proj = nn.Linear(vis_dim, attn_dim)
        self.box_pe = nn.Linear(6, attn_dim)
        self.mha = nn.MultiheadAttention(attn_dim, num_heads, batch_first=True)

        # span→region gate
        self.gate = nn.Sequential(
            nn.Linear(span_dim + attn_dim, attn_dim), nn.ReLU(), nn.Linear(attn_dim, 1)
        )

        # project back to 2H span dim
        self.out_proj = nn.Linear(attn_dim, span_dim)

    def forward(self, span_rep, region_feats, region_boxes, region_mask):
        # span_rep: (1, span_dim)
        q = self.q_proj(span_rep).unsqueeze(1)   # (1,1,A)
        k = self.k_proj(region_feats) + self.box_pe(region_boxes)
        v = self.v_proj(region_feats) + self.box_pe(region_boxes)

        kpm = (region_mask == 0)  # (1,R)
        z, _ = self.mha(q, k, v, key_padding_mask=kpm)  # (1,1,A)
        z = z.squeeze(1)

        g = torch.sigmoid(self.gate(torch.cat([span_rep, z], dim=-1)))
        fused = span_rep + g * (self.out_proj(z) - span_rep)
        return fused  # (1, span_dim)

# ---------------------------------------------------------
# Knowledge Cross Attention
# ---------------------------------------------------------
class KnowledgeCrossAttn(nn.Module):
    def __init__(self, text_dim, know_dim, attn_dim=256, num_heads=4):
        super().__init__()
        self.q_proj = nn.Linear(text_dim, attn_dim)
        self.k_proj = nn.Linear(know_dim, attn_dim)
        self.v_proj = nn.Linear(know_dim, attn_dim)
        self.mha = nn.MultiheadAttention(attn_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(attn_dim, text_dim)

    def forward(self, t_feat, k_feat, k_mask):
        q = self.q_proj(t_feat)
        k = self.k_proj(k_feat)
        v = self.v_proj(k_feat)
        kpm = (k_mask == 0)
        z, _ = self.mha(q, k, v, key_padding_mask=kpm)
        return self.out_proj(z)

# ---------------------------------------------------------
# Main Model
# ---------------------------------------------------------
class Context_EnhancedMNER(nn.Module):
    def __init__(self, num_labels, vis_dim, tokenizer,
                 model_name="roberta-base",
                 use_knowledge=True,
                 use_context=True,
                 use_region_attn=True,
                 use_span_injection=True, 
                 use_span_cls=False, use_span_refine=True):
        super().__init__()

        self.use_knowledge = use_knowledge
        self.use_context = use_context
        self.use_region_attn = use_region_attn
        self.use_span_injection = use_span_injection
        self.use_span_cls = use_span_cls
        self.use_span_refine = use_span_refine
        self.tokenizer = tokenizer

        self.encoder = RobertaModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size
        self.H = H

        # --- span classification head---
        if use_span_cls:
            self.span_cls_head = nn.Linear(2 * H, num_labels)
            self.lambda_span = 0.5

        # --- knowledge vector projection ---
        self.knowledge_proj = nn.Linear(H, 2 * H)
        self.knowledge_gate = nn.Sequential(
            nn.Linear(4 * H, H),
            nn.ReLU(),
            nn.Linear(H, 1)
        )

        with torch.no_grad():
            self.knowledge_gate[-1].bias.fill_(-3.0)

        # Knowledge encoder
        self.knowledge_encoder = RobertaModel.from_pretrained(model_name)
        for p in self.knowledge_encoder.parameters():
            p.requires_grad = False
        self.know_cross = KnowledgeCrossAttn(text_dim=H, know_dim=H)

        # Boundary heads (focal loss)
        self.start_mlp = nn.Linear(H, 1)
        self.end_mlp   = nn.Linear(H, 1)

        # Span context
        self.span_ctx_proj = nn.Linear(4 * H, 2 * H)
        self.ctx_gate = nn.Sequential(nn.Linear(4 * H, H), nn.ReLU(), nn.Linear(H, 1))

        # Region cross-attn
        self.region_cross = SpanRegionCrossAttn(span_dim=2*H, vis_dim=vis_dim)

        # span→token
        self.span_to_token = nn.Linear(2 * H, H)

        # Final token classifier
        self.token_head = nn.Linear(H, num_labels)

        self.know_global_proj = nn.Linear(self.H, self.H)
        self.know_global_gate = nn.Sequential(
            nn.Linear(self.H * 2, self.H),
            nn.ReLU(),
            nn.Linear(self.H, 1)
        )
        with torch.no_grad():
            self.know_global_gate[-1].bias.fill_(-2.0)  # conservative gating

        self.span_inject_gate = nn.Sequential(
            nn.Linear(2 * self.H, self.H),
            nn.ReLU(),
            nn.Linear(self.H, 1)
        )
        self.lambda_contrast = 0.1  
        self.temperature = 0.07


    def encode_text(self, input_ids, attention_mask):
        return self.encoder(input_ids, attention_mask).last_hidden_state

    def encode_knowledge(self, texts, device):
        safe = [t if (t and t.strip() != "") else "[EMPTY]" for t in texts]
        k_in = self.tokenizer(safe, padding=True, truncation=True, return_tensors="pt").to(device)
        out = self.knowledge_encoder(**k_in)
        hidden = out.last_hidden_state       # [B, K, H]
        mask   = k_in["attention_mask"]      # [B, K]
        pooled = hidden[:, :4].mean(dim=1)        # [B, H]
        knowledge_vec = self.knowledge_proj(pooled)  # [B, 2H]
        return hidden, mask, knowledge_vec

    def predict_boundaries(self, Hs):
        return self.start_mlp(Hs).squeeze(-1), self.end_mlp(Hs).squeeze(-1)

    def decode_spans(self, start_logits, end_logits, attention_mask,
                      max_len=30, thr_s=0.5, thr_e=0.5):
        s_prob, e_prob = torch.sigmoid(start_logits), torch.sigmoid(end_logits)
        B, T = s_prob.size()
        spans = []
        for b in range(B):
            s_idx = torch.nonzero((s_prob[b] >= thr_s) & attention_mask[b].bool()).squeeze(-1)
            e_idx = torch.nonzero((e_prob[b] >= thr_e) & attention_mask[b].bool()).squeeze(-1)
            if len(s_idx)==0 or len(e_idx)==0:
                spans.append([(0, int(attention_mask[b].sum())-1)])
                continue
            cand=[]
            for s in s_idx:
                for e in e_idx:
                    if e>=s and (e-s+1)<=max_len:
                        cand.append((int(s), int(e)))
            spans.append(cand if cand else [(0, int(attention_mask[b].sum())-1)])
        return spans

    # ------------------ span encoding
    def encode_spans(self, Hs, spans, knowledge_vec=None):
        B, T, H = Hs.size()
        reps, span_map = [], []

        for b in range(B):
            hb = Hs[b]               # [T, H]
            kv = knowledge_vec[b] if knowledge_vec is not None else None  # [2H]

            for (s, e) in spans[b]:
                sv, ev = hb[s], hb[e]

                # left/right context
                lv = hb[max(0, s-2):s].mean(0) if s > 0 else torch.zeros_like(sv)
                rv = hb[e+1:min(T, e+3)].mean(0) if e < T-1 else torch.zeros_like(ev)

                # 4H → 2H projection (context)
                ctx4 = torch.cat([sv, ev, lv, rv], dim=-1)
                base2 = torch.cat([sv, ev], dim=-1)  # 2H
                ctx2  = self.span_ctx_proj(ctx4)     # 2H
                g_ctx = torch.sigmoid(self.ctx_gate(ctx4))
                rep2H = base2 + g_ctx * (ctx2 - base2)
            
                # --- Knowledge refinement ---
                if kv is not None and self.use_span_refine:
                    gate_in = torch.cat([rep2H, kv], dim=-1)
                    g_k = torch.sigmoid(self.knowledge_gate(gate_in))
                    rep2H = rep2H + g_k * (kv - rep2H)

                reps.append(rep2H)
                span_map.append((b, s, e))

        if len(reps) == 0:
            return torch.zeros(0, 2 * H), []

        return torch.stack(reps), span_map


    # ------------------ region fusion
    def apply_region(self, span_reps, span_map, region_feats, region_boxes, region_mask):
        if region_feats is None: return span_reps
        new=[]
        for i,(b,s,e) in enumerate(span_map):
            rep = span_reps[i:i+1]
            fused = self.region_cross(rep, region_feats[b:b+1], region_boxes[b:b+1], region_mask[b:b+1])
            new.append(fused.squeeze(0))
        return torch.stack(new)

    # ------------------ span→token injection
    def inject_spans(self, Hs, span_reps, span_map):
        if span_reps.size(0) == 0:
            return Hs
        B, T, H = Hs.size()
        fused = torch.zeros_like(Hs)
        cnt = torch.zeros(B, T, device=Hs.device)
        proj = self.span_to_token(span_reps)
        for i, (b, s, e) in enumerate(span_map):
            fused[b, s:e+1] += proj[i]
            cnt[b, s:e+1] += 1
        cnt = cnt.clamp_min(1).unsqueeze(-1)
        return Hs + fused / cnt

    # ------------------ token classifier
    def predict_token_bio(self, Hs):
        return self.token_head(Hs)

    def compute_token_loss(self, token_logits, labels, attention_mask):
        labels = labels.masked_fill(attention_mask == 0, -100)
        return F.cross_entropy(
            token_logits.view(-1, token_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
    def contrastive_loss(self, span_reps, knowledge_vec, span_map):
        if span_reps.size(0) == 0:
            return 0.0
        span_norm = F.normalize(span_reps, dim=-1)     # [N, 2H]
        know_norm = F.normalize(knowledge_vec, dim=-1) # [B, 2H]

        # similarity matrix: [N, B]
        logits = torch.matmul(span_norm, know_norm.t()) / self.temperature
        labels = torch.tensor([b for (b,_,_) in span_map], device=logits.device)

        return F.cross_entropy(logits, labels)

    # ------------------ forward (full pipeline)
    def forward(
        self,
        input_ids,
        attention_mask,
        start_labels=None,
        end_labels=None,
        region_feats=None,
        region_boxes=None,
        region_mask=None,
        knowledge_texts=None,
        token_labels=None,
        max_span_len=30,
        thr_start=0.6,
        thr_end=0.6
    ):
        device = input_ids.device
        Hs = self.encode_text(input_ids, attention_mask)

        # ---- Knowledge
        if self.use_knowledge and knowledge_texts is not None:
            k_hidden, k_mask, knowledge_vec = self.encode_knowledge(knowledge_texts, device)
            if k_hidden is not None:
                Hs = Hs + self.know_cross(Hs, k_hidden, k_mask)

        # ---- Boundary
        start_logits, end_logits = self.predict_boundaries(Hs)

        # # Focal loss for boundaries (only if labels provided)
        total_loss = 0.0
        if start_labels is not None:
            total_loss += sigmoid_focal_with_logits(start_logits, start_labels.float())

        if end_labels is not None:
            total_loss += sigmoid_focal_with_logits(end_logits, end_labels.float())

        # ---- Decode spans
        spans = self.decode_spans(start_logits, end_logits, attention_mask,
                                  max_len=max_span_len,
                                  thr_s=thr_start,
                                  thr_e=thr_end)

        # ---- Span contextual encoding
        span_reps, span_map = self.encode_spans(Hs, spans, knowledge_vec=knowledge_vec)

        # ---- Region
        span_reps = self.apply_region(span_reps, span_map, region_feats, region_boxes, region_mask)

        # ---- Inject spans → tokens
        Hs_fused = self.inject_spans(Hs, span_reps, span_map)

        # ---- Final token predictions
        token_logits = self.predict_token_bio(Hs_fused)

        if token_labels is not None:
            total_loss += self.compute_token_loss(token_logits, token_labels, attention_mask)
        if self.use_span_cls and span_reps.size(0) > 0 and token_labels is not None:
            span_logits = self.span_cls_head(span_reps)

            span_labels = []
            for (b, s, e) in span_map:
                lab_ids = token_labels[b, s:e+1]
                lab_ids = lab_ids[lab_ids != -100]
                if len(lab_ids) == 0:
                    span_labels.append(0)
                else:
                    span_labels.append(int(lab_ids[0]))

            span_labels = torch.tensor(span_labels, device=span_logits.device)

            span_loss = F.cross_entropy(span_logits, span_labels)
            total_loss += self.lambda_span * span_loss


        return {
            "loss": total_loss,
            "token_logits": token_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "spans": spans,
            "span_reps": span_reps,
        }
