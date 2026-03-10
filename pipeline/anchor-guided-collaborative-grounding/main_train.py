from src.dataset import GroundingDataset, InferenceDataset
from src.collator import collate_fn, collate_fn_infer
from src.model_origin import GroundingModel, train_one_epoch
from src.evaluate import evaluate_dev, inference_and_save
from config import train_json, dev_json, test_json, npz_dir, img_dir
import torch
from torch.utils.data import DataLoader
import os
import random
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "best_model.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    def format_bucket_acc(bucket_acc):
        if not bucket_acc:
            return "none"
        parts = []
        for k in sorted(bucket_acc.keys()):
            s = bucket_acc[k]
            parts.append(
                f"E={k}:acc={s['acc']:.3f}({s['correct']}/{s['total']},n={s['samples']})"
            )
        return " | ".join(parts)

    try:
        from config import use_xml_clip_regions
    except Exception:
        use_xml_clip_regions = False
    try:
        from config import use_clip_region_encoder
    except Exception:
        use_clip_region_encoder = False
    try:
        from config import xml_dir
    except Exception:
        xml_dir = None
    try:
        from config import clip_model_name
    except Exception:
        clip_model_name = "openai/clip-vit-base-patch32"
    try:
        from config import clip_device
    except Exception:
        clip_device = "cpu"

    region_dim_default = "768" if (bool(use_xml_clip_regions) or bool(use_clip_region_encoder)) else "2048"
    region_dim = int(os.getenv("REGION_DIM", region_dim_default))

    train_ds = GroundingDataset(
        train_json, npz_dir, img_dir,
        use_xml_clip_regions=use_xml_clip_regions,
        use_clip_region_encoder=use_clip_region_encoder,
        xml_dir=xml_dir,
        clip_model_name=clip_model_name,
        clip_device=clip_device,
    )
    dev_ds   = GroundingDataset(
        dev_json, npz_dir, img_dir,
        use_xml_clip_regions=use_xml_clip_regions,
        use_clip_region_encoder=use_clip_region_encoder,
        xml_dir=xml_dir,
        clip_model_name=clip_model_name,
        clip_device=clip_device,
    )
    test_ds  = GroundingDataset(
        test_json, npz_dir, img_dir,
        use_xml_clip_regions=use_xml_clip_regions,
        use_clip_region_encoder=use_clip_region_encoder,
        xml_dir=xml_dir,
        clip_model_name=clip_model_name,
        clip_device=clip_device,
    )

    batch_size = int(os.getenv("BATCH_SIZE", "4"))
    epochs = int(os.getenv("EPOCHS", "6"))
    lr = float(os.getenv("LR", "1e-5")) # 1e-5
    temperature = float(os.getenv("TEMPERATURE", "0.8"))
    seed = int(os.getenv("SEED", "13"))
    save_best = os.getenv("SAVE_BEST", "0").lower() in {"1", "true", "yes", "y"}
    save_path = os.getenv("SAVE_PATH", SAVE_PATH)
    test_save_path = os.getenv("TEST_SAVE_PATH", "").strip()

    ce_w = float(os.getenv("LOSS_CE_W", "1.0"))
    cons_w = float(os.getenv("LOSS_CONS_W", "0.3"))
    excl_w = float(os.getenv("LOSS_EXCL_W", "0.5"))
    margin_w = float(os.getenv("LOSS_MARGIN_W", "1.0"))
    info_nce_w = float(os.getenv("LOSS_INFO_NCE_W", "0.0"))
    info_nce_tau = float(os.getenv("INFO_NCE_TAU", "0.07"))
    info_nce_topk_neg = int(os.getenv("INFO_NCE_TOPK_NEG", "0"))
    cons_alpha = float(os.getenv("CONS_ALPHA", "0.65"))
    cons_sim_thr = float(os.getenv("CONS_SIM_THR", "0.5"))
    cons_target_pool = os.getenv("CONS_TARGET_POOL", "mean").strip().lower()
    cons_hard_topk = int(os.getenv("CONS_HARD_TOPK", "0"))
    cons_include_intra = os.getenv("CONS_INCLUDE_INTRA", "0").lower() in {"1", "true", "yes", "y"}
    cons_weighted_sim = os.getenv("CONS_WEIGHTED_SIM", "0").lower() in {"1", "true", "yes", "y"}
    cons_sim_tau = float(os.getenv("CONS_SIM_TAU", "0.2"))
    cons_sim_min = float(os.getenv("CONS_SIM_MIN", "-1.0"))
    cons_sim_weight_cap = float(os.getenv("CONS_SIM_WEIGHT_CAP", "5.0"))
    excl_margin = float(os.getenv("EXCL_MARGIN", "0.3"))
    excl_same_type_w = float(os.getenv("EXCL_SAME_TYPE_W", "1.0"))
    excl_diff_type_w = float(os.getenv("EXCL_DIFF_TYPE_W", "1.0"))
    excl_topk = int(os.getenv("EXCL_TOPK", "0"))
    excl_mode = os.getenv("EXCL_MODE", "relu_margin").strip().lower()
    excl_variant = os.getenv("EXCL_VARIANT", "anchor").strip().lower()
    excl_ent_name_w = float(os.getenv("EXCL_ENT_NAME_W", "0.7"))
    excl_ent_type_w = float(os.getenv("EXCL_ENT_TYPE_W", "0.3"))
    excl_pair_alpha = float(os.getenv("EXCL_PAIR_ALPHA", "0.5"))
    excl_pair_beta = float(os.getenv("EXCL_PAIR_BETA", "0.3"))
    excl_pair_region_sim = os.getenv("EXCL_PAIR_REGION_SIM", "none").strip().lower()
    excl_pair_type_mode = os.getenv("EXCL_PAIR_TYPE_MODE", "neutral").strip().lower()
    excl_pair_type_factor = float(os.getenv("EXCL_PAIR_TYPE_FACTOR", "1.5"))
    excl_pair_name_thr = float(os.getenv("EXCL_PAIR_NAME_THR", "0.5"))
    excl_pair_region_thr = float(os.getenv("EXCL_PAIR_REGION_THR", "0.5"))
    excl_pair_easy_scale = float(os.getenv("EXCL_PAIR_EASY_SCALE", "0.5"))
    excl_pair_hard_scale = float(os.getenv("EXCL_PAIR_HARD_SCALE", "1.5"))
    rank_margin = float(os.getenv("RANK_MARGIN", "0.5"))
    rank_topk = int(os.getenv("RANK_TOPK", "5"))
    ung_push_w = float(os.getenv("LOSS_UNG_PUSH_W", "0.0"))
    ung_push_margin = float(os.getenv("UNG_PUSH_MARGIN", "0.0"))
    rt_pull_w = float(os.getenv("RT_PULL_W", "0.0"))
    rt_push_w = float(os.getenv("RT_PUSH_W", "0.0"))
    rt_neg_margin = float(os.getenv("RT_NEG_MARGIN", "0.2"))
    rt_push_sim_weighted = os.getenv("RT_PUSH_SIM_WEIGHTED", "0").lower() in {"1", "true", "yes", "y"}
    rt_push_topk = int(os.getenv("RT_PUSH_TOPK", "0"))
    rt_push_weight_power = float(os.getenv("RT_PUSH_WEIGHT_POWER", "1.0"))
    rt_push_detach_weights = os.getenv("RT_PUSH_DETACH_WEIGHTS", "1").lower() in {"1", "true", "yes", "y"}
    rt_groundability_weighted = os.getenv("RT_GROUNDABILITY_WEIGHTED", "0").lower() in {"1", "true", "yes", "y"}
    rt_groundability_power = float(os.getenv("RT_GROUNDABILITY_POWER", "1.0"))
    rt_groundability_min = float(os.getenv("RT_GROUNDABILITY_MIN", "0.0"))
    rt_groundability_detach = os.getenv("RT_GROUNDABILITY_DETACH", "1").lower() in {"1", "true", "yes", "y"}
    rt_mid_pull_w = float(os.getenv("RT_MID_PULL_W", "0.0"))
    rt_mid_alpha = float(os.getenv("RT_MID_ALPHA", "0.5"))
    anchor_align_w = float(os.getenv("LOSS_ANCHOR_ALIGN_W", "0.0"))
    anchor_align_margin = float(os.getenv("ANCHOR_ALIGN_MARGIN", "0.0"))
    anchor_align_target_pool = os.getenv("ANCHOR_ALIGN_TARGET_POOL", "max").strip().lower()
    anchor_align_detach_target = os.getenv("ANCHOR_ALIGN_DETACH_TARGET", "1").lower() in {"1", "true", "yes", "y"}
    anchor_align_include_anchor_entity = os.getenv("ANCHOR_ALIGN_INCLUDE_ANCHOR_ENTITY", "0").lower() in {"1", "true", "yes", "y"}
    proto_cons_w = float(os.getenv("LOSS_PROTO_CONS_W", "0.0"))
    proto_cons_alpha = float(os.getenv("PROTO_CONS_ALPHA", "0.7"))
    proto_min_support = int(os.getenv("PROTO_MIN_SUPPORT", "2"))
    mod_proto_w = float(os.getenv("LOSS_MOD_PROTO_W", "0.0"))
    mod_proto_alpha = float(os.getenv("MOD_PROTO_ALPHA", "0.7"))
    mod_proto_min_support = int(os.getenv("MOD_PROTO_MIN_SUPPORT", "2"))
    tripod_w = float(os.getenv("LOSS_TRIPOD_W", "0.0"))
    tripod_margin = float(os.getenv("TRIPOD_MARGIN", "0.2"))
    tripod_neg_topk = int(os.getenv("TRIPOD_NEG_TOPK", "3"))
    tripod_score_weighted_neg = os.getenv("TRIPOD_SCORE_WEIGHTED_NEG", "1").lower() in {"1", "true", "yes", "y"}
    multi_proto_cons_w = float(os.getenv("LOSS_MULTI_PROTO_CONS_W", "0.0"))
    multi_proto_alpha = float(os.getenv("MULTI_PROTO_ALPHA", "0.7"))
    multi_proto_k = int(os.getenv("MULTI_PROTO_K", "3"))
    multi_proto_min_support = int(os.getenv("MULTI_PROTO_MIN_SUPPORT", "2"))
    multi_proto_assign_iters = int(os.getenv("MULTI_PROTO_ASSIGN_ITERS", "2"))
    hybrid_proto_cons_w = float(os.getenv("LOSS_HYBRID_PROTO_CONS_W", "0.0"))
    hybrid_proto_alpha = float(os.getenv("HYBRID_PROTO_ALPHA", "0.7"))
    hybrid_proto_k = int(os.getenv("HYBRID_PROTO_K", "3"))
    hybrid_proto_min_support = int(os.getenv("HYBRID_PROTO_MIN_SUPPORT", "2"))
    hybrid_proto_assign_iters = int(os.getenv("HYBRID_PROTO_ASSIGN_ITERS", "2"))
    hybrid_proto_beta = float(os.getenv("HYBRID_PROTO_BETA", "0.5"))
    anchor_select_mode = os.getenv("ANCHOR_SELECT_MODE", "mlp").strip().lower()
    anchor_topk = int(os.getenv("ANCHOR_TOPK", "1"))
    anchor_softmax_temp = float(os.getenv("ANCHOR_SOFTMAX_TEMP", "1.0"))
    anchor_use_reliability = os.getenv("ANCHOR_USE_RELIABILITY", "0").lower() in {"1", "true", "yes", "y"}
    anchor_rel_gap_temp = float(os.getenv("ANCHOR_REL_GAP_TEMP", "0.10"))
    use_multi_view = os.getenv("USE_MULTI_VIEW", "0").lower() in {"1", "true", "yes", "y"}
    mv_w_ent = float(os.getenv("MV_W_ENT", "0.65"))
    mv_w_know = float(os.getenv("MV_W_KNOW", "0.20"))
    mv_w_type = float(os.getenv("MV_W_TYPE", "0.15"))
    mv_w_ctx = float(os.getenv("MV_W_CTX", "0.0"))
    set_seed(seed)
    dl_generator = torch.Generator()
    dl_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        generator=dl_generator,
    )
    dev_loader   = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_infer, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_infer, num_workers=0)

    try:
        from config import model_name
    except Exception:
        model_name = "roberta-large"

    model = GroundingModel(
        temperature=temperature,
        model_name=model_name,
        region_dim=region_dim,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    print(
        f"[HP] batch={batch_size} epochs={epochs} lr={lr} temp={temperature} "
        f"| seed={seed} save_best={int(save_best)} "
        f"| use_xml_clip_regions={int(bool(use_xml_clip_regions))} use_clip_region_encoder={int(bool(use_clip_region_encoder))} "
        f"region_dim={region_dim} clip_model={clip_model_name} "
        f"| w(ce={ce_w},cons={cons_w},excl={excl_w},margin={margin_w}) "
        f"| w(info_nce={info_nce_w}) | info_nce_tau={info_nce_tau} info_nce_topk_neg={info_nce_topk_neg} "
        f"| w(ung_push={ung_push_w}) "
        f"| w(rt_pull={rt_pull_w},rt_push={rt_push_w},rt_mid_pull={rt_mid_pull_w},anchor_align={anchor_align_w},proto_cons={proto_cons_w},mod_proto={mod_proto_w},tripod={tripod_w},multi_proto_cons={multi_proto_cons_w},hybrid_proto_cons={hybrid_proto_cons_w}) "
        f"| cons_alpha={cons_alpha} cons_sim_thr={cons_sim_thr} "
        f"cons_target_pool={cons_target_pool} cons_hard_topk={cons_hard_topk} cons_include_intra={int(cons_include_intra)} "
        f"cons_weighted_sim={int(cons_weighted_sim)} cons_sim_tau={cons_sim_tau} cons_sim_min={cons_sim_min} cons_sim_weight_cap={cons_sim_weight_cap} "
        f"excl_margin={excl_margin} excl_same_type_w={excl_same_type_w} excl_diff_type_w={excl_diff_type_w} excl_topk={excl_topk} excl_mode={excl_mode} "
        f"excl_variant={excl_variant} excl_ent_name_w={excl_ent_name_w} excl_ent_type_w={excl_ent_type_w} "
        f"excl_pair_alpha={excl_pair_alpha} excl_pair_beta={excl_pair_beta} "
        f"excl_pair_region_sim={excl_pair_region_sim} excl_pair_type_mode={excl_pair_type_mode} excl_pair_type_factor={excl_pair_type_factor} "
        f"excl_pair_name_thr={excl_pair_name_thr} excl_pair_region_thr={excl_pair_region_thr} "
        f"excl_pair_easy_scale={excl_pair_easy_scale} excl_pair_hard_scale={excl_pair_hard_scale} "
        f"rank_margin={rank_margin} rank_topk={rank_topk} "
        f"ung_push_margin={ung_push_margin} "
        f"rt_neg_margin={rt_neg_margin} rt_push_sim_weighted={int(rt_push_sim_weighted)} "
        f"rt_push_topk={rt_push_topk} rt_push_weight_power={rt_push_weight_power} "
        f"rt_push_detach_weights={int(rt_push_detach_weights)} "
        f"rt_groundability_weighted={int(rt_groundability_weighted)} rt_groundability_power={rt_groundability_power} "
        f"rt_groundability_min={rt_groundability_min} rt_groundability_detach={int(rt_groundability_detach)} "
        f"rt_mid_alpha={rt_mid_alpha} "
        f"anchor_align_margin={anchor_align_margin} anchor_align_target_pool={anchor_align_target_pool} "
        f"anchor_align_detach_target={int(anchor_align_detach_target)} "
        f"anchor_align_include_anchor_entity={int(anchor_align_include_anchor_entity)} "
        f"proto_cons_alpha={proto_cons_alpha} proto_min_support={proto_min_support} "
        f"mod_proto_alpha={mod_proto_alpha} mod_proto_min_support={mod_proto_min_support} "
        f"tripod_margin={tripod_margin} tripod_neg_topk={tripod_neg_topk} tripod_score_weighted_neg={int(tripod_score_weighted_neg)} "
        f"multi_proto_alpha={multi_proto_alpha} multi_proto_k={multi_proto_k} "
        f"multi_proto_min_support={multi_proto_min_support} multi_proto_assign_iters={multi_proto_assign_iters} "
        f"hybrid_proto_alpha={hybrid_proto_alpha} hybrid_proto_k={hybrid_proto_k} "
        f"hybrid_proto_min_support={hybrid_proto_min_support} hybrid_proto_assign_iters={hybrid_proto_assign_iters} "
        f"hybrid_proto_beta={hybrid_proto_beta} "
        f"anchor_select_mode={anchor_select_mode} "
        f"anchor_topk={anchor_topk} anchor_softmax_temp={anchor_softmax_temp} "
        f"anchor_use_reliability={int(anchor_use_reliability)} anchor_rel_gap_temp={anchor_rel_gap_temp} "
        f"use_multi_view={int(use_multi_view)} mv_w(ent={mv_w_ent},know={mv_w_know},type={mv_w_type},ctx={mv_w_ctx})"
    )

    best_f1, best_state = 0.0, None

    for epoch in range(epochs):
        loss_dict = train_one_epoch(
            model,
            DEVICE,
            train_loader,
            opt,
            ce_w=ce_w,
            cons_w=cons_w,
            excl_w=excl_w,
            margin_w=margin_w,
            info_nce_w=info_nce_w,
            info_nce_tau=info_nce_tau,
            info_nce_topk_neg=info_nce_topk_neg,
            cons_alpha=cons_alpha,
            cons_sim_thr=cons_sim_thr,
            cons_target_pool=cons_target_pool,
            cons_hard_topk=cons_hard_topk,
            cons_include_intra=cons_include_intra,
            cons_weighted_sim=cons_weighted_sim,
            cons_sim_tau=cons_sim_tau,
            cons_sim_min=cons_sim_min,
            cons_sim_weight_cap=cons_sim_weight_cap,
            excl_margin=excl_margin,
            excl_same_type_w=excl_same_type_w,
            excl_diff_type_w=excl_diff_type_w,
            excl_topk=excl_topk,
            excl_mode=excl_mode,
            excl_variant=excl_variant,
            excl_ent_name_w=excl_ent_name_w,
            excl_ent_type_w=excl_ent_type_w,
            excl_pair_alpha=excl_pair_alpha,
            excl_pair_beta=excl_pair_beta,
            excl_pair_region_sim=excl_pair_region_sim,
            excl_pair_type_mode=excl_pair_type_mode,
            excl_pair_type_factor=excl_pair_type_factor,
            excl_pair_name_thr=excl_pair_name_thr,
            excl_pair_region_thr=excl_pair_region_thr,
            excl_pair_easy_scale=excl_pair_easy_scale,
            excl_pair_hard_scale=excl_pair_hard_scale,
            rank_margin=rank_margin,
            rank_topk=rank_topk,
            ung_push_w=ung_push_w,
            ung_push_margin=ung_push_margin,
            rt_pull_w=rt_pull_w,
            rt_push_w=rt_push_w,
            rt_neg_margin=rt_neg_margin,
            rt_push_sim_weighted=rt_push_sim_weighted,
            rt_push_topk=rt_push_topk,
            rt_push_weight_power=rt_push_weight_power,
            rt_push_detach_weights=rt_push_detach_weights,
            rt_groundability_weighted=rt_groundability_weighted,
            rt_groundability_power=rt_groundability_power,
            rt_groundability_min=rt_groundability_min,
            rt_groundability_detach=rt_groundability_detach,
            rt_mid_pull_w=rt_mid_pull_w,
            rt_mid_alpha=rt_mid_alpha,
            anchor_align_w=anchor_align_w,
            anchor_align_margin=anchor_align_margin,
            anchor_align_target_pool=anchor_align_target_pool,
            anchor_align_detach_target=anchor_align_detach_target,
            anchor_align_include_anchor_entity=anchor_align_include_anchor_entity,
            proto_cons_w=proto_cons_w,
            proto_cons_alpha=proto_cons_alpha,
            proto_min_support=proto_min_support,
            mod_proto_w=mod_proto_w,
            mod_proto_alpha=mod_proto_alpha,
            mod_proto_min_support=mod_proto_min_support,
            tripod_w=tripod_w,
            tripod_margin=tripod_margin,
            tripod_neg_topk=tripod_neg_topk,
            tripod_score_weighted_neg=tripod_score_weighted_neg,
            multi_proto_cons_w=multi_proto_cons_w,
            multi_proto_alpha=multi_proto_alpha,
            multi_proto_k=multi_proto_k,
            multi_proto_min_support=multi_proto_min_support,
            multi_proto_assign_iters=multi_proto_assign_iters,
            hybrid_proto_cons_w=hybrid_proto_cons_w,
            hybrid_proto_alpha=hybrid_proto_alpha,
            hybrid_proto_k=hybrid_proto_k,
            hybrid_proto_min_support=hybrid_proto_min_support,
            hybrid_proto_assign_iters=hybrid_proto_assign_iters,
            hybrid_proto_beta=hybrid_proto_beta,
            anchor_select_mode=anchor_select_mode,
            anchor_topk=anchor_topk,
            anchor_softmax_temp=anchor_softmax_temp,
            anchor_use_reliability=anchor_use_reliability,
            anchor_rel_gap_temp=anchor_rel_gap_temp,
            use_multi_view=use_multi_view,
            mv_w_ent=mv_w_ent,
            mv_w_know=mv_w_know,
            mv_w_type=mv_w_type,
            mv_w_ctx=mv_w_ctx,
        )

        p, r, f1, dev_bucket_acc = evaluate_dev(model, DEVICE, dev_loader, iou_thresh=0.5)
        print(
            f"[Epoch {epoch}] Loss={loss_dict['total']:.4f} "
            f"(ce={loss_dict['ce']:.4f}, cons={loss_dict['cons']:.4f}, excl={loss_dict['excl']:.4f}, "
            f"margin={loss_dict['margin']:.4f}, info_nce={loss_dict['info_nce']:.4f}, ung_push={loss_dict['ung_push']:.4f}, rt_pull={loss_dict['rt_pull']:.4f}, rt_push={loss_dict['rt_push']:.4f}, rt_mid_pull={loss_dict['rt_mid_pull']:.4f}, anchor_align={loss_dict['anchor_align']:.4f}, "
            f"proto_cons={loss_dict['proto_cons']:.4f}, mod_proto={loss_dict['mod_proto']:.4f}, tripod={loss_dict['tripod']:.4f}, multi_proto_cons={loss_dict['multi_proto_cons']:.4f}, "
            f"hybrid_proto_cons={loss_dict['hybrid_proto_cons']:.4f}) "
            f"| Dev P={p:.3f} R={r:.3f} F1={f1:.3f}"
        )
        print(f"  [Dev-by-EntCount] {format_bucket_acc(dev_bucket_acc)}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            if save_best:
                torch.save(best_state, save_path)
                print(f"  ✅ Best model saved at {save_path}")
            else:
                print("  ℹ️ Best model updated in-memory (disk save is OFF)")

    if best_state:
        model.load_state_dict(best_state)

    p, r, f1, test_bucket_acc = evaluate_dev(model, DEVICE, test_loader, iou_thresh=0.5)
    print(f"[TEST] P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print(f"[TEST-by-EntCount] {format_bucket_acc(test_bucket_acc)}")
    if test_save_path:
        inference_and_save(model, DEVICE, test_loader, save_path=test_save_path)
        print(f"[TEST] Inference jsonl saved: {test_save_path}")


if __name__ == "__main__":
    main()

