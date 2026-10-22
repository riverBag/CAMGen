#!/usr/bin/env python3
"""
对每个 repo 的 core entities 及其相似实体进行聚类。

输入：core_entities.jsonl（enriched core entities，兼容旧拼写 core_eneities.jsonl）、filtered_entities.json
流程：
1. 用 CodeBERT 对所有 entities 的 code 向量化
2. 对 embedding 得到的向量进行 L2 normalize
3. normalize 后进行 UMAP 降维
4. 对降维后的向量进行 HDBSCAN 聚类
5. 输出若干包含 core entity 的 clusters
6. 仅保留含至少一个 core entity 的 cluster，不含 core entity 的 cluster 丢弃

输出：clusters.json，记录每个 cluster 的 entity 数量、core entity 数量、entities 列表（格式同 filtered_entities.json）

支持断点续跑：已存在 clusters.json 的 repo 跳过。
"""

# 强制离线：在导入 transformers 之前设置
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import logging
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = PROJECT_ROOT.parent.parent
DEFAULT_REPOS_DIR = PACKAGE_ROOT / "outputs" / "view2" / "repositories"
FILTERED_FILE = "filtered_entities.json"
CORE_ENTITIES_JSONL = "core_entities.jsonl"
OUTPUT_FILE = "clusters_hdbscan.json"
LOG_FILE = PACKAGE_ROOT / "outputs" / "view2" / "cluster_core_entities_umap.log"

CODEBERT_MODEL = "microsoft/codebert-base"
MAX_CODE_LENGTH = 512
DEFAULT_BATCH_SIZE = 64

# UMAP 默认参数
UMAP_N_COMPONENTS = 20
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.0
# HDBSCAN 默认参数
HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES = 1


def load_core_entities(repo_dir: Path) -> List[Dict[str, Any]] | None:
    """加载 enriched core entities，兼容旧文件名；错误记录返回 None。"""
    path = repo_dir / CORE_ENTITIES_JSONL
    if not path.is_file():
        path = repo_dir / "core_eneities.jsonl"  # Accept the original filename spelling.
    if not path.is_file():
        return None
    entities = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("error") is True:
                return None
            entities.append(obj)
    return entities if entities else None


def load_filtered_entities(repo_dir: Path) -> List[Dict[str, Any]] | None:
    """加载 filtered_entities.json 的 entities 列表。"""
    path = repo_dir / FILTERED_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("entities") or []
    except Exception:
        return None


def load_codebert_runtime(model_name: str = CODEBERT_MODEL):
    """只加载一次 CodeBERT runtime，返回 tokenizer/model/device/torch。"""
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
    except ImportError as e:
        raise ImportError("需要安装 transformers 和 torch: pip install transformers torch") from e

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device, torch


def get_codebert_embeddings(
    entities: List[Dict[str, Any]],
    tokenizer,
    model,
    device: str,
    torch_mod,
    max_length: int = MAX_CODE_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """用已加载的 CodeBERT runtime 对 entities 的 code 向量化，返回向量和阶段耗时。"""

    t_prepare_start = time.perf_counter()
    texts = []
    for ent in entities:
        code = ent.get("code") or ""
        if not isinstance(code, str):
            code = str(code)
        texts.append(code[:5000])
    prepare_sec = time.perf_counter() - t_prepare_start

    use_cuda = (device == "cuda")
    gpu_embeddings = []
    tokenize_sec = 0.0
    h2d_sec = 0.0
    forward_sec = 0.0
    d2h_sec = 0.0
    cat_sec = 0.0
    t_embed_start = time.perf_counter()
    with torch_mod.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            t0 = time.perf_counter()
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenize_sec += time.perf_counter() - t0

            if use_cuda:
                torch_mod.cuda.synchronize()
            t1 = time.perf_counter()
            moved = {}
            for k, v in enc.items():
                if use_cuda:
                    v = v.pin_memory()
                moved[k] = v.to(device, non_blocking=use_cuda)
            enc = moved
            if use_cuda:
                torch_mod.cuda.synchronize()
            h2d_sec += time.perf_counter() - t1

            if use_cuda:
                torch_mod.cuda.synchronize()
            t2 = time.perf_counter()
            if use_cuda:
                with torch_mod.autocast(device_type="cuda", dtype=torch_mod.float16):
                    out = model(**enc)
            else:
                out = model(**enc)
            if use_cuda:
                torch_mod.cuda.synchronize()
            forward_sec += time.perf_counter() - t2

            # 保留在 GPU，避免每个 batch 都做一次 D2H 同步
            pool = out.last_hidden_state[:, 0, :]
            gpu_embeddings.append(pool)

    if use_cuda:
        torch_mod.cuda.synchronize()
    t3 = time.perf_counter()
    all_pool = torch_mod.cat(gpu_embeddings, dim=0)
    if use_cuda:
        torch_mod.cuda.synchronize()
    cat_sec = time.perf_counter() - t3

    if use_cuda:
        torch_mod.cuda.synchronize()
    t4 = time.perf_counter()
    embeddings = all_pool.float().cpu().numpy()
    if use_cuda:
        torch_mod.cuda.synchronize()
    d2h_sec = time.perf_counter() - t4
    embed_total_sec = time.perf_counter() - t_embed_start

    stats = {
        "batch_size": float(batch_size),
        "prepare_sec": prepare_sec,
        "tokenize_sec": tokenize_sec,
        "h2d_sec": h2d_sec,
        "forward_sec": forward_sec,
        "cat_sec": cat_sec,
        "d2h_sec": d2h_sec,
        "embed_total_sec": embed_total_sec,
    }
    return embeddings, stats


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """L2 归一化，按行。"""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def entity_key(ent: Dict[str, Any]) -> str:
    """唯一标识 entity。"""
    return ent.get("qualifiedName") or str(ent.get("id", ""))


def process_repo(
    repo_dir: Path,
    tokenizer,
    model,
    device: str,
    torch_mod,
    umap_n_components: int = UMAP_N_COMPONENTS,
    umap_n_neighbors: int = UMAP_N_NEIGHBORS,
    umap_min_dist: float = UMAP_MIN_DIST,
    hdbscan_min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    hdbscan_min_samples: int = HDBSCAN_MIN_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> bool:
    """处理单个 repo，返回是否成功。"""
    name = repo_dir.name
    out_path = repo_dir / OUTPUT_FILE
    if out_path.is_file():
        logging.info("已存在 %s，跳过: %s", OUTPUT_FILE, name)
        return True

    core_entities = load_core_entities(repo_dir)
    if not core_entities:
        logging.info("[%s] 无有效 core entities，跳过", name)
        return False

    all_entities = load_filtered_entities(repo_dir)
    if not all_entities:
        logging.info("[%s] 无 filtered_entities，跳过", name)
        return False

    qn_to_idx: Dict[str, int] = {}
    for i, ent in enumerate(all_entities):
        qn = entity_key(ent)
        if qn:
            qn_to_idx[qn] = i

    core_indices: List[int] = []
    for ce in core_entities:
        qn = entity_key(ce)
        if qn in qn_to_idx:
            core_indices.append(qn_to_idx[qn])
        else:
            logging.warning("[%s] core entity 不在 filtered 中: %s", name, qn[:60])

    if not core_indices:
        logging.warning("[%s] 无 core entity 可匹配，跳过", name)
        return False

    logging.info("[%s] 获取 CodeBERT 向量...", name)
    cur_bs = batch_size
    while True:
        try:
            vecs, emb_stats = get_codebert_embeddings(
                all_entities,
                tokenizer=tokenizer,
                model=model,
                device=device,
                torch_mod=torch_mod,
                batch_size=cur_bs,
            )
            logging.info(
                "[%s] CodeBERT阶段耗时: bs=%d prepare=%.2fs tokenize=%.2fs h2d=%.2fs forward=%.2fs cat=%.2fs d2h=%.2fs total=%.2fs",
                name,
                cur_bs,
                emb_stats["prepare_sec"],
                emb_stats["tokenize_sec"],
                emb_stats["h2d_sec"],
                emb_stats["forward_sec"],
                emb_stats["cat_sec"],
                emb_stats["d2h_sec"],
                emb_stats["embed_total_sec"],
            )
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and device == "cuda" and cur_bs > 8:
                if torch_mod.cuda.is_available():
                    torch_mod.cuda.empty_cache()
                next_bs = max(8, cur_bs // 2)
                if next_bs == cur_bs:
                    logging.exception("[%s] OOM 且无法继续降低 batch_size: %s", name, e)
                    return False
                logging.warning("[%s] CUDA OOM，batch_size 从 %d 降到 %d 重试", name, cur_bs, next_bs)
                cur_bs = next_bs
                continue
            logging.exception("[%s] CodeBERT 向量化失败: %s", name, e)
            return False
        except Exception as e:
            logging.exception("[%s] CodeBERT 向量化失败: %s", name, e)
            return False

    # 2. L2 normalize
    t_l2 = time.perf_counter()
    vecs = l2_normalize(vecs.astype(np.float64))
    logging.info("[%s] L2 normalize 耗时: %.2f 秒", name, time.perf_counter() - t_l2)

    logging.info("[%s] UMAP 降维 (n_components=%s)...", name, umap_n_components)
    t_umap = time.perf_counter()
    try:
        import umap
        reducer = umap.UMAP(
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric="euclidean",
            random_state=None,
            n_jobs=-1,
        )
        reduced = reducer.fit_transform(vecs)
    except ImportError as e:
        logging.exception("[%s] 未安装 umap-learn: pip install umap-learn", name)
        return False
    except Exception as e:
        logging.exception("[%s] UMAP 失败: %s", name, e)
        return False
    logging.info("[%s] UMAP 阶段耗时: %.2f 秒", name, time.perf_counter() - t_umap)

    n_entities = reduced.shape[0]
    logging.info("[%s] HDBSCAN 聚类，参与实体数: %s", name, n_entities)
    t_hdbscan = time.perf_counter()
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(reduced)
    except ImportError as e:
        logging.exception("[%s] 未安装 hdbscan: pip install hdbscan", name)
        return False
    except Exception as e:
        logging.exception("[%s] HDBSCAN 失败: %s", name, e)
        return False
    logging.info("[%s] HDBSCAN 阶段耗时: %.2f 秒", name, time.perf_counter() - t_hdbscan)

    t_post = time.perf_counter()
    core_set = set(core_indices)
    # HDBSCAN 总 cluster 数（排除噪声 label=-1）
    unique_labels = {int(l) for l in labels if l >= 0}
    logging.info("[%s] HDBSCAN 共得到 %s 个 cluster（含无 core 的，排除噪声）", name, len(unique_labels))

    # core 即使是噪声点，也不能丢弃：分配到最近的非噪声 cluster
    effective_labels = labels.copy()
    non_noise_labels = sorted(unique_labels)
    if non_noise_labels:
        centroids = {
            lb: reduced[labels == lb].mean(axis=0)
            for lb in non_noise_labels
        }
        reassigned = 0
        for ci in core_indices:
            if effective_labels[ci] >= 0:
                continue
            v = reduced[ci]
            nearest = min(
                non_noise_labels,
                key=lambda lb: np.linalg.norm(v - centroids[lb]),
            )
            effective_labels[ci] = nearest
            reassigned += 1
        if reassigned:
            logging.info("[%s] 将 %s 个噪声 core entity 重新分配到最近非噪声 cluster", name, reassigned)
    else:
        # 极端情况：HDBSCAN 全部为噪声。为 core entity 建立保底 cluster，保证 core 不丢失。
        next_label = 0
        for ci in core_indices:
            if effective_labels[ci] < 0:
                effective_labels[ci] = next_label
                next_label += 1
        logging.warning("[%s] HDBSCAN 无非噪声 cluster；已为 core entities 创建 %s 个保底 cluster", name, next_label)

    cluster_to_entities: Dict[int, List[Dict[str, Any]]] = {}
    for i, label in enumerate(effective_labels):
        if label < 0:
            continue
        label = int(label)
        if label not in cluster_to_entities:
            cluster_to_entities[label] = []
        cluster_to_entities[label].append(all_entities[i])

    core_clusters: List[Dict[str, Any]] = []
    for label, ents in cluster_to_entities.items():
        core_count = sum(1 for e in ents if qn_to_idx.get(entity_key(e)) in core_set)
        if core_count > 0:
            core_clusters.append({
                "entity_count": len(ents),
                "core_entity_count": core_count,
                "entities": ents,
            })

    if not core_clusters:
        logging.warning("[%s] 无包含 core entity 的 cluster", name)
        return False

    result = {"clusters": core_clusters}
    t_save = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logging.info("[%s] 写出 clusters.json 耗时: %.2f 秒", name, time.perf_counter() - t_save)
    logging.info("[%s] 后处理阶段耗时: %.2f 秒", name, time.perf_counter() - t_post)

    logging.info(
        "[%s] 完成: %s 个 clusters, 共 %s 个 entities",
        name,
        len(core_clusters),
        sum(c["entity_count"] for c in core_clusters),
    )
    return True


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="对 core entities 做 CodeBERT + UMAP + HDBSCAN 聚类")
    parser.add_argument("-d", "--repos-dir", type=Path, default=DEFAULT_REPOS_DIR, help="目标 repo 文件夹路径")
    parser.add_argument("--umap-components", type=int, default=UMAP_N_COMPONENTS, help="UMAP 降维维度")
    parser.add_argument("--umap-neighbors", type=int, default=UMAP_N_NEIGHBORS, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=UMAP_MIN_DIST, help="UMAP min_dist")
    parser.add_argument("--hdbscan-min-size", type=int, default=HDBSCAN_MIN_CLUSTER_SIZE, help="HDBSCAN min_cluster_size")
    parser.add_argument("--hdbscan-min-samples", type=int, default=HDBSCAN_MIN_SAMPLES, help="HDBSCAN min_samples")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="CodeBERT 推理 batch size（OOM 自动回退）")
    args = parser.parse_args()

    repos_dir = args.repos_dir
    if not repos_dir.is_absolute():
        repos_dir = repos_dir.resolve()
    else:
        repos_dir = repos_dir.resolve()

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    if not repos_dir.is_dir():
        logging.error("目标文件夹不存在: %s", repos_dir)
        sys.exit(1)

    repo_dirs = sorted(p for p in repos_dir.iterdir() if p.is_dir())
    total = len(repo_dirs)
    done = 0
    skipped = 0
    failed = 0

    logging.info("目标文件夹: %s, 共 %s 个 repo", repos_dir, total)
    logging.info("加载 CodeBERT 模型（仅一次）...")
    try:
        tokenizer, model, device, torch_mod = load_codebert_runtime(CODEBERT_MODEL)
        logging.info("CodeBERT 已加载到设备: %s", device)
    except Exception as e:
        logging.exception("CodeBERT 初始化失败: %s", e)
        sys.exit(1)

    for idx, repo_dir in enumerate(repo_dirs, start=1):
        name = repo_dir.name
        logging.info("[%s/%s] 处理: %s", idx, total, name)
        t0 = time.perf_counter()
        try:
            if process_repo(
                repo_dir,
                tokenizer=tokenizer,
                model=model,
                device=device,
                torch_mod=torch_mod,
                umap_n_components=args.umap_components,
                umap_n_neighbors=args.umap_neighbors,
                umap_min_dist=args.umap_min_dist,
                hdbscan_min_cluster_size=args.hdbscan_min_size,
                hdbscan_min_samples=args.hdbscan_min_samples,
                batch_size=args.batch_size,
            ):
                done += 1
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            logging.exception("[%s/%s] 失败 %s: %s", idx, total, name, e)
        elapsed = time.perf_counter() - t0
        logging.info("[%s/%s] %s 耗时: %.2f 秒", idx, total, name, elapsed)

    logging.info("结束: 成功 %s, 跳过 %s, 失败 %s", done, skipped, failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
