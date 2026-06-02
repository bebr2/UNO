import argparse
import copy
import json
import os
from collections import Counter


HOME_PATH = os.getenv("HOME_PATH", ".")


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(HOME_PATH, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def dump_json(data, path):
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)


def index_by_id(items, file_path):
    indexed = {}
    for item in items:
        item_id = item.get("id")
        if item_id is None:
            raise ValueError(f"Missing id in {file_path}")
        indexed[item_id] = item
    return indexed


def choose_primary_modes(cluster_to_ids, win_rate_path, win_rate_threshold):
    win_rates = load_json(win_rate_path)
    cluster_to_mode = {}

    for cluster_key in cluster_to_ids:
        cluster_rates = win_rates.get(cluster_key, {})
        best_checkpoint = None
        best_win_rate = -1.0

        for checkpoint, win_rate in cluster_rates.items():
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_checkpoint = checkpoint

        if best_checkpoint is None or best_win_rate < win_rate_threshold:
            cluster_to_mode[cluster_key] = "init"
            continue

        checkpoint_steps = sorted(int(name.split("-")[-1]) for name in cluster_rates)
        best_epoch = checkpoint_steps.index(int(best_checkpoint.split("-")[-1]))
        cluster_to_mode[cluster_key] = best_epoch

    return cluster_to_mode


def apply_reflective_selection(cluster_to_mode, novelty_path, novel_min_score):
    novelty_scores = load_json(novelty_path)
    uno_single = copy.deepcopy(cluster_to_mode)
    uno = copy.deepcopy(cluster_to_mode)

    for cluster_key in cluster_to_mode:
        if cluster_key not in novelty_scores:
            raise ValueError(f"Missing novelty score for cluster {cluster_key} in {novelty_path}")

        min_novelty_score = novelty_scores[cluster_key].get("min_novelty_score", 0.0)
        if min_novelty_score >= novel_min_score:
            uno_single[cluster_key] = "init"
            uno[cluster_key] = "reverse"
        elif uno_single[cluster_key] == "init":
            uno[cluster_key] = "reverse"

    return uno_single, uno


def load_epoch_results(cluster_dir, epoch, cache, allow_epoch0_fallback=False):
    if epoch not in cache:
        path = os.path.join(cluster_dir, f"test_results_epoch{epoch}.json")
        if not os.path.exists(path) and allow_epoch0_fallback:
            path = os.path.join(cluster_dir, "test_results_epoch0.json")
        cache[epoch] = index_by_id(load_json(path), path)
    return cache[epoch]


def build_outputs(test_items, modes, init_by_id, reverse_by_id, cluster_dir, allow_epoch0_fallback=False):
    epoch_cache = {}
    outputs = []
    source_counts = Counter()

    for item in test_items:
        item_id = item["id"]
        cluster_key = item["cluster_key"]
        mode = modes[cluster_key]

        if mode == "init":
            if item_id not in init_by_id:
                raise ValueError(f"Missing init result for {item_id}")
            selected = copy.deepcopy(init_by_id[item_id])
            source = "init"
            source_file = "test_results.json"
        elif mode == "reverse":
            if item_id not in reverse_by_id:
                raise ValueError(f"Missing reflective result for {item_id}")
            selected = copy.deepcopy(reverse_by_id[item_id])
            source = "reflective"
            source_file = "test_results_reverse.json"
        else:
            epoch_results = load_epoch_results(
                cluster_dir=cluster_dir,
                epoch=mode,
                cache=epoch_cache,
                allow_epoch0_fallback=allow_epoch0_fallback,
            )
            if item_id not in epoch_results:
                raise ValueError(f"Missing primary epoch {mode} result for {item_id}")
            selected = copy.deepcopy(epoch_results[item_id])
            source = "primary"
            source_file = f"test_results_epoch{mode}.json"

        selected["selected_cluster_key"] = cluster_key
        selected["selected_path"] = source
        selected["selected_mode"] = mode
        selected["selected_source_file"] = source_file
        outputs.append(selected)
        source_counts[source] += 1

    return outputs, dict(source_counts)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate primary, reflective, and init outputs into UNO and UNO-Single result files."
    )
    parser.add_argument("--model", type=str, default="Qwen3-8B")
    parser.add_argument("--data_type", type=str, default="Long-Long")
    parser.add_argument("--output_root_path", type=str, required=True)
    parser.add_argument("--init_root_path", type=str)
    parser.add_argument("--cluster_dir_name", type=str, default="cluster")
    parser.add_argument("--win_rate_threshold", type=float, default=0.53)
    parser.add_argument("--novel_min_score", type=float, default=0.45)
    parser.add_argument("--reverse_file_name", type=str, default="test_results_reverse.json")
    parser.add_argument(
        "--allow_epoch0_fallback",
        action="store_true",
        help="Use test_results_epoch0.json if the selected primary epoch file is absent. Intended for SMALL_TEST only.",
    )
    args = parser.parse_args()

    if args.init_root_path is None:
        if args.model == "Qwen3-8B":
            args.init_root_path = "./UNO/code/output_init_0"
        else:
            args.init_root_path = f"./UNO/code/output_{args.model}_init_0"

    output_task_dir = os.path.join(resolve_path(args.output_root_path), args.data_type)
    cluster_dir = os.path.join(output_task_dir, args.cluster_dir_name)
    init_cluster_dir = os.path.join(resolve_path(args.init_root_path), args.data_type, args.cluster_dir_name)

    test_config_path = os.path.join(cluster_dir, "test_set_clustered_qa_epoch0.json")
    if not os.path.exists(test_config_path):
        test_config_path = os.path.join(cluster_dir, "test_set_clustered_qa.json")
    test_items = load_json(test_config_path)

    cluster_to_ids = {}
    for item in test_items:
        cluster_to_ids.setdefault(item["cluster_key"], []).append(item["id"])

    primary_modes = choose_primary_modes(
        cluster_to_ids=cluster_to_ids,
        win_rate_path=os.path.join(cluster_dir, "cluster_to_win_rate_by_bleu.json"),
        win_rate_threshold=args.win_rate_threshold,
    )
    uno_single_modes, uno_modes = apply_reflective_selection(
        cluster_to_mode=primary_modes,
        novelty_path=os.path.join(cluster_dir, "common_rules_info_scores.json"),
        novel_min_score=args.novel_min_score,
    )

    init_path = os.path.join(init_cluster_dir, "test_results.json")
    reverse_path = os.path.join(cluster_dir, args.reverse_file_name)
    if not os.path.exists(reverse_path) and args.reverse_file_name == "test_results_reverse.json":
        reverse_path = os.path.join(cluster_dir, "test_results_reverse_small.json")

    init_by_id = index_by_id(load_json(init_path), init_path)
    reverse_by_id = index_by_id(load_json(reverse_path), reverse_path)

    uno_single_outputs, uno_single_counts = build_outputs(
        test_items=test_items,
        modes=uno_single_modes,
        init_by_id=init_by_id,
        reverse_by_id=reverse_by_id,
        cluster_dir=cluster_dir,
        allow_epoch0_fallback=args.allow_epoch0_fallback,
    )
    uno_outputs, uno_counts = build_outputs(
        test_items=test_items,
        modes=uno_modes,
        init_by_id=init_by_id,
        reverse_by_id=reverse_by_id,
        cluster_dir=cluster_dir,
        allow_epoch0_fallback=args.allow_epoch0_fallback,
    )

    uno_single_path = os.path.join(cluster_dir, "uno_single_test_results.json")
    uno_path = os.path.join(cluster_dir, "uno_test_results.json")
    selection_path = os.path.join(cluster_dir, "uno_path_selection.json")

    dump_json(uno_single_outputs, uno_single_path)
    dump_json(uno_outputs, uno_path)
    dump_json(
        {
            "model": args.model,
            "data_type": args.data_type,
            "win_rate_threshold": args.win_rate_threshold,
            "novel_min_score": args.novel_min_score,
            "test_config": test_config_path,
            "init_results": init_path,
            "reverse_results": reverse_path,
            "uno_single_modes": uno_single_modes,
            "uno_modes": uno_modes,
            "uno_single_counts": uno_single_counts,
            "uno_counts": uno_counts,
        },
        selection_path,
    )

    print(f"Saved UNO-Single results to {uno_single_path}")
    print(f"Saved UNO results to {uno_path}")
    print(f"Saved path selection metadata to {selection_path}")


if __name__ == "__main__":
    main()
