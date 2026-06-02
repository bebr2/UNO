import argparse
import json
import os


def build_sft_rules(rules_root_path, small_test=False):
    dataset_names = sorted(
        entry.name for entry in os.scandir(rules_root_path) if entry.is_dir()
    )

    for dataset_name in dataset_names:
        rules_path = os.path.join(rules_root_path, dataset_name, "with_feedback_rules.json")
        if not os.path.exists(rules_path):
            continue

        with open(rules_path, "r", encoding="utf-8") as fin:
            rules_data = json.load(fin)

        results = []
        for item in rules_data:
            try:
                rules = item["_raw_response"]["rules"]
                prompt = (
                    "Provide suggestions to make the answer better. Your output must be some rules, "
                    "using number to order them.\n\n"
                    f"## User Query\n\n{item['q']}\n\n"
                    f"## Initial Answer\n\n{item['old_a']}"
                )
                answer = "\n".join(f"{idx + 1}. {rule}" for idx, rule in enumerate(rules))
                results.append({
                    "test_idx": item["test_idx"],
                    "prompt": prompt,
                    "answer": answer,
                })
            except (KeyError, TypeError):
                continue

        if small_test:
            results = results[:20]

        output_path = os.path.join(rules_root_path, dataset_name, "sft_rules.json")
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(results, fout, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Convert distilled feedback rules into SFT data for the reflective Critic LoRA."
    )
    parser.add_argument("rules_root_path", help="Directory containing per-dataset with_feedback_rules.json files.")
    parser.add_argument(
        "small_test",
        nargs="?",
        default="False",
        help="Whether to keep only a small subset. Accepts True/False.",
    )
    args = parser.parse_args()

    build_sft_rules(args.rules_root_path, args.small_test.lower() == "true")


if __name__ == "__main__":
    main()
