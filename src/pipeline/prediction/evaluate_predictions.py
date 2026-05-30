#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Any, List, Sequence

from datasets import Dataset

from bespokelabs import curator

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.answers import extract_prediction_answer
from common.datasets import REPO_ROOT, get_dataset_config, load_examples, resolve_column
from common.prediction_io import dump_json, find_eval_path, find_prediction_path, load_json
from config.task_prompts import CHARTMUSEUM_PROMPTS


def parse_chart_filters(chart_args: Sequence[str] | None, charts_arg: str | None) -> List[str]:
    filters: List[str] = []
    for value in chart_args or []:
        filters.extend(part.strip() for part in str(value).split(",") if part.strip())
    if charts_arg:
        filters.extend(part.strip() for part in str(charts_arg).split(",") if part.strip())
    return list(dict.fromkeys(filters))


def example_matches_chart(example: Any, filters: Sequence[str]) -> bool:
    if not filters:
        return True
    haystack_parts: List[str] = []
    for key in ("image", "chart_id", "question_id", "chart_data_json", "_image_stem"):
        value = example.get(key) if hasattr(example, "get") else None
        if value is not None:
            haystack_parts.append(str(value))
    haystack = "\n".join(haystack_parts)
    return any(chart in haystack for chart in filters)


class AnswerCompareGenerator(curator.LLM if curator is not None else object):
    def prompt(self, input: dict) -> str:
        return (
            CHARTMUSEUM_PROMPTS["COMPARE_ANSWER"]
            .replace("[QUESTION]", input["question"])
            .replace("[ANSWER1]", input["answer1"])
            .replace("[ANSWER2]", input["answer2"])
        )

    def parse(self, input: dict, response) -> dict:
        input["equal_answer"] = 1 if "yes" in str(response).lower() else 0
        return input


class AnswerEvaluator:
    def __init__(self, dataset: str, split: str, model: str, judge_model: str) -> None:
        if curator is None:
            raise RuntimeError(
                "bespokelabs is not installed. Install it in the target environment to use evaluate.py."
            )

        self.dataset = dataset
        self.split = split
        self.model = model
        self.judge_model = judge_model
        self.cfg = get_dataset_config(dataset)
        self.compare_answer_generator = AnswerCompareGenerator(
            model_name=judge_model,
            generation_params={"temperature": 0},
        )

    def construct_dataset_for_answer_compare(
        self,
        prediction_path: Path,
        indices: Sequence[int] | None = None,
    ):
        benchmark = load_examples(self.dataset, self.split)
        predictions = load_json(prediction_path)

        question_col = resolve_column(self.cfg, benchmark, "question_col", "question_cols")
        answer_col = resolve_column(self.cfg, benchmark, "answer_col", "answer_cols")

        n = min(len(benchmark), len(predictions))
        selected_indices = list(indices) if indices is not None else list(range(n))
        selected_indices = [i for i in selected_indices if i < n]
        questions: List[str] = []
        ground_truths: List[str] = []
        pred_answers: List[str] = []

        for i in selected_indices:
            ex = benchmark[i]
            questions.append(str(ex.get(question_col, "")))
            ground_truths.append(str(ex.get(answer_col, "")))
            pred_answers.append(extract_prediction_answer(self.dataset, predictions[i]))

        judge_dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer1": ground_truths,
                "answer2": pred_answers,
            }
        )
        return benchmark, judge_dataset, selected_indices

    def evaluate(self, prediction_path: Path, indices: Sequence[int] | None = None) -> dict:
        benchmark, judge_dataset, selected_indices = self.construct_dataset_for_answer_compare(
            prediction_path,
            indices=indices,
        )
        curator_cache_dir = REPO_ROOT / ".cache" / "curator"
        curator_response = self.compare_answer_generator(judge_dataset, working_dir=str(curator_cache_dir))
        equal_answer = list(curator_response.dataset["equal_answer"])
        accuracy = (sum(equal_answer) / len(equal_answer)) if equal_answer else 0.0
        return {
            "benchmark": benchmark,
            "equal_answer": equal_answer[:len(selected_indices)],
            "indices": selected_indices,
            "accuracy": accuracy,
        }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--judge-model", "--judge_model", dest="judge_model", required=True)
    ap.add_argument(
        "--chart",
        action="append",
        default=None,
        help="Chart stem/filter to evaluate and replace in place. Can be repeated or comma-separated.",
    )
    ap.add_argument(
        "--charts",
        default=None,
        help="Comma-separated chart stems/filters to evaluate and replace in place.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Evaluate a limited prediction file from a larger dataset run.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    pred_path = find_prediction_path(args.dataset, args.split, args.model, limit=args.limit)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    save_path = find_eval_path(pred_path, args.model)
    chart_filters = parse_chart_filters(args.chart, args.charts)
    selected_indices = None
    existing_eval = None

    if chart_filters:
        benchmark = load_examples(args.dataset, args.split)
        selected_indices = [
            i for i in range(len(benchmark))
            if example_matches_chart(benchmark[i], chart_filters)
        ]
        if not selected_indices:
            raise ValueError(f"No examples matched chart filter(s): {', '.join(chart_filters)}")
        if not save_path.exists():
            raise FileNotFoundError(
                "Chart-filtered eval replacement requires an existing full eval file: "
                f"{save_path}"
            )
        existing_eval = load_json(save_path)
        if not isinstance(existing_eval, list) or len(existing_eval) != len(benchmark):
            raise ValueError(
                "Chart-filtered eval replacement requires an existing full eval file with "
                f"{len(benchmark)} rows. Found {len(existing_eval) if isinstance(existing_eval, list) else 'non-list'}."
            )
        print(
            "[Chart filter] replacing "
            f"{len(selected_indices)} eval rows matching: {', '.join(chart_filters)}"
        )

    evaluator = AnswerEvaluator(
        dataset=args.dataset,
        split=args.split,
        model=args.model,
        judge_model=args.judge_model,
    )
    results = evaluator.evaluate(pred_path, indices=selected_indices)

    if chart_filters:
        merged = list(existing_eval)
        for idx, label in zip(results["indices"], results["equal_answer"]):
            merged[idx] = label
        dump_json(save_path, merged)
    else:
        dump_json(save_path, results["equal_answer"])
    print(f"Predictions: {pred_path}")
    print(f"Saved eval labels: {save_path}")
    print(f"Compared {len(results['equal_answer'])} examples")
    print(f"Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
