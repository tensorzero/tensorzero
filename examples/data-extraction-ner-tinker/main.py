"""CLI entry point for the NER data extraction example using Tinker."""

import argparse

import tinker
from tinker_cookbook import renderers, tokenizer_utils

from data import load_dataset
from evaluate import evaluate_batch, print_comparison
from inference import run_inference
from train import TrainConfig, train

DEFAULT_MODEL = "Qwen/Qwen3-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NER data extraction with LoRA fine-tuning via Tinker"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Base model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default=None,
        help="Renderer name for chat template (default: auto-detect from model name)",
    )
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-val", type=int, default=500)
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline evaluation",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (baseline eval only)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to conllpp.csv (default: ../data-extraction-ner/data/conllpp.csv)",
    )
    return parser.parse_args()


def detect_renderer_name(model_name: str) -> str:
    """Guess the renderer name from the model identifier.

    Uses non-thinking variants by default since NER extraction needs clean
    JSON output without <think> tags.
    """
    lower = model_name.lower()
    if "qwen3" in lower:
        return "qwen3_disable_thinking"
    if "llama" in lower:
        return "llama3"
    if "deepseek" in lower:
        return "deepseekv3"
    # Default fallback
    return "llama3"


def main() -> None:
    args = parse_args()

    # ── Load dataset ──────────────────────────────────────────────
    load_kwargs = {"num_train": args.num_train, "num_val": args.num_val}
    if args.data_path:
        load_kwargs["path"] = args.data_path

    train_df, val_df = load_dataset(**load_kwargs)
    print(f"Train: {len(train_df)} examples, Val: {len(val_df)} examples")

    # ── Tinker setup ──────────────────────────────────────────────
    service_client = tinker.ServiceClient()
    tokenizer = tokenizer_utils.get_tokenizer(args.model)
    renderer_name = args.renderer or detect_renderer_name(args.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Model: {args.model}  (renderer: {renderer_name})")

    results: dict[str, dict[str, float]] = {}

    # ── Baseline evaluation ───────────────────────────────────────
    if not args.skip_baseline:
        print("\n--- Baseline Evaluation ---")
        baseline_sampling = service_client.create_sampling_client(
            base_model=args.model,
        )
        baseline_preds = run_inference(
            baseline_sampling, renderer, list(val_df["input"])
        )
        baseline_metrics = evaluate_batch(baseline_preds, list(val_df["output"]))
        results["baseline"] = baseline_metrics
        print(
            f"Baseline — valid: {baseline_metrics['valid_output']:.1%}, "
            f"exact_match: {baseline_metrics['exact_match']:.1%}, "
            f"jaccard: {baseline_metrics['jaccard_similarity']:.1%}"
        )

    # ── Fine-tuning ───────────────────────────────────────────────
    if not args.skip_training:
        print("\n--- LoRA Fine-Tuning ---")
        config = TrainConfig(
            rank=args.rank,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        ft_sampling = train(
            service_client, args.model, renderer, train_df, config
        )

        print("\n--- Fine-Tuned Evaluation ---")
        ft_preds = run_inference(ft_sampling, renderer, list(val_df["input"]))
        ft_metrics = evaluate_batch(ft_preds, list(val_df["output"]))
        results["fine-tuned"] = ft_metrics
        print(
            f"Fine-tuned — valid: {ft_metrics['valid_output']:.1%}, "
            f"exact_match: {ft_metrics['exact_match']:.1%}, "
            f"jaccard: {ft_metrics['jaccard_similarity']:.1%}"
        )

    # ── Comparison ────────────────────────────────────────────────
    if results:
        print_comparison(results)


if __name__ == "__main__":
    main()
