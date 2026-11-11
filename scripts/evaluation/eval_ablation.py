"""
Results .jsonl schema:
{
    "task_id": "string",
    "model": "string",
    "response": "string",
    "label": "string",
}
"""
import json
import re
import os
import glob
import evaluate
from fire import Fire
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

if __package__:
    from .metric.cider import Cider
    from .sbert.sbert import calculate_sbert_similarity
else:
    from metric.cider import Cider
    from sbert.sbert import calculate_sbert_similarity


def tokenize(text):
    """Modified version of tokenizer_13a specifically for commit messages."""
    tokenzier_13a = Tokenizer13a()
    processed_text = tokenzier_13a(text)
    original_tokens = processed_text.split()

    def split_symbols(token):
        modified = re.sub(r"([^a-zA-Z0-9])", r" \1 ", token)
        return modified.split()

    def split_camel_case(token):
        modified = re.sub(r"(?<!^)(?=[A-Z][a-z])", " ", token)
        return modified.split()

    new_tokens = []
    for token in original_tokens:
        symbol_parts = split_symbols(token)
        for part in symbol_parts:
            camel_parts = split_camel_case(part)
            new_tokens.extend(camel_parts)

    new_tokens = [token.lower() for token in new_tokens]

    return new_tokens


def clean_response(response: str) -> str:
    """
    Cleans the response by removing <message> and </message> markers
    and surrounding whitespace.
    """
    if response is None:
        return ""
    if not isinstance(response, str):
        response = str(response)
    cleaned = response.replace("<message>", "").replace("</message>", "")
    return cleaned.strip()


def clean_label(label: str) -> str:
    """
    Cleans the label by removing issue identifiers like [CAMEL-1234], CAMEL-1234, CAMEL-1234:, etc.
    """
    if label is None:
        return ""
    if not isinstance(label, str):
        label = str(label)
    cleaned = re.sub(r'^(\[?\w+-\d+\]?:?\s*)+', '', label)
    return cleaned.strip()


def process_single_file(result_jsonl: str, output_cleaned_jsonl: str = None, report_file: str = None):
    print(f"Processing file: {result_jsonl}")
    with open(result_jsonl, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    predictions = [clean_response(item["response"]) for item in results]
    references = [clean_label(item["label"]) for item in results]
    missing_response_count = sum(1 for item in results if item.get("response") is None)

    if output_cleaned_jsonl:
        os.makedirs(os.path.dirname(output_cleaned_jsonl), exist_ok=True)
        with open(output_cleaned_jsonl, "w", encoding="utf-8") as f_out:
            for item, pred, ref in zip(results, predictions, references):
                cleaned_entry = {
                    "task_id": item["task_id"],
                    "cleaned_label": ref,
                    "cleaned_response": pred
                }
                f_out.write(json.dumps(cleaned_entry) + "\n")
        print(f"Cleaned data saved to: {output_cleaned_jsonl}")

    google_bleu = evaluate.load("google_bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    cider = Cider()

    bleu_result = google_bleu.compute(
        predictions=predictions,
        references=[[ref] for ref in references],
        tokenizer=tokenize,
    )
    rouge_result = rouge.compute(predictions=predictions, references=references)
    meteor_result = meteor.compute(predictions=predictions, references=references)

    gts = {
        result["task_id"]: [" ".join(tokenize(clean_label(result["label"])))] for result in results
    }
    res = {
        result["task_id"]: [" ".join(tokenize(clean_response(result["response"])))] for result in results
    }

    cider_result, _ = cider.compute_score(gts, res)

    sbert_result = calculate_sbert_similarity(predictions, references)

    model_name = results[0].get('model', 'Unknown Model')

    report_lines = []
    report_lines.append(f"filename: {result_jsonl}")
    report_lines.append(f"Evaluation Report for > {model_name} <")
    report_lines.append("=" * 30)
    report_lines.append(f"Samples         : {len(results)}")
    report_lines.append(f"Missing response: {missing_response_count}")
    report_lines.append(f"BLEU           : {bleu_result['google_bleu']*100:.2f}")
    report_lines.append(f"ROUGE-L        : {rouge_result['rougeL']*100:.2f}")
    report_lines.append(f"METEOR         : {meteor_result['meteor']*100:.2f}")
    report_lines.append(f"CIDEr          : {cider_result*10:.2f}")
    report_lines.append(f"SBERT Cosine   : {sbert_result['sbert_cosine']*100:.2f}")
    report_lines.append("=" * 30)
    report_lines.append("\n")

    report_content = "\n".join(report_lines)
    print(report_content)

    if report_file:
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(report_content)


def eval(
    result_path: str,
    output_dir: str = None,
    report_file: str = "ablation_evaluation_report.txt"
):
    if os.path.exists(report_file):
        os.remove(report_file)

    if os.path.isdir(result_path):
        files = glob.glob(os.path.join(result_path, "*.jsonl"))
        if not files:
            print(f"No .jsonl files found in {result_path}")
            return

        if not output_dir:
            output_dir = os.path.join(result_path, "cleaned_results")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for file_path in files:
            filename = os.path.basename(file_path)
            output_file_path = os.path.join(output_dir, filename)
            process_single_file(file_path, output_file_path, report_file)

    elif os.path.isfile(result_path):
        process_single_file(result_path, output_dir, report_file)
    else:
        print(f"Error: {result_path} is not a valid file or directory")


if __name__ == "__main__":
    Fire(eval)
