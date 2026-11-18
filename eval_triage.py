"""
Evaluation script for the Office Request Triage Tool
----------------------------------------------------
What it does:
    Runs every test case through office_helper.classify() (the exact function
    the tool uses), compares the AI's answer with your expected labels, and
    reports accuracy, emergencies caught, latency, tokens and cost.
    With --rag, the closest office policies are added to each prompt, and the
    report lists which policies each case retrieved.

Input:   data/eval_cases.csv
         columns: id, message, expected_category, expected_urgency, set
         (expected_category may list two acceptable answers as "A|B")
Output:  two NEW files in results/ per run (old runs are never overwritten):
             eval_<version>_<date>_<time>.csv       one row per case (RAG off)
             eval_<version>_rag_<date>_<time>.csv   one row per case (RAG on)
         plus a .txt file with the same name holding the printed summary.

Run (inside the virtual environment, key typed in hidden). Leave out --rag
for the "before RAG" score:
    source envoy_env/bin/activate
    python -c "import getpass, os, runpy; os.environ['ANTHROPIC_API_KEY'] = getpass.getpass('API key: ').strip(); runpy.run_path('eval_triage.py', run_name='__main__')" --rag

The expected labels are YOUR ground truth. Read through the case file and
change any label you disagree with before you trust the scores.
"""

import argparse
import csv
import statistics
import sys
import time
from datetime import datetime

import anthropic
import chromadb

import office_helper  # reuses classify(), review_reasons(), read_csv_rows(), settings
import policy_search  # only for the name of the emergency policy, in the report

# All test cases live in one file. Its "set" column says where each case came
# from; scores are shown for all cases AND per set, so the original 50 stay
# comparable with earlier runs.
#   original  the first 50 cases (v1 and v2 were scored on these)
#   extra     13 cases added with v3: more emergency types, plus messages
#             that SOUND urgent but aren't
#   holdout   15 cases written by other people after the prompt and policies
#             were finished. Never use them to tune the prompt: they are the
#             fair test of how the tool does on requests it hasn't "seen".
CASES_FILE = "eval_cases.csv"
REQUIRED_COLUMNS = ["id", "message", "expected_category", "expected_urgency", "set"]

# Prices in US dollars per million tokens. Check the current price for your
# model on Anthropic's pricing page and edit these two numbers if they differ.
INPUT_PRICE_PER_MTOK = 1.00
OUTPUT_PRICE_PER_MTOK = 5.00


def run_case(
    client: anthropic.Anthropic,
    message: str,
    policy_index: chromadb.Collection | None,
) -> tuple[dict, float, int, int, list[dict]]:
    """Send one test message through the real tool's classify(), with timing.

    Returns: answer, seconds taken, input tokens, output tokens, policies used.
    The time includes the policy search when RAG is on.
    """
    start = time.perf_counter()
    answer, usage, policies = office_helper.classify(client, message, policy_index)
    seconds = time.perf_counter() - start
    return answer, seconds, usage.input_tokens, usage.output_tokens, policies


def percentile(values: list[float], pct: float) -> float:
    """The value below which pct% of the values fall (e.g. 95th percentile latency)."""
    values = sorted(values)
    index = max(0, min(len(values) - 1, round(pct / 100 * len(values)) - 1))
    return values[index]


def accuracy(rows: list[dict]) -> tuple[float, float, int]:
    """Return (category accuracy, urgency accuracy, number of urgency-scored cases).

    Cases labeled Unclear have no expected urgency, so they're left out of
    urgency accuracy.
    """
    category_accuracy = sum(row["category_correct"] for row in rows) / len(rows)
    scored = [row for row in rows if row["urgency_correct"] != ""]
    urgency_accuracy = sum(row["urgency_correct"] for row in scored) / len(scored) if scored else 0
    return category_accuracy, urgency_accuracy, len(scored)


def emergency_stats(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """The safety numbers.

    expected_high: cases you labeled high (we want the AI to catch ALL of these)
    missed:        expected high but the AI said lower (the dangerous mistake)
    over:          expected medium/low but the AI said high (extra work, not dangerous)
    """
    expected_high = [row for row in rows if row["expected_urgency"] == "high"]
    missed = [row for row in expected_high if row["urgency"] != "high"]
    over = [row for row in rows
            if row["expected_urgency"] in ("medium", "low") and row["urgency"] == "high"]
    return expected_high, missed, over


def score_case(case: dict, client: anthropic.Anthropic,
               policy_index: chromadb.Collection | None) -> dict:
    """Run one case and return a result row comparing the answer with the label."""
    error = ""
    answer = {}
    seconds = 0.0
    input_tokens = output_tokens = 0
    policies = []
    try:
        answer, seconds, input_tokens, output_tokens, policies = run_case(
            client, case["message"], policy_index)
    except anthropic.AuthenticationError:
        # A bad key fails every case. Stop now rather than report a fake 0%.
        sys.exit("Stopped: the API key was rejected (401). Nothing was scored. "
                 "Check the key and run again.")
    except anthropic.APIError as api_error:
        error = f"API error: {api_error}"
    except (ValueError, KeyError, IndexError) as parse_error:
        # The reply wasn't the JSON we asked for (ValueError includes JSON errors).
        error = f"parse error: {parse_error}"

    category = answer.get("category", "")
    urgency = answer.get("urgency", "")
    # Same human-review rules as the real tool, plus any error.
    reasons = [error] if error else office_helper.review_reasons(answer, policies)
    expected_category = case["expected_category"]
    expected_urgency = case["expected_urgency"]
    # Some cases have two acceptable categories, written "A|B" in the CSV
    # (e.g. a stuck elevator: Safety & Security|Facilities). Either one counts.
    accepted_categories = expected_category.split("|")

    return {
        "set": case["set"],
        "id": case["id"],
        "message": case["message"],
        "expected_category": expected_category,
        "expected_urgency": expected_urgency,
        "category": category,
        "urgency": urgency,
        "reason": answer.get("reason", ""),
        "policies": office_helper.policy_names(policies),
        "category_correct": category in accepted_categories,
        # Urgency is only scored when the case has an expected urgency ("" = not scored).
        "urgency_correct": (urgency == expected_urgency) if expected_urgency else "",
        "flagged_for_human": bool(reasons),
        "review_reason": "; ".join(reasons),
        "seconds": round(seconds, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error": error,
    }


def case_mark(row: dict) -> str:
    """Short label printed next to each case; "ok" only when category AND urgency are right."""
    if row["error"]:
        return "ERROR"
    if row["expected_urgency"] == "high" and row["urgency"] != "high":
        return "MISSED-HIGH"  # the most important kind of miss
    if not row["category_correct"]:
        return "MISS-CAT"
    if row["urgency_correct"] is False:
        return "MISS-URG"
    return "ok"


def build_report(rows: list[dict], use_rag: bool) -> list[str]:
    """Turn the result rows into summary lines, so they can be printed AND saved."""
    total_cases = len(rows)
    category_accuracy, urgency_accuracy, urgency_scored = accuracy(rows)

    unclear_cases = [row for row in rows if row["expected_category"] == "Unclear"]
    real_requests = [row for row in rows if row["expected_category"] != "Unclear"]
    unclear_caught = sum(row["category"] == "Unclear" for row in unclear_cases)
    false_unclear = sum(row["category"] == "Unclear" for row in real_requests)

    expected_high, missed_high, over_high = emergency_stats(rows)

    times = [row["seconds"] for row in rows if not row["error"]]
    total_in = sum(row["input_tokens"] for row in rows)
    total_out = sum(row["output_tokens"] for row in rows)
    total_cost = (total_in * INPUT_PRICE_PER_MTOK + total_out * OUTPUT_PRICE_PER_MTOK) / 1_000_000
    errors = sum(bool(row["error"]) for row in rows)

    rag_text = f"on (top {office_helper.POLICIES_PER_REQUEST} policies added)" if use_rag else "off"
    missed_ids = ", ".join("#" + row["id"] for row in missed_high)
    over_ids = ", ".join("#" + row["id"] for row in over_high)

    report = [
        "=== Results (all cases) ===",
        f"Prompt version:            {office_helper.PROMPT_VERSION}",
        f"RAG:                       {rag_text}",
        f"Model:                     {office_helper.MODEL}",
        f"Cases:                     {total_cases}  (data/{CASES_FILE})",
        f"Category accuracy:         {category_accuracy:.0%}",
        f"Urgency accuracy:          {urgency_accuracy:.0%}  (on {urgency_scored} cases with an expected urgency)",
        f"Emergencies caught:        {len(expected_high) - len(missed_high)} of {len(expected_high)} high-urgency cases rated high"
        + (f"  MISSED: {missed_ids}" if missed_high else ""),
        f"Over-escalated to high:    {len(over_high)} medium/low cases rated high"
        + (f"  ({over_ids})" if over_high else ""),
        f"Unclear caught:            {unclear_caught} of {len(unclear_cases)} meaningless inputs sent to a human",
        f"False 'Unclear':           {false_unclear} of {len(real_requests)} real requests wrongly sent to a human",
        f"Flagged for human review:  {sum(row['flagged_for_human'] for row in rows)} of {total_cases}",
        f"Errors:                    {errors}",
    ]
    if times:
        report.append(f"Latency (seconds):         average {statistics.mean(times):.2f}, "
                      f"median {statistics.median(times):.2f}, 95th percentile {percentile(times, 95):.2f}")
    report.append(f"Tokens:                    {total_in} in, {total_out} out")
    report.append(f"Cost:                      ${total_cost:.4f} total, ${total_cost / total_cases:.5f} per request")

    if use_rag:
        # How often the two RAG-only rules kicked in.
        danger_ids = ["#" + row["id"] for row in rows if policy_search.mentions_danger(row["message"])]
        no_match = "no related office policy found"
        flagged_real_ids = ["#" + row["id"] for row in real_requests if no_match in row["review_reason"]]
        report.append(f"Danger word found:         {len(danger_ids)} cases (emergency policy sent first): "
                      + ", ".join(danger_ids))
        report.append(f"No-related-policy flag:    "
                      f"{sum(no_match in row['review_reason'] for row in unclear_cases)} of "
                      f"{len(unclear_cases)} meaningless inputs, "
                      f"{len(flagged_real_ids)} of {len(real_requests)} real requests"
                      + (f" ({', '.join(flagged_real_ids)})" if flagged_real_ids else ""))

    # Per-set breakdown, so the original 50 can be compared with v1/v2.
    report.append("")
    report.append("=== By set ===")
    set_names = dict.fromkeys(row["set"] for row in rows)  # each set name once, in file order
    for set_name in set_names:
        set_rows = [row for row in rows if row["set"] == set_name]
        set_category, set_urgency, _ = accuracy(set_rows)
        set_high, set_missed, set_over = emergency_stats(set_rows)
        report.append(f"{set_name} ({len(set_rows)} cases): category {set_category:.0%}, "
                      f"urgency {set_urgency:.0%}, emergencies caught "
                      f"{len(set_high) - len(set_missed)} of {len(set_high)}, "
                      f"over-escalated {len(set_over)}")

    misses = [row for row in rows if not row["category_correct"] or row["urgency_correct"] is False]
    if misses:
        report.append("")
        report.append("=== Misses (review each: is the AI wrong, or is your label debatable?) ===")
        for row in misses:
            report.append(f"#{row['id']}: \"{row['message']}\"")
            report.append(f"    expected {row['expected_category']}/{row['expected_urgency'] or '-'}"
                          f", got {row['category']}/{row['urgency']} - {row['reason'] or row['error']}")
            if row["policies"]:
                report.append(f"    policies used: {row['policies']}")
    return report


def save_results(rows: list[dict], report: list[str], use_rag: bool) -> str:
    """Save one CSV (a row per case) and one .txt (the summary).

    The file name has a timestamp, so no run ever overwrites an earlier one,
    and "_rag" when RAG was on, so before/after runs are easy to tell apart.
    Returns the shared file name (without extension).
    """
    office_helper.RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    label = office_helper.PROMPT_VERSION + ("_rag" if use_rag else "")
    base_path = office_helper.RESULTS_DIR / f"eval_{label}_{stamp}"

    with open(f"{base_path}.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with open(f"{base_path}.txt", "w", encoding="utf-8") as text_file:
        text_file.write("\n".join(report) + "\n")
    return base_path.name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score the triage tool on the test cases.")
    parser.add_argument("--rag", action="store_true",
                        help="add the closest office policies to each prompt")
    args = parser.parse_args(argv)

    cases = office_helper.read_csv_rows(office_helper.DATA_DIR / CASES_FILE, REQUIRED_COLUMNS)
    policy_index = office_helper.open_policies_if_requested(args.rag)
    client = office_helper.make_client()

    print(f"Prompt {office_helper.PROMPT_VERSION}, RAG {'on' if args.rag else 'off'}, "
          f"model {office_helper.MODEL}, {len(cases)} cases\n")

    rows = []
    for case in cases:
        row = score_case(case, client, policy_index)
        rows.append(row)
        print(f"[{case_mark(row):<11}] #{row['id']:>2} expected "
              f"{row['expected_category']}/{row['expected_urgency'] or '-'}"
              f"  got {row['category']}/{row['urgency']}")

    report = build_report(rows, args.rag)
    print("\n" + "\n".join(report))

    file_name = save_results(rows, report, args.rag)
    print(f"\nSaved results/{file_name}.csv and results/{file_name}.txt")


if __name__ == "__main__":
    main()
