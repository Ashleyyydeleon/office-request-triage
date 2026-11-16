"""
Evaluation script for the Office Request Triage Tool
----------------------------------------------------
Runs every case in data/eval_cases.csv through office_helper.classify() (the
exact function the tool uses), compares the AI's answer to your expected
labels, and reports accuracy, emergency catching, latency, tokens and cost.

Each run saves two NEW files in results/ (old runs are never overwritten):
    eval_<prompt version>_<date>_<time>.csv   one row per case
    eval_<prompt version>_<date>_<time>.txt   the summary printed below

Run (inside the virtual environment, key typed in hidden):
    source envoy_env/bin/activate
    python -c "import getpass, os, runpy; os.environ['ANTHROPIC_API_KEY'] = getpass.getpass('API key: ').strip(); runpy.run_path('eval_triage.py', run_name='__main__')"

The expected labels are YOUR ground truth. Read through the case file and
change any label you disagree with before you trust the scores.
"""

import csv
import json
import statistics
import sys
import time
from datetime import datetime

import anthropic

import office_helper  # reuses classify(), PROMPT_VERSION, CATEGORIES, URGENCIES, folders

# All test cases live in one file in data/. Its "set" column says where each
# case came from, and scores are shown for all cases AND per set, so the
# original 50 stay comparable with earlier runs.
#   original  the first 50 cases (v1 and v2 were scored on these)
#   extra     13 cases added with v3: more emergency types, plus messages
#             that SOUND urgent but aren't
CASES_FILE = "eval_cases.csv"

# Prices in US dollars per million tokens. Check the current price for your
# model on Anthropic's pricing page and edit these two numbers if they differ.
INPUT_PRICE_PER_MTOK = 1.00
OUTPUT_PRICE_PER_MTOK = 5.00


def run_case(client, message):
    """Send one message through the real tool's classify(), with timing.

    We call office_helper.classify() instead of keeping a copy of the API
    call here, so the eval always tests the tool exactly as it runs.
    """
    start = time.perf_counter()
    result, usage = office_helper.classify(client, message)
    seconds = time.perf_counter() - start
    return result, seconds, usage.input_tokens, usage.output_tokens


def percentile(values, pct):
    values = sorted(values)
    index = max(0, min(len(values) - 1, round(pct / 100 * len(values)) - 1))
    return values[index]


def accuracy(rows):
    """Return (category accuracy, urgency accuracy, number of urgency-scored cases)."""
    cat_acc = sum(r["category_correct"] for r in rows) / len(rows)
    scored = [r for r in rows if r["urgency_correct"] != ""]
    urg_acc = sum(r["urgency_correct"] for r in scored) / len(scored) if scored else 0
    return cat_acc, urg_acc, len(scored)


def emergency_stats(rows):
    """The safety numbers.

    caught: expected high AND the AI said high (we want ALL of these)
    missed: expected high but the AI said lower (the dangerous mistake)
    over:   expected medium/low but the AI said high (extra attention, not dangerous)
    """
    expected_high = [r for r in rows if r["expected_urgency"] == "high"]
    missed = [r for r in expected_high if r["urgency"] != "high"]
    over = [r for r in rows if r["expected_urgency"] in ("medium", "low") and r["urgency"] == "high"]
    return expected_high, missed, over


def main():
    client = office_helper.make_client()

    with open(office_helper.DATA_DIR / CASES_FILE, newline="", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))

    print(f"Prompt {office_helper.PROMPT_VERSION}, model {office_helper.MODEL}, {len(cases)} cases\n")

    rows = []
    for case in cases:
        error = ""
        category = urgency = reason = ""
        seconds = 0.0
        in_tok = out_tok = 0
        try:
            result, seconds, in_tok, out_tok = run_case(client, case["message"])
            category = result.get("category", "")
            urgency = result.get("urgency", "")
            reason = result.get("reason", "")
        except anthropic.AuthenticationError:
            # A bad key fails every case. Stop now rather than report a fake 0%.
            sys.exit("Stopped: the API key was rejected (401). Nothing was scored. "
                     "Check the key and run again.")
        except anthropic.APIError as e:
            error = f"API error: {e}"
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            error = f"parse error: {e}"

        valid = category in office_helper.CATEGORIES and urgency in office_helper.URGENCIES
        # A human sees this item if the answer is invalid or the AI said Unclear.
        flagged = (not valid) or category == "Unclear" or bool(error)

        expected_cat = case["expected_category"]
        expected_urg = case["expected_urgency"]
        cat_ok = category == expected_cat
        # Urgency is only scored when a case has an expected urgency.
        urg_ok = (urgency == expected_urg) if expected_urg else ""
        rows.append({
            "set": case["set"],
            "id": case["id"],
            "message": case["message"],
            "expected_category": expected_cat,
            "expected_urgency": expected_urg,
            "category": category,
            "urgency": urgency,
            "reason": reason,
            "category_correct": cat_ok,
            "urgency_correct": urg_ok,
            "flagged_for_human": flagged,
            "seconds": round(seconds, 3),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "error": error,
        })
        # "ok" only when BOTH category and urgency are right.
        if error:
            mark = "ERROR"
        elif expected_urg == "high" and urgency != "high":
            mark = "MISSED-HIGH"  # the most important kind of miss
        elif not cat_ok:
            mark = "MISS-CAT"
        elif urg_ok is False:
            mark = "MISS-URG"
        else:
            mark = "ok"
        print(f"[{mark:<11}] #{case['id']:>2} expected {expected_cat}/{expected_urg or '-'}"
              f"  got {category}/{urgency}")

    # ---- Scores ----
    n = len(rows)
    cat_acc, urg_acc, n_urg = accuracy(rows)

    unclear_cases = [r for r in rows if r["expected_category"] == "Unclear"]
    normal_cases = [r for r in rows if r["expected_category"] != "Unclear"]
    caught = sum(r["category"] == "Unclear" for r in unclear_cases)
    false_unclear = sum(r["category"] == "Unclear" for r in normal_cases)

    expected_high, missed_high, over_high = emergency_stats(rows)

    times = [r["seconds"] for r in rows if not r["error"]]
    total_in = sum(r["input_tokens"] for r in rows)
    total_out = sum(r["output_tokens"] for r in rows)
    total_cost = (total_in * INPUT_PRICE_PER_MTOK + total_out * OUTPUT_PRICE_PER_MTOK) / 1_000_000
    errors = sum(bool(r["error"]) for r in rows)

    # Build the summary as a list of lines so we can print it AND save it.
    report = [
        "=== Results (all cases) ===",
        f"Prompt version:            {office_helper.PROMPT_VERSION}",
        f"Model:                     {office_helper.MODEL}",
        f"Cases:                     {n}  (data/{CASES_FILE})",
        f"Category accuracy:         {cat_acc:.0%}",
        f"Urgency accuracy:          {urg_acc:.0%}  (on {n_urg} cases with an expected urgency)",
        f"Emergencies caught:        {len(expected_high) - len(missed_high)} of {len(expected_high)} high-urgency cases rated high"
        + (f"  MISSED: {', '.join('#' + r['id'] for r in missed_high)}" if missed_high else ""),
        f"Over-escalated to high:    {len(over_high)} medium/low cases rated high"
        + (f"  ({', '.join('#' + r['id'] for r in over_high)})" if over_high else ""),
        f"Unclear caught:            {caught} of {len(unclear_cases)} meaningless inputs sent to a human",
        f"False 'Unclear':           {false_unclear} of {len(normal_cases)} real requests wrongly sent to a human",
        f"Flagged for human review:  {sum(r['flagged_for_human'] for r in rows)} of {n}",
        f"Errors:                    {errors}",
    ]
    if times:
        report.append(f"Latency (seconds):         average {statistics.mean(times):.2f}, "
                      f"median {statistics.median(times):.2f}, 95th percentile {percentile(times, 95):.2f}")
    report.append(f"Tokens:                    {total_in} in, {total_out} out")
    report.append(f"Cost:                      ${total_cost:.4f} total, ${total_cost / n:.5f} per request")

    # Per-set breakdown, so the original 50 can be compared with v1/v2.
    report.append("")
    report.append("=== By set ===")
    for set_name in dict.fromkeys(r["set"] for r in rows):  # each set once, in file order
        set_rows = [r for r in rows if r["set"] == set_name]
        s_cat, s_urg, _ = accuracy(set_rows)
        s_high, s_missed, s_over = emergency_stats(set_rows)
        report.append(f"{set_name} ({len(set_rows)} cases): category {s_cat:.0%}, urgency {s_urg:.0%}, "
                      f"emergencies caught {len(s_high) - len(s_missed)} of {len(s_high)}, "
                      f"over-escalated {len(s_over)}")

    misses = [r for r in rows if not r["category_correct"] or r["urgency_correct"] is False]
    if misses:
        report.append("")
        report.append("=== Misses (review each: is the AI wrong, or is your label debatable?) ===")
        for r in misses:
            report.append(f"#{r['id']}: \"{r['message']}\"")
            report.append(f"    expected {r['expected_category']}/{r['expected_urgency'] or '-'}"
                          f", got {r['category']}/{r['urgency']} - {r['reason'] or r['error']}")

    print("\n" + "\n".join(report))

    # Save with a timestamp so no run ever overwrites an earlier one.
    office_helper.RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base = office_helper.RESULTS_DIR / f"eval_{office_helper.PROMPT_VERSION}_{stamp}"

    with open(f"{base}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with open(f"{base}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print(f"\nSaved results/{base.name}.csv and results/{base.name}.txt")


if __name__ == "__main__":
    main()
