"""
Office Request Triage Tool
--------------------------
Reads workplace requests from data/requests.csv, asks an LLM (Claude) to sort
each one into a category and urgency, checks the answer is valid (a
"guardrail"), and writes the results to results/triaged_requests.csv.

This mirrors a real internal tool: an office team gets messy requests in
Slack or email and wants them routed automatically.

Run (inside the virtual environment, key typed in hidden):
    source envoy_env/bin/activate
    python -c "import getpass, os, runpy; os.environ['ANTHROPIC_API_KEY'] = getpass.getpass('API key: ').strip(); runpy.run_path('office_helper.py', run_name='__main__')"
"""

import csv
import json
import sys
from pathlib import Path

import anthropic

# Folders, found relative to this file so the script works from any directory.
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"        # inputs: requests.csv, eval_cases.csv
RESULTS_DIR = PROJECT_DIR / "results"  # outputs written by the scripts

# Only answers we accept. Anything else gets flagged for a human.
CATEGORIES = ["Visitors", "Rooms & Desks", "Deliveries", "Facilities", "Kitchen", "Other", "Unclear"]
URGENCIES = ["high", "medium", "low"]

MODEL = "claude-haiku-4-5-20251001"  # small, fast, cheap model

# Bump this whenever PROMPT changes, so eval results can be compared by version.
#   v1: first prompt + one-line "Unclear" rule (baseline: 88% category, 65% urgency)
#   v2: one-line description per category, narrower "Unclear", medium/low defined
#       (96% category, 67% urgency, 11 of 11 high cases caught)
#   v3: emergency examples under "high", plus two tie-breakers:
#       unsure if high -> high (safety first); unsure medium vs low -> low
PROMPT_VERSION = "v3"

# Everything the model needs to know must be INSIDE this text;
# code comments are never sent to the model.
# We ask for JSON, not a paragraph, so the output is machine-readable.
PROMPT = """You triage office requests for a workplace team.

Pick exactly one category:
- Visitors: guests, visitor registration, front-desk check-in, visitor kiosks and forms
- Rooms & Desks: meeting rooms, desk bookings, room equipment, seating
- Deliveries: packages, mail, couriers, returns, the mailroom
- Facilities: the building itself: leaks, heating and cooling, lights, elevators, restrooms, doors, badge readers, safety hazards
- Kitchen: food, drinks, snacks, kitchen appliances, kitchen cleanliness
- Other: any real workplace question or request that fits none of the above (policies, HR letters, office hours, events, lost items, IT problems)
- Unclear: ONLY when you cannot tell what the person wants (random characters, "test", a bare greeting, a joke). If the message is a real question or problem, never use Unclear: use the closest category, or Other.

Pick exactly one urgency:
- high: a safety risk, or people are blocked right now. For example: fire or smoke,
  a gas smell, flooding, an injury or medical emergency, someone trapped, or a security threat.
- medium: affects people's work and should be handled within about a day, but nobody is at risk or blocked right now
- low: questions, restocks, small inconveniences, and requests for later
If you are unsure whether something is high, choose high: missing a real emergency is the worst mistake.
If you are unsure between medium and low, choose low.

Reply with ONLY a JSON object, no other text, like:
{"category": "Facilities", "urgency": "high", "reason": "short reason"}

Request: """


def make_client():
    """Create the API client. Reads ANTHROPIC_API_KEY from the environment.

    max_retries: the Anthropic library automatically retries rate-limit (429),
    server (5xx) and connection errors, waiting longer each time (exponential
    backoff). The default is 2 retries; we raise it to 5.
    """
    return anthropic.Anthropic(max_retries=5)


def classify(client, message):
    """Send one request to the LLM.

    Returns two things: the parsed answer (a dict) and the token usage
    (response.usage, which has .input_tokens and .output_tokens).
    eval_triage.py calls this same function, so whatever we change here
    (for example adding RAG later) is exactly what the evals measure.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": PROMPT + message}],
    )
    text = response.content[0].text.strip()
    # Models sometimes wrap JSON in ```json ... ``` fences; remove them.
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text), response.usage


def validate(result):
    """Guardrail: allowed values only, and 'Unclear' goes to a human."""
    return (
        result.get("category") in CATEGORIES
        and result.get("urgency") in URGENCIES
        and result.get("category") != "Unclear"
    )


def main():
    client = make_client()

    with open(DATA_DIR / "requests.csv", newline="", encoding="utf-8") as f:
        requests = list(csv.DictReader(f))

    results = []
    for row in requests:
        try:
            result, _usage = classify(client, row["message"])  # usage not needed here
            status = "ok" if validate(result) else "needs_review"
        except anthropic.AuthenticationError:
            # A bad key fails every request, so stop instead of trying them all.
            sys.exit("Stopped: the API key was rejected (401). Check the key and try again.")
        except anthropic.APIError as error:
            # Still failing after the automatic retries: flag this one, keep going.
            result = {"category": "", "urgency": "", "reason": f"API error: {error}"}
            status = "needs_review"
        except (json.JSONDecodeError, KeyError, IndexError) as error:
            result = {"category": "", "urgency": "", "reason": f"Could not parse AI reply: {error}"}
            status = "needs_review"

        results.append({
            "id": row["id"],
            "message": row["message"],
            "category": result.get("category", ""),
            "urgency": result.get("urgency", ""),
            "reason": result.get("reason", ""),
            "status": status,
        })
        print(f"[{status}] #{row['id']} -> {result.get('category')} / {result.get('urgency')}")

    # Most urgent first, so the team knows what to handle now.
    order = {"high": 0, "medium": 1, "low": 2}
    results.sort(key=lambda r: order.get(r["urgency"], 3))

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "triaged_requests.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    flagged = sum(r["status"] == "needs_review" for r in results)
    print(f"\nDone. {len(results)} requests sorted, {flagged} flagged for a human.")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
