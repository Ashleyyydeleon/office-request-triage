"""
Office Request Triage Tool
--------------------------
What it does:
    Reads workplace requests, asks an LLM (Claude) to sort each one into a
    category and an urgency, checks that the answer is valid (a "guardrail"),
    and saves the results with the most urgent requests first. This mirrors a
    real internal tool: an office team gets messy requests in Slack or email
    and wants them routed automatically.

Optional RAG (retrieval-augmented generation), switched on with --rag:
    Before asking the LLM, look up 2 office policies for the request (see
    policy_search.py) and add them to the prompt, so the answer can follow the
    office's own rules. Requests with a danger word (fire, gas, collapsed...)
    always get the emergency policy first.

Input:   data/requests.csv              columns: id, message
Output:  results/triaged_requests.csv   columns: id, message, category,
                                        urgency, reason, status,
                                        review_reason, policies

Human in the loop: a request is marked "needs_review" (with the reasons in
review_reason) when the AI's answer is invalid or "Unclear", the call failed,
or, with RAG, no policy is even close to the message (likely nonsense).

Run (inside the virtual environment, key typed in hidden). Leave out --rag to
run without policies:
    source envoy_env/bin/activate
    python -c "import getpass, os, runpy; os.environ['ANTHROPIC_API_KEY'] = getpass.getpass('API key: ').strip(); runpy.run_path('office_helper.py', run_name='__main__')" --rag
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import anthropic
import chromadb

import policy_search

# Folders, found relative to this file so the script works from any directory.
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"        # inputs: requests.csv, eval_cases.csv
RESULTS_DIR = PROJECT_DIR / "results"  # outputs written by the scripts

# The only answers we accept. Anything else is flagged for a human.
CATEGORIES = ["Visitors", "Rooms & Desks", "Deliveries", "Facilities", "Kitchen",
              "Safety & Security", "Other", "Unclear"]
URGENCIES = ["high", "medium", "low"]

MODEL = "claude-haiku-4-5-20251001"  # small, fast, cheap model
POLICIES_PER_REQUEST = 2             # how many policies RAG adds to the prompt

# Backup human-review flag (RAG only). If even the closest policy is this far
# away in meaning, the message probably isn't a real office request ("asdf",
# "Hi"), so a human checks it no matter what the AI answered. Set on the test
# cases: the furthest real request scored 0.833 and the closest nonsense 0.864,
# so 0.85 sits between them. Margins are small: recheck after editing policies.
NO_MATCH_DISTANCE = 0.85

# Bump this whenever INSTRUCTIONS change, so eval results can be compared by version.
# (Whether RAG was on is recorded separately, in the eval report and file name.)
#   v1: first prompt + one-line "Unclear" rule (baseline: 88% category, 65% urgency)
#   v2: one-line description per category, narrower "Unclear", medium/low defined
#       (96% category, 67% urgency, 11 of 11 high cases caught)
#   v3: emergency examples under "high", plus two tie-breakers:
#       unsure if high -> high (safety first); unsure medium vs low -> low
#       (all 63 cases: 92% category, 89% urgency; with RAG 94%, 96%)
#   v4: new "Safety & Security" category for emergencies involving PEOPLE
#       (medical, injuries, security threats); building hazards stay with the
#       team that owns the space. Test cases #28, #58, #59 relabeled to match.
PROMPT_VERSION = "v4"

# Everything the model needs to know must be INSIDE these strings;
# code comments are never sent to the model.
# We ask for JSON, not a paragraph, so the answer is machine-readable.
INSTRUCTIONS = """You triage office requests for a workplace team.

Pick exactly one category:
- Visitors: guests, visitor registration, front-desk check-in, visitor kiosks and forms
- Rooms & Desks: meeting rooms, desk bookings, room equipment, seating
- Deliveries: packages, mail, couriers, returns, the mailroom
- Facilities: the building itself: leaks, heating and cooling, lights, elevators, restrooms, doors, badge readers, safety hazards
- Kitchen: food, drinks, snacks, kitchen appliances, kitchen cleanliness
- Safety & Security: emergencies involving people: someone collapsed, hurt or injured, intruders or unknown people without a badge, theft, threats. Building hazards like fire, gas or leaks go to the team that owns the space.
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

"""

# Only used with RAG. Search always returns its closest matches, even for a
# message like "asdf", so we tell the model the policies may not apply.
POLICY_INTRO = """Office policies that may help. Use them only if they apply to this request; otherwise ignore them.

"""


def build_prompt(message: str, policies: list[dict]) -> str:
    """Put the prompt together: instructions, then any policies, then the request.

    With no policies (RAG off) this is just the instructions plus the request,
    so results with and without RAG can be compared fairly.
    """
    prompt = INSTRUCTIONS
    if policies:
        prompt += POLICY_INTRO
        for policy in policies:
            prompt += f"--- Policy: {policy['name']} ---\n{policy['text'].strip()}\n\n"
    prompt += "Request: " + message
    return prompt


def make_client() -> anthropic.Anthropic:
    """Create the API client. It reads ANTHROPIC_API_KEY from the environment.

    max_retries: the Anthropic library automatically retries rate-limit (429),
    server (5xx) and connection errors, waiting longer each time (exponential
    backoff). The default is 2 retries; we raise it to 5.
    """
    return anthropic.Anthropic(max_retries=5)


def open_policies_if_requested(use_rag: bool) -> chromadb.Collection | None:
    """Open the policy index when RAG is on; return None when it is off.

    Stops the script with a clear message if the index hasn't been built.
    """
    if not use_rag:
        return None
    try:
        return policy_search.open_policy_index()
    except FileNotFoundError as error:
        sys.exit(f"Error: {error}")


def classify(
    client: anthropic.Anthropic,
    message: str,
    policy_index: chromadb.Collection | None = None,
) -> tuple[dict, anthropic.types.Usage, list[dict]]:
    """Send one request to the LLM and return its answer.

    Pass a policy_index to use RAG; leave it out (None) to run without policies.
    eval_triage.py calls this same function, so the eval measures exactly what
    the tool does.

    Returns three things:
        answer    the parsed JSON, e.g. {"category": ..., "urgency": ..., "reason": ...}
        usage     token counts (usage.input_tokens, usage.output_tokens)
        policies  the policies added to the prompt, each with "name" and
                  "distance" (an empty list without RAG)
    """
    policies = []
    if policy_index is not None:
        policies = policy_search.find_policies_for_request(policy_index, message, POLICIES_PER_REQUEST)

    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": build_prompt(message, policies)}],
    )
    text = response.content[0].text.strip()
    # Models sometimes wrap JSON in ```json ... ``` fences; remove them.
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    answer = json.loads(text)  # raises ValueError if the reply isn't valid JSON
    if not isinstance(answer, dict):
        raise ValueError(f"expected a JSON object, got: {text[:80]}")

    return answer, response.usage, policies


def review_reasons(answer: dict, policies: list[dict]) -> list[str]:
    """Guardrail: list every reason a human should check this request.

    An empty list means the answer can be used as is.
    """
    reasons = []
    if answer.get("category") not in CATEGORIES or answer.get("urgency") not in URGENCIES:
        reasons.append("AI answer not in the allowed values")
    if answer.get("category") == "Unclear":
        reasons.append("AI could not tell what was asked")
    # Backup check that doesn't depend on the AI (only possible with RAG).
    if policies and min(policy["distance"] for policy in policies) > NO_MATCH_DISTANCE:
        reasons.append("no related office policy found")
    return reasons


def policy_names(policies: list[dict]) -> str:
    """The policy names as one text cell, e.g. "emergencies_and_safety; kitchen_and_pantry"."""
    return "; ".join(policy["name"] for policy in policies)


def read_csv_rows(path: Path, required_columns: list[str]) -> list[dict[str, str]]:
    """Read a CSV file into a list of rows (one dict per row).

    Stops the script with a clear message if the file is missing, empty,
    or lacks a column the script needs.
    """
    if not path.exists():
        sys.exit(f"Error: can't find {path}")

    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        columns = reader.fieldnames or []

    missing = [column for column in required_columns if column not in columns]
    if missing:
        sys.exit(f"Error: {path.name} is missing the column(s): {', '.join(missing)}")
    if not rows:
        sys.exit(f"Error: {path.name} has no rows to process")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Triage the requests in data/requests.csv.")
    parser.add_argument("--rag", action="store_true",
                        help="add the closest office policies to each prompt")
    args = parser.parse_args(argv)

    requests = read_csv_rows(DATA_DIR / "requests.csv", ["id", "message"])
    policy_index = open_policies_if_requested(args.rag)
    client = make_client()

    results = []
    for row in requests:
        policies = []
        try:
            answer, _usage, policies = classify(client, row["message"], policy_index)
            reasons = review_reasons(answer, policies)
        except anthropic.AuthenticationError:
            # A bad key fails every request, so stop instead of trying them all.
            sys.exit("Stopped: the API key was rejected (401). Check the key and try again.")
        except anthropic.APIError as error:
            # Still failing after the automatic retries: flag this one, keep going.
            answer = {}
            reasons = [f"API error: {error}"]
        except (ValueError, KeyError, IndexError) as error:
            # The reply wasn't the JSON we asked for (ValueError includes JSON errors).
            answer = {}
            reasons = [f"could not read the AI's reply: {error}"]

        status = "needs_review" if reasons else "ok"
        results.append({
            "id": row["id"],
            "message": row["message"],
            "category": answer.get("category", ""),
            "urgency": answer.get("urgency", ""),
            "reason": answer.get("reason", ""),        # the AI's explanation
            "status": status,
            "review_reason": "; ".join(reasons),        # why a human should look
            "policies": policy_names(policies),
        })
        print(f"[{status}] #{row['id']} -> {answer.get('category')} / {answer.get('urgency')}")

    # Most urgent first, so the team knows what to handle now.
    # Anything without a valid urgency goes last (3).
    urgency_order = {"high": 0, "medium": 1, "low": 2}
    results.sort(key=lambda result: urgency_order.get(result["urgency"], 3))

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "triaged_requests.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    flagged = sum(result["status"] == "needs_review" for result in results)
    print(f"\nDone. {len(results)} requests sorted, {flagged} flagged for a human.")
    print(f"RAG: {'on' if args.rag else 'off'}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
