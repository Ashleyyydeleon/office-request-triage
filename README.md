# Office Request Triage Tool

Sorts messy workplace requests ("the elevator is stuck", "can we get more oat milk?")
into a category and urgency using Claude (Haiku 4.5), checks each answer with a
guardrail, and sends anything invalid or meaningless to a human.

## Folder layout

```
office_helper.py    the tool: prompt, classify(), validate(), main()
eval_triage.py      scores the tool on the test cases in data/
policy_search.py    RAG retrieval: indexes policies/ in ChromaDB and searches it
policies/           9 short office policy documents (the knowledge RAG looks up)
chroma_db/          ChromaDB index, rebuilt by `python policy_search.py` (not shared)
models/             local embedding model, downloaded on first build (not shared)
data/
  requests.csv      sample requests the tool triages
  eval_cases.csv    63 hand-labeled test cases (my ground truth); the "set" column
                    marks the first 50 as "original" and 13 later ones as "extra"
                    (emergency types + urgent-sounding non-emergencies)
results/            everything the scripts write; eval runs are timestamped
requirements.txt    Python packages
envoy_env/          virtual environment (not shared)
```

## How to run

Always use the virtual environment (the Anaconda base env conflicts with `anthropic`):

```bash
cd path/to/this/project
source envoy_env/bin/activate
```

Build the policy search index once (and again after editing `policies/`).
This runs locally and needs no API key:

```bash
python policy_search.py
python policy_search.py "the microwave is smoking"   # try a search
```

The key is typed in hidden and never saved to a file:

```bash
python -c "import getpass, os, runpy; os.environ['ANTHROPIC_API_KEY'] = getpass.getpass('API key: ').strip(); runpy.run_path('eval_triage.py', run_name='__main__')"
```

Swap `eval_triage.py` for `office_helper.py` to triage `data/requests.csv`.

Add `--rag` at the very end of either command (after the closing `"`) to add
the 2 closest policies to each prompt. RAG eval runs are saved as
`results/eval_<version>_rag_<date>_<time>`, next to the plain
`eval_<version>_<date>_<time>` runs.

## What is built and tested

- LLM classification into 8 categories (incl. "Safety & Security" for
  emergencies involving people, and "Unclear") and 3 urgency levels, as JSON
- Guardrail / human in the loop: invalid or "Unclear" answers, failed calls, and
  (with RAG) messages with no related policy are marked `needs_review`, with the
  reasons in a `review_reason` column
- Eval harness: category/urgency accuracy, emergencies caught (high-urgency recall),
  over-escalation to high, Unclear catch rate, false-Unclear rate, latency, tokens,
  cost per request, scores per test file; each run saved with a timestamp
- Error handling: a rejected API key stops the run; rate-limit (429), server (5xx)
  and connection errors are retried by the Anthropic SDK with exponential backoff
  (configured to 5 retries); a request that still fails is flagged, not crashed on
- RAG retrieval: 9 policy documents embedded locally with all-MiniLM-L6-v2 and
  stored in ChromaDB; a request returns its 2 closest policies. Checked offline
  on the eval messages: for 46 of the 48 cases that belong to a named team
  (Visitors, Rooms & Desks, Deliveries, Facilities, Kitchen), that team's policy
  is in the top 2. No API key needed.
- RAG switch (`--rag`): the 2 retrieved policies are added to the prompt, and
  the eval logs which policies each case used. With RAG off the prompt is
  word-for-word v3, so before/after scores are comparable.
- Safety-first retrieval: requests containing a danger word (a short list taken
  from the emergency policy: fire, smoke, gas, leaking, collapsed, trapped...)
  always get the emergency policy first and the closest team policy second
  (e.g. "the microwave sparked and smoke is coming out" -> emergencies, kitchen).
  Triggers on 15 of the 63 test messages, including the two fire-drill
  questions, where the emergency policy itself says drills are low urgency.
  Keyword matching is brittle: "the power just went out" is not matched (its
  team policy still says outages are high).
- No-related-policy flag: if even the closest policy is further than 0.85
  (cosine distance), a human reviews the request whatever the AI said. On the
  test cases it flags 5 of 7 nonsense inputs and 0 of 56 real requests; the
  threshold was set on these same cases and the margins are small.
- All of the above tested offline with a fake AI client (27 checks: prompt
  contents and order, saved files, flags, bad replies, missing files, missing
  index), then run against the real API (results below).
- The code currently runs prompt v4; v3 scored higher on category (see below).

## Results so far

Scored on the 50 "original" cases in `data/eval_cases.csv` (the eval prints this as "By set: original"):

| Prompt | Category | Urgency | Emergencies caught | Unclear caught | False Unclear | Cost / request |
|---|---|---|---|---|---|---|
| v1 (baseline) | 88% | 65% | 11 / 11 | 7 / 7 | 5 / 43 | $0.00037 |
| v2 | 96% | 67% | 11 / 11 | 7 / 7 | 0 / 43 | $0.00054 |
| v3 | 98% | 86% | 9 / 11 | 7 / 7 | 0 / 43 | $0.00061 |
| v3 + RAG | 98% | 98% | 11 / 11 | 7 / 7 | 0 / 43 | $0.00118 |
| v4 | 94% | 84% | 9 / 11 | 7 / 7 | 0 / 43 | $0.00066 |
| v4 + RAG | 88% | 95% | 11 / 11 | 7 / 7 | 0 / 43 | $0.00125 |

All 63 cases (original + extra), v3 vs v3 + RAG, run 2026-11-16:

| | Category | Urgency | Emergencies caught | Over-escalated to high | Median latency | Cost / request |
|---|---|---|---|---|---|---|
| v3 | 92% | 89% | 18 / 20 | 0 | 0.73 s | $0.00061 |
| v3 + RAG | 94% | 96% | 20 / 20 | 0 | 0.84 s | $0.00118 |

What RAG changed (v3 -> v3 + RAG):
- Fixed both missed emergencies: #3 (client waiting, host not told) and #8
  (someone in a booked room). Both had dropped to medium in v3, after v2 had
  caught them.
- Fixed urgency on #24, #33, #40 and category on #55 (toaster fire -> Kitchen)
  and #60 (fire drill -> Facilities).
- New misses: #57 (guest locked in lobby -> Facilities instead of Visitors;
  the emergency policy, sent first, frames it as a door problem) and #61 (exit
  sign -> low, although the facilities policy says one business day = medium).
- Still missed in both: #18, #19 (debatable labels), #58, #59 (medical and
  security emergencies labeled Other; the AI picks Facilities, whose
  description includes "safety hazards").
- About twice the input tokens (2 policies per prompt), so about twice the
  cost. A few calls took 3-8 s; the median barely moved, so this looks like
  API delay rather than the local search, but that is not confirmed.
- The no-related-policy flag fired on 5 of 7 nonsense inputs and 0 real
  requests, but the AI had already labeled all 7 Unclear, so in this run it
  caught nothing extra.

v4 (new "Safety & Security" category), all 63 cases, run 2026-11-16:

| | Category | Urgency | Emergencies caught | Over-escalated to high | Median latency | Cost / request |
|---|---|---|---|---|---|---|
| v4 | 87% | 88% | 18 / 20 | 0 | 0.72 s | $0.00066 |
| v4 + RAG | 81% | 95% | 20 / 20 | 0 | 0.80 s | $0.00125 |

Label change with v4: #28 (someone slipped and fell), #58 (someone collapsed)
and #59 (stranger without a badge) were relabeled from Facilities / Other to
Safety & Security. v1-v3 were scored with the old labels, so v4 category
scores are not directly comparable on those 3 cases; urgency is unaffected.

What v4 changed (an honest regression in category accuracy):
- The 3 relabeled cases (#28, #58, #59) are now right, with and without RAG.
- But the AI also sends building hazards and trapped people to Safety &
  Security, even though the prompt says building hazards go to the team that
  owns the space: fire (#35, #51, #55), gas (#25, RAG only), chemical leak
  (#56), and people trapped in the elevator or lobby (#26, #57).
- RAG makes this worse (94% -> 88% category on the original 50): the emergency
  policy, sent first, frames these as safety incidents; #8 (someone in a
  booked room) even became Safety & Security.
- Urgency held up: 95% with RAG, all 20 emergencies caught, 0 over-escalated.
  No case was harmed in the way that matters most (missing an emergency).
- Latency was normal in both runs (95th percentile about 1 s), which supports
  that the slow calls in the v3 + RAG run were API-side.
- Open decision for v5: send every emergency to Safety & Security (and relabel),
  make the "team that owns the space" rule stronger, or go back to v3's
  categories.

Prompt changes are described next to `PROMPT_VERSION` in `office_helper.py`.
v1 details: `results/baseline_prompt_v1_2026-11-04.txt`; v2 and later: `results/eval_v*.txt`.

Caveat: v2 and v3 were written after looking at earlier misses on these same 50
cases, so those scores are somewhat optimistic. The 13 extra cases were written
at the same time as v3, so they are not a true unseen test either; a fair check
needs new cases written by someone who hasn't seen the prompt.
The policy documents in `policies/` were also written with the test cases in
view (without copying their wording), so the same caveat applies to RAG scores.
Several RAG fixes come from policy lines that match a test label closely
(e.g. "smells or old food are low" for #33, "a host was not notified and a
guest is waiting ... high" for #3). In a real deployment the policies would
come from the company, which is how RAG should work, but it makes these
scores optimistic for unseen requests.

## Not built yet

- A held-out test set written by someone who hasn't seen the prompt or policies
- FastAPI service and Google Cloud Run deployment
- Multi-agent version
- Input-side guardrail (checking whether the request itself makes sense before
  calling the model)
- The retry setting has not been tested against real 429/500 errors
