# Office Request Triage Tool

Sorts messy workplace requests ("the elevator is stuck", "can we get more oat milk?")
into a category and urgency using Claude (Haiku 4.5), checks each answer with a
guardrail, and sends anything invalid or meaningless to a human.

## Folder layout

```
office_helper.py    the tool: prompt, classify(), validate(), main()
eval_triage.py      scores the tool on the test cases in data/
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

The key is typed in hidden and never saved to a file:

```bash
python -c "import getpass, os, runpy; os.environ['ANTHROPIC_API_KEY'] = getpass.getpass('API key: ').strip(); runpy.run_path('eval_triage.py', run_name='__main__')"
```

Swap `eval_triage.py` for `office_helper.py` to triage `data/requests.csv`.

## What is built and tested

- LLM classification into 7 categories (incl. "Unclear") and 3 urgency levels, as JSON
- Guardrail: invalid or "Unclear" answers are marked `needs_review` for a human
- Eval harness: category/urgency accuracy, emergencies caught (high-urgency recall),
  over-escalation to high, Unclear catch rate, false-Unclear rate, latency, tokens,
  cost per request, scores per test file; each run saved with a timestamp
- Error handling: a rejected API key stops the run; rate-limit (429), server (5xx)
  and connection errors are retried by the Anthropic SDK with exponential backoff
  (configured to 5 retries); a request that still fails is flagged, not crashed on

## Results so far

Scored on the 50 "original" cases in `data/eval_cases.csv` (the eval prints this as "By set: original"):

| Prompt | Category | Urgency | Emergencies caught | Unclear caught | False Unclear | Cost / request |
|---|---|---|---|---|---|---|
| v1 (baseline) | 88% | 65% | 11 / 11 | 7 / 7 | 5 / 43 | $0.00037 |
| v2 | 96% | 67% | 11 / 11 | 7 / 7 | 0 / 43 | $0.00054 |
| v3 | not run yet | | | | | |

Prompt changes are described next to `PROMPT_VERSION` in `office_helper.py`.
v1 details: `results/baseline_prompt_v1_2026-11-04.txt`; v2 and later: `results/eval_v*.txt`.

Caveat: v2 and v3 were written after looking at earlier misses on these same 50
cases, so those scores are somewhat optimistic. The 13 extra cases were written
at the same time as v3, so they are not a true unseen test either; a fair check
needs new cases written by someone who hasn't seen the prompt.

## Not built yet

- RAG over office policy documents (ChromaDB)
- FastAPI service and Google Cloud Run deployment
- Multi-agent version
- Input-side guardrail (checking whether the request itself makes sense before
  calling the model)
- The retry setting has not been tested against real 429/500 errors
