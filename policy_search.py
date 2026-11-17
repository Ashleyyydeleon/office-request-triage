"""
Policy search (the "retrieval" part of RAG)
-------------------------------------------
What it does:
    Stores the office policy documents in a local ChromaDB database and finds
    the policies closest in meaning to a request. office_helper.py uses this
    to add the most relevant policies to the AI prompt.

How it works, in plain words:
    1. A small embedding model turns each policy into a list of numbers that
       captures its meaning (an "embedding").
    2. ChromaDB saves those numbers on disk.
    3. To search, the request is turned into numbers the same way, and
       ChromaDB returns the policies whose numbers are closest.
    4. Safety-first rule: if the request contains a danger word (fire, gas,
       collapsed...), the emergency policy is always sent first.

Inputs:   policies/*.md         one policy per file
Outputs:  chroma_db/            the saved index (rebuild it any time)
          models/               the embedding model, downloaded on first build

Everything runs on this computer: no API key and no Anthropic API calls.
The embedding model (all-MiniLM-L6-v2, about 80 MB) comes from Chroma's
official server and is checked against a SHA-256 fingerprint.

Usage (inside the virtual environment):
    python policy_search.py                              rebuild the index
    python policy_search.py "the microwave is smoking"   show the 2 closest policies
"""

import argparse
import re
import sys
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

PROJECT_DIR = Path(__file__).parent
POLICIES_DIR = PROJECT_DIR / "policies"   # the documents we search
DB_DIR = PROJECT_DIR / "chroma_db"        # ChromaDB's saved index
MODEL_DIR = PROJECT_DIR / "models"        # the downloaded embedding model

COLLECTION_NAME = "office_policies"  # a "collection" is like a table in ChromaDB
REBUILD_HINT = "Build it first with: python policy_search.py"

# Safety-first rule. Meaning-based search alone is not reliable enough for
# emergencies: with this small model, "a strong smell of gas" is no closer to
# the emergency policy than "we're out of sparkling water". So when a request
# contains a danger word, the emergency policy is always sent FIRST, and the
# closest team policy (e.g. kitchen) second. The words come from the
# emergency policy itself. Matching is on whole words or phrases, lowercase.
EMERGENCY_POLICY = "emergencies_and_safety"
DANGER_WORDS = [
    # fire and smoke
    "fire", "smoke", "smoking", "smoky", "burning", "flames", "sparked", "sparking", "sparks",
    # gas, water, power
    "gas", "flood", "flooding", "flooded", "pouring", "leak", "leaking",
    "power outage", "power went out",
    # people hurt or trapped
    "collapsed", "unconscious", "unresponsive", "not breathing", "injured", "injury",
    "bleeding", "fell", "slipped", "trapped", "stuck in", "stuck between",
    "locked in", "locked inside", "can't get out",
    # security and hazardous materials
    "intruder", "without a badge", "threat", "weapon", "break-in", "chemical", "hazardous",
]


class LocalMiniLM(ONNXMiniLM_L6_V2):
    """Chroma's default embedding model, saved inside this project.

    Chroma normally downloads the model to the home folder (~/.cache). The only
    way to change that is to override this one setting, which is the whole
    reason this small class exists. The model itself is unchanged.
    """
    DOWNLOAD_PATH = MODEL_DIR / "all-MiniLM-L6-v2"


def connect_to_database() -> ClientAPI:
    """Open (or create) the ChromaDB database stored in chroma_db/."""
    # anonymized_telemetry=False stops Chroma from sending usage statistics.
    return chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def build_policy_index() -> int:
    """Read every policy file and rebuild the index from scratch.

    Returns the number of policies indexed.
    """
    policy_files = sorted(POLICIES_DIR.glob("*.md"))
    if not policy_files:
        raise FileNotFoundError(f"No .md policy files found in {POLICIES_DIR}")

    database = connect_to_database()

    # Delete the old index first, so a renamed or deleted policy file
    # doesn't stay searchable.
    existing_names = [collection.name for collection in database.list_collections()]
    if COLLECTION_NAME in existing_names:
        database.delete_collection(COLLECTION_NAME)

    policy_index = database.create_collection(
        COLLECTION_NAME,
        embedding_function=LocalMiniLM(),
        # Cosine distance: 0 means identical meaning; larger means less related.
        configuration={"hnsw": {"space": "cosine"}},
    )

    # Each policy is short, so each file is stored whole.
    # The file name without ".md" (e.g. "kitchen_and_pantry") is its id.
    policy_index.add(
        ids=[policy_file.stem for policy_file in policy_files],
        documents=[policy_file.read_text(encoding="utf-8") for policy_file in policy_files],
    )
    return len(policy_files)


def open_policy_index() -> chromadb.Collection:
    """Open the saved index so it can be searched.

    Open it once and reuse it: loading the embedding model takes a moment.
    """
    if not DB_DIR.exists():
        raise FileNotFoundError(f"The policy index hasn't been built yet. {REBUILD_HINT}")

    database = connect_to_database()
    existing_names = [collection.name for collection in database.list_collections()]
    if COLLECTION_NAME not in existing_names:
        raise FileNotFoundError(f"The policy index is empty. {REBUILD_HINT}")

    return database.get_collection(COLLECTION_NAME, embedding_function=LocalMiniLM())


def search_policies(policy_index: chromadb.Collection, message: str, how_many: int = 2) -> list[dict]:
    """Return the policies closest in meaning to the message, closest first.

    Each match is a dict:
        name      the policy file name without .md
        text      the full policy
        distance  how far apart in meaning (lower = more related)
    """
    found = policy_index.query(query_texts=[message], n_results=how_many)

    # Chroma can search for several messages at once, so every result is a
    # list of lists (one inner list per message). We sent one message: take [0].
    names = found["ids"][0]
    texts = found["documents"][0]
    distances = found["distances"][0]

    matches = []
    for name, text, distance in zip(names, texts, distances):
        matches.append({"name": name, "text": text, "distance": round(distance, 3)})
    return matches


def mentions_danger(message: str) -> bool:
    """True if the message contains any danger word or phrase (see DANGER_WORDS)."""
    text = message.lower()
    for word in DANGER_WORDS:
        # \b means "word boundary", so "gas" matches "gas leak" but not "Vegas".
        if re.search(r"\b" + re.escape(word) + r"\b", text):
            return True
    return False


def find_policies_for_request(policy_index: chromadb.Collection, message: str,
                              how_many: int = 2) -> list[dict]:
    """Pick the policies to show the AI for one request (2 by default).

    Danger word found: emergency policy first, then the closest other policies
                       (e.g. "the microwave is smoking" -> emergencies, kitchen)
    Otherwise:         the closest policies by meaning
    """
    # Ask for every policy (there are only a few), already sorted closest first.
    all_policies = search_policies(policy_index, message, policy_index.count())

    if not mentions_danger(message):
        return all_policies[:how_many]

    emergency = [policy for policy in all_policies if policy["name"] == EMERGENCY_POLICY]
    others = [policy for policy in all_policies if policy["name"] != EMERGENCY_POLICY]
    return emergency + others[:how_many - 1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the policy index, or search it if you give a request."
    )
    parser.add_argument(
        "request", nargs="*",
        help="a request to search for (leave out to rebuild the index)",
    )
    args = parser.parse_args()
    request = " ".join(args.request)  # lets the request be typed with or without quotes

    try:
        if request:
            if mentions_danger(request):
                print("Danger word found: emergency policy goes first.")
            for match in find_policies_for_request(open_policy_index(), request):
                print(f"{match['distance']:.3f}  {match['name']}")
        else:
            count = build_policy_index()
            print(f"Indexed {count} policies from {POLICIES_DIR.name}/ into {DB_DIR.name}/")
    except FileNotFoundError as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()
