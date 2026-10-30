"""Add new questions to data/question_bank.json (checks them first).

Usage: python add_questions.py data/new_questions.json
"""
import json
import sys

from agent import tools


def main(path):
    with open(tools.BANK_PATH, encoding="utf-8-sig") as f:
        bank = json.load(f)
    with open(path, encoding="utf-8-sig") as f:
        new = json.load(f)
    ids = {q["id"] for q in bank}
    for q in new:
        if not tools.is_valid_question(q):
            sys.exit(f"Invalid question: {q.get('id', q)}")
        if q["id"] in ids:
            sys.exit(f"Duplicate id: {q['id']}")
        ids.add(q["id"])
    bank.extend(new)
    with open(tools.BANK_PATH, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=1, ensure_ascii=False)
    print(f"Added {len(new)} questions. Bank now has {len(bank)}.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python add_questions.py <new_questions.json>")
    main(sys.argv[1])
