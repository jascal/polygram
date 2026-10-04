"""Deterministic, auditable minimal pairs with compositional holdouts."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import random
import re
from pathlib import Path


VOCAB = {
    "subject_object": (
        "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
        "Sarah", "James", "Kate", "Laura",
    ),
    "variable_value": tuple("ABCDEFGHIJKL"),
    "modifier_attachment": (
        "red", "blue", "green", "yellow", "black", "white", "orange", "purple",
        "silver", "gold", "pink", "brown",
    ),
}
DEV_VOCAB = {
    "subject_object": ("Peter", "Mary", "John", "Susan", "Tom", "Anna"),
    "variable_value": tuple("UVWXYZ"),
    "modifier_attachment": ("gray", "cyan", "tan", "red", "blue", "green"),
}
ROLES = {
    "subject_object": ("agent", "patient", "observer"),
    "variable_value": ("x", "y", "z"),
    "modifier_attachment": ("cup_color", "bowl_color", "plate_color"),
}
SYSTEM = "Answer the question using exactly one word or number, without punctuation."


@dataclass(frozen=True)
class Example:
    id: str
    pair_id: str
    task: str
    split: str
    template: int
    member: int
    context: str
    question: str
    answer: str
    unrelated_question: str
    unrelated_answer: str
    candidates: tuple[str, ...]
    bindings: tuple[tuple[str, str], ...]

    def prompt(self, unrelated: bool = False) -> str:
        question = self.unrelated_question if unrelated else self.question
        return f"{self.context}\n{question}"


def _render(task: str, template: int, a: str, b: str, c: str):
    if task == "subject_object":
        contexts = (
            f"{a} praised {b} while {c} watched.",
            f"While {c} watched, {a} thanked {b}.",
            f"{b} was helped by {a} while {c} watched.",
        )
        question = (
            "Who did the praising?", "Who did the thanking?", "Who did the helping?"
        )[template]
        unrelated = "Who watched?"
    elif task == "variable_value":
        contexts = (
            f'x = "{a}"\ny = "{b}"\nz = "{c}"',
            f'z = "{c}"; x = "{a}"; y = "{b}"',
            f"The value of y is {b}. The value of z is {c}. The value of x is {a}.",
        )
        question, unrelated = "What is the value of x?", "What is the value of z?"
    elif task == "modifier_attachment":
        contexts = (
            f"The {a} cup is beside the {b} bowl. The {c} plate is on the shelf.",
            f"On the shelf is a {c} plate. A {a} cup is beside a {b} bowl.",
            f"Beside the bowl, which is {b}, is the cup, which is {a}. The plate is {c}.",
        )
        question, unrelated = "What color is the cup?", "What color is the plate?"
    else:
        raise ValueError(f"Unknown task: {task}")
    return contexts[template], question, unrelated


def make_pair(task: str, split: str, template: int, values: tuple[str, str, str]):
    canonical = (task, split, template, *values)
    pid = hashlib.sha256(json.dumps(canonical).encode()).hexdigest()[:16]
    a, b, c = values
    candidates = tuple(sorted(values))
    out = []
    for member, (x, y) in enumerate(((a, b), (b, a))):
        context, question, unrelated = _render(task, template, x, y, c)
        out.append(Example(
            id=f"{pid}:{member}", pair_id=pid, task=task, split=split,
            template=template, member=member, context=context, question=question,
            answer=x, unrelated_question=unrelated, unrelated_answer=c,
            candidates=candidates, bindings=tuple(zip(ROLES[task], (x, y, c))),
        ))
    return out


def generate(seed: int = 17, train_pairs: int = 192, eval_pairs: int = 48,
             dev_pairs: int = 12) -> list[Example]:
    rng = random.Random(seed)
    result = []
    for task, vocab in VOCAB.items():
        for split, count in (("dev", dev_pairs), ("train", train_pairs),
                             ("validation", eval_pairs), ("test_combinations", eval_pairs),
                             ("test_templates", eval_pairs)):
            pool = DEV_VOCAB[task] if split == "dev" else vocab
            triples = []
            for a, b, c in itertools.permutations(pool, 3):
                if a > b:
                    continue
                active = {a, b}
                if split in ("train", "test_templates") and not active <= set(vocab[:8]):
                    continue
                if split == "validation" and not (
                    len(active & set(vocab[8:10])) == 1
                    and len(active & set(vocab[:8])) == 1
                ):
                    continue
                if split == "test_combinations" and not (
                    len(active & set(vocab[10:])) == 1
                    and len(active & set(vocab[:8])) == 1
                ):
                    continue
                templates = (2,) if split == "test_templates" else (0, 1)
                triples.extend((template, (a, b, c)) for template in templates)
            if count > len(triples):
                raise ValueError(f"Requested {count} {task}/{split} pairs; only {len(triples)} exist")
            rng.shuffle(triples)
            for template, values in triples[:count]:
                result.extend(make_pair(task, split, template, values))
    audit(result)
    return result


def audit(examples: list[Example]) -> dict:
    ids, prompts, groups = set(), set(), {}
    for ex in examples:
        if ex.id in ids or ex.prompt() in prompts:
            raise ValueError("Duplicate example or prompt")
        ids.add(ex.id)
        prompts.add(ex.prompt())
        groups.setdefault(ex.pair_id, []).append(ex)
    for pair in groups.values():
        if len(pair) != 2 or {ex.member for ex in pair} != {0, 1}:
            raise ValueError("Incomplete pair")
        a, b = pair
        if (a.split, a.task, a.template) != (b.split, b.task, b.template):
            raise ValueError("Pair crosses partitions")
        def words(s):
            return Counter(re.findall(r"\w+", s))
        if words(a.prompt()) != words(b.prompt()):
            raise ValueError("Pair changes lexical content")
        if a.answer == b.answer or a.unrelated_answer != b.unrelated_answer:
            raise ValueError("Invalid counterfactual or unrelated target")
        if a.answer not in a.candidates or a.unrelated_answer not in a.candidates:
            raise ValueError("Missing answer candidate")
    heldout_counts = {}
    for task in VOCAB:
        bindings = {
            split: {binding for ex in examples if ex.task == task and ex.split == split
                    for binding in ex.bindings}
            for split in ("train", "validation", "test_combinations", "test_templates")
        }
        fit = bindings["train"]
        fit_roles, fit_fillers = {r for r, _ in fit}, {f for _, f in fit}
        for split in ("validation", "test_combinations", "test_templates"):
            for role, filler in bindings[split]:
                if role not in fit_roles or filler not in fit_fillers:
                    raise ValueError(f"Unseen individual symbol in {task}/{split}: {role}, {filler}")
        val_new = bindings["validation"] - fit
        test_new = bindings["test_combinations"] - fit
        if not val_new or not test_new or val_new & test_new:
            raise ValueError("Missing or overlapping held-out combinations")
        if bindings["test_templates"] - fit:
            raise ValueError("Template test also introduces new combinations")
        for ex in examples:
            if ex.task != task or ex.split not in ("validation", "test_combinations"):
                continue
            if not set(ex.bindings) - fit:
                raise ValueError("Combination example contains no withheld combination")
        heldout_counts[task] = {"validation": len(val_new), "test": len(test_new)}
    return {"examples": len(examples), "pairs": len(groups), "heldout": heldout_counts}


def write_dataset(path: Path, examples: list[Example]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(asdict(ex), sort_keys=True) + "\n" for ex in examples)
    path.write_text(data)
    return {**audit(examples), "sha256": hashlib.sha256(data.encode()).hexdigest()}


def read_dataset(path: Path) -> list[Example]:
    out = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        row["candidates"] = tuple(row["candidates"])
        row["bindings"] = tuple(tuple(b) for b in row["bindings"])
        out.append(Example(**row))
    audit(out)
    return out
