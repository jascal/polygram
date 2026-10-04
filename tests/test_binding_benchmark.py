"""Scientific invariants for the binding-preservation research dataset."""

from dataclasses import replace

import pytest

from examples.binding_benchmark.data import (
    ROLES, VOCAB, audit, generate, read_dataset, write_dataset,
)


@pytest.fixture(scope="module")
def examples():
    return generate()


def test_all_tasks_have_known_symbols_but_reserved_bindings(examples):
    audit(examples)
    for task, vocab in VOCAB.items():
        train = {b for ex in examples if ex.task == task and ex.split == "train"
                 for b in ex.bindings}
        for filler in vocab[8:]:
            assert (ROLES[task][2], filler) in train
            assert (ROLES[task][0], filler) not in train
            assert (ROLES[task][1], filler) not in train


def test_minimal_pairs_change_only_the_two_target_bindings(examples):
    for a, b in zip(examples[::2], examples[1::2]):
        assert a.bindings[0][1] == b.bindings[1][1] == a.answer
        assert a.bindings[1][1] == b.bindings[0][1] == b.answer
        assert a.bindings[2] == b.bindings[2]
        assert a.question == b.question
        assert a.unrelated_question == b.unrelated_question


def test_duplicate_prompt_and_cross_split_pair_rejected(examples):
    with pytest.raises(ValueError, match="Duplicate"):
        audit(examples + [replace(examples[0], id="different", pair_id="different")])
    broken = list(examples)
    broken[0] = replace(broken[0], split="test_combinations")
    with pytest.raises(ValueError, match="crosses"):
        audit(broken)


def test_reproducible_manifest_and_roundtrip(examples, tmp_path):
    assert examples == generate()
    path = tmp_path / "dataset.jsonl"
    manifest = write_dataset(path, examples)
    assert read_dataset(path) == examples
    assert manifest["pairs"] == 1044
    assert len(manifest["sha256"]) == 64


def test_summary_uses_both_members_for_pair_gate():
    from examples.binding_benchmark.host import summarize

    rows = [dict(task="t", split="dev", pair_id=str(i // 2), correct=i != 0,
                 first_token_correct=False, margin_nats=0) for i in range(20)]
    summary = summarize(rows)["t/dev"]
    assert summary["accuracy"] == 0.95
    assert summary["pair_accuracy"] == 0.9
    assert summary["competence_pass"]
