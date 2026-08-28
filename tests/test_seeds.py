from src.seeds import set_seed, ALLOWED_SEEDS
import random


def test_allowed_seed_ok():
    assert set_seed(42) == 42


def test_reject_bad_seed():
    try:
        set_seed(7)
    except ValueError:
        return
    assert False, "should reject non-allowed seed in strict mode"


def test_reproducible():
    set_seed(13); a = [random.random() for _ in range(5)]
    set_seed(13); b = [random.random() for _ in range(5)]
    assert a == b
