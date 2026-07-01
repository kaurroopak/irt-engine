from irt import bloom_mapper
from irt.bloom_mapper import BloomInfo, UnknownBloomLevelError, bucket_for, describe, difficulty_for
import pytest


def test_difficulty_for_matches_config_values():
    assert difficulty_for("Remember") == -2.0
    assert difficulty_for("understand") == -1.0
    assert difficulty_for("Apply") == 0.0
    assert difficulty_for("analyze") == 1.0
    assert difficulty_for("EVALUATE") == 2.0
    assert difficulty_for("Create") == 2.5


def test_difficulty_for_is_case_and_whitespace_insensitive():
    assert difficulty_for("  Apply  ") == difficulty_for("apply")


def test_difficulty_for_unknown_level_raises():
    with pytest.raises(UnknownBloomLevelError):
        difficulty_for("anaylze")  # common typo, should NOT silently pass


def test_difficulty_for_none_or_empty_raises():
    with pytest.raises(UnknownBloomLevelError):
        difficulty_for(None)
    with pytest.raises(UnknownBloomLevelError):
        difficulty_for("   ")


def test_bucket_for_easy_medium_hard_partition():
    assert bucket_for("Remember") == "easy"
    assert bucket_for("Understand") == "easy"
    assert bucket_for("Apply") == "medium"
    assert bucket_for("Analyze") == "hard"
    assert bucket_for("Evaluate") == "hard"
    assert bucket_for("Create") == "hard"


def test_bucket_for_unknown_level_raises():
    with pytest.raises(UnknownBloomLevelError):
        bucket_for("nonexistent_level")


def test_describe_bundles_difficulty_and_bucket():
    info = describe("Analyze")
    assert isinstance(info, BloomInfo)
    assert info.bloom_level == "analyze"
    assert info.difficulty == 1.0
    assert info.bucket == "hard"


def test_every_configured_bloom_level_has_both_difficulty_and_bucket():
    # Guards against config.py drifting out of sync (map vs buckets).
    from irt.config import BLOOM_DIFFICULTY_BUCKETS, BLOOM_DIFFICULTY_MAP

    for level in BLOOM_DIFFICULTY_MAP:
        assert level in BLOOM_DIFFICULTY_BUCKETS, f"{level} missing from buckets"
        # should not raise
        bloom_mapper.describe(level)
