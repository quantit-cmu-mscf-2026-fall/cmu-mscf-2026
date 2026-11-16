import pytest

from capstone.factors import store
from capstone.factors.research_contracts import digest
from capstone.factors.research_records import load_artifact, put_artifact


def test_typed_artifacts_replay_idempotently_beside_legacy_records(tmp_path):
    with store.connect(tmp_path / "research.db") as connection:
        store.add_paper(connection, "existing-paper", "Existing paper")
        claim = {"mechanism": "", "prediction": "A testable draft.", "required_inputs": ["returns"]}
        key = put_artifact(connection, "idea_claim_v2", claim)
        assert key == put_artifact(connection, "idea_claim_v2", dict(reversed(claim.items())))
        assert load_artifact(connection, key, kind="idea_claim_v2") == claim
        assert connection.execute("SELECT count(*) FROM research_artifacts").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM papers").fetchone()[0] == 1
        legacy = {"version": 1, "method": "old"}
        assert put_artifact(connection, "method_spec", legacy) == digest(legacy)


def test_artifact_kind_and_corruption_never_load_or_overwrite(tmp_path):
    with store.connect(tmp_path / "research.db") as connection:
        key = put_artifact(connection, "idea_claim_v2", {"value": 1})
        with pytest.raises(ValueError, match="kind"):
            load_artifact(connection, key, kind="assessment_v1")
        connection.execute(
            "UPDATE research_artifacts SET payload=? WHERE id=?", ('{"value":2}', key)
        )
        with pytest.raises(ValueError, match="changed"):
            load_artifact(connection, key, kind="idea_claim_v2")
        with pytest.raises(ValueError, match="differs"):
            put_artifact(connection, "idea_claim_v2", {"value": 1})


def test_reading_missing_artifact_does_not_create_a_table(tmp_path):
    with store.connect(tmp_path / "research.db") as connection:
        with pytest.raises(ValueError, match="missing"):
            load_artifact(connection, "none", kind="idea_claim_v2")
        assert not connection.execute(
            "SELECT name FROM sqlite_master WHERE name='research_artifacts'"
        ).fetchall()
