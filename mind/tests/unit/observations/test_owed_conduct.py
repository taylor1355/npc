"""FS10b lane 0b: ``relationship.owed`` and ``conduct`` on visible entities."""

import pytest
from pydantic import ValidationError

from mind.cognitive_architecture.observations import (
    ConductState,
    EntityData,
    RecalledDeed,
    RelationshipState,
    VisionObservation,
)
from tests.unit.observations.test_observation_rendering import _base_observation


def _text(entity: EntityData) -> str:
    return str(_base_observation(vision=VisionObservation(visible_entities=[entity])))


def _entity(**kw) -> EntityData:
    return EntityData(entity_id="alice", display_name="Alice", position=(6, 10), **kw)


class TestOwedParsing:
    def test_absent_owed_defaults_to_zero(self):
        rel = RelationshipState.model_validate(
            {"familiarity": 0.5, "sentiment": 0.1, "interaction_count": 2}
        )
        assert rel.owed == 0.0

    def test_present_owed_parses(self):
        rel = RelationshipState.model_validate({"familiarity": 0.5, "sentiment": 0.1, "owed": 0.4})
        assert rel.owed == pytest.approx(0.4)

    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_owed_is_bounded(self, bad):
        with pytest.raises(ValidationError):
            RelationshipState(familiarity=0.5, sentiment=0.0, owed=bad)


class TestConductParsing:
    def test_full_wire_shape_parses(self):
        # Shape of EntityData.Conduct.to_wire: notable replaced last_act/patient/place/minutes_ago.
        c = ConductState.model_validate(
            {
                "reputation": 0.4,
                "evidence": 2.0,
                "notable": [
                    {
                        "act": "gift",
                        "patients": ["Bob"],
                        "place": "Market",
                        "minutes_ago": 12.0,
                        "role": "patient",
                    },
                    {"act": "tip", "patients": [], "minutes_ago": 900.0, "role": "perceiver"},
                ],
                "last_cell": [3, 4],
            }
        )
        assert [d.act for d in c.notable] == ["gift", "tip"]
        assert c.notable[0] == RecalledDeed(
            act="gift", patients=["Bob"], place="Market", minutes_ago=12.0, role="patient"
        )
        assert c.notable[1].place is None
        assert c.last_cell == (3, 4)

    def test_minimal_wire_shape_parses_with_parts_absent(self):
        c = ConductState.model_validate({"reputation": -0.2, "evidence": 0.0})
        assert c.notable == []
        assert c.last_cell is None

    def test_retired_deed_fields_are_gone(self):
        assert not hasattr(ConductState(), "last_act")
        assert "minutes_ago" not in ConductState.model_fields

    def test_unknown_keys_are_ignored(self):
        assert ConductState.model_validate({"reputation": 0.0, "future": 1}).reputation == 0.0

    @pytest.mark.parametrize("bad", [-1.01, 1.01])
    def test_reputation_is_bounded_like_the_simulation_clamp(self, bad):
        with pytest.raises(ValidationError):
            ConductState(reputation=bad)

    def test_entity_without_conduct_key_has_none(self):
        assert _entity().conduct is None

    def test_entity_parses_conduct_key(self):
        e = EntityData.model_validate(
            {
                "entity_id": "a",
                "display_name": "A",
                "position": [1, 2],
                "conduct": {"reputation": 0.3, "evidence": 1.0},
            }
        )
        assert e.conduct is not None and e.conduct.evidence == 1.0


class TestRendering:
    def test_owed_renders_on_the_relationship_line(self):
        text = _text(
            _entity(relationship=RelationshipState(familiarity=0.6, sentiment=0.3, owed=0.45))
        )
        assert "you owe them 0.45" in text

    def test_no_owed_is_byte_identical_to_the_legacy_line(self):
        with_default = _text(
            _entity(relationship=RelationshipState(familiarity=0.6, sentiment=0.3))
        )
        assert "owe" not in with_default
        expected = "    Relationship: familiarity 0.60, sentiment +0.30, 0 shared interactions"
        assert expected in with_default.splitlines()

    def test_conduct_absent_is_byte_identical(self):
        base = _text(_entity())
        assert "Conduct:" not in base
        assert base == _text(_entity(conduct=None))

    def test_conduct_with_deed_renders(self):
        text = _text(
            _entity(
                conduct=ConductState(
                    reputation=0.4,
                    evidence=2.0,
                    notable=[
                        RecalledDeed(
                            act="gift",
                            patients=["Bob"],
                            place="Market",
                            minutes_ago=12.0,
                            role="patient",
                        ),
                        RecalledDeed(act="tip", minutes_ago=900.0),
                    ],
                    last_cell=(3, 4),
                )
            )
        )
        assert (
            "Conduct: reputation +0.40 (evidence 2.0); remembers: "
            "gift to Bob at Market (done to you), just now; tip, earlier today; "
            "last seen at cell (3, 4)" in text
        )

    def test_conduct_unjudged_without_deed(self):
        assert "Conduct: seen acting, not yet judged" in _text(_entity(conduct=ConductState()))
