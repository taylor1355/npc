"""The ``entity_memory`` root block survives the boundary (NPC-1504).

The simulation began emitting this block on every NPC observation at sim
``dadd2a5aa`` (2026-09-14). ``Observation`` is ``extra="forbid"`` and did not
declare it, so every ``decide_action`` was refused with ``extra_forbidden`` and
every MCP NPC fell back to WAIT. Row shapes below are transcribed from
``entity_memory_observation.gd`` (``Remembered.to_dict`` / ``get_data``) at
simulation ``origin/main`` @ ``56e6df500`` via the wire fixtures.
"""

import logging

import pytest
from pydantic import ValidationError

from mind.cognitive_architecture.observations import EntityMemoryObservation, Observation
from tests.fixtures.observations import wire_entity_memory_block, wire_remembered_entity


def _observation(entity_memory: dict) -> dict:
    return {"entity_id": "npc_a", "current_simulation_time": 100, "entity_memory": entity_memory}


class TestEntityMemoryRootBlock:
    def test_root_payload_with_entity_memory_validates(self):
        """Should accept the block the simulation sends on every cycle.

        The regression itself: before the field existed this raised
        ``extra_forbidden`` at the ROOT and refused the whole observation.
        """
        block = wire_entity_memory_block(
            [
                wire_remembered_entity("apple_002", "an apple", (46, 6), present=True, age=42),
                wire_remembered_entity("bed_001", "a bed", (3, 9), present=False, age=180),
            ]
        )

        obs = Observation.model_validate(_observation(block))

        assert obs.entity_memory is not None
        assert obs.entity_memory.contract_version == 1
        assert obs.entity_memory.known_total == 2
        first, second = obs.entity_memory.remembered
        assert first.entity_id == "apple_002"
        assert first.name == "an apple"
        assert first.last_cell == (46, 6)
        assert first.present is True
        assert first.age_minutes == 42
        assert second.present is False

    def test_empty_memory_block_validates(self):
        """Should accept the block an NPC that remembers nothing still emits"""
        obs = Observation.model_validate(_observation(wire_entity_memory_block([])))

        assert obs.entity_memory is not None
        assert obs.entity_memory.remembered == []
        assert obs.entity_memory.known_total == 0

    def test_block_is_optional(self):
        """Should read None when the simulation sends no block (non-substrate entity)"""
        obs = Observation.model_validate({"entity_id": "npc_a", "current_simulation_time": 1})

        assert obs.entity_memory is None

    def test_truncated_list_keeps_the_uncapped_total(self):
        """Should carry ``known_total`` rather than re-deriving it from the capped list"""
        block = wire_entity_memory_block(
            [wire_remembered_entity("apple_002", "an apple", (46, 6))], known_total=9
        )

        mem = EntityMemoryObservation.model_validate(block)

        assert mem.known_total == 9
        assert len(mem.remembered) == 1


class TestEntityMemoryContract:
    def test_undeclared_block_key_is_refused(self):
        """The block ROOT stays strict under a known version: the contract promises
        additive growth for rows, not for the block."""
        block = wire_entity_memory_block([])
        block["confidence"] = 0.5

        with pytest.raises(ValidationError):
            EntityMemoryObservation.model_validate(block)

    def test_additive_row_key_degrades_rather_than_refusing(self, caplog):
        """The contract promises "rows grow keys under the *same*
        ``contract_version``", so an unlearned row key is dropped with a WARNING
        naming it, and the row still parses."""
        row = wire_remembered_entity("apple_002", "an apple", (46, 6), age=7)
        row["expected_yield"] = 3.0

        with caplog.at_level(logging.WARNING):
            obs = Observation.model_validate(_observation(wire_entity_memory_block([row])))

        parsed = obs.entity_memory.remembered[0]
        assert parsed.entity_id == "apple_002"
        assert parsed.age_minutes == 7
        warnings = [r for r in caplog.records if "'expected_yield'" in r.getMessage()]
        assert len(warnings) == 1
        assert warnings[0].levelno == logging.WARNING
        assert "RememberedEntity" in warnings[0].getMessage()

    @pytest.mark.parametrize("key", ["entity_id", "last_cell", "present", "age_minutes"])
    def test_evidence_fields_are_required(self, key):
        """Should refuse a row missing a field no default could honestly fill.

        ``present`` in particular: a default of True fabricates a belief and
        False fabricates a disproof.
        """
        row = wire_remembered_entity("apple_002", "an apple", (46, 6))
        del row[key]

        with pytest.raises(ValidationError):
            EntityMemoryObservation.model_validate(wire_entity_memory_block([row]))

    def test_unknown_contract_version_warns_and_degrades(self, caplog):
        """A simulation ahead of this mind must not be able to kill the cycle.

        The probe carries an unknown version AND a novel root key, which is what
        a real additive future version looks like: the key is shed, the declared
        fields still parse, and the warning names both versions.
        """
        block = wire_entity_memory_block(
            [wire_remembered_entity("apple_002", "an apple", (46, 6), age=5)]
        )
        block["contract_version"] = 2
        block["a_key_from_the_future"] = {"anything": True}

        with caplog.at_level(logging.WARNING):
            obs = Observation.model_validate(_observation(block))

        assert obs.entity_memory.contract_version == 2
        assert obs.entity_memory.remembered[0].entity_id == "apple_002"
        assert "unknown contract_version 2" in caplog.text
        assert "[1]" in caplog.text

    def test_known_version_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING):
            EntityMemoryObservation.model_validate(wire_entity_memory_block([]))

        assert "contract_version" not in caplog.text
