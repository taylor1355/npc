"""The ``taste`` root block survives the boundary and reaches the prompt (NPC-1308).

The simulation sends an NPC's learned tastes as band TOKENS, never numbers, in
their own block. Row shapes are transcribed from ``taste_observation.gd``
(``get_data``) via the wire fixtures. The mind owns the words: tokens render
through a phrase table here, and a token with no wording is withheld and logged,
not refused and not rendered raw.
"""

import logging

import pytest
from pydantic import ValidationError

from mind.cognitive_architecture.observations import Observation, TasteObservation
from tests.fixtures.observations import wire_taste_block, wire_taste_row


def _observation(taste: dict) -> dict:
    return {"entity_id": "npc_a", "current_simulation_time": 100, "taste": taste}


class TestTasteRootBlock:
    def test_root_payload_with_tastes_validates(self):
        block = wire_taste_block(
            [
                wire_taste_row("likes", "consume", ("stew_001", "Stew")),
                wire_taste_row("avoids", "tend_plant"),
            ]
        )

        obs = Observation.model_validate(_observation(block))

        assert obs.taste is not None
        assert obs.taste.contract_version == 1
        first, second = obs.taste.tastes
        assert (first.band, first.interaction_type, first.target_id, first.target_label) == (
            "likes",
            "consume",
            "stew_001",
            "Stew",
        )
        assert second.band == "avoids"
        assert second.interaction_label == "tend plant"
        assert second.target_id is None and second.target_label is None

    def test_empty_block_validates_and_is_not_absent(self):
        """An NPC with no settled tastes sends [], which is not the same as no block"""
        obs = Observation.model_validate(_observation(wire_taste_block([])))

        assert obs.taste is not None
        assert obs.taste.tastes == []

    def test_block_is_optional(self):
        obs = Observation.model_validate({"entity_id": "npc_a", "current_simulation_time": 1})

        assert obs.taste is None

    def test_an_undeclared_row_key_degrades_and_is_logged(self, caplog):
        row = wire_taste_row("likes", "consume")
        row["intensity_word"] = "strongly"

        with caplog.at_level(logging.WARNING):
            obs = Observation.model_validate(_observation(wire_taste_block([row])))

        assert obs.taste is not None and obs.taste.tastes[0].band == "likes"
        assert any("intensity_word" in r.getMessage() for r in caplog.records)

    def test_an_undeclared_block_root_key_under_a_known_version_is_refused(self):
        block = wire_taste_block([])
        block["total"] = 0

        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(block))

    def test_an_unknown_contract_version_degrades(self, caplog):
        block = wire_taste_block([wire_taste_row("likes", "consume")])
        block["contract_version"] = 99
        block["total"] = 1

        with caplog.at_level(logging.WARNING):
            obs = Observation.model_validate(_observation(block))

        assert obs.taste is not None and len(obs.taste.tastes) == 1
        assert any("unknown contract_version" in r.getMessage() for r in caplog.records)


class TestTasteRendering:
    def test_tokens_render_as_second_person_words_with_targets(self):
        taste = TasteObservation.model_validate(
            wire_taste_block(
                [
                    wire_taste_row("likes", "consume", ("stew_001", "Stew")),
                    wire_taste_row("leans_toward", "sleep"),
                    wire_taste_row("leans_away", "conversation", ("npc_bob", "Bob")),
                    wire_taste_row("avoids", "tend_plant"),
                ]
            )
        )

        assert taste.render_summary() == (
            "Tastes: you like consume (Stew), lean toward sleep, "
            "lean away from conversation (Bob), avoid tend plant."
        )

    def test_the_prompt_carries_the_tastes_and_no_number(self):
        obs = Observation.model_validate(
            _observation(wire_taste_block([wire_taste_row("likes", "consume", ("s", "Stew"))]))
        )

        assert "Tastes: you like consume (Stew)." in str(obs)
        line = obs.taste.render_summary()
        assert line == "Tastes: you like consume (Stew)."
        assert not any(ch.isdigit() for ch in line)

    def test_no_tastes_renders_nothing(self):
        """Control arm: an empty block leaves the prompt byte-identical to no block"""
        base = {"entity_id": "npc_a", "current_simulation_time": 1}
        empty = {**base, "taste": wire_taste_block([])}

        assert str(Observation.model_validate(empty)) == str(Observation.model_validate(base))

    def test_an_unworded_band_is_withheld_and_logged_once_not_rendered_raw(self, caplog):
        with caplog.at_level(logging.WARNING):
            taste = TasteObservation.model_validate(
                wire_taste_block(
                    [wire_taste_row("adores", "consume"), wire_taste_row("likes", "sleep")]
                )
            )
            text = taste.render_summary()
            taste.render_summary()

        assert text == "Tastes: you like sleep."
        assert "adores" not in text
        assert sum("adores" in r.getMessage() for r in caplog.records) == 1

    def test_only_unworded_bands_render_nothing(self):
        taste = TasteObservation.model_validate(wire_taste_block([wire_taste_row("adores", "x")]))

        assert taste.render_summary() == ""
