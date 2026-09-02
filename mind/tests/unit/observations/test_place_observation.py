"""The ``place`` observation block (NPC-1299).

NPC-1299 adds a ``PlaceObservation`` to the composite the simulation sends to
MCP. ``Observation`` is ``extra="forbid"``, so without a declared field the
block would not merely go unread -- the whole observation would be REFUSED at
the parse boundary, taking every MCP NPC to wait every cycle until a deploy.

This module is therefore written AHEAD of its producer, against the approved
specification; see the provenance note on ``PLACE_BLOCK_CONTRACT_SAMPLE`` for
what re-derives it once that producer lands.
"""

import pytest
from pydantic import ValidationError

from mind.cognitive_architecture.observations import (
    MarkBudgetState,
    Observation,
    PlaceDescriptor,
    PlaceKnowledgeSource,
    PlaceObservation,
)

# The ``place`` block's wire shape.
#
# PROVENANCE: pinned against SHIPPED simulation code, re-derived from
# ``src/minds/observations/place_observation.gd::get_data`` and
# ``place_descriptor.gd::to_dict`` on the NPC-1299 branch carrying NPC-1473.
#
# It was previously pinned against an APPROVED SPEC whose producer had not
# merged (``docs/plans/NPC-1299.md`` section 2.1), and the note here said to
# re-derive once it landed and to treat any mismatch as a contract
# disagreement rather than a test to edit. That is exactly what happened: the
# implementation emitted ``current_place`` / ``known_places`` / ``target_place``
# where the plan illustrated ``here`` / ``known`` / ``target``, and sent a FULL
# descriptor where the model had narrowed ``current_place`` to three fields.
# With extra="forbid" on both this model and Observations, that raised and
# refused the WHOLE observation -- every MCP NPC to wait, every cycle. The
# distinction the old note drew is what caught it, so keep drawing it.
#
# ``told_by`` is absent from the non-TOLD descriptors ON PURPOSE: the wire omits
# the key except for TOLD, and test_told_by_is_omitted_except_for_told pins it.
# ``witnessed`` gates ``affords`` / ``provider_count`` / ``witnessed_age_minutes``
# the same way -- the producer omits all three when it is false.
PLACE_BLOCK_CONTRACT_SAMPLE = {
    "contract_version": 2,
    # A FULL descriptor: the producer sends current_place.to_dict() off the same
    # PlaceDescriptor it puts in known_places, where this place also appears.
    "current_place": {
        "zone_id": "zone_berry",
        "name": "the berry grounds",
        "kind": "gathering_ground",
        "anchor": [12, 34],
        "distance": 0,
        "witnessed": True,
        "affords": ["harvest"],
        "provider_count": 4,
        "witnessed_age_minutes": 5,
        "source": "created",
        "age_minutes": 120,
        "beyond_vision": False,
        "confidence": 0.95,
    },
    "known_places": [
        {
            "zone_id": "zone_berry",
            "name": "the berry grounds",
            "kind": "gathering_ground",
            "anchor": [12, 34],
            "distance": 0,
            "witnessed": True,
            "affords": ["harvest"],
            "provider_count": 4,
            "witnessed_age_minutes": 5,
            "source": "created",
            "age_minutes": 120,
            "beyond_vision": False,
            "confidence": 0.95,
        },
        {
            "zone_id": "zone_pond",
            "name": "the pond bend",
            "kind": "gathering_ground",
            "anchor": [30, 34],
            "distance": 18,
            "witnessed": True,
            "affords": ["drink"],
            "provider_count": 1,
            "witnessed_age_minutes": 40,
            "source": "told",
            "told_by": "npc_bo",
            "age_minutes": 40,
            "beyond_vision": True,
            "confidence": 0.4,
        },
        {
            "zone_id": "zone_green",
            "name": "the green",
            "kind": "gathering_ground",
            "anchor": [43, 34],
            "distance": 31,
            "witnessed": True,
            "affords": [],
            "provider_count": 0,
            "witnessed_age_minutes": 900,
            "source": "visited",
            "age_minutes": 900,
            "beyond_vision": True,
            "confidence": 0.7,
        },
    ],
    "known_total": 7,
    "target_place": {
        "zone_id": "zone_pond",
        "name": "the pond bend",
        "kind": "gathering_ground",
        "anchor": [30, 34],
        "distance": 18,
        "witnessed": True,
        "affords": ["drink"],
        "provider_count": 1,
        "witnessed_age_minutes": 40,
        "source": "told",
        "told_by": "npc_bo",
        "age_minutes": 40,
        "beyond_vision": True,
        "confidence": 0.4,
    },
    # Verbatim from section 2.1's illustration, including the ``active < cap``
    # with a positive wait. NOTE FOR THE SIMULATION SIDE: ``MarkBudget.
    # minutes_until_next_slot`` returns ``-1.0`` whenever a slot is free, so this
    # pair cannot arise from that accessor. It is pinned as written because the
    # spec is frozen; the renderer reads occupancy from active/cap and therefore
    # renders correctly under BOTH readings. See MarkBudgetState's docstring.
    "mark_budget": {"active": 1, "cap": 3, "next_slot_in_minutes": 42.0},
}


def _observation(place: dict | None) -> dict:
    payload = {"entity_id": "npc_alice", "current_simulation_time": 100}
    if place is not None:
        payload["place"] = place
    return payload


class TestPlaceBlockParsing:
    def test_the_contract_sample_parses_into_the_observation(self):
        observation = Observation.model_validate(_observation(PLACE_BLOCK_CONTRACT_SAMPLE))

        place = observation.place
        assert place is not None
        assert place.contract_version == 2
        assert place.known_total == 7
        assert place.current_place.name == "the berry grounds"
        assert [p.zone_id for p in place.known_places] == ["zone_berry", "zone_pond", "zone_green"]
        assert place.target_place.zone_id == "zone_pond"
        assert place.mark_budget.cap == 3

    def test_an_undeclared_root_key_is_still_refused(self):
        """Control arm: the test above could have gone red.

        ``extra="forbid"`` is what made the place block a hard parse failure
        rather than a silent drop, and it is still live -- so the sample parsing
        cleanly is evidence that ``place`` is DECLARED, not evidence that the
        observation stopped checking. Without this arm the test above would pass
        just as happily against a model that forbade nothing.
        """
        with pytest.raises(ValidationError):
            Observation.model_validate(
                {
                    "entity_id": "npc_alice",
                    "current_simulation_time": 100,
                    "place_knowledge": {"contract_version": 1},
                }
            )

    def test_the_place_block_is_optional(self):
        """The property that removes every cross-repository ordering constraint.

        This model may merge and deploy before the simulation emits the block,
        and the simulation may merge first. Making it required would manufacture
        exactly the merge-order hazard this lane was designed to avoid.
        """
        observation = Observation.model_validate(_observation(None))
        assert observation.place is None

    def test_an_undeclared_key_inside_the_place_block_is_refused(self):
        """``extra="forbid"`` on the block itself, matching every Goal* model."""
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation({"contract_version": 1, "invented_key": 1}))

    def test_an_unknown_contract_version_degrades_rather_than_raising(self, caplog):
        """A simulation ahead of this mind must not be able to kill the cycle.

        Raising would collapse ``decide_action`` into an error response -- an NPC
        that silently stops acting because a version number moved. Undeclared
        root keys are shed so a purely additive v2 parses despite forbid.
        """
        observation = Observation.model_validate(
            _observation(
                {
                    "contract_version": 99,
                    "known_total": 2,
                    "a_key_from_the_future": {"anything": True},
                }
            )
        )
        assert observation.place.known_total == 2
        assert "unknown contract_version 99" in caplog.text

    def test_told_by_is_omitted_except_for_told(self):
        """Provenance is read from ``source``, never from the emptiness of told_by."""
        place = PlaceObservation.model_validate(PLACE_BLOCK_CONTRACT_SAMPLE)
        by_id = {p.zone_id: p for p in place.known_places}

        assert by_id["zone_pond"].source == PlaceKnowledgeSource.TOLD
        assert by_id["zone_pond"].told_by == "npc_bo"
        assert by_id["zone_berry"].source == PlaceKnowledgeSource.CREATED
        assert by_id["zone_berry"].told_by == ""

    def test_an_unknown_source_is_refused(self):
        """``PlaceKnowledge.Source`` is a closed three-member enum with a shipped
        save vocabulary, so a fourth value is a breaking change and deserves to
        fail loudly -- the ``ValenceBand`` posture, not the free-string one used
        for simulation-owned open registries like interaction names."""
        with pytest.raises(ValidationError):
            PlaceDescriptor.model_validate({"zone_id": "z", "source": "overheard"})

    def test_the_anchor_reads_the_wire_pair(self):
        """Godot has no JSON vector type; every observation converts to ``[x, y]``."""
        descriptor = PlaceDescriptor.model_validate({"zone_id": "z", "anchor": [12, 34]})
        assert descriptor.anchor == (12, 34)


class TestMarkBudgetRendering:
    def test_a_free_slot_renders_no_wait(self):
        assert (
            MarkBudgetState(active=1, cap=3, next_slot_in_minutes=-1.0).render_summary()
            == "You are holding 1 of 3 marks."
        )

    def test_a_full_budget_names_the_wait(self):
        assert (
            MarkBudgetState(active=3, cap=3, next_slot_in_minutes=42.0).render_summary()
            == "You are holding 3 of 3 marks; the next frees in 42 minutes."
        )

    def test_the_negative_sentinel_never_renders_as_a_duration(self):
        """``-1.0`` is a flag wearing a number's clothes.

        The simulation returns it whenever a slot is free, deliberately not
        ``0.0``, because a genuinely-zero wait is a real answer. Formatting it
        would put "the next frees in -1 minutes" in front of the model.
        """
        rendered = MarkBudgetState(active=3, cap=3, next_slot_in_minutes=-1.0).render_summary()
        assert "-1" not in rendered
        assert rendered == "You are holding 3 of 3 marks."


class TestPlaceRendering:
    def _rendered(self) -> str:
        return str(Observation.model_validate(_observation(PLACE_BLOCK_CONTRACT_SAMPLE)))

    def test_the_block_reaches_the_prompt_prose(self):
        """``str(state.observation)`` is the ONLY channel by which an observation
        reaches the LLM, so a parsed-but-unrendered block is perceived by
        nothing."""
        rendered = self._rendered()
        assert "You are at the berry grounds." in rendered
        assert "Places you know" in rendered

    def test_the_listing_is_name_first(self):
        """A name is what one NPC can say to another; a zone id is not."""
        rendered = self._rendered()
        assert "the pond bend (18 away" in rendered
        assert "zone_pond" not in rendered

    def test_a_truncated_listing_says_so(self):
        """ "I know three places" and "I know thirty and was shown three" are
        different facts, and only ``known_total`` separates them."""
        assert "(3 of 7)" in self._rendered()

    def test_an_untruncated_listing_omits_the_count(self):
        """The scope clause is spent only when it tells the model something it
        cannot already see -- these tokens are uncached, every cycle."""
        sample = dict(PLACE_BLOCK_CONTRACT_SAMPLE, known_total=3)
        rendered = str(Observation.model_validate(_observation(sample)))
        assert "Places you know:" in rendered
        assert " of 3)" not in rendered

    def test_provenance_and_affordances_are_carried(self):
        rendered = self._rendered()
        assert "told by npc_bo" in rendered
        assert "harvest x4" in rendered
        assert "you named it" in rendered

    def test_the_place_you_stand_on_is_marked_here_not_by_distance(self):
        assert "the berry grounds (here" in self._rendered()

    def test_an_empty_place_block_renders_nothing(self):
        """An NPC that knows no places reads byte-identically to the pre-place
        world, which is what lets every existing fixture stand as a control."""
        with_empty = str(Observation.model_validate(_observation({"contract_version": 1})))
        without = str(Observation.model_validate(_observation(None)))
        assert with_empty == without

    def test_rendering_survives_a_nameless_place(self):
        """Runs on the prompt path, where an exception collapses the cycle."""
        rendered = PlaceObservation.model_validate(
            {"current_place": {"zone_id": "zone_x"}, "known_places": [{"zone_id": "zone_x"}]}
        ).render_summary()
        assert "zone_x" in rendered
