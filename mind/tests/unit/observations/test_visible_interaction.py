"""Tests for VisibleInteraction -- the interaction a VISIBLE entity is in.

The simulation has emitted this per visible entity since 2026-08-11, and the
mind tier declared no field for it, so pydantic's ``extra="ignore"`` discarded
it silently on arrival (NPC-1323). These tests pin the four things that made
that failure possible and would make it possible again:

* the payload parses at all, from the simulation's real key set;
* the idle sentinel ``{}`` does NOT become a phantom interaction;
* the rendered line reaches ``Observation.__str__``, which is the only place
  the LLM can see it;
* this model is not confused with ``StatusObservation.current_interaction``,
  which shares its wire key and nothing else.
"""

import pytest

from mind.cognitive_architecture.observations import (
    EntityData,
    Observation,
    StatusObservation,
    VisibleInteraction,
    VisionObservation,
)

# The exact key set of the simulation's InteractionSummary.to_dict(), including
# `extra`, which this model deliberately does not declare. Written out in full
# rather than built by a helper: the point of the fixture is to be a faithful
# copy of the wire, so drift between the two shows up here as a failure.
WIRE_PAYLOAD = {
    "interaction_id": "int_42",
    "interaction_name": "Conversation",
    "participant_ids": ["npc_alice", "npc_bob"],
    "participant_names": ["Alice", "Bob"],
    "participant_count": 2,
    "max_participants": 4,
    "min_participants": 2,
    "duration_minutes": 3.4,
    "is_joinable": True,
    "joinable_reason": "",
    "extra": {"message_count": 7},
}


def _entity(current_interaction):
    return EntityData(
        entity_id="npc_alice",
        display_name="Alice",
        position=(3, 4),
        current_interaction=current_interaction,
    )


class TestParsing:
    def test_parses_the_simulations_full_payload(self):
        entity = _entity(WIRE_PAYLOAD)

        assert entity.current_interaction is not None
        assert entity.current_interaction.interaction_name == "Conversation"
        assert entity.current_interaction.participant_names == ["Alice", "Bob"]
        assert entity.current_interaction.participant_count == 2
        assert entity.current_interaction.max_participants == 4
        assert entity.current_interaction.duration_minutes == pytest.approx(3.4)
        assert entity.current_interaction.is_joinable is True

    def test_undeclared_extra_is_dropped_without_raising(self):
        """`extra` is the simulation's subclass escape hatch, documented as
        not-for-generic-consumers, so dropping it is intended. Pinned so a
        future ``model_config = ConfigDict(extra="forbid")`` cannot silently
        turn every visible entity's payload into a ValidationError -- which on
        the prompt path collapses the cycle into the WAIT fallback."""
        entity = _entity(WIRE_PAYLOAD)

        assert not hasattr(entity.current_interaction, "extra")

    def test_absent_key_is_none(self):
        assert _entity(None).current_interaction is None


class TestIdleSentinel:
    """The simulation emits ``{}`` for an idle entity rather than omitting the key.

    Every field has a default, so without the before-validator ``{}`` parses
    into an all-default model and the renderer announces a nameless,
    zero-participant interaction for every idle entity in view.
    """

    @pytest.mark.parametrize("idle", [{}, None])
    def test_idle_payload_is_none_not_an_empty_interaction(self, idle):
        assert _entity(idle).current_interaction is None

    def test_idle_entity_renders_no_interaction_line(self):
        obs = Observation(
            entity_id="npc_observer",
            current_simulation_time=0,
            vision=VisionObservation(visible_entities=[_entity({})]),
        )

        assert "In " not in str(obs)
        assert "joinable" not in str(obs)


class TestRenderSummary:
    def test_joinable_names_participants_and_capacity(self):
        line = VisibleInteraction(**WIRE_PAYLOAD).render_summary()

        assert "In Conversation" in line
        assert "with Alice, Bob" in line
        assert "2 of 4" in line
        assert "3 min so far" in line
        assert line.endswith("joinable")
        assert "not joinable" not in line

    def test_refusal_reason_is_surfaced_before_the_bid(self):
        """The reason is the load-bearing half: an NPC deciding whether to
        approach a group needs it BEFORE bidding, not as a rejected bid."""
        payload = WIRE_PAYLOAD | {
            "participant_count": 4,
            "is_joinable": False,
            "joinable_reason": "at_capacity",
        }

        line = VisibleInteraction(**payload).render_summary()

        assert "not joinable: at capacity" in line

    def test_not_joinable_without_a_reason_says_so_plainly(self):
        payload = WIRE_PAYLOAD | {"is_joinable": False, "joinable_reason": ""}

        line = VisibleInteraction(**payload).render_summary()

        assert line.endswith("not joinable")

    def test_unnamed_interaction_falls_back_rather_than_rendering_empty(self):
        payload = WIRE_PAYLOAD | {"interaction_name": "   "}

        assert "In interaction" in VisibleInteraction(**payload).render_summary()

    def test_solo_and_zero_duration_omit_their_clauses(self):
        payload = WIRE_PAYLOAD | {
            "participant_names": [],
            "duration_minutes": 0.0,
            "max_participants": 1,
            "participant_count": 1,
        }

        line = VisibleInteraction(**payload).render_summary()

        assert "with" not in line
        assert "so far" not in line
        assert "1 of 1" in line


class TestReachesThePrompt:
    """A declared field the renderer never prints is the same failure as an
    undeclared one: the LLM cannot see either."""

    def test_interaction_line_appears_in_observation_str(self):
        obs = Observation(
            entity_id="npc_observer",
            current_simulation_time=0,
            vision=VisionObservation(visible_entities=[_entity(WIRE_PAYLOAD)]),
        )

        rendered = str(obs)

        # "with Bob", not "with Alice, Bob" -- the entity is excluded from its
        # own companion list; see TestSelfExclusion.
        assert "In Conversation with Bob" in rendered
        assert "joinable" in rendered

    def test_rendered_under_the_entity_it_describes(self):
        """Two visible entities, one interacting -- the line must attach to the
        right one. The prompt is flat text, so misattribution is invisible to
        any assertion that only greps the whole blob."""
        idle = EntityData(entity_id="npc_bob", display_name="Bob", position=(9, 9))
        obs = Observation(
            entity_id="npc_observer",
            current_simulation_time=0,
            vision=VisionObservation(visible_entities=[_entity(WIRE_PAYLOAD), idle]),
        )

        lines = str(obs).splitlines()
        alice_at = next(n for n, ln in enumerate(lines) if "Alice (ID: npc_alice" in ln)
        bob_at = next(n for n, ln in enumerate(lines) if "Bob (ID: npc_bob" in ln)
        interaction_at = next(n for n, ln in enumerate(lines) if "In Conversation" in ln)

        assert alice_at < interaction_at < bob_at


class TestNotConfusedWithStatusObservation:
    """``StatusObservation.current_interaction`` is the OBSERVER's own
    interaction and carries ``Interaction.to_dict()`` -- a different shape
    behind the same wire key. Conflating the two is the NPC-1278 shape one
    boundary over."""

    def test_the_two_fields_have_different_types(self):
        status = StatusObservation(position=(0, 0), current_interaction={"name": "Eating"})
        entity = _entity(WIRE_PAYLOAD)

        assert isinstance(status.current_interaction, dict)
        assert isinstance(entity.current_interaction, VisibleInteraction)

    def test_status_shaped_payload_does_not_masquerade_as_a_visible_interaction(self):
        """An ``Interaction.to_dict()`` payload carries none of
        VisibleInteraction's keys. It parses (every field defaults) but must not
        claim a name it never carried -- the renderer falls back rather than
        reaching for ``name``."""
        status_shaped = {"name": "Eating", "description": "eating an apple"}

        parsed = _entity(status_shaped).current_interaction

        assert parsed is not None
        assert parsed.interaction_name == ""
        assert "In interaction" in parsed.render_summary()
        assert "Eating" not in parsed.render_summary()


class TestSelfExclusion:
    """The simulation's participant lists include the observed entity itself,
    so a verbatim render says "Alice ... In Conversation with Alice, Bob".
    Caught by reading the rendered prompt, not by any assertion that existed
    at the time -- which is why these exist now.
    """

    def test_observed_entity_is_not_listed_as_its_own_companion(self):
        line = VisibleInteraction(**WIRE_PAYLOAD).render_summary("npc_alice")

        assert "with Bob" in line
        assert "Alice" not in line

    def test_occupancy_count_is_not_adjusted_by_the_exclusion(self):
        """The count is capacity information. An observer deciding whether to
        join needs real occupancy, not occupancy-minus-one."""
        line = VisibleInteraction(**WIRE_PAYLOAD).render_summary("npc_alice")

        assert "2 of 4" in line

    def test_without_an_id_every_participant_is_listed(self):
        line = VisibleInteraction(**WIRE_PAYLOAD).render_summary()

        assert "with Alice, Bob" in line

    def test_unknown_id_excludes_nobody(self):
        line = VisibleInteraction(**WIRE_PAYLOAD).render_summary("npc_stranger")

        assert "with Alice, Bob" in line

    def test_exclusion_is_by_id_not_by_display_name(self):
        """Two participants sharing a display name must not both vanish."""
        payload = WIRE_PAYLOAD | {
            "participant_ids": ["npc_alice", "npc_alice_2"],
            "participant_names": ["Alice", "Alice"],
        }

        line = VisibleInteraction(**payload).render_summary("npc_alice")

        assert "with Alice" in line
        assert "Alice, Alice" not in line

    def test_misaligned_lists_keep_every_name_rather_than_guessing(self):
        """Pairing is unsafe when the lists disagree in length. A redundant
        name reads oddly; a wrongly dropped one misinforms."""
        payload = WIRE_PAYLOAD | {"participant_ids": ["npc_alice"]}

        line = VisibleInteraction(**payload).render_summary("npc_alice")

        assert "with Alice, Bob" in line

    def test_solo_interaction_renders_no_companion_clause(self):
        payload = WIRE_PAYLOAD | {
            "participant_ids": ["npc_alice"],
            "participant_names": ["Alice"],
            "participant_count": 1,
        }

        line = VisibleInteraction(**payload).render_summary("npc_alice")

        assert "with" not in line
        assert "In Conversation" in line

    def test_observation_str_passes_the_entity_id_through(self):
        """The exclusion is only useful if the renderer actually supplies the
        id -- a correct helper called without its argument is inert."""
        obs = Observation(
            entity_id="npc_observer",
            current_simulation_time=0,
            vision=VisionObservation(visible_entities=[_entity(WIRE_PAYLOAD)]),
        )

        rendered = str(obs)

        assert "In Conversation with Bob" in rendered
        assert "with Alice, Bob" not in rendered
