"""Rendering tests for Observation.__str__

``str(observation)`` is the ``observation_text`` the reflection prompt
consumes, so anything that does not render here is invisible to the LLM no
matter how faithfully it crosses the wire.
"""

from mind.cognitive_architecture.observations import (
    ArousalBand,
    ConversationMessage,
    ConversationObservation,
    EntityData,
    GoalDetail,
    GoalObservation,
    InventoryObservation,
    MoodObservation,
    NeedsObservation,
    Observation,
    RelationshipState,
    StatusObservation,
    ValenceBand,
    VisionObservation,
)


def _base_observation(**overrides) -> Observation:
    """Observation with no enrichment — the control arm."""
    fields = {
        "entity_id": "explorer_npc",
        "current_simulation_time": 100,
        "status": StatusObservation(position=(5, 10), movement_locked=False),
        "needs": NeedsObservation(needs={"hunger": 20.0, "energy": 60.0}),
        "vision": VisionObservation(
            visible_entities=[
                EntityData(
                    entity_id="berry_bush_01",
                    display_name="Berry Bush",
                    position=(6, 10),
                    interactions={
                        "gather_berries": {
                            "name": "gather_berries",
                            "description": "Gather berries for food",
                            "needs_filled": ["hunger"],
                            "needs_drained": [],
                        }
                    },
                )
            ]
        ),
    }
    fields.update(overrides)
    return Observation(**fields)


# Exact pre-change output, captured by executing str() against the model before
# the goal field existed. This is the control arm of the enrichment A/B: it
# proves an observation carrying no goal pays zero additional prompt tokens.
GOLDEN_UNENRICHED = (
    "Position: (5, 10)\n"
    "\n"
    "Movement locked: False\n"
    "\n"
    "Needs: hunger: 20%, energy: 60%\n"
    "\n"
    "Visible entities:\n"
    "\n"
    "  - Berry Bush (ID: berry_bush_01, Position: (6, 10))\n"
    "    Interactions: gather_berries: Gather berries for food [+hunger]"
)


class TestControlArmRendering:
    """An observation without enrichment must render exactly as it always did"""

    def test_unenriched_observation_matches_golden_string(self):
        assert str(_base_observation()) == GOLDEN_UNENRICHED

    def test_goal_present_but_inactive_renders_nothing_extra(self):
        obs = _base_observation(goal=GoalObservation())

        assert str(obs) == GOLDEN_UNENRICHED


class TestSubstrateGoalRendering:
    """The active goal must reach the prompt text, not just the model"""

    def test_active_goal_renders_label_urgency_and_drive(self):
        obs = _base_observation(
            goal=GoalObservation(
                active_goal=GoalDetail(
                    label="Find something to eat",
                    urgency=1.21,
                    drive_source="hunger",
                    template_id="satisfy_hunger",
                ),
            )
        )

        text = str(obs)

        assert "Subconscious pull: Find something to eat" in text
        assert "urgency 1.21" in text
        assert "from your hunger drive" in text

    def test_active_goal_without_drive_source_omits_the_clause(self):
        obs = _base_observation(
            goal=GoalObservation(active_goal=GoalDetail(label="Wander a while", urgency=0.4))
        )

        text = str(obs)

        assert "Subconscious pull: Wander a while (urgency 0.40)" in text
        assert "drive" not in text.split("Subconscious pull:")[1]

    def test_goal_line_is_additive(self):
        """Enrichment must not perturb any pre-existing line"""
        obs = _base_observation(
            goal=GoalObservation(active_goal=GoalDetail(label="Eat", urgency=0.9))
        )

        for line in GOLDEN_UNENRICHED.split("\n\n"):
            assert line in str(obs)


def _stressed_mood() -> MoodObservation:
    return MoodObservation(
        valence=-0.42,
        arousal=0.81,
        valence_band=ValenceBand.NEG,
        arousal_band=ArousalBand.HIGH,
        label="stressed",
        valence_baseline=-0.05,
        arousal_baseline=0.5,
    )


class TestMoodRendering:
    """Mood must reach observation_text, not merely survive validation"""

    def test_mood_renders_label_value_and_baseline(self):
        text = str(_base_observation(mood=_stressed_mood()))

        assert "Mood: stressed" in text
        assert "valence -0.42" in text
        assert "resting -0.05" in text
        assert "arousal 0.81" in text
        assert "resting 0.50" in text

    def test_absent_mood_renders_nothing(self):
        assert str(_base_observation()) == GOLDEN_UNENRICHED


class TestRelationshipRendering:
    """A relationship line appears only where there is shared history"""

    def test_known_entity_renders_its_relationship(self):
        obs = _base_observation(
            vision=VisionObservation(
                visible_entities=[
                    EntityData(
                        entity_id="alice_npc",
                        display_name="Alice",
                        position=(6, 10),
                        relationship=RelationshipState(
                            familiarity=0.62, sentiment=0.31, interaction_count=14
                        ),
                    )
                ]
            )
        )

        text = str(obs)

        assert "familiarity 0.62" in text
        assert "sentiment +0.31" in text
        assert "14 shared interactions" in text

    def test_stranger_renders_no_relationship_line(self):
        obs = _base_observation(
            vision=VisionObservation(
                visible_entities=[
                    EntityData(entity_id="stranger", display_name="Stranger", position=(6, 10))
                ]
            )
        )

        assert "Relationship:" not in str(obs)

    def test_negative_sentiment_keeps_its_sign(self):
        obs = _base_observation(
            vision=VisionObservation(
                visible_entities=[
                    EntityData(
                        entity_id="rival",
                        display_name="Rival",
                        position=(6, 10),
                        relationship=RelationshipState(
                            familiarity=0.4, sentiment=-0.2, interaction_count=3
                        ),
                    )
                ]
            )
        )

        assert "sentiment -0.20" in str(obs)


def _conversation(*messages) -> ConversationObservation:
    return ConversationObservation(
        interaction_id="conv_1",
        interaction_name="conversation",
        participants=["explorer_npc", "alice_npc"],
        conversation_history=list(messages),
    )


class TestMessageDeclarationRendering:
    """A speaker's declarations must reach the prompt, not merely survive validation.

    NPC-1278: the mind has never perceived a farewell. The simulation has always
    carried the speaker's own annotation on the message, this model never
    declared the field, and pydantic's default ``extra="ignore"`` dropped it
    silently — so an NPC could be told goodbye and read only the words.

    Declarations render by their ``kind`` key verbatim. There is deliberately no
    kind vocabulary in Python: a kind registered in the simulation must reach the
    LLM with no change here.
    """

    def test_declaration_renders_as_its_kind_verbatim(self):
        obs = _base_observation(
            conversations=[
                _conversation(
                    ConversationMessage(
                        id="message_test_observation_rendering_1",
                        speaker_id="alice_npc",
                        speaker_name="Alice",
                        message="Well, I should be going.",
                        declarations=[{"kind": "farewell"}],
                    )
                )
            ]
        )

        assert "Alice: Well, I should be going. [farewell]" in str(obs)

    def test_an_unknown_kind_needs_no_python_change(self):
        obs = _base_observation(
            conversations=[
                _conversation(
                    ConversationMessage(
                        id="message_test_observation_rendering_2",
                        speaker_id="alice_npc",
                        speaker_name="Alice",
                        message="Do you agree?",
                        declarations=[{"kind": "question"}],
                    )
                )
            ]
        )

        assert "[question]" in str(obs)

    def test_system_and_declarations_render_together(self):
        obs = _base_observation(
            conversations=[
                _conversation(
                    ConversationMessage(
                        id="message_test_observation_rendering_3",
                        speaker_id="sim",
                        speaker_name="System",
                        message="Bob left.",
                        is_system=True,
                        declarations=[{"kind": "farewell"}],
                    )
                )
            ]
        )

        assert "System: Bob left. [system] [farewell]" in str(obs)

    def test_own_message_keeps_its_you_marker(self):
        obs = _base_observation(
            conversations=[
                _conversation(
                    ConversationMessage(
                        id="message_test_observation_rendering_4",
                        speaker_id="explorer_npc",
                        speaker_name="Bob",
                        message="Goodbye then.",
                        declarations=[{"kind": "farewell"}],
                    )
                )
            ]
        )

        assert "[YOU] Bob: Goodbye then. [farewell]" in str(obs)

    def test_undeclared_message_renders_exactly_as_before(self):
        obs = _base_observation(
            conversations=[
                _conversation(
                    ConversationMessage(
                        id="message_test_observation_rendering_5",
                        speaker_id="alice_npc",
                        speaker_name="Alice",
                        message="Hello.",
                    )
                )
            ]
        )

        assert "Alice: Hello." in str(obs)
        assert "[" not in str(obs).split("Conversation:")[1]

    def test_malformed_declaration_is_skipped_rather_than_raising(self):
        # Never raise on the prompt path: a malformed entry costs one marker,
        # not the whole decision cycle.
        obs = _base_observation(
            conversations=[
                _conversation(
                    ConversationMessage(
                        id="message_test_observation_rendering_6",
                        speaker_id="alice_npc",
                        speaker_name="Alice",
                        message="Bye.",
                        declarations=[{"no_kind": True}, {"kind": ""}, {"kind": "farewell"}],
                    )
                )
            ]
        )

        assert "Alice: Bye. [farewell]" in str(obs)


class TestFullyEnrichedArm:
    """Every enriched field reaches the prompt text in one pass"""

    def test_all_enrichment_renders(self):
        from tests.fixtures.observations import create_enriched_observation

        text = str(create_enriched_observation())

        assert "Subconscious pull: Find something to eat" in text
        assert "Mood: stressed" in text
        assert "familiarity 0.62" in text
        # The unfamiliar traveller carries no relationship, so exactly one
        # relationship line is rendered across two visible entities.
        assert text.count("Relationship:") == 1


class TestInventoryRendering:
    """Carried items reach the prompt text (NPC-1116).

    A model that parses but renders nothing is still invisible to the LLM,
    which is the half of the bug that a validation test alone would miss.
    """

    def test_inventory_renders_into_observation_text(self):
        """Should name each carried item and its affordances"""
        from tests.fixtures.observations import create_carrying_observation

        text = str(create_carrying_observation())

        assert "Carrying (2 of 4):" in text
        assert "Ripe Apple (ID: apple_001) [consume]" in text
        assert "Smooth Pebble (ID: pebble_001)" in text

    def test_item_without_affordances_renders_no_bracket(self):
        """Should omit the affordance bracket rather than render an empty one"""
        from tests.fixtures.observations import create_carrying_observation

        text = str(create_carrying_observation())

        assert "Smooth Pebble (ID: pebble_001) []" not in text

    def test_empty_inventory_renders_nothing(self):
        """Should be byte-identical to carrying no inventory block at all"""
        from tests.fixtures.observations import wire_inventory_block

        empty = _base_observation(
            inventory=InventoryObservation.model_validate(wire_inventory_block(items=[]))
        )

        assert str(empty) == str(_base_observation())
        assert str(empty) == GOLDEN_UNENRICHED
