"""Unit tests for action models"""

from mind.cognitive_architecture.actions import Action


class TestActionModel:
    """Test Action model"""

    def test_create_basic_action(self):
        """Should create action with type"""
        action = Action(action="wait")

        assert action.action == "wait"
        assert action.parameters == {}

    def test_create_action_with_parameters(self):
        """Should create action with parameters"""
        action = Action(action="move_to", parameters={"x": 10, "y": 20})

        assert action.action == "move_to"
        assert action.parameters["x"] == 10
        assert action.parameters["y"] == 20

    def test_create_interaction_action(self):
        """Should create interact_with action"""
        action = Action(
            action="interact_with",
            parameters={"entity_id": "food_1", "interaction_name": "eat"},
        )

        assert action.action == "interact_with"
        assert action.parameters["entity_id"] == "food_1"
        assert action.parameters["interaction_name"] == "eat"
