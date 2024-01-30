from .action import Action
from .openai_api import get_llm_completion
from .utils import extract_tags

class Gamemaster:
    def __init__(self):
        self.name = "Gamemaster"
        self.current_action = None  

    def write_action_description(self, agent_intent, simulator):
        prompt_format = "\n".join([
        "The agent is attempting to take an action. Their intent is written in <action_intent></action_intent> tags. As gamemaster, please describe how the execution of the action ended up going when the agent initiated it and what effects (intended or unintended) it had. Consider both the states of the agent and the room in your description. Here are some guidelines you need to follow:"
        "- The action should be plausible as a single action in the Sims (with all imaginable expansion packs and additional content enabled). If the user tries to do a long multi-step action then you should break it up into multiple actions and just describe the execution of the first one.",
        "- Be sure to keep in mind the locations of different entities in the room. For example, if the agent wants to use an object, then the object needs to be available to use and the agent may need to move to the object.",
        "- The description should be detailed, but not overly granular or flowery.",
        "- The description should be written in past tense third person.",
        "- The length of the description should be at least one sentence and at most two paragraphs. If the action is very simple then you can write a single sentence, but if it is complex then you should write up to two paragraphs.",
        "- Do not fixate on irrelevant details which do not affect the simulation and would not be interesting to a human observer. For example, if the agent sits down you don not need to describe the exact motions of the agent or the cushions.",
        "- You are describing the final outcome of the action. You should not use words like 'possibly', 'potentially', or 'may' in your description.",
        "",
        "The agent's intent can be found below in the <action_intent></action_intent> tags. This is only their intention, not the actual action they took. You should describe the actual action they took in your description.",
        "<agent_intent>",
        "{agent_intent}", 
        "</agent_intent>",
        "",
        "The state of the room, including all the entities it contains, can be found below in the <room_state></room_state> tags:",
        "<room_state>",
        "{room_state}",
        "</room_state>",
        "",
        "And the state of the agent can be found below in the <agent_state></agent_state> tags:",
        "<agent_state>",
        "{agent_state}",
        "</agent_state>",
        "",
        ])
        prompt = prompt_format.format(
            agent_intent=agent_intent,
            room_state=simulator.room.state_str(),
            agent_state=simulator.agent.state_str()
        )
        system_prompt = "You are the gamemaster of a simulation of a room with an agent. Think of the simulation as a sophisticated version of the Sims. The agent can take open-ended actions in the room within reason (based on its capabilities, the state of the room, and the laws of physics)."
        llm_output = get_llm_completion(prompt, system_prompt)
        return llm_output

    def determine_action_effects(self, action_description, simulator):
        prompt_format = "\n".join([
            "The agent has just taken an action. A detailed description of how this action unfolded can be found in <action_effects></action_effects> tags. I would like you to describe how the room state and/or agent state has changed as a consequence of the action. To do this, please follow this procedure:",
            "1. Create a numbered JSON list of all the ways in which the action affects the state of the room and/or the agent. Put this analysis inside <effect_list></effect_list> tags. Here are some guidelines:",
            "    - Each element of the list should be a JSON object with three fields: 'effect_name', 'description', and 'deliberation'. In the 'deliberation' field, you should briefly play devil's advocate for why the effect might be inconsistent with the simulation state, then decide whether to implement it.",
            "    - Each effect should be an atomic (i.e., not able to be broken up into smaller effects) state change.",
            "    - Keep in mind that the general 'vibe' of the room will not change unless the action is very significant. These effects should mainly simulate what a game engine would do to the state of the room and/or agent in response to the action. If the 'vibe' does change, then the reason should be explicitly stated in the description of the effect (e.g., if music causes the change, then the change should be described as a consequence of the music).",
            "    - Do not get bogged down in the details. For example, if the agent sits down, you do not need to describe the exact motions of the agent or the cushions.",
            "    - If the agent does something which affects both them and an entity in the room, then you should have effects for both the agent and the entity. For example, if the agent sits down, then (among other effects) the agent's state should change to sitting down on the chair and the entity's state should change to occupied by the agent.",
            "2. Break down each effect into individual changes to the room and agent. Do not include any effects that do not pass the deliberation step. Put this breakdown inside <final_effects></final_effects> tags.",
            "3. Record how many minutes (as a positive integer) the action takes inside <time_taken></time_taken> tags. All actions must take at least 1 minute to complete.",
            ""
            "Here's a detailed descrioption of how the action unfolded in <action_description></action_description> tags:",
            "<action_description>",
            "{action_description}",
            "</action_description>",
            "",
            "Here's the current state state of the room in <room_state></room_state> tags:",
            "<room_state>",
            "{room_state}",
            "</room_state>",
            "",
            "And finally here's the current state of the agent in <agent_state></agent_state> tags:",
            "<agent_state>",
            "{agent_state}",
            "</agent_state>",
            "",
            "Now, using the information above and meticulously following the procedure, please provide the effect_list, implemented_effects, final_effects, and time_taken in the appropriate tags.",
        ])
        prompt = prompt_format.format(
            action_description=action_description, 
            room_state=simulator.room.state_str(),
            agent_state=simulator.agent.state_str()
        )
        system_prompt = "You are the gamemaster of a simulation of a room with an agent. Think of the simulation as a sophisticated version of the Sims. The agent can take open-ended actions in the room within reason (based on its capabilities, the state of the room, and the laws of physics)."
        llm_output = get_llm_completion(prompt, system_prompt)

        tags = extract_tags(llm_output) 
        if "final_effects" not in tags:
            print("Warning: LLM output does not contain a final_effects tag.")
            tags["final_effects"] = ""

        if "time_taken" not in tags:
            print("Warning: LLM output does not contain a time_taken tag.")
            time_taken = 1
        try:
            time_taken = int(tags["time_taken"])
        except:
            print("Warning: LLM output contains a malformed time_taken tag.")
            time_taken = 1
        if time_taken < 1:
            print("Warning: LLM output contains a time_taken tag with a value less than 1.")
            time_taken = 1
        
        metadata = {
            'effect_list': tags.get("effect_list", None),
        }
        return tags["final_effects"], time_taken, metadata
    
    def write_state_updating_code(self, action_effects, simulator):
        prompt_format = "\n".join([
            "The agent has just taken an action. The detailed effects of this action have been described to you in <action_effects></action_effects> tags. I need you to translate these qualitative effects into updates to the state of the simulation. To do this, please implement each effect using a line of Python code which will manipulate the state dictionaries corresponding to the room and/or the agent. The Python dictionaries will be called room_state_dict and agent_state_dict, respectively. Put these lines of code inside <state_updating_code></state_updating_code> tags.",
            "- Each line of code will be independently executed, so do not use multiline statements.",
            "",
            "The current state of the room (as a Python dictionary) is inside <room_state_dict></room_state_dict> tags:",
            "<room_state_dict>",
            "{room_state_dict}",
            "</room_state_dict>",
            "",
            "And the current state of the agent (as a Python dictionary) is inside <agent_state_dict></agent_state_dict> tags:",
            "<agent_state_dict>",
            "{agent_state_dict}",
            "</agent_state_dict>",
            "",
            "The detailed effects of the action are inside <action_effects></action_effects> tags:",
            "<action_effects>",
            "{action_effects}",
            "</action_effects>",
        ])
        prompt = prompt_format.format(
            action_effects=action_effects,
            room_state_dict=simulator.room.state_str(),
            agent_state_dict=simulator.agent.state_str()
        )
        system_prompt = "You are the gamemaster of a simulation of a room with an agent. Think of the simulation as a sophisticated version of the Sims. The agent can take open-ended actions in the room within reason (based on its capabilities, the state of the room, and the laws of physics)."
        llm_output = get_llm_completion(prompt, system_prompt)

        tags = extract_tags(llm_output) 
        if "state_updating_code" not in tags:
            print("Warning: LLM output does not contain a state_updating_code tag.")
        
        return tags.get("state_updating_code", "")

    # TODO: instead of instructing LLM to compress the state, ask it to break a field up into multiple fields if needed
    def generate_state_correction_action(self, simulator):
        prompt_format = "\n".join([
            "Your job is to correct any major issues in the simulation state Python dictionaries: 'room_state_dict' and 'agent_state_dict'. These dictionaries might contain inconsistencies, repetitive elements, formatting, or grammatical errors. The task is in two parts:",
            "1. In <issues></issues> tags, write a JSON list of potential issues with the simulation state. Each element of the list should be a JSON object with three fields: 'issue', 'deliberation', and 'severity'. See details below:",
            "    - 'description': describe the issue",
            "    - 'deliberation': play devil's advocate for why your diagnosis is mistaken and/or why it is not worth a state update (which inherently risks introducing new issues)",
            "    - 'severity': rate it as being 'serious', 'minor', or a 'non-issue'.",
            "2. For each 'serious' issue you identified, write a line of Python code to fix it. The code should perform the necessary modification on the 'room_state_dict' and 'agent_state_dict' dictionaries. You should put these lines of code inside <state_updating_code></state_updating_code> tags.",
            "    - Each line of code will be independently executed, so do not use multiline statements.",
            "    - If you do not identify any serious issues, then you should leave the state_updating_code tags empty.",
            "",
            "Here are some guidelines on what sorts of edits you should make:",
            "- Your duty is to ensure that the state of the simulation is not corrupted over time. This means that (1) you should be very sure you are addressing actual issues and not imagined ones and (2) all edits should preserve the intended meaning and consistency of the state dictionaries.",
            "- Remove any formatting or grammatical errors.",
            "- Remove any overly repetitive language.",
            "- If you notice any inconsistencies in the state dictionaries, correct them.",
            "- If you notice any missing information in the state dictionaries that is easily inferable from the context, add it.",
            "",
            "The current state of the room (as a Python dictionary) is inside <room_state_dict></room_state_dict> tags:",
            "<room_state_dict>",
            "{room_state_dict}",
            "</room_state_dict>",
            "",
            "And the current state of the current agent (as a Python dictionary) is inside <agent_state_dict></agent_state_dict> tags:",
            "<agent_state_dict>",
            "{agent_state_dict}",
            "</agent_state_dict>",
            "",
            "Please provide the edits and corresponding lines of Python code.",
        ])

        prompt = prompt_format.format(
            room_state_dict=simulator.room.state_str(),
            agent_state_dict=simulator.agent.state_str()
        )
        system_prompt = "You are the gamemaster of a simulation of a room with NPC agents. Think of the simulation as a sophisticated version of the Sims. Your current job involves maintain the integrity of the simulation state by correcting any errors that arise over time. You are given two Python dictionaries: 'room_state_dict' and 'agent_state_dict', which represent the state of the current room and agent."
        llm_output = get_llm_completion(prompt, system_prompt)

        tags = extract_tags(llm_output)
        if "state_updating_code" not in tags:
            print("Warning: LLM output does not contain a state_updating_code tag.")

        return Action(
            start_time=simulator.timestep, 
            end_time=simulator.timestep,
            state_updating_code=tags.get("state_updating_code", ""),
            acting_agent=self, 
            metadata={
                "issues": tags.get("issues", None),
            }
        )

    def generate_user_action(self, agent_intent, simulator):
        # Determining action success: 
        # 1. Judge whether it is plausible that one or more aspects of the agent's intent could fail to be achieved
        # 2. Imagine alternative outcomes for each of these aspects and assign a likelihood to each
        # 3. Sample from the possible outcomes then pass to the description prompt with the outcomes given as constraints
        # Prompt llm to describe in detail what the action would look like and whether it is successful (can eventually use dice rolls like dnd skill checks)

        # TODO: sometimes the description misses details like the agent needing to move somewhere else to use the computer
        action_description = self.write_action_description(agent_intent, simulator)

        # Prompt llm to describe each effect the action has on the agent or room state
        action_effects, time_taken, action_effects_metadata = self.determine_action_effects(action_description, simulator)

        # Prompt llm to write code which updates the state of the room and/or agent
        state_updating_code = self.write_state_updating_code(action_effects, simulator)

        return Action(
            start_time=simulator.timestep, 
            end_time=simulator.timestep + time_taken,
            state_updating_code=state_updating_code,
            acting_agent=simulator.agent, 
            metadata={
                "agent_intent": agent_intent,
                "description": action_description,
                **action_effects_metadata,
                "action_effects": action_effects,
            }
        )