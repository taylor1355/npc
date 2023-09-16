from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Agent:
    def __init__(self, verbose=False):
        
        model_path = "TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # TODO: The agent should have a separate instance of a Memory struct for each environment it is currently "playing" (like a video game)
        # This can serve 2 purposes. 1) Easily training the weights of the same agent on multiple very long time horizon environmnents,
        # 2) Allowing "prompt ensembling" in which the same environment generates many slight variations of each prompt or even more extreme
        # variations like only giving certain information with a certain probability. The different variations of the same agent can vote on the
        # best action, or the base model itself can compare memories side by side to choose the best action using a tournament system
        self.action_stream = ''
        self.thought_stream = ''
        self.observation_stream = ''
        
        self.verbose = verbose
    
    # TODO: the information passing between the environment and agent is too messy. Maybe package all this stuff into
    # an EnvState struct or something then just unpack that object. Or directly pass the env to the agent
    def update(self, state, reward, timestamp, thought_prompt_factory, state_to_str):
        self.update_observation_stream(state, reward, timestamp, state_to_str)
        self.update_thought_stream(thought_prompt_factory, timestamp)

    def parse_action(self, response, action_space):
        # Parse for description
        for action_id, description in action_space.items():
            if description.lower() in response.lower():
                return action_id

        # Parse for action id
        for action_id, description in action_space.items():
            if str(action_id) in response:
                return action_id

        return None

    def take_action(self, timestamp, action_prompt_factory, action_space):
        prompt = action_prompt_factory(self.thought_stream)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # TODO: stream a token at a time and stop as soon as an action id or action description is matched
        output = self.llm.generate(inputs=input_ids, temperature=0.7, do_sample=True, max_new_tokens=8)
        generated_tokens = output[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(generated_tokens)
        if self.verbose:
            print(prompt)
            print(response)
        
        action = self.parse_action(response, action_space)
        action_str = f'Invalid action "response"' if action is None else action_space[action]
        self.action_stream += f"{action_str} <t= {timestamp}>\n"
        return action

    def update_thought_stream(self, thought_prompt_factory, timestamp):
        prompt = thought_prompt_factory(self.observation_stream, self.thought_stream)
        if self.verbose:
            print(prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        output = self.llm.generate(inputs=input_ids, temperature=0.75, repetition_penalty=1.25, do_sample=True, max_new_tokens=50)
        thought = self.tokenizer.decode(output[0][input_ids.shape[-1]:])
        self.thought_stream +=  f"{thought} <t={timestamp}> "

    # Limited to most recent observation for now
    def update_observation_stream(self, state, reward, timestamp, state_to_str):
        state_str = state_to_str(state)
        self.observation_stream = f"State: [{state_str}], Reward: {reward} <t={timestamp}> "