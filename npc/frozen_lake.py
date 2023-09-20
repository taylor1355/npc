from .agent_args import AgentArgs
from .agent import Agent

from transformers import AutoModelForCausalLM, AutoTokenizer
import gymnasium as gym
import torch

# TODO: randomly change prompt and information that is provided as a form of domain randomization. You can have 
# gpt-4 do this to avoid needing manual effort to come up with the prompts
# TODO: also let gpt-4 pick random grounding questions such as "Which directions should you avoid because of a hole or wall?"
# TODO: create a pool of prompt setups then train for a certain number of simulations on each one. Then feed the statistics to
# gpt-4 to have it combine the best parts of the good ones while avoiding the bad ones (plus a couple random ones). Penalize longer
# agent responses, episode times, and prompt lengths with a small negative reward
class FrozenLake:
    def __init__(self, is_slippery=False):
        self.is_slippery = is_slippery
        self.env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=is_slippery)
        self.timestamp = 0
        self.done = False

        self.prompt_prefix = "<|im_start|>"
        self.prompt_suffix = "<|im_end|>"
        self.env_description = ''.join([
            'I am OrcaPhi. The following is my internal dialogue as an intelligent AI agent.',
            'I am playing a video game where I must avoid walking into any holes while',
            ' traveling over a frozen lake to find the goal tile. I can safely move onto floor tiles but cannot move through walls. I ',
            ' am able to see both the goal and any holes from one tile away. The faster I reach the goal the better.'
        ])
        if self.is_slippery:
            self.env_description += ' Due to the slippery nature of the frozen lake, I may accidentally move perpendicular to the direction I intend to move sometimes '

        self.action_space = {
            0: 'West',
            1: 'South',
            2: 'East',
            3: 'North'
        }
        
        model_path = "Open-Orca/oo-phi-1_5"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        agent_args = AgentArgs.builder() \
            .with_llm(llm) \
            .with_tokenizer(tokenizer) \
            .build()
        self.agent = Agent(agent_args)

    # TODO: decouple system prompt, user prompt, assistant prompt. create a helper file shared across envs
    def thought_prompt_factory(self, observation_stream, thought_stream):
        return '\n'.join([
            self.prompt_prefix + 'system',
            self.env_description,
            f'My observations: {observation_stream}',
            f'My past thoughts: {thought_stream}',
            f'Possible Actions: {self.action_space}' + self.prompt_suffix,
            self.prompt_prefix + 'user',
            f'Choose which action you will take.' + self.prompt_suffix,
            self.prompt_prefix + 'assistant\n'
        ])

    def action_prompt_factory(self, thought_stream):
        return '\n'.join([
            self.prompt_prefix + 'system',
            self.env_description,
            f'My plans: {thought_stream}',
            f'Possible Actions: {self.action_space}' + self.prompt_suffix,
            self.prompt_prefix + 'user',
            f'What action are you going to take?' + self.prompt_suffix,
            self.prompt_prefix + 'assistant',
            'I will take action "'
        ])

    def describe_tile(self, row, col):
        if row < 0 or row >= 3 or col < 0 or col >= 3:
            return 'wall'
        else:
             env_letter = self.env.desc[row, col].decode('utf-8')
             match env_letter:
                 case 'S':
                     return 'start'
                 case 'F':
                     return 'floor'
                 case 'H':
                     return 'hole'
                 case _:
                     return 'unknown'

    def state_to_str(self, state):
        row, col = state // 4, state % 4
        surroundings = [
            f"North: {self.describe_tile(row - 1, col)}",
            f"South: {self.describe_tile(row + 1, col)}",
            f"West: {self.describe_tile(row, col - 1)}",
            f"East: {self.describe_tile(row, col + 1)}",
        ]
        return f"Row {row}, Col {col}, {', '.join(surroundings)}"

    def run(self):
        state, _ = self.env.reset()
        reward = 0

        while not self.done:
            input("Press Enter to take a step...")
            print('\n\n' + '=' * 80)

            self.agent.update(
                state, reward, self.timestamp,
                self.thought_prompt_factory,
                self.state_to_str
            )
            action = self.agent.take_action(
                self.timestamp,
                self.action_prompt_factory,
                self.action_space
            )
            state, reward, self.done, _, _ = self.env.step(action)
            self.env.render()

            self.timestamp += 1

            print(f'| t={self.timestamp}')
            print(f'| Observation Stream: [{self.agent.observation_stream}]')
            print(f'| Thought Stream: [{self.agent.thought_stream}]')
            print(f'| Action Stream: [{self.agent.action_stream}]')

if __name__ == '__main__':
    env = FrozenLake()
    env.run()