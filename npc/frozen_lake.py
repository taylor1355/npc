from .agent import Agent
import gymnasium as gym

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

        self.env_description = ''.join([
            'You are an intelligent human playing a video game where you must avoid walking into any holes while',
            ' traveling over the frozen lake to find the goal. You can safely move onto floor tiles but cannot move through walls. You ',
            ' are able to see both the goal and any holes from one tile away. The faster you reach the goal the better.'
        ])
        if self.is_slippery:
            self.env_description += ' You may move perpendicular to the intended direction sometimes due to the slippery nature of the frozen lake.'

        self.action_space = {
            0: 'West',
            1: 'South',
            2: 'East',
            3: 'North'
        }
        
        self.agent = Agent(verbose=True)

    def thought_prompt_factory(self, observation_stream, thought_stream):
        return '\n'.join([
            f'### Human:',
            self.env_description,
            f'Your observations: {observation_stream}',
            f'Past thoughts: {thought_stream}',
            f'Action space: {self.action_space}',
            f'Briefly choose which action you will take. You cannot do nothing.',
            '\n### Assistant:\n'
        ])

    def action_prompt_factory(self, thought_stream):
        return '\n'.join([
            f'### Human:',
            f'Your plans: {thought_stream}',
            f'Action space: {self.action_space}',
            f'What action are you going to take?',
            '\n### Assistant:\n'
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