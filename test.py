import gymnasium as gym

def main():
    env = gym.make("FrozenLake-v1", render_mode='human')
    env.reset()
    env.render()

    done = False
    while not done:
        input("Press Enter to take a step...")
        action = env.action_space.sample()

        # state is a value representing the player’s current position as current_row * nrows + current_col (where both the row and col start at 0)
        # state, reward, done, _, _ = env.step(action)
        print(env.step(action))
        env.render()

if __name__ == "__main__":
    main()