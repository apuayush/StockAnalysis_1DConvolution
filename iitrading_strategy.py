import numpy as np

from iireinforcement import Agent
from gym import spaces
from btgym import BTgymEnv


class Strategy:
    def __init__(self, state_shape, start_cash):
        # data1.csv contains sequence of 30 days chunks of data
        self.env = BTgymEnv(
            filename='data1.csv',
            state_shape={'raw_state': spaces.Box(low=-100, high=100, shape=state_shape)},
            skip_frame=5,
            start_cash=start_cash,
            fixed_stake=100,
            drawdown_call=90,
            render_ylable='Price Lines',
            render_size_episode=(12, 8),
            render_size_human=(8, 3.5),
            render_size_state=(10, 3.5),
            render_dpi=75,
            verbose=0
        )
        self.generate_strategy(state_shape)

    def generate_strategy(self, state_shape):
        gamma = 0.9
        epsilon = 0.95

        trials = 100
        trial_len = 1000

        agent = Agent(env=self.env, state_shape=state_shape)
        steps = []

        for trial in range(trials):
            cur_state = np.array(list(self.env.reset().items())[0][1])
            cur_state = np.reshape(cur_state, state_shape)

            for step in range(trial_len):
                action = agent.act(cur_state)
                new_state, reward, done, _ = self.env.step(action)
                if not done:
                    reward *= 10
                else:
                    reward = -10
                new_state = np.reshape(list(new_state.items())[0][1], state_shape)
                steps.append([new_state, reward])
                agent.target_train()

                cur_state = new_state

                if done:
                    break

                print("Completed trial %d" % trial)
                agent.render_all_modes(self.env)
                agent.save_model("model/model"+str(trial))

        return steps