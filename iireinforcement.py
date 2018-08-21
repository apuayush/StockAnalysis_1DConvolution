import IPython.display as Display
import PIL.Image as Image
import cnn
import numpy as np

from collections import deque


class Agent:
    def __init__(self, env, state_shape, is_eval=False, model_name=""):
        self.state_shape = state_shape  # for my model 30 days
        self.action_size = 3  # sit, buy, sell

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125
        self.memory = deque(maxlen=10000)

        self.inventory = []
        self.model_name = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        cnn_model = cnn.CNN(layers=[
            {
                'type': 'sepconv1D',
                'args': {
                    'filters': 32,
                    'kernel_size': 5,
                    'activation': 'relu',
                    'input_shape': self.state_shape
                }
            },
            {
                'type': 'maxpool1D',
                'args': {
                    'pool_size': 2
                }
            },
            {
                'type': 'conv1D',
                'args': {
                    'filters': 128,
                    'kernel_size': 3,
                    'activation': 'relu'
                }
            },
            {
                'type': 'maxpool1D',
                'args': {
                    'pool_size': 2
                }
            },
            {
                'type': 'dropout',
                'args': {
                    'ratio': 0.15
                }
            },
            {
                'type': 'flatten',
                'args': None
            },
            {
                'type': 'dense',
                'args': {
                    'output': 250
                }
            },
            {
                'type': 'dropout',
                'args': {
                    'ratio': 0.2
                }
            },
            {
                'type': 'activation',
                'args': {
                    'function': 'relu'
                }
            },
            {
                'type': 'dense',
                'args': {
                    'output': 250
                }
            },
            {
                'type': 'dense',
                'args': {
                    'output': 125
                }
            },
            {
                'type': 'dense',
                'args': {
                    'output': 32
                }
            },

            {
                'type': 'dense',
                'args': {
                    'output': self.action_size
                }
            },
            {
                'type': 'activation',
                'args': {
                    'function': 'linear'
                }
            },
        ])
        cnn_model.build_model()
        cnn_model.compile_model()
        return cnn_model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def target_train(self):
        wts = self.model.get_weights()
        trgt_wts = self.target_model.get_weights()
        for i in range(len(trgt_wts)):
            trgt_wts[i] = wts[i] * self.tau + trgt_wts[i] * (1-self.tau)
        self.target_model.set_weights(trgt_wts)

    def show_rendered_img(self, rgb_array):
        """
        convert numpy array to rgb and display it on Main.ipynb
        """
        Display.display(Image.fromarray(rgb_array))

    def render_all_modes(self, env):
        """
        Show environment renderings for all supported modes
        :param env: gym environment
        :return: None
        """
        for mode in self.metadata['render.modes']:
            print('[{}] mode'.format(mode))
            self.show_rendered_img(self.env.render(mode))

    def save_model(self, fn):
        self.model.save(fn)
