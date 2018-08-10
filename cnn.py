import numpy as np

class CNN(object):
    def __init__(self, layers, input_shape):
        self.layers = layers
        # self.weights_initializer()

    # def weights_initializer(self):
    #     for layer in layers:
    #         # filter_dim
    #         wts = np.random.randn(*filter_dim) * np.sqrt(2.0 / (sum(filter_dim))).astype(np.float32)

    def conv_forward(self, X, W):
        '''
        Arguments:
        X -- output activations of the previous layer, numpy array of shape (n_H_prev, n_W_prev) assuming input channels = 1
        W -- Weights, numpy array of size (f, f) assuming number of filters = 1

        Returns:
        H -- conv output, numpy array of size (n_H, n_W)
        cache -- cache of values needed for conv_backward() function
        '''

        # Retrieving previous kernel size and filters from X's shape
        X = X.T
        W = W.T
        (n_f_prev, n_k_prev) = X.shape

        # Retrieving dimensions from W's shape
        f, k = W.shape

        # Compute the output dimensions assuming no padding and stride = 1
        n_k = n_k_prev - k + 1
        n_f = f
        print(n_k)

        # Initialize the output H with zeros (filters, kernels) for processing
        H = np.zeros((n_f, n_k))

        # Looping over vertical(h) and horizontal(w) axis of output volume
        for filter in range(n_f):
            print(filter)
            for kernel in range(n_k):
                x_slice = X[0, kernel:kernel + k]
                # x_slice = x_slice.reshape(-1, 1)
                H[filter, kernel] = np.sum(W[filter] * x_slice)

        # Saving information in 'cache' for backprop
        cache = (X, W)

        return H.T, cache

cnn = CNN([],2)
H, cache = cnn.conv_forward(np.random.randn(30, 1), W=np.random.rand(5, 32))
print(H.shape, len(cache))