import numpy as np

from cs231n.layers import *
#from cs231n.fast_layers import *
from cs231n.layer_utils import *

class MultiLayersCNN:
    def __init__(self, num_filters, filter_sizes, hidden_dims,
                 input_dim=(3, 32, 32), num_classes=10, weight_scale=1e-3,
                 reg=0.0, dtype=np.float32):
        """
        num_filters: list, filter value for each conn.
        filter_size: list, filter size for each conn. It's one on one match with num_filters.
        hidden_dims: list, after cnn, the affines hidden layers.
        """
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.reg = reg
        pool_times = len(self.filter_sizes) / 2
        
        C, H, W = input_dim
        FC_input_dim = (H/(2*pool_times)) * (W/(2*pool_times)) * self.num_filters[-1] if pool_times else C*H*W
        
        # initial parameters
        self.params = {}
        param_counter = 1
        # conn layers
        for i in xrange(len(self.num_filters)):
            _C = C if param_counter == 1 else self.num_filters[i-1]
            num_filter = self.num_filters[i]
            filter_size = filter_sizes[i]
            self.params['W{}'.format(param_counter)] = np.random.randn(
                num_filter, _C, filter_size, filter_size
                ) * weight_scale
            self.params['b{}'.format(param_counter)] = np.zeros(num_filter)
            param_counter += 1
            
        # affine layers
        # because the CONV layers won't change size and only
        # pool layers will change the size down.
        FC_dims = [FC_input_dim,] + hidden_dims + [num_classes]
                   
        for i in xrange(1, len(FC_dims)):
            self.params['W{}'.format(param_counter)] = np.random.randn(
                FC_dims[i-1], FC_dims[i]
            ) * weight_scale
            self.params['b{}'.format(param_counter)] = np.zeros(FC_dims[i])
            param_counter += 1
                
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
                    
    def loss(self, X, y=None):
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
                        
        out = {}
        cache = {}
        param_counter = 1
        # every two conv will have a pool layer
        for i in xrange(len(self.num_filters)):
            conv_param = {'stride': 1, 'pad': (self.filter_sizes[i] - 1)/2}

            _X = X if param_counter == 1 else out[param_counter-1]
    
            if param_counter % 2 != 0:
                conv_out, conv_cache = conv_relu_forward(
                    _X,
                    self.params['W{}'.format(param_counter)],
                    self.params['b{}'.format(param_counter)],
                    conv_param
                )
            else:
                conv_out, conv_cache = conv_relu_pool_forward(
                    _X,
                    self.params['W{}'.format(param_counter)],
                    self.params['b{}'.format(param_counter)],
                    conv_param,
                    pool_param
                )
            out[param_counter] = conv_out
            cache[param_counter] = conv_cache
            param_counter += 1
                                
        # affine + relu layer
        for i in xrange(len(self.hidden_dims)):
            
            _X = X if param_counter == 1 else out[param_counter-1]

            ar_out, ar_cache = affine_relu_forward(
                _X,
                self.params['W{}'.format(param_counter)],
                self.params['b{}'.format(param_counter)],
            )
            out[param_counter] = ar_out
            cache[param_counter] = ar_cache
            param_counter += 1
                                    
        # last affine layer
        scores, a_cache = affine_forward(
            out[param_counter-1],
            self.params['W{}'.format(param_counter)],
            self.params['b{}'.format(param_counter)]
        )
        out[param_counter] = scores
        cache[param_counter] = a_cache
                                    
        if y is None:
            return scores
        
        # calculate loss and sofemax dout 
        dout = {}
        loss, dscores = softmax_loss(out[param_counter], y)
        W_sum = 0.0
        for k, v in self.params.iteritems():
            if k.startswith('W'):
                W_sum += np.sum(self.params[k] * self.params[k])
                                            
        loss += 0.5 * self.reg * W_sum

        grads = {}
        # back propogation
        # last affine
        _dout, _dw, _db = affine_backward(dscores, cache[param_counter])
        dout[param_counter] = _dout
        grads['b{}'.format(param_counter)] = _db
        grads['W{}'.format(param_counter)] = _dw + self.reg * self.params['W{}'.format(param_counter)]
        
        # affline + relu
        for i in xrange(len(self.hidden_dims)):
            param_counter -= 1
            _dout, _dw, _db, = affine_relu_backward(dout[param_counter+1], cache[param_counter])
            dout[param_counter] = _dout
            grads['b{}'.format(param_counter)] = _db
            grads['W{}'.format(param_counter)] = _dw + self.reg * self.params['W{}'.format(param_counter)]
            
        # conv + relu [+ pool]
        for i in xrange(len(self.num_filters)):
            param_counter -= 1

            if param_counter % 2 == 0:
                _dout, _dw, _db = conv_relu_pool_backward(dout[param_counter+1], cache[param_counter])
            else:
                _dout, _dw, _db = conv_relu_backward(dout[param_counter+1], cache[param_counter])
            dout[param_counter] = _dout
            grads['b{}'.format(param_counter)] = _db
            grads['W{}'.format(param_counter)] = _dw + self.reg * self.params['W{}'.format(param_counter)]
        self.out = out
        self.cache = cache
        self.dout = dout
        return loss, grads 
