# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:09:28 2018

@author: Fong Zhi Kang
"""

import numpy as np
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nnet.nnet import sigmoid
from theano import function, shared, config
import theano.tensor as T

class RBM:
    """A class for Restricted Boltzmann Machines"""
    def __init__(self, num_hid, verbose=True, batchsize=100, num_epoch=10, 
                 learning_rate=0.1, learner="cd", cd_n=1, pcd_num_particles=100):
        self.batchsize = batchsize
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        
        self.learner = learner
        self.cd_n = cd_n
        self.pcd_num_particles = pcd_num_particles
        self.pcd_particles = None
        
        self.num_hid = shared(num_hid, 'num_hid')
        self.num_vis = shared(0, 'num_vis')
        self.num_data = shared(0, 'num_data')
        
        self.verbose = verbose
        self.rng = RandomStreams(seed=101)
        
        self.__init_weights()
        self.__init_graphs()
        
    #def gibbs(self, vis):
    #   return self.__gibbs(vis)[0]
        
    def __init_graphs(self):
        v_0 = T.matrix('v_0') # num_data x vis
        h_0, h_mean_0, pre_sigmoid_h_0 = self.__get_h_given_v(v_0) # num_data x hid
        E_v_h_0 = T.dot(v_0.transpose(), h_0) / self.batchsize # vis x hid
        
        if self.learner == 'cd':
            h_n = h_0.copy()
            
        elif self.learner == 'pcd':
            if self.pcd_particles is None:
                self.pcd_particles = h_0.copy()
            
            h_n = self.pcd_particles
        
        ## To replace with theano.scan
        for i in range(self.cd_n):
            (h_n, h_mean_n, pre_sigmoid_h_n, 
             v_n, v_mean_n, pre_sigmoid_v_n) = self.__gibbs_from_hid(h_n)
            
        if self.learner == 'pcd':
            self.pcd_particles = h_n
            
        
        E_v_h_n = T.dot(v_n.transpose(), h_n) / self.batchsize # vis x hid
        
        dW = E_v_h_0 - E_v_h_n
        db = T.mean(h_0, axis=0) - T.mean(h_n, axis=0) # use soft probabilities to speed up training
        dc = T.mean(v_0, axis=0) - T.mean(v_n, axis=0) # ditto with db
        
        v_mean_1, hidden_rep = self.__gibbs_soft(v_0) # num_data x vis
        diff = v_0 - v_mean_1
        sq_err = diff.norm(2, axis=1).mean()
        
        self.__update = function([v_0], 
                                 [], 
                                 updates=[(self.W, self.W + self.learning_rate*dW), 
                                          (self.b, self.b + self.learning_rate*db),
                                          (self.c, self.c + self.learning_rate*dc)], 
                                 allow_input_downcast=True)
        
        
        self.__sq_error = function([v_0],
                                   [sq_err],
                                   allow_input_downcast=True)
        self.__predict = function([v_0],
                                  [hidden_rep],
                                  allow_input_downcast=True)
        
        return self
    
    def __get_h_given_v(self, v):
        pre_sigmoid_h = T.dot(v, self.W) + self.b
        h_mean = sigmoid(pre_sigmoid_h)
        h = self.__bernoulli_sample(h_mean)
        
        return h, h_mean, pre_sigmoid_h
    
    def __get_v_given_h(self, h):
        pre_sigmoid_v = T.dot(h, self.W.transpose()) + self.c
        v_mean = sigmoid(pre_sigmoid_v)
        v = self.__bernoulli_sample(v_mean)
        
        return v, v_mean, pre_sigmoid_v
        
        
    def __gibbs_from_vis(self, v_0):
        h_0, h_mean_0, pre_sigmoid_h_0 = self.__get_h_given_v(v_0)
        v_1, v_mean_1, pre_sigmoid_v_1 = self.__get_v_given_h(h_0)
        
        return v_1, v_mean_1, pre_sigmoid_v_1, h_0, h_mean_0, pre_sigmoid_h_0
    
    def __gibbs_from_hid(self, h_0):
        v_1, v_mean_1, pre_sigmoid_v_1 = self.__get_v_given_h(h_0)
        h_1, h_mean_1, pre_sigmoid_h_1 = self.__get_h_given_v(v_1)
        
        return h_1, h_mean_1, pre_sigmoid_h_1, v_1, v_mean_1, pre_sigmoid_v_1
        
    def __gibbs_soft(self, X):
        soft_hid = sigmoid(T.dot(X, self.W) + self.b) # num_data x hid
        soft_vis = sigmoid(T.dot(soft_hid, self.W.transpose()) + self.c) # num_data x vis
        
        return soft_vis, soft_hid
    
    def __bernoulli_sample(self, probs):
        return self.rng.binomial(p=probs, dtype=config.floatX)
  
    def __batch_gen(self, X):
        num_batchs = (self.num_data.get_value() * self.num_epoch // self.batchsize) + 1
        num_data = int(self.num_data.get_value())
        random_indices = np.random.permutation(num_data)
        
        for batch_num in range(num_batchs):
            batch_index_start = (batch_num * self.batchsize) % num_data
            batch_index_end = batch_index_start + self.batchsize
            
            if batch_index_end <= num_data:
                batch_index = random_indices[batch_index_start:batch_index_end]
                
            else:
                batch_index_end = batch_index_end % num_data
                batch_index = np.concatenate([random_indices[batch_index_start:], random_indices[:batch_index_end]])
            
            batch_data = X[batch_index, :]
            yield batch_data
            
    def fit(self, X):
        self.num_data.set_value(X.shape[0])
        self.num_vis.set_value(X.shape[1])
        self.__reset_weights()
        
        for minibatch in self.__batch_gen(X):
            self.__update(minibatch)
            
            if self.verbose:
                self.__print_info(minibatch)
            
        return self
    
    def __init_weights(self):
        self.W = shared(np.random.normal(0, 0.1, (self.num_vis.get_value(), self.num_hid.get_value())).astype('float32'), 'W')
        self.b = shared(np.zeros((self.num_hid.get_value(),)).astype('float32'), 'b')
        self.c = shared(np.zeros((self.num_vis.get_value(),)).astype('float32'), 'c')
        
        return self
    
    def __reset_weights(self):
        self.W.set_value(np.random.normal(0, 0.1, (self.num_vis.get_value(), self.num_hid.get_value())).astype('float32'))
        self.b.set_value(np.zeros((self.num_hid.get_value(),)).astype('float32'))
        self.c.set_value(np.zeros((self.num_vis.get_value(),)).astype('float32'))
        
        return self
    
    
    def __print_info(self, minibatch):
        print("Mean squared error for minibatch: {}".format(self.__sq_error(minibatch)[0]))
        print()
        
    def predict(self, batch):
        return self.__predict(batch)[0]
    
        
        
        
    
            
        
        
        
        
        
        
        
        
            
        