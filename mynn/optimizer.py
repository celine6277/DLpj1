from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.v = {} # 字典，用来存储每一层参数的速度 (velocity)
        
        # 初始化所有可优化参数的速度为 0
        for idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                self.v[idx] = {}
                for key in layer.params.keys():
                    self.v[idx][key] = np.zeros_like(layer.params[key])
    
    def step(self):
        for idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                for key in layer.params.keys():
                    # 1. 权重衰减 (Weight Decay / L2 Regularization)
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    
                    # 2. 计算并更新速度 (velocity)
                    # v = mu * v + lr * grad
                    self.v[idx][key] = self.mu * self.v[idx][key] + self.init_lr * layer.grads[key]
                    
                    # 3. 更新参数
                    # param = param - v
                    layer.params[key] = layer.params[key] - self.v[idx][key]