from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self):
        super().__init__()
        self.layers = []
        
        # 第 1 层：特征提取
        # 输入 28x28x1 -> 输出 28x28x16
        self.layers.append(conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1))
        self.layers.append(ReLU())
        
        # 第 2 层：第一次空间降维
        # 输入 28x28x16 -> 输出 14x14x32
        self.layers.append(conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1))
        self.layers.append(ReLU())
        
        # 第 3 层：第二次空间降维 
        # 输入 14x14x32 -> 输出 7x7x32
        self.layers.append(conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1))
        self.layers.append(ReLU()) # 注意：这里现在需要加 ReLU 了
        
        # 第 4 层：展平层
        # 输入 7x7x32 的特征图 -> 展平为长度 1568 的一维向量 (7 * 7 * 32 = 1568)
        self.layers.append(Flatten())
        
        # 第 5 层：全连接层映射到 10 个类别
        # 输入 1568 -> 输出 10 (作为 Logits)
        self.layers.append(Linear(in_dim=1568, out_dim=10))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        X: [batch_size, 28, 28] for MNIST - need to reshape to [batch_size, 1, 28, 28]
        """
        # Reshape input if necessary
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, 28, 28)
        elif len(X.shape) == 3:
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        # Apply all layers in sequence
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            param_list = pickle.load(f)
        
        idx = 0
        for layer in self.layers:
            if layer.optimizable: # 只有包含参数的层(如 conv2D, Linear)才需要加载
                layer.params['W'] = param_list[idx]['W']
                layer.params['b'] = param_list[idx]['b']
                layer.W = param_list[idx]['W']
                layer.b = param_list[idx]['b']
                idx += 1
        
    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if layer.optimizable: # 提取需要优化的参数
                param_list.append({
                    'W': layer.params['W'], 
                    'b': layer.params['b']
                })
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)