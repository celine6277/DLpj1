from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        fan_in = in_dim
        std = np.sqrt(2.0 / fan_in)
        self.W = np.random.normal(0, std, size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.dot(X, self.params['W']) + self.params['b']
    
    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = np.dot(self.input.T, grad) / self.input.shape[0]
        self.grads['b'] = np.mean(grad, axis=0, keepdims=True)
        return np.dot(grad, self.params['W'].T)
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        
        # Weight shape: [out_channels, in_channels, kernel_size, kernel_size]
        # 计算 fan_in (当前卷积核感受野内输入的元素总数)
        fan_in = in_channels * kernel_size * kernel_size
        # 使用 Kaiming (He) 初始化，代替原来的 * 0.01
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size)) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros((1, out_channels, 1, 1))
        
        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}
        
        self.input = None
        self.input_padded = None
        self.output_shape = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        self.input = X
        N, C, H, W = X.shape
        F, _, HH, WW = self.params['W'].shape  # F 是 out_channels
        
        # 1. 处理 Padding
        p = self.padding
        if p > 0:
            self.input_padded = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        else:
            self.input_padded = X
            
        # 2. 计算输出维度并初始化
        out_H = (H + 2 * p - HH) // self.stride + 1
        out_W = (W + 2 * p - WW) // self.stride + 1
        out = np.zeros((N, F, out_H, out_W))
        
        # 3. 仅保留空间维度的双层循环
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                h_end = h_start + HH
                w_start = j * self.stride
                w_end = w_start + WW
                
                x_slice = self.input_padded[:, :, h_start:h_end, w_start:w_end]
                
                # 向量化计算
                out[:, :, i, j] = np.sum(
                    x_slice[:, np.newaxis, :, :, :] * self.params['W'][np.newaxis, :, :, :, :], 
                    axis=(2, 3, 4)
                )
                
        # 巧妙利用广播机制在最后加上 bias
        return out + self.params['b']

    def backward(self, grads):
        N, F, out_H, out_W = grads.shape
        _, C, HH, WW = self.params['W'].shape
        
        self.grads['W'] = np.zeros_like(self.params['W'])
        dX_padded = np.zeros_like(self.input_padded)
        
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                h_end = h_start + HH
                w_start = j * self.stride
                w_end = w_start + WW
                
                x_slice = self.input_padded[:, :, h_start:h_end, w_start:w_end] 
                g_slice = grads[:, :, i, j] 
                
                self.grads['W'] += np.sum(
                    x_slice[:, np.newaxis, :, :, :] * g_slice[:, :, np.newaxis, np.newaxis, np.newaxis],
                    axis=0
                )
                
                dX_padded[:, :, h_start:h_end, w_start:w_end] += np.sum(
                    self.params['W'][np.newaxis, :, :, :, :] * g_slice[:, :, np.newaxis, np.newaxis, np.newaxis],
                    axis=1
                )
        
        # 🚨 极其关键：除以 Batch Size，与 Linear 层保持数学一致！
        self.grads['W'] /= N
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=True) / N
        
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded
            
        return dX
    
    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

class GAP2D(Layer):
    """
    全局平均池化层 (Global Average Pooling 2D)
    输入: [batch_size, channels, H, W]
    输出: [batch_size, channels]
    """
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.params = {} 
        self.grads = {}

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        # X 维度为 (batch_size, channels, H, W)
        # 沿着 H(axis=2) 和 W(axis=3) 维度求均值
        return np.mean(X, axis=(2, 3))

    def backward(self, grads):
        # grads 是从 Loss 传回来的梯度，维度为: [batch_size, channels]
        batch_size, channels, H, W = self.input_shape
        
        # 将 grads 增加两个维度，变成 [batch_size, channels, 1, 1]，方便后续广播
        grads_reshaped = grads.reshape(batch_size, channels, 1, 1)
        
        # 因为前向是求平均 (除以 H*W)，所以反向传播时，每个元素的梯度就是 grads / (H*W)
        # 利用 np.ones 构建原始形状的矩阵，再乘上梯度
        dX = np.ones(self.input_shape) * (grads_reshaped / (H * W))
        
        return dX

class Flatten(Layer):
    """
    展平层 (Flatten Layer)
    输入: [batch_size, channels, H, W]
    输出: [batch_size, channels * H * W]
    """
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        batch_size = X.shape[0]
        # 保持 batch_size 不变，把后面的维度全部展平
        return X.reshape(batch_size, -1)

    def backward(self, grads):
        # 将上一层传回的一维梯度，恢复成原本的四维形状传给卷积层
        return grads.reshape(self.input_shape)
    
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.predicts = None
        self.labels = None
        self.grads = None
        self.optimizable = False
        self.probs = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels
        self.grads = None
        
        # Compute softmax probabilities for numerical stability
        eps = 1e-7
        x_max = np.max(predicts, axis=1, keepdims=True)
        x_exp = np.exp(predicts - x_max)
        self.probs = x_exp / (np.sum(x_exp, axis=1, keepdims=True) + eps)
        
        # Get probabilities for correct labels and compute cross-entropy loss
        batch_size = predicts.shape[0]
        correct_logprobs = -np.log(self.probs[np.arange(batch_size), labels] + eps)
        loss = np.mean(correct_logprobs)
        
        return loss
    
    def backward(self):
        # Compute the gradient: softmax - one_hot_labels
        self.grads = self.probs - np.eye(self.predicts.shape[1])[self.labels]
        
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition