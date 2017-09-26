#coding: utf-8

import numpy as np


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-8)) / batch_size


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=img.dtype)#dtypeをfloat32に設定しました。

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1), dtype=np.float32)#dtypeをfloat32に設定しました。
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def x2pad(x, stride):
    x_size = x.shape
    x_padded = np.zeros((x_size[0], x_size[1], (x_size[2]-1)*stride+3, (x_size[3]-1)*stride+3), dtype=np.float32)
    x_padded[:, :, 1:-1:stride, 1:-1:stride] = x
    return x_padded


def pad2x(pad,stride):
    return pad[:, :, 1:-1:stride, 1:-1:stride]


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Elu:
    def __init__(self,alpha=1.0):
        self.alpha = np.float32(alpha)
        self.mask = None
        self.x = None

    def forward(self, x):
        self.x = x
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = self.alpha * (np.exp(out[self.mask]) - 1)

        return out

    def backward(self, dout):
        len_dout = len(dout)
        mask = self.mask[:len_dout]
        dx = dout.copy()
        dx[mask] = dout[mask] * self.alpha * np.exp(self.x[:len_dout][mask])

        return dx
    

class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = np.tanh(x)
        self.out = out

        return out / 2 + 1

    def backward(self, dout):
        dx = dout * (1 - np.power(self.out,2)) / 2

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b, output_shape=None):
        self.W = W
        self.b = b
        self.output_shape = output_shape
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(x, self.W) + self.b

        if self.output_shape is not None:
            out = out.reshape(x.shape[0], *self.output_shape)

        return out

    def backward(self, dout):
        len_dout = len(dout)
        dout = dout.reshape(dout.shape[0], -1)
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x[:len_dout].T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(len_dout, *self.original_x_shape[1:])

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx * dout / batch_size

        return dx


class BatchNormalization:
    def __init__(self, gamma, beta, init_mu=0, init_std=1):
        self.gamma = gamma
        self.beta = beta
        self.input_shape = None  # Conv層の場合は4次元、全結合層の場合は2次元
        self.init_mu = init_mu
        self.init_std = init_std

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)#(N, C*H*W)に変形
        
        if x.shape[0] == 1:
            xn = (x.copy() - self.init_mu) / self.init_std
            out = self.gamma * xn + self.beta
            out = out.reshape(*self.input_shape)
            return out
        
        out = self.__forward(x)

        return out.reshape(*self.input_shape)#"*"はタプルの展開

    def __forward(self, x):
        mu = x.mean(axis=0)
        xc = x - mu
        var = np.mean(xc ** 2, axis=0)
        std = np.sqrt(var + 10e-7)
        xn = xc / std

        self.batch_size = x.shape[0]
        self.xc = xc
        self.xn = xn
        self.std = std
        self.init_mu = mu
        self.init_std = std

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)
            
        if dout.shape[0] == 1:
            dx = dout.copy().reshape(len(dout), *self.input_shape[1:])
            return dx

        dx = self.__backward(dout)

        dx = dx.reshape(len(dout), *self.input_shape[1:])
        return dx

    def __backward(self, dout):
        len_dout = len(dout)
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn[:len_dout] * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc[:len_dout]) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / len_dout) * self.xc[:len_dout] * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / len_dout

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中間データ（backward時に使用）
        self.x = None
        self.col = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape

        col = im2col(x, FH, FW, self.stride, self.pad)

        out = np.tensordot(col, self.W, ((1, 2, 3), (1, 2, 3))).astype(x.dtype, copy=False) + self.b
        out = np.rollaxis(out, 3, 1)

        self.x = x
        self.col = col

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        len_dout = len(dout)

        self.dW = np.tensordot(dout, self.col[:len_dout], ((0, 2, 3), (0, 4, 5))).astype(dout.dtype, copy=False)
        self.db = dout.sum(axis=(0, 2, 3))

        dcol = np.tensordot(dout, self.W, (1, 0)).astype(dout.dtype, copy=False).transpose(0, 3, 4, 5, 1, 2)
        #input_shape = (len_dout, *self.x.shape[1:])
        #python2だとこの記法はうまくいきません
        #関数の引数で使うときは問題ないのですが。
        input_shape = (len_dout, self.x.shape[1], self.x.shape[2], self.x.shape[3])
        dx = col2im(dcol, input_shape, FH, FW, self.stride, self.pad)

        return dx


class Deconvolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.out_h = None
        self.out_w = None

        # 中間データ（backward時に使用）
        self.x = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, h, w = x.shape
        self.x = x

        if self.out_h is None:
            self.out_h = self.stride * (h - 1) + FH - 2 * self.pad
        if self.out_w is None:
            self.out_w = self.stride * (w - 1) + FW - 2 * self.pad

        dcol = np.tensordot(self.W, x, (1, 1)).astype(x.dtype, copy=False)
        dcol = np.rollaxis(dcol, 3)

        y = col2im(dcol, (N, FN, self.out_h, self.out_w), FH, FW, stride=self.stride, pad=self.pad)

        y += self.b.reshape(1, self.b.size, 1, 1)
        return y

    def backward(self, dout):
        FH, FW = self.W.shape[2:]
        col = im2col(dout, FH, FW, stride=self.stride, pad=self.pad)

        self.dW = np.tensordot(self.x, col, ([0, 2, 3], [0, 4, 5])).astype(self.W.dtype, copy=False)
        self.dW = np.rollaxis(self.dW, 1)
        self.db = dout.sum(axis=(0, 2, 3))

        dx = np.tensordot(col, self.W, ([1, 2, 3], [0, 2, 3])).astype(dout.dtype, copy=False)
        dx = np.rollaxis(dx, 3, 1)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad).transpose(0,1,4,5,2,3)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, C, out_h, out_w)
        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        
        dmax = dmax.reshape(dout.shape + (self.pool_h,self.pool_w))

        dcol = dmax.transpose(0, 3, 4, 5, 1, 2)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx