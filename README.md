# npgrad

A slow and inefficient autograd engine built on NumPy, offering a PyTorch-like API for some basic neural net operations (linear transformation, 2D convolution and pooling). This is only meant to be a proof-of-concept built for fun, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

### Functioning

`npgrad` is based on an `Array` class wrapping a NumPy `ndarray` (and implementing the [`__array__` method](https://numpy.org/devdocs/user/basics.interoperability.html#the-array-method)), with some extra bits to handle the gradient computation. Some of the main NumPy functions (`add`, `multiply`, `log`, `reshape`, ...) have an implementation which works with `Array` and can be called from either `npgrad` or `numpy` itself. `Array` also supports some of the same attributes and methods provided by PyTorch's `Tensor` (`requires_grad`, `grad`, `backward()`) and can be used with the few functions from `npgrad.nn`.
### Usage example

```python
import numpy as np
import npgrad as npg

a = npg.array([1, 2, 3])

# both b and c are Array instances
b = npg.log(a)
c = np.log(a)
print(b.__class__, c.__class__)  # prints <class 'npgrad.array.Array'> for both

# start tracking operations on a for grad computation
a.requires_grad = True
(a ** 2).sum().backward()   # populate a.grad

# nn example
n, c = 4, 3
data = np.random.rand(n, 3, 16, 16)
lbls = np.random.randint(0, c, n)
w1 = npg.asarray(np.random.rand(2, 3, 4, 4)).requires_grad_()
x = npg.nn.functional.conv2d(data, w1, stride=2)
x = npg.nn.functional.max_pool2d(x, 4, padding=1)
x = x.reshape((n, -1))  # -> (n, 8)
w2 = npg.asarray(np.random.rand(8, c)).requires_grad_()
x = x @ w2  # -> (n, c)
loss = npg.nn.functional.cross_entropy(x, lbls).mean()
loss.backward()  # populate w1.grad and w2.grad

# support modules too
linear = npg.nn.Linear(16, 5)
conv = npg.nn.Conv2d(4, 3, kernel_size=5, stride=3, padding=2)
```
