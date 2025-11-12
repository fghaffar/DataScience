import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this notebook we are creating a NN and backpropagation from scratch baed on https://github.com/karpathy/micrograd
    """)
    return


@app.cell
def _():
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    return math, np, plt


@app.function
def f(x):
    return 3*x**2 - 4*x + 5


@app.cell
def _():
    f(3.0)
    return


@app.cell
def _(np, plt):
    xs = np.arange(-5,5,0.25)
    ys = f(xs)
    plt.plot(xs,ys)
    return


@app.cell
def _():
    # what is derivative of f(x) at x=3.0
    # derative measures the slope of the response when there is small change in the input
    # it tells us whether slope goes up or down at that point after the change
    # let's say change h is very very small like 0.0001
    _h = 0.001
    x = 3.0
    f(x + _h)
    # at x=3.0 the slope is positive
    print(f"when  is +3, slope = {(f(x + _h) - f(x)) / _h}")  # divide by the change to get the slope
    # output 14.00300000000243
    #now make the change very very small
    _h = 1e-08
    print((f(x + _h) - f(x)) / _h)
    #output is 14.00000009255109
    # now make the x negative
    x = -3.0
    #slope is negative
    # output is -22.00000039920269
    (f(x + _h) - f(x)) / _h
    return


@app.cell
def _():
    # get a bit more complex more variables 
    a = 2.0
    b = -3.0
    c = 10.0
    d = a*b+c
    print(d)
    return a, b, c


@app.cell
def _(a, b, c):
    _h = 0.001
    d1 = a * b + c
    c_1 = c + _h
    d2 = a * b + c_1
    print('d1', d1)
    print('d2', d2)
    print('slope', (d2 - d1) / _h)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Neural Network
    NN are massive mathematical expressions and we need a data structure to maintain those expressions.
    so we build a value object
    """)
    return


@app.cell
def _():
    # a container to represent the value object
    class Value:

        def __init__(self, data):
            self.data = data

        def __repr__(self) -> str:
            return f'Value(data={self.data})'
    a_1 = Value(5.0)
    a_1
    return (Value,)


@app.cell
def _(Value):
    # how about adding two value objects
    # it will throw an error, since python does not know how to add two value objects
    a_2 = Value(5.0)
    b_1 = Value(3.0)
    a_2 + b_1
    return


@app.cell
def _(Value):
    # we define the add method to add two value objects
    class Value_1:

        def __init__(self, data):
            self.data = data

        def __repr__(self) -> str:
            return f'Value(data={self.data})'

        def __add__(self, other):
            return Value(self.data + other.data)

        def __mul__(self, other):
            return Value(self.data * other.data)
    a_3 = Value_1(2.0)
    b_2 = Value_1(-3.0)
    c_2 = Value_1(10.0)
    a_3 + b_2
    a_3 * b_2
    # now add multuplication
    a_3.__mul__(b_2).__add__(c_2)
    # we can also by objects
    a_3 * b_2 + c_2
    return (Value_1,)


@app.cell
def _(Value_1):
    # now let add a structure to keep track of what value produces what resutls
    # this is to get the DAG
    class Value_2:

        def __init__(self, data, _children=()):
            self.data = data
            self._prev = set(_children)

        def __repr__(self) -> str:
            return f'Value(data={self.data})'

        def __add__(self, other):
            return Value_1(self.data + other.data, (self, other))

        def __mul__(self, other):
            return Value_1(self.data * other.data, (self, other))
    a_4 = Value_2(2.0)
    b_3 = Value_2(-3.0)
    c_3 = Value_2(10.0)
    d_1 = a_4 * b_3 + c_3
    print(d_1)
    print(d_1._prev)
    return Value_2, d_1


@app.cell
def _(d_1):
    d_1._prev
    return


@app.cell
def _(Value_2):
    # now let's add name of operation that created the output
    class Value_3:

        def __init__(self, data, _children=(), _op=''):
            self.data = data
            self._prev = set(_children)
            self._op = _op

        def __repr__(self) -> str:
            return f'Value(data={self.data})'

        def __add__(self, other):
            return Value_2(self.data + other.data, (self, other), '+')

        def __mul__(self, other):
            return Value_2(self.data * other.data, (self, other), '*')
    a_5 = Value_3(2.0)
    b_4 = Value_3(-3.0)
    c_4 = Value_3(10.0)
    d_2 = a_5 * b_4 + c_4
    return Value_3, a_5, b_4, c_4, d_2


@app.cell
def _(d_2):
    d_2._op
    return


@app.cell
def _(d_2):
    d_2._prev
    return


@app.cell
def _():
    # visualise the DAG
    from graphviz import Digraph

    def _trace(root):
        nodes, edges = (set(), set())

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
        build(root)
        return (nodes, edges)

    def draw_dot(root):
        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
        nodes, edges = _trace(root)
        for n in nodes:  # for any value in the graph, create rectangualr representation of the node
            uid = str(id(n))
            dot.node(name=uid, label='{data %0.4f }' % (n.data,), shape='record')
            if n._op:
                dot.node(name=uid + n._op, label=n._op)
                dot.edge(uid + n._op, uid)
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)  # connect n1 to op node of n2
        return dot
    return Digraph, draw_dot


@app.cell
def _(a_5, b_4, c_4, draw_dot):
    draw_dot(a_5 * b_4 + c_4)
    return


@app.cell
def _(Value_3):
    # now let's add labels
    class Value_4:

        def __init__(self, data, _children=(), _op='', label=''):
            self.data = data
            self._prev = set(_children)
            self._op = _op
            self.label = label

        def __repr__(self) -> str:
            return f'Value(data={self.data})'

        def __add__(self, other):
            return Value_3(self.data + other.data, (self, other), '+')

        def __mul__(self, other):
            return Value_3(self.data * other.data, (self, other), '*')
    a_6 = Value_4(2.0, label='a')
    b_5 = Value_4(-3.0, label='b')
    c_5 = Value_4(10.0, label='c')
    e = a_6 * b_5
    e.label = 'e'
    d_3 = e + c_5
    # now let make a layer deep and make out f
    d_3.label = 'd'
    d_3
    f_1 = Value_4(-2.0, label='f')
    L = d_3 * f_1
    L.label = 'L'
    L
    return L, Value_4, d_3


@app.cell
def _(Digraph):
    def _trace(root):
        nodes, edges = (set(), set())

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
        build(root)
        return (nodes, edges)

    def draw_dot_1(root):
        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
        nodes, edges = _trace(root)
        for n in nodes:
            uid = str(id(n))
            dot.node(name=uid, label='{%s | data %0.4f }' % (n.label, n.data), shape='record')
            if n._op:
                dot.node(name=uid + n._op, label=n._op)
                dot.edge(uid + n._op, uid)
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        return dot
    return (draw_dot_1,)


@app.cell
def _(d_3, draw_dot_1):
    draw_dot_1(d_3)
    return


@app.cell
def _(L, draw_dot_1):
    draw_dot_1(L)
    return


@app.cell
def _(Value_4):
    # in NN, L is the loss function, we want to find the derivative of L with respect to a,b,c,d,e,f
    # this deraovative is called gradient, stored in variable grad
    # now let's add labels
    class Value_5:

        def __init__(self, data, _children=(), _op='', label=''):
            self.data = data
            self._prev = set(_children)
            self._op = _op
            self.label = label
            self.grad = 0.0

        def __repr__(self) -> str:
            return f'Value(data={self.data})'

        def __add__(self, other):
            return Value_4(self.data + other.data, (self, other), '+')

        def __mul__(self, other):
            return Value_4(self.data * other.data, (self, other), '*')
    a_7 = Value_5(2.0, label='a')
    b_6 = Value_5(-3.0, label='b')
    c_6 = Value_5(10.0, label='c')
    e_1 = a_7 * b_6
    e_1.label = 'e'
    d_4 = e_1 + c_6
    # now let make a layer deep and make out f
    d_4.label = 'd'
    d_4
    f_2 = Value_5(-2.0, label='f')
    L_1 = d_4 * f_2
    L_1.label = 'L'
    L_1
    return L_1, Value_5, a_7, b_6, c_6, d_4, e_1, f_2


@app.cell
def _(Digraph, L_1):
    def _trace(root):
        nodes, edges = (set(), set())

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
        build(root)
        return (nodes, edges)

    def draw_dot_2(root):
        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
        nodes, edges = _trace(root)
        for n in nodes:
            uid = str(id(n))
            dot.node(name=uid, label='{%s | data %0.4f | grad %.4f }' % (n.label, n.data, n.grad), shape='record')
            if n._op:
                dot.node(name=uid + n._op, label=n._op)
                dot.edge(uid + n._op, uid)
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        return dot
    draw_dot_2(L_1)
    return (draw_dot_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    L = d* f
    dL / dd =? f
    definition of derative is:
    (f(x+h) - f(x))/h
    replace d with d+h
    ((d+h)*f - d* f)/h
    (d*f + h*f - d*f)/h
    f

    **The '+' operator is local gradient that just passses the information without any change. Change happens with "*" operator**

    How L is sensitive to c?
    via chain rule

    dd / dc ?
    d = c + e

    definition of derative is:
    (f(x+h) - f(x))/h

    ((c+h + e) - (c+e))/h
    (c+h+e-c-e)/h
    h/h = 1.0

    similarly
    dd /de = 1.0

    This is local derative with the + node

    WANRT:
    dL / dc

    KNOW:
    dL / dd
    dd /dc

    By chain rule

    dL / dc = (dL / dd) * (dd /dc)
    """)
    return


@app.cell
def _(c_6, d_4, e_1, f_2):
    #L.grad = 1.0
    d_4.grad = f_2.data
    f_2.grad = d_4.data
    c_6.grad = -2.0
    e_1.grad = -2.0
    return


@app.cell
def _(L_1, draw_dot_2):
    draw_dot_2(L_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now applying chain rule all the way back to start of the above graph

    dL / de = -2.0

    dL /da = (dL / de) * (de / da)
    """)
    return


@app.cell
def _(a_7, b_6, e_1):
    a_7.grad = b_6.data * e_1.grad
    b_6.grad = a_7.data * e_1.grad
    return


@app.cell
def _(Value_5):
    def lol():
        _h = 0.001
        a = Value_5(2.0, label='a')
        b = Value_5(-3.0, label='b')
        c = Value_5(10.0, label='c')
        e = a * b
        e.label = 'e'
        d = e + c
        d.label = 'd'
        f = Value_5(-2.0, label='f')
        L = d * f
        L.label = 'L'
        L1 = L.data
        a = Value_5(2.0, label='a')
        b = Value_5(-3.0, label='b')
        c = Value_5(10.0, label='c')
        e = a * b
        e.label = 'e'
        e.data = e.data + _h
        d = e + c
        d.label = 'd'
        f = Value_5(-2.0, label='f')
        L = d * f
        L.label = 'L'
        L2 = L.data
        print((L2 - L1) / _h)
    lol()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Backpropagation
    It is the recursive application of chain rule backwards through the computation graph
    Use chain-rule to find the derative of oeprators that are one or moe step away

    SUmmarised chain rule is: " if a  car travels twice as fast as a bucylce and bicycle is 4 times as fast as a walking amn, then car travels 2 x 4 = 8 times as fast as the man."
    """)
    return


@app.cell
def _(a_7, b_6, c_6, f_2):
    step = 0.01
    a_7.data = a_7.data + step * a_7.grad
    b_6.data = b_6.data + step * b_6.grad
    c_6.data = c_6.data + step * c_6.grad
    f_2.data = f_2.data + step * f_2.grad
    e_2 = a_7 * b_6
    # forward pass
    d_5 = e_2 + c_6
    L_2 = d_5 * f_2
    print(L_2.data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Backpropagation in NN
    NN ahve input vector multiplied by weights (synapses) added and passed though an activation function (squashing functions)
    """)
    return


@app.cell
def _(np, plt):
    plt.plot(np.arange(-5,5,0.2), np.tanh(np.arange(-5,5,0.2))); plt.grid()
    return


@app.cell
def _(Value_5, draw_dot_2):
    # inputs x1,x2
    x1 = Value_5(2.0, label='x1')
    x2 = Value_5(0.0, label='x2')
    # weights w1,w2
    w1 = Value_5(-3.0, label='w1')
    w2 = Value_5(1.0, label='w2')
    # bias of the neuron
    b_7 = Value_5(6.7, label='b')
    # x1*w1 + x2*w2 + b
    x1w1 = x1 * w1
    x1w1.label = 'x1*w1'
    x2w2 = x2 * w2
    x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b_7
    n.label = 'n'
    draw_dot_2(n)
    return


@app.cell
def _(Value_5, math):
    # first implement the sigmoid function in the Value class
    class Value_6:

        def __init__(self, data, _children=(), _op='', label=''):
            self.data = data
            self._prev = set(_children)
            self._op = _op
            self.label = label
            self.grad = 0.0

        def __repr__(self) -> str:
            return f'Value(data={self.data})'

        def __add__(self, other):
            return Value_5(self.data + other.data, (self, other), '+')

        def __mul__(self, other):
            return Value_5(self.data * other.data, (self, other), '*')

        def tanh(self):
            x = self.data
            t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
            out = Value_5(t, (self,), 'tanh')
            return out
    a_8 = Value_6(2.0, label='a')
    b_8 = Value_6(-3.0, label='b')
    c_7 = Value_6(10.0, label='c')
    e_3 = a_8 * b_8
    e_3.label = 'e'
    d_6 = e_3 + c_7
    # now let make a layer deep and make out f
    d_6.label = 'd'
    d_6
    f_3 = Value_6(-2.0, label='f')
    L_3 = d_6 * f_3
    L_3.label = 'L'
    L_3
    return (Value_6,)


@app.cell
def _(Value_6):
    # inputs x1,x2
    x1_1 = Value_6(2.0, label='x1')
    x2_1 = Value_6(0.0, label='x2')
    # weights w1,w2
    w1_1 = Value_6(-3.0, label='w1')
    w2_1 = Value_6(1.0, label='w2')
    # bias of the neuron
    b_9 = Value_6(6.881373587019543, label='b')
    # x1*w1 + x2*w2 + b
    x1w1_1 = x1_1 * w1_1
    x1w1_1.label = 'x1*w1'
    x2w2_1 = x2_1 * w2_1
    x2w2_1.label = 'x2*w2'
    x1w1x2w2_1 = x1w1_1 + x2w2_1
    x1w1x2w2_1.label = 'x1*w1 + x2*w2'
    n_1 = x1w1x2w2_1 + b_9
    n_1.label = 'n'
    o = n_1.tanh()
    o.label = 'o'
    return b_9, n_1, o, w1_1, w2_1, x1_1, x1w1_1, x1w1x2w2_1, x2_1, x2w2_1


@app.cell
def _(draw_dot_2, o):
    draw_dot_2(o)
    return


@app.cell
def _(n_1, o):
    o.grad = 1.0
    # o = tanh(h)
    #do/dn = 1-o**2 
    n_1.grad = 1 - o.data ** 2
    return


@app.cell
def _(
    b_9,
    draw_dot_2,
    n_1,
    o,
    w1_1,
    w2_1,
    x1_1,
    x1w1_1,
    x1w1x2w2_1,
    x2_1,
    x2w2_1,
):
    n_1.grad
    # now + operator just distributes the gradient to the children, therefore
    x1w1x2w2_1.grad = n_1.grad
    b_9.grad = n_1.grad
    x1w1_1.grad = x1w1x2w2_1.grad
    x2w2_1.grad = x1w1x2w2_1.grad
    x1_1.grad = x1w1_1.grad * w1_1.data
    w1_1.grad = x1w1_1.grad * x1_1.data
    x2_1.grad = x2w2_1.grad * w2_1.data
    w2_1.grad = x2w2_1.grad * x2_1.data
    draw_dot_2(o)
    return


@app.cell
def _(Value_6, math):
    class Value_7:

        def __init__(self, data, _children=(), _op='', label=''):
            self.data = data
            self._prev = set(_children)
            self._op = _op
            self.label = label
            self.grad = 0.0
            self._backward = lambda: None

        def __repr__(self) -> str:
            return f'Value(data={self.data})'

        def __add__(self, other):
            out = Value_6(self.data + other.data, (self, other), '+')

            def _backward():
                self.grad = 1.0 * out.grad
                other.grad = 1.0 * out.grad
            out._backward = _backward()
            return out

        def __mul__(self, other):
            out = Value_6(self.data * other.data, (self, other), '*')

            def _backward():
                self.grad = other.data * out.grad
                other.grad = self.data * out.grad
            out._backward = _backward()
            return out

        def tanh(self):
            x = self.data
            t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
            out = Value_6(t, (self,), 'tanh')

            def _backward():
                self.grad = (1 - t ** 2) * out.grad
            out._backward = _backward()
            return out

        def backward(self):
            self.grad = 1.0
            if self._op == 'tanh':
                self._prev[0].grad = self.grad * (1 - self.data ** 2)
            else:
                for _p in self._prev:
                    _p.grad = _p.grad + self.grad
    a_9 = Value_7(2.0, label='a')
    b_10 = Value_7(-3.0, label='b')
    c_8 = Value_7(10.0, label='c')
    e_4 = a_9 * b_10
    e_4.label = 'e'
    d_7 = e_4 + c_8
    d_7.label = 'd'
    d_7
    f_4 = Value_7(-2.0, label='f')
    L_4 = d_7 * f_4
    L_4.label = 'L'
    L_4
    return (Value_7,)


@app.cell
def _(Value_7, draw_dot_2, math):
    class Value_8:

        def __init__(self, data, _children=(), _op='', label=''):
            self.data = data
            self.grad = 0.0
            self._backward = lambda: None
            self._prev = set(_children)
            self._op = _op
            self.label = label

        def __repr__(self):
            return f'Value(data={self.data})'

        def __add__(self, other):
            other = other if isinstance(other, Value_7) else Value_7(other)  # this is to faciliate adding scalar to Value
            out = Value_7(self.data + other.data, (self, other), '+')

            def _backward():
                self.grad = self.grad + 1.0 * out.grad
                other.grad = other.grad + 1.0 * out.grad
            out._backward = _backward
            return out

        def __mul__(self, other):
            other = other if isinstance(other, Value_7) else Value_7(other)
            out = Value_7(self.data * other.data, (self, other), '*')

            def _backward():
                self.grad = self.grad + other.data * out.grad
                other.grad = other.grad + self.data * out.grad  # += is used to accumulate the gradient otherwise it will be overwritten
            out._backward = _backward
            return out

        def __rmul__(self, other):
            return self.__mul__(other)

        def tanh(self):
            x = self.data  # python calls this method when the left operand is not a Value object. e.g. 2 * Value(3.0)
            t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
            out = Value_7(t, (self,), 'tanh')

            def _backward():
                self.grad = self.grad + (1 - t ** 2) * out.grad
            out._backward = _backward
            return out

        def exp(self):
            x = self.data
            t = math.exp(x)
            out = Value_7(t, (self,), 'exp')

            def _backward():
                self.grad = self.grad + t * out.grad
            out._backward = _backward
            return out

        def __pow__(self, other):
            assert isinstance(other, (int, float)), 'only integer or float power is supported'
            out = Value_7(self.data ** other, (self,), f'**{other}')

            def _backward():
                self.grad = self.grad + other * self.data ** (other - 1) * out.grad
            out._backward = _backward

        def __truediv__(self, other):
            return self * other ** (-1)

        def __neg__(self):
            return self * -1

        def __sub__(self, other):
            return self + -other

        def backward(self):  # -self
            topo = []
            visited = set()
      # self - other
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            build_topo(self)
            self.grad = 1.0
            for node in reversed(topo):
                node._backward()
    x1_2 = Value_8(2.0, label='x1')
    x2_2 = Value_8(0.0, label='x2')
    w1_2 = Value_8(-3.0, label='w1')
    w2_2 = Value_8(1.0, label='w2')
    b_11 = Value_8(6.881373587019543, label='b')
    x1w1_2 = x1_2 * w1_2
    x1w1_2.label = 'x1*w1'
    x2w2_2 = x2_2 * w2_2
    x2w2_2.label = 'x2*w2'
    # inputs x1,x2
    x1w1x2w2_2 = x1w1_2 + x2w2_2
    x1w1x2w2_2.label = 'x1*w1 + x2*w2'
    # weights w1,w2
    n_2 = x1w1x2w2_2 + b_11
    n_2.label = 'n'
    # bias of the neuron
    e_5 = (2 * n_2).exp()
    # x1*w1 + x2*w2 + b
    o_1 = (e_5 - 1) / (e_5 + 1)
    o_1.label = 'o'
    o_1.backward()
    #------
    draw_dot_2(o_1)
    return Value_8, b_11, n_2, o_1, x1w1_2, x1w1x2w2_2, x2w2_2


@app.cell
def _(draw_dot_2, o_1):
    draw_dot_2(o_1)
    return


@app.cell
def _(b_11, n_2, o_1, x1w1_2, x1w1x2w2_2, x2w2_2):
    # now no need to set the gradient manually
    o_1.grad = 1.0
    o_1._backward()
    n_2._backward()
    b_11._backward()
    x1w1x2w2_2._backward()
    x1w1_2._backward()
    x2w2_2._backward()
    return


@app.cell
def _(o_1):
    # instead of calling backward on each node, we will build  a topological order of the nodes and call backward on each node in reverse order
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(o_1)
    topo
    return


@app.cell
def _(o_1):
    o_1.grad = 1.0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There is bug in the first v esion if the variable is used more than once. so we need to do multivariate backpropagation. therefore we need to accumulate the gradients.
    """)
    return


@app.cell
def _(Value_8, math):
    class Value_9:

        def __init__(self, data, _children=(), _op='', label=''):
            self.data = data
            self.grad = 0.0
            self._backward = lambda: None
            self._prev = set(_children)
            self._op = _op
            self.label = label

        def __repr__(self):
            return f'Value(data={self.data})'

        def __add__(self, other):
            other = other if isinstance(other, Value_8) else Value_8(other)
            out = Value_8(self.data + other.data, (self, other), '+')

            def _backward():
                self.grad = self.grad + 1.0 * out.grad
                other.grad = other.grad + 1.0 * out.grad
            out._backward = _backward
            return out

        def __mul__(self, other):
            other = other if isinstance(other, Value_8) else Value_8(other)
            out = Value_8(self.data * other.data, (self, other), '*')

            def _backward():
                self.grad = self.grad + other.data * out.grad
                other.grad = other.grad + self.data * out.grad
            out._backward = _backward
            return out

        def __pow__(self, other):
            assert isinstance(other, (int, float)), 'only supporting int/float powers for now'
            out = Value_8(self.data ** other, (self,), f'**{other}')

            def _backward():
                self.grad = self.grad + other * self.data ** (other - 1) * out.grad
            out._backward = _backward
            return out

        def __rmul__(self, other):
            return self * other

        def __truediv__(self, other):  # other * self
            return self * other ** (-1)

        def __neg__(self):  # self / other
            return self * -1

        def __sub__(self, other):  # -self
            return self + -other

        def __radd__(self, other):  # self - other
            return self + other

        def tanh(self):  # other + self
            x = self.data
            t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
            out = Value_8(t, (self,), 'tanh')

            def _backward():
                self.grad = self.grad + (1 - t ** 2) * out.grad
            out._backward = _backward
            return out

        def exp(self):
            x = self.data
            out = Value_8(math.exp(x), (self,), 'exp')

            def _backward():
                self.grad = self.grad + out.data * out.grad
            out._backward = _backward
            return out

        def backward(self):  # NOTE: in the video I incorrectly used = instead of +=. Fixed here.
            topo = []
            visited = set()

            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            build_topo(self)
            self.grad = 1.0
            for node in reversed(topo):
                node._backward()
    return (Value_9,)


@app.cell
def _(Value_9, draw_dot_2):
    # inputs x1,x2
    x1_3 = Value_9(2.0, label='x1')
    x2_3 = Value_9(0.0, label='x2')
    # weights w1,w2
    w1_3 = Value_9(-3.0, label='w1')
    w2_3 = Value_9(1.0, label='w2')
    # bias of the neuron
    b_12 = Value_9(6.881373587019543, label='b')
    # x1*w1 + x2*w2 + b
    x1w1_3 = x1_3 * w1_3
    x1w1_3.label = 'x1*w1'
    x2w2_3 = x2_3 * w2_3
    x2w2_3.label = 'x2*w2'
    #------
    x1w1x2w2_3 = x1w1_3 + x2w2_3
    x1w1x2w2_3.label = 'x1*w1 + x2*w2'
    n_3 = x1w1x2w2_3 + b_12
    n_3.label = 'n'
    e_6 = (2 * n_3).exp()
    o_2 = (e_6 - 1) / (e_6 + 1)
    o_2.label = 'o'
    o_2.backward()
    draw_dot_2(o_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pytorch

    create a class of NN
    """)
    return


@app.cell
def _():
    import torch
    import random
    return random, torch


@app.cell
def _(torch):
    x1_4 = torch.Tensor([2.0]).double()
    x1_4.requires_grad = True  #use double to convert to float64 scalar python by default uses double precision floating point
    x2_4 = torch.Tensor([0.0]).double()
    x2_4.requires_grad = True
    w1_4 = torch.Tensor([-3.0]).double()
    w1_4.requires_grad = True
    w2_4 = torch.Tensor([1.0]).double()
    w2_4.requires_grad = True
    b_13 = torch.Tensor([6.881373587019543]).double()
    b_13.requires_grad = True
    n_4 = x1_4 * w1_4 + x2_4 * w2_4 + b_13
    o_3 = torch.tanh(n_4)
    print(o_3.data.item())
    o_3.backward()
    print('x1', x1_4.grad.item())
    print('x2', x2_4.grad.item())
    print('w1', w1_4.grad.item())
    print('w2', w2_4.grad.item())
    return


@app.cell
def _(Value_9, random):
    class Neuron:

        def __init__(self, nin):
            self.w = [Value_9(random.uniform(-1, 1)) for _ in range(nin)]
            self.b = Value_9(random.uniform(-1, 1))

        def __call__(self, x):
            act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
            out = act.tanh()
            return out

        def parameters(self):
            return self.w + [self.b]

    class Layer:

        def __init__(self, nin, nout):
            self.neurons = [Neuron(nin) for _ in range(nout)]

        def __call__(self, x):
            outs = [n(x) for n in self.neurons]
            return outs[0] if len(outs) == 1 else outs

        def parameters(self):
            return [_p for neuron in self.neurons for _p in neuron.parameters()]

    class MLP:

        def __init__(self, nin, nouts):
            sz = [nin] + nouts
            self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

        def parameters(self):
            return [_p for layer in self.layers for _p in layer.parameters()]
    x_1 = [2.0, 3.0, -1.0]
    n_5 = MLP(4, [4, 4, 1])
    n_5(x_1)
    return n_5, x_1


@app.cell
def _(draw_dot_2, n_5, x_1):
    draw_dot_2(n_5(x_1))
    return


@app.cell
def _():
    xs_1 = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys_1 = [1.0, -1.0, -1.0, 1.0]  # desired targets
    return xs_1, ys_1


@app.cell
def _(n_5, xs_1):
    ypred = [n_5(x) for x in xs_1]
    ypred
    return (ypred,)


@app.cell
def _(ypred, ys_1):
    loss = sum(((yout - ygt) ** 2 for yout, ygt in zip(ypred, ys_1)))
    loss
    return (loss,)


@app.cell
def _(loss):
    loss.backward()
    return


@app.cell
def _(n_5):
    n_5.layers[0].neurons[0].w[0].grad
    return


@app.cell
def _(n_5):
    n_5.layers[0].neurons[0].w[0].data
    return


@app.cell
def _(draw_dot_2, loss):
    draw_dot_2(loss)
    return


@app.cell
def _(n_5):
    for _p in n_5.parameters():
        _p.data = _p.data + -0.01 * _p.grad
    return


@app.cell
def _(n_5):
    n_5.layers[0].neurons[0].w[0].data
    return


@app.cell
def _(n_5, xs_1, ys_1):
    for k in range(20):
        ypred_1 = [n_5(x) for x in xs_1]
        loss_1 = sum(((yout - ygt) ** 2 for yout, ygt in zip(ypred_1, ys_1)))
        loss_1.backward()
        for _p in n_5.parameters():
            _p.grad = 0
            _p.data = _p.data + -0.01 * _p.grad
        print(k, loss_1.data)
    return (ypred_1,)


@app.cell
def _(ypred_1):
    ypred_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
