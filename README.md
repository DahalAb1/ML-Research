# ML Research

This is my collection of machine learning algorithms that I'm implementing to
really understand how they work, not just how to call them. I'm doing this under
the guidance of my professor.

Everything is built in **PyTorch**, but I go through it at two levels. First I
write things out the low-level way, using raw tensors, declaring the parameters
myself, and running the training loop by hand. That way I can actually see the
mechanics. Then I rebuild the same idea with the high-level tools like
`nn.Module`, optimizers, and built-in loss functions, which do all of that in
just a few lines. It's the same library and the same ideas, I'm just peeling back
the abstraction to see what's underneath.

I'm using PyTorch because it's what most of the industry and research world
actually uses, so what I learn here carries over to real work and to reading
other people's code. It also goes way beyond these small examples: it can compute
gradients automatically for models with millions of parameters, run on a GPU, and
hand you ready-made layers and optimizers. That's exactly why I think it's worth
understanding the low-level mechanics first, before leaning on the shortcuts.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Contents

| File | What it covers | Key ideas |
|---|---|---|
| `linear_regression.ipynb` | Linear regression written by hand | autograd, MSE, manual gradient descent (slope first, then slope + bias) |
| `linear_regression_pytorch.py` | The same idea with PyTorch's high-level tools | `nn.Module`, `nn.Linear`, `MSELoss`, `optim.SGD` |

If you're reading this to learn, you can stay on this page. Each section below
walks through what the code does and why, and shows what comes out when you run
it. Open the actual files only when you want to see the line-by-line code.

---

## Linear Regression (`linear_regression.ipynb`)

This notebook builds linear regression the slow way, without `nn.Module` or a
built-in optimizer, so that every piece of the training loop is out in the open.
The goal isn't really to fit a line. It's to see how a model learns one.

### Making data we already know the answer to

Before training anything, I need data. I create it myself so I already know the
true answer the model is supposed to discover.

```python
X = torch.arange(-5, 5, 0.1).view(-1, 1)   # inputs from -5 to 5
func = -5 * X                              # the true line: y = -5x
```

So `func` is a perfect straight line with a slope of **-5**. Real data is never
that clean, though, so I add a bit of Gaussian noise to scatter the points around
the line:

```python
Y = func + 0.4 * torch.randn(X.size())     # noisy version of the true line
```

Now I have a cloud of points that roughly follow `y = -5x`. The question the rest
of the notebook answers is: can the model recover that slope of -5 just by
looking at the noisy data?

### The model

The model is about as simple as it gets, a single weight `w` and no bias term:

```python
w = torch.tensor(-10.0, requires_grad=True)   # deliberately wrong starting guess

def forward(x):
    return w * x
```

I start `w` at -10, far from the true -5, so I can watch it travel to the right
answer. The `requires_grad=True` part tells PyTorch to track what happens to `w`
so it can work out the gradients for me later.

### Measuring how wrong we are

Before it can improve, the model needs to know how badly it's currently doing.
For that I use **Mean Squared Error**: take the difference between the prediction
and the truth, square it so positives and negatives don't cancel out, and average
over all the points.

```python
def criterion(y_pred, y):
    return torch.mean((y_pred - y) ** 2)
```

### Learning by gradient descent

This is the heart of the notebook. Each iteration does four things:

```python
Y_pred = forward(X)              # 1. predict
loss = criterion(Y_pred, Y)      # 2. measure how wrong we are
loss.backward()                  # 3. compute the gradient of the loss w.r.t. w
w.data = w.data - step_size * w.grad.data   # 4. step toward a lower loss
w.grad.data.zero_()              # reset the gradient for the next round
```

Step 4 is the one worth slowing down on. Here's the intuition I keep in my head:

> The gradient (`w.grad`) points in the direction of steepest *ascent*, the way
> that makes the loss climb fastest from where I'm standing right now. If I know
> the direction that goes up the hill, then the exact opposite direction goes down
> it. So I flip the sign of the gradient to head toward the minimum.
>
> Moving anywhere needs two things, a direction and a distance:
>
> * `w.grad.data` gives the direction (I flip its sign to go downhill)
> * `step_size`, the learning rate, decides how far I step each time
>
> ```
> w_new = w_old - learning_rate * w.grad.data
> ```

Two small details that are easy to miss:

* `w.data` and `w.grad.data`: I reach for `.data` so this manual update isn't
  itself recorded by autograd. I don't want it computing gradients *of the update*.
* `w.grad.data.zero_()`: PyTorch adds up gradients by default, so if I don't reset
  them to zero each round, the next gradient just piles on top of the old one.

### The result

Over 20 iterations, `w` climbs from its wrong starting point to almost exactly the
true slope:

```
iter   loss        w
0      0.1758     -4.998
...
19     0.1758     -4.999
```

It lands at about **-4.999**, so it recovered the -5 I hid in the data. The loss
settles around **0.176** and stops dropping, but that floor isn't a failure. It's
the noise I added myself. The model fits the underlying line as well as it
possibly can, and what's left over is the randomness it was never going to be able
to predict. Plotting the loss per iteration shows it falling fast and then
flattening out, which is what convergence looks like.

### Adding a second parameter, the bias

The model so far is `y = w·x`. Because there's no constant term, that line is
stuck going through the origin: when `x = 0`, it has to output `y = 0`. Most real
lines don't pass through the origin, so I add a second learnable parameter, the
**bias** `b`:

```
y = w·x + b
```

Now `w` controls the slope and `b` shifts the whole line up or down. To make sure
the bias is actually pulling its weight, this time the true line has an intercept
of its own: `y = -5x + 1`.

The training loop barely changes. I just keep two parameters, give each one its
own gradient step, and zero out both gradients each round:

```python
w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-20.0, requires_grad=True)

def forward(x):
    return w * x + b              # the only real change is the "+ b"

loss.backward()                   # fills in BOTH w.grad and b.grad
w.data = w.data - step_size * w.grad.data
b.data = b.data - step_size * b.grad.data
w.grad.data.zero_()
b.grad.data.zero_()
```

The thing that clicked for me here: a single `loss.backward()` call computes a
separate gradient for *every* parameter that has `requires_grad=True`. Each one
then takes its own step downhill, with `w` rotating the line to the right slope
and `b` sliding it to the right height. By the end the model recovers both
numbers, `w ≈ -5` and `b ≈ 1`.

---

## The same idea with PyTorch's high-level tools (`linear_regression_pytorch.py`)

Everything above was deliberately manual. I declared `w` and `b` myself, wrote out
the MSE formula, and updated each parameter by hand. That's the right way to learn
what's happening, but it isn't how anyone actually writes PyTorch day to day.

So now that the mechanics are clear, this script rebuilds the same idea the normal
way: I let PyTorch hold the parameters, compute the loss, and run the update for
me. It's the same forward, loss, backward, step loop, just with a lot less code
because the library handles the bookkeeping. Seeing both side by side is really
the whole point. I know exactly what each library call is doing underneath,
because I already did it by hand.

A couple of things change from the notebook so this example stands on its own.
Instead of a synthetic noisy line, the data is three clean points that follow the
rule `y = 2x`. The model never sees that rule. It only sees the inputs and the
answers, and it has to recover the rule on its own:

```python
x_data = torch.tensor([[1.0], [2.0], [3.0]])   # inputs (the questions)
y_data = torch.tensor([[2.0], [4.0], [6.0]])   # correct outputs (the answer key)
```

Every manual step from the notebook now has a one-line equivalent:

| By hand (the notebook) | With the high-level tools (this script) |
|---|---|
| `w`, `b` as tensors with `requires_grad=True` | `nn.Linear(1, 1)` holds them for you |
| `w.data = w.data - lr * w.grad.data` | `optimizer.step()` |
| `w.grad.data.zero_()` | `optimizer.zero_grad()` |
| hand-written MSE | `nn.MSELoss()` |

The training loop is the same forward, loss, zero, backward, step sequence, run
500 times.

### What it prints

```
epoch 0, loss 24.913528442382812
epoch 50, loss 0.03636964410543442
epoch 100, loss 0.01763598434627056
...
epoch 450, loss 0.00011117422400275245
learned: y = 1.995x + 0.011
predict (after training) 4 7.991497993469238
```

Reading that output top to bottom tells the whole story:

* The loss collapses from about 25 to about 0.0001, so the model goes from wildly
  wrong to nearly perfect.
* `learned: y = 1.995x + 0.011` means it recovered the true rule `y = 2x` (slope
  near 2, bias near 0) without ever being told what the rule was.
* `predict ... 4 -> 7.99`: handed a brand new input it never trained on, it
  correctly predicts about 8. That's the proof it actually learned the pattern
  instead of just memorizing the three examples.
