# ML Research

Hands-on implementations of machine learning algorithms, built from scratch to
understand what's actually happening under the hood rather than leaning on
high-level APIs. Work conducted under the guidance of my professor.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Contents

| Notebook | Topic | Key concepts |
|---|---|---|
| `linear_regression.ipynb` | Single-parameter linear regression | autograd, MSE, manual gradient descent |

---

## Linear Regression (`linear_regression.ipynb`)

This notebook builds linear regression in PyTorch the slow way — no `nn.Module`,
no built-in optimizer — so that every piece of the training loop is out in the
open. The goal isn't to fit a line; it's to *see* how a model learns one.

### The setup: make data we already know the answer to

Before training anything, we need data. We create it ourselves so we know the
"true" answer the model is supposed to discover.

```python
X = torch.arange(-5, 5, 0.1).view(-1, 1)   # inputs from -5 to 5
func = -5 * X                              # the true line: y = -5x
```

`func` is a perfect straight line with slope **-5**. But real-world data is
never that clean, so we add a bit of Gaussian noise to scatter the points
around the line:

```python
Y = func + 0.4 * torch.randn(X.size())     # noisy version of the true line
```

Now we have a cloud of points that *roughly* follow `y = -5x`. The whole point
of the exercise: can the model recover that slope of -5 just by looking at the
noisy data?

### The model

Our model is as simple as it gets — a single weight `w`, and no bias term:

```python
w = torch.tensor(-10.0, requires_grad=True)   # deliberately wrong starting guess

def forward(x):
    return w * x
```

We start `w` at -10, far from the true -5, so we can watch it travel to the
right answer. `requires_grad=True` tells PyTorch to track operations on `w` so
it can compute gradients later.

### Measuring how wrong we are

To improve, the model first needs to know how badly it's doing. We use **Mean
Squared Error** — take the difference between prediction and truth, square it
(so positives and negatives don't cancel out), and average over all points:

```python
def criterion(y_pred, y):
    return torch.mean((y_pred - y) ** 2)
```

### Learning by gradient descent

This is the heart of the notebook. Each iteration does four things:

```python
Y_pred = forward(X)              # 1. predict
loss = criterion(Y_pred, Y)      # 2. measure error
loss.backward()                  # 3. compute gradient of loss w.r.t. w
w.data = w.data - step_size * w.grad.data   # 4. step toward lower loss
w.grad.data.zero_()              # reset gradient for the next round
```

**Why does step 4 look like that?** Here's the intuition:

> The gradient (`w.grad`) tells you the direction of steepest *ascent* — the way
> that makes the loss go *up* the fastest from where you're currently standing.
> If you know the direction that climbs the hill, then the exact opposite
> direction goes *down* it. So we negate the gradient to head toward the minimum.
>
> Moving anywhere needs two things: a **direction** and a **distance**.
> - `w.grad.data` → the direction (we flip its sign to descend)
> - `step_size` (the learning rate) → how far we step each time
>
> ```
> w_new = w_old - learning_rate * w.grad.data
> ```

Two small but important details:

- **`w.data` / `w.grad.data`** — we touch `.data` so this manual update isn't
  itself recorded by autograd (we don't want to compute gradients *of the
  update*).
- **`w.grad.data.zero_()`** — PyTorch *accumulates* gradients by default, so if
  we don't reset to zero each iteration, the next gradient piles on top of the
  old one.

### The result

Over 20 iterations, `w` climbs from its wrong starting point to almost exactly
the true slope:

```
iter   loss        w
0      0.1758     -4.998
...
19     0.1758     -4.999
```

`w` lands at **≈ -4.999** — it recovered the -5 we hid in the data. Notice the
loss settles at about **0.176** and stops dropping. That floor isn't a failure:
it's the noise we added ourselves. The model fits the underlying line as well as
it possibly can, and what's left over is the randomness it was never meant to
predict.

Plotting the loss per iteration shows it falling fast and then flattening — the
visual signature of a model that has converged.

---

## Training the Model for Two Parameters

The model so far is `y = w·x`. Because it has no constant term, that line is
*forced through the origin* — when `x = 0` it has to output `y = 0`. Most real
lines don't pass through the origin, so we add a second learnable parameter, the
**bias** `b`:

```
y = w·x + b
```

`w` controls the **slope** and `b` shifts the whole line **up or down**. To show
the bias is actually doing something, the true line this time has an intercept of
its own: `y = -5x + 1`.

The training loop barely changes. We now keep two parameters, give each its own
gradient step, and zero both gradients each round:

```python
w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-20.0, requires_grad=True)

def forward(x):
    return w * x + b              # the only change: a "+ b" term

loss.backward()                   # fills BOTH w.grad and b.grad
w.data = w.data - step_size * w.grad.data
b.data = b.data - step_size * b.grad.data
w.grad.data.zero_()
b.grad.data.zero_()
```

The key idea: one `loss.backward()` call computes a separate gradient for *every*
parameter that has `requires_grad=True`. Each parameter then takes its own step
in the direction that lowers the loss — `w` rotates the line to the right slope,
`b` slides it to the right height. By the end, the model recovers both numbers:
`w ≈ -5` and `b ≈ 1`.
