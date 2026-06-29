# Regression

This is where I started. The goal of these notebooks isn't really to fit a line —
it's to watch a model *learn* one, with every moving part out in the open. So I
built linear regression from nothing first (no `nn.Module`, no optimizer, gradients
I derived myself), then rebuilt the exact same thing with PyTorch's tools, and
finally scaled it up to real data with many features at once.

| Notebook | What I did | What I was trying to see |
|---|---|---|
| `linear_regression_manual.ipynb` | Linear regression entirely by hand | the actual mechanics of gradient descent, one piece at a time |
| `linear_regression_pytorch.ipynb` | The same idea with PyTorch's tools | how each line of library code maps to something I did by hand |
| `multivariate_regression.ipynb` | Regression on real, many-feature data | what changes when the inputs are messy and there's more than one |

---

## Doing it all by hand (`linear_regression_manual.ipynb`)

I wanted to see every step, so I refused to use any shortcuts here. The first thing
I needed was data — and I made it myself, so I'd already know the answer the model
was supposed to find:

```python
X = torch.arange(-5, 5, 0.1).view(-1, 1)   # inputs from -5 to 5
func = -5 * X                              # the true line: y = -5x
```

A perfectly straight line with slope **-5**. Real data is never that clean, so I
scattered the points around it with a bit of noise:

```python
Y = func + 0.4 * torch.randn(X.size())     # noisy version of the true line
```

Now I had a cloud of points that *roughly* follow `y = -5x`, and a clear question:
can the model recover that -5 just from the noisy points?

### The model and how I measure it being wrong

I made it as simple as possible — one weight, no bias yet — and started it at a
deliberately wrong `-10` so I could watch it travel to the right answer:

```python
w = torch.tensor(-10.0, requires_grad=True)   # deliberately wrong starting guess

def forward(x):
    return w * x
```

The `requires_grad=True` is me telling PyTorch to keep track of `w` so it can hand
me the gradient later. To know *how* wrong I am, I use Mean Squared Error — the gap
between prediction and truth, squared so signs don't cancel, averaged over all
points:

```python
def criterion(y_pred, y):
    return torch.mean((y_pred - y) ** 2)
```

### The part worth slowing down on

The actual learning is four lines, repeated:

```python
Y_pred = forward(X)              # 1. predict
loss = criterion(Y_pred, Y)      # 2. measure how wrong we are
loss.backward()                  # 3. compute the gradient of the loss w.r.t. w
w.data = w.data - step_size * w.grad.data   # 4. step toward a lower loss
w.grad.data.zero_()              # reset the gradient for next round
```

Step 4 is the one that took me a minute to really *get*. The way I hold it in my
head now:

> The gradient points in the direction of steepest *ascent* — the way that makes
> the loss climb fastest from where I'm standing. So the exact opposite direction
> goes downhill. I flip its sign to head toward the minimum. Moving anywhere needs
> a direction *and* a distance: the gradient gives the direction, and `step_size`
> (the learning rate) decides how far I step.
>
> ```
> w_new = w_old - learning_rate * w.grad.data
> ```

Two small things that bit me here:

- **why `.data`** — I update `w.data`, not `w`, so this manual step isn't itself
  recorded by autograd. I don't want it computing gradients *of my update*.
- **why zero the gradient** — PyTorch *adds up* gradients by default. If I don't
  reset to zero each round, the next gradient just piles on top of the last one.

### What came out

Over 20 iterations `w` climbed from -10 to about **-4.999** — it found the -5 I'd
hidden. The loss dropped fast, then flattened around **0.176** and stopped. That
floor isn't failure: it's exactly the noise I added by hand. The model fit the real
line as well as anything could, and what's left over is randomness it was never
going to predict.

### Then I added the bias

`y = w·x` is stuck through the origin — at `x = 0` it's forced to output `0`. Most
real lines aren't, so I added a second learnable parameter, the bias `b`, and made
the true line `y = -5x + 1` so the bias actually had to do something:

```python
def forward(x):
    return w * x + b              # the only real change is the "+ b"

loss.backward()                   # fills in BOTH w.grad and b.grad
w.data = w.data - step_size * w.grad.data
b.data = b.data - step_size * b.grad.data
w.grad.data.zero_()
b.grad.data.zero_()
```

The thing that clicked: one `loss.backward()` call fills in a separate gradient for
*every* parameter with `requires_grad=True`. Each takes its own step — `w` rotating
the line to the right slope, `b` sliding it to the right height — and the model
recovers both, `w ≈ -5` and `b ≈ 1`.

---

## The same idea, the normal way (`linear_regression_pytorch.ipynb`)

Everything above was deliberately manual, and that's the right way to *learn* it —
but it's not how anyone actually writes PyTorch. So once the mechanics were clear, I
rebuilt the same thing letting the library hold the parameters, compute the loss,
and run the update. Seeing both side by side is the whole point: I know exactly what
every library call does underneath, because I just did it by hand.

This time the data is three clean points following `y = 2x`. The model never sees
that rule — only the inputs and answers — and has to recover it:

```python
x_data = torch.tensor([[1.0], [2.0], [3.0]])   # the questions
y_data = torch.tensor([[2.0], [4.0], [6.0]])   # the answer key
```

Every manual step now collapses to one line:

| By hand (the manual notebook) | With the tools |
|---|---|
| `w`, `b` as tensors with `requires_grad=True` | `nn.Linear(1, 1)` holds them |
| `w.data = w.data - lr * w.grad.data` | `optimizer.step()` |
| `w.grad.data.zero_()` | `optimizer.zero_grad()` |
| hand-written MSE | `nn.MSELoss()` |

Run that loop 500 times and the output tells the whole story:

```
epoch 0, loss 24.91...
epoch 450, loss 0.0001...
learned: y = 1.995x + 0.011
predict x=4 → 7.99  (true answer: 8.0)
```

Loss collapses from ~25 to ~0.0001; it recovered `y = 2x` without being told the
rule; and handed a brand-new `x=4` it never trained on, it predicts ~8 — proof it
learned the pattern instead of memorizing three points.

---

## Scaling up to real data (`multivariate_regression.ipynb`)

The first two fit a line through one input. Real problems have many inputs at once,
so I moved to the **California Housing** dataset (from the 1990 census): 8 features
per block — median income, house age, average rooms, population, location — used to
predict the median house value.

The model is the same linear idea, just wider — one weight per feature, all held by
a single layer:

```python
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1)   # 8 features in, 1 price out
    def forward(self, x):
        return self.linear(x)
```

Two things mattered here that the toy data let me ignore:

- **Feature scaling.** The 8 features live on wildly different scales (income 0–15,
  population in the thousands). Left alone, the gradient updates go haywire and
  training breaks. So I standardize each feature to mean 0 / std 1 — computing those
  stats on the **training set only**, then applying them to the test set, so the
  test data stays genuinely unseen:

  ```python
  mean = X_train.mean(dim=0)
  std  = X_train.std(dim=0)
  X_train = (X_train - mean) / std
  X_test  = (X_test  - mean) / std   # train stats, not test stats
  ```

- **A held-out test set.** The model only ever learns from `X_train`. Afterward I
  check it on `X_test`, data it never saw. If the two losses are close, it actually
  learned the pattern; if test loss is much higher, it just memorized the training
  data (overfitting):

  ```python
  with torch.no_grad():
      test_pred = multivariate_regression(X_test)
      test_loss = criterion(test_pred, Y_test)
  ```

The training loop itself is the familiar forward → loss → `zero_grad` → `backward`
→ `step`, run 500 epochs with `nn.MSELoss` and `optim.SGD`. Same machinery as the
single-feature PyTorch notebook — the real jump was going from 1 weight to 8, and
learning that messy real-world data has to be prepped before any of it works.
