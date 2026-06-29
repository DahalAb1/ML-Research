# Gradient Descent (SGD)

In the regression folder I trained with full-batch gradient descent: every single
step looked at *all* the data, averaged the gradient over every point, and then
moved. That works fine for 100 points. It falls apart when you have millions,
because you can't afford to sweep the whole dataset just to take one step.

So this folder is where I switched to **stochastic** gradient descent. The idea
that took me a second to trust: instead of the exact gradient over all the data, I
use a rough estimate from *one random point* (or a small handful). Each step is
noisier and sometimes flat-out wrong, but it's cheap, and over many steps the
noise mostly averages out and I still drift toward the answer. Noisy-but-cheap
beats exact-but-unaffordable once the data gets big.

Like everywhere in this repo, I did it by hand first — derived the gradient on
paper, wrote the update myself — and only then let PyTorch do it for me.

| Notebook | What I did | What I was trying to see |
|---|---|---|
| `SGD_SingleFeature.ipynb` | SGD from scratch, one feature | the bare mechanics, and how close I land vs sklearn |
| `SGD_MultiFeature.ipynb` | SGD from scratch, two features | the same idea as a plane, and how to *see* it |
| `SGD_pytorch.ipynb` | SGD with PyTorch on real data | how little code it takes once the library does the work |

---

## Starting simple: one feature by hand (`SGD_SingleFeature.ipynb`)

I started with the simplest thing that still counts as SGD: a straight line
`y = w·x + b`, one weight, one bias, fit to 100 noisy points from
`make_regression`. I deliberately set a wrong starting guess (`w=2, b=3`) and
plotted it against the data so I could watch a clearly-bad line and, hopefully,
watch it get better.

### Deriving the gradient myself

The whole point of doing this by hand is that I don't get to call `.backward()` —
I have to know which way is downhill. For one point the squared loss is

```
loss = (y_pred - y)²  =  (w·x + b - y)²
```

To take a step I need to know how the loss changes as I nudge `w`, and as I nudge
`b`. That's just the derivative of that expression with respect to each one
(chain rule — bring the square down, then differentiate the inside):

```
∂loss/∂w = 2 (w·x + b - y) · x      ← the ·x falls out because w is multiplied by x
∂loss/∂b = 2 (w·x + b - y) · 1      ← b just adds on, so its derivative is 1
```

Both share that `(w·x + b - y)` chunk. I lean on that below.

### The residual, and a distinction that tripped me up

That shared chunk, `(w·x + b - y)`, is the **residual**: how far my prediction sits
from the answer I actually have, for this one point. Once I have it, both
gradients are tiny:

```
r      = y_pred - y       # the residual
grad_w = 2 · r · x
grad_b = 2 · r
```

Here's the thing I had to slow down on, because I'd been using "error" and
"residual" like they meant the same thing. They don't:

- the **error** is my prediction minus the *true* underlying value — the clean,
  noise-free truth the data came from
- the **residual** is my prediction minus the *observed* value — the actual number
  sitting in my dataset

The catch is I deliberately added noise to this data, so the "true" line and the
points I can see aren't the same thing. I never get to touch the real error — the
truth is hidden behind the noise. The residual is all I can actually compute, and
it's what the gradient is built from. Calling it "error" in my code was hiding
that, so this is a residual everywhere now.

(My professor flagged the same kind of sloppiness in my naming — more on that
below.)

### The update, and the result

With the gradients in hand, each step just walks both parameters a little way
downhill — opposite the gradient, scaled by a learning rate `lr`:

```
w_new = w - lr · grad_w
b_new = b - lr · grad_b
```

I run that for a bunch of random points, stash the loss each step so I can plot
it, and at the end I check myself against scikit-learn's
`SGDRegressor(penalty=None)` — plain SGD, no regularization, so it's a fair
comparison to what I built. If my `w` and `b` land near sklearn's, I did it right.

### What my professor pushed back on (still in progress)

I sent this to my professor and got three good corrections. I'm working through
them:

1. **Naming.** I'd been passing the residual into a function parameter I'd named
   `Loss`, which is genuinely confusing because they're different things. The clean
   vocabulary:
   - **residual** = `y_pred - y`
   - **per-sample loss** = `(y_pred - y)²`
   - **average loss** (also called the *criterion*, or in the theory the *empirical
     risk*) = `(1/n) Σ (y_pred_i - y_i)²`

   My gradient math was already right — this is purely about calling things by
   their real names so the code stops lying about what it holds.

2. **Minibatch.** Right now each step uses exactly one random point, which is as
   noisy as SGD gets. The better version lets me pick a `batch_size`: grab that
   many random points and average their gradients. Batch of 1 is pure SGD, batch of
   *n* is full-batch gradient descent, and anything in between is minibatch — the
   knob that trades noise for cost.

3. **A learning rate that shrinks over time.** I noticed my loss curve spikes late
   in training instead of settling. The reason: a fixed `lr` keeps taking the same
   big step even when I'm right next to the minimum, so I overshoot and bounce. The
   fix is to let the step size decay as training goes:

   ```
   lr_t = C / sqrt(t)      # t = the iteration number, C is some constant > 0
   ```

   Big steps early when I'm far away, tiny steps late when I'm fine-tuning. There's
   real theory under this (it's the condition that guarantees SGD actually
   converges): the steps have to shrink, but *not too fast* — slow enough that they
   still add up to infinity and can carry me any distance, yet fast enough that the
   leftover noise dies out. `1/sqrt(t)` sits in that sweet spot.

---

## Same idea, one more feature (`SGD_MultiFeature.ipynb`)

Once the single-feature version made sense, I added a second input to make sure I
actually understood it and wasn't just pattern-matching. The model becomes a plane:

```
y = w1·x1 + w2·x2 + b
```

The derivation barely changes. Each weight gets its own gradient, and each one
picks up the input it's attached to:

```
grad_w1 = 2 · r · x1
grad_w2 = 2 · r · x2
grad_b  = 2 · r
```

where `r = w1·x1 + w2·x2 + b - y` is the same residual idea as before. That
symmetry — every weight's gradient is just `2·r·(its own input)` — is the part
that made it finally click that this scales to any number of features.

The fun part here was that with two inputs the fit is a *surface*, not a line, so I
could finally *see* learning happen:

- a **predicted-vs-actual** scatter, where points landing on the diagonal mean
  perfect predictions — drawn before training (a red mess) and after (green,
  hugging the diagonal)
- the **3D planes** themselves — my initial guess plane (red) and the learned plane
  (green) floating over the data cloud, so you watch the surface tilt into place

And again I checked the final numbers against `SGDRegressor` to make sure I wasn't
fooling myself.

---

## Letting PyTorch do it (`SGD_pytorch.ipynb`)

After grinding through the math by hand twice, I rebuilt the same thing the way
people actually write it — and on real data this time, the California Housing
dataset (8 features, predicting median house value). PyTorch holds the parameters,
autograd computes the gradients (no paper derivation), and `torch.optim.SGD` runs
the update:

```python
model = nn.Linear(8, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

Two things bit me here that the toy data never did:

- **You have to scale the features first.** Income runs 0–15, population runs in
  the thousands — wildly different ranges. Without standardizing everything to
  mean 0 / std 1 (`StandardScaler`, fit on the training data only), SGD just
  diverges. The big-range features completely dominate the gradient.
- **Stochastic really does mean one point at a time.** Each step pulls a single
  random house, which makes the loss curve genuinely jagged — and means it needs
  *a lot* of steps (3000 here) to learn from ~16k rows, since any one step sees
  almost nothing.

The honest takeaway from doing it both ways: every line of this short version maps
onto something I'd done by hand. `optimizer.step()` is my `w_new = w - lr·grad_w`.
`loss.backward()` is the gradient I derived on paper. `nn.Linear(8, 1)` is the
`w1, w2, …, b` I used to declare myself. None of it is magic anymore, which was the
entire point.
