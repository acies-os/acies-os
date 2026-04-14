# ICT Fixedlag5 Algorithm

Date: `2026-04-14`

## Purpose

This note describes:

- the old `fixedlag5` algorithm used in the `2026-04-13` continuity evaluation
- the new pooled-median `fixedlag5` algorithm
- a compact pseudocode view
- a step-by-step walkthrough of what happens when `1 s` of sensor data arrives

Related note:

- trace comparison against the old baseline: [`2026-04-14-ict-baseline-vs-pooled-median-traces.md`](/home/kara4/demo/acies-os/docs/design/ict_tracker/2026-04-14-ict-baseline-vs-pooled-median-traces.md)

## What "Trained" Means Here

This pipeline is not using a deep model or a heavy learned tracker.

When this note says "trained", it mostly means:

- compute simple summary statistics from labeled ICT data
- store those statistics as a template bank
- use those templates inside a hand-designed continuity decoder

Concretely, the learned pieces are small:

- old baseline: per-loop-node mean energy template
- new candidate: per-loop-node median energy template
- lower-layer template fallback inside `hybrid_mic`: simple station/side template matching from labeled examples
- optional transition preferences in some exploratory tracker baselines elsewhere in the repo

The main runtime algorithm itself is still hand-designed:

- fixed `1 s` energy windowing
- fixed loop lattice geometry
- fixed transition rules
- fixed lag budget of `5`
- fixed scoring weights and penalties chosen from offline evaluation

So this is better understood as:

- template estimation from labeled data
- plus a structured dynamic-programming decoder

not as:

- a complex end-to-end trained model

## High-Level Structure

The old and new algorithms share the same runtime structure:

1. convert raw sensor traces into per-second `mic` energy features
2. produce a coarse station/side prior called `hybrid_mic`
3. score all loop-lattice nodes using:
- an observation-template score
- an anchor bias from the coarse prior
4. run a continuity-constrained decoder on the loop lattice
5. emit a bounded-lag `fixedlag5` path estimate

The only change is the observation template:

- old algorithm: one pooled mean template per loop node
- new algorithm: one pooled coordinate-wise median template per loop node

Everything else stays the same:

- same `hybrid_mic` prior
- same loop lattice
- same transition penalties
- same fixed-lag decoding logic

## Geometry And State

The continuity layer uses a discrete `40`-node loop lattice:

- `8` loop sensors
- `5` points per sensor

Each loop node is a discrete candidate vehicle position.

The lattice order follows the ICT loop:

- `S1- -> S2- -> S3- -> S4- -> S4+ -> S3+ -> S2+ -> S1+`

The decoder does not try to estimate continuous XY directly. It chooses a loop node, then maps that node to XY.

## Old Fixedlag5

### Observation model

The "training" step here is very simple:

- align labeled samples to loop nodes
- aggregate the observed `mic` feature vectors at each loop node
- store one summary vector per node

For each loop node:

- collect all training rows whose ground-truth node is that loop node
- take the mean of the `mic` feature vector
- use that mean vector as the node template

At runtime:

- compare the current `mic` feature vector to every node template
- convert that into a per-node emission score

### Continuity model

The continuity layer adds:

- a bias toward the coarse `hybrid_mic` station/side anchor
- penalties for large node jumps
- direction-aware rewards and penalties
- a reset option with a strong penalty

Then it runs a fixed-lag smoothed decoder:

- sample `t` is usually finalized when sample `t+5` arrives

Important clarification:

- the decoder itself is not trained
- it is a hand-specified dynamic program with fixed penalties such as hop penalty, stay penalty, reset penalty, and direction bonuses
- those values were selected empirically from offline evaluation, not learned by gradient-based training

## New Pooled-Median Fixedlag5

The new version changes only the template-estimation step.

Again, the "training" step is still simple summary-statistics estimation, not a complex model fit.

For each loop node:

- collect all training rows whose ground-truth node is that loop node
- take the coordinate-wise median of the `mic` feature vector
- use that median vector as the node template

Everything else is unchanged:

- same energy features
- same `hybrid_mic` anchor prior
- same lattice
- same transition scoring
- same lag-5 commit rule

Why this helps:

- the median is less sensitive to noisy passes or outlier loops
- so the per-node emission score is slightly more robust on the harder runs

So the actual algorithm change is:

- old: replace each node's data cloud with its mean vector
- new: replace each node's data cloud with its median vector

That is the whole "training" difference.

## Compact Pseudocode

```text
inputs:
  raw sensor traces for all ICT mic nodes
  trained loop-node template bank
  loop lattice with 40 nodes
  fixed-lag budget L = 5

offline training:
  for each labeled sample:
    compute aligned 1 s mic-energy feature vector x_t
    map sample to ground-truth loop node g_t

  for each loop node k:
    collect X_k = {x_t : g_t == k}
    old baseline template[k] = mean(X_k)
    new pooled-median template[k] = coordwise_median(X_k)

  choose decoder weights / penalties from offline evaluation
  do not fit a complex learned model

runtime / evaluation at time t:
  1. compute current 1 s mic-energy feature vector x_t

  2. compute coarse hybrid prior:
     station_t, side_t, direction_t = hybrid_mic(x_t)

  3. for each loop node k:
     template_score[k] = -distance(x_t, template[k])
     anchor_score[k] = compatibility(k, station_t, side_t, station_margin_t)
     emission[k] = template_weight * zscore(template_score[k]) + anchor_score[k]

  4. update dynamic program:
     for each destination node dst:
       best_score[dst] = max(
         emission[dst] - reset_penalty,
         max over allowed src:
           prev_score[src]
           + emission[dst]
           + transition_score(src -> dst, direction_t)
       )

  5. fixed-lag commit:
     when sample t is processed, backtrack through the DP
     commit the state for sample t - 5

output:
  committed loop node
  mapped station / side / XY for that committed node
```

In other words, the only "model parameters" learned from data in the main continuity layer are the node templates themselves.

## Step-By-Step Walkthrough For One Second Of Data

Assume we just received roughly `1 s` of raw mic trace from each ICT node.

### 1. Window the raw traces

For each sensor node:

- take the current `1 s` window
- compute mean-square energy
- take log-energy

This produces one scalar energy per sensor for this second.

So the current observation becomes a feature vector like:

```text
x_t = [
  rs1__mic,
  rs2__mic,
  rs3__mic,
  ...
  rs10__mic
]
```

### 2. Build the coarse `hybrid_mic` prior

The lower layer inspects the same energy vector and estimates:

- likely station
- likely side
- likely direction
- station margin or confidence

This does not produce the final path output. It only says something like:

```text
coarse prior at time t:
  pred_station = 3
  pred_side = positive_cross
  pred_direction = toward_S1
  pred_station_margin = 0.58
```

That coarse prior is also not a complex learned model.

It is built from simple energy-based heuristics and a lightweight template fallback:

- pair-aware station estimate from current energies
- simple template lookup when the station margin is weak
- direction from the smoothed station-centroid trend

### 3. Score every loop node by template match

Now the continuity layer compares `x_t` to every loop-node template.

For the old algorithm:

```text
template_score[k] = -mean_square_error(x_t, mean_template[k])
```

For the new pooled-median algorithm:

```text
template_score[k] = -mean_square_error(x_t, median_template[k])
```

This gives one raw score per loop node.

### 4. Add anchor bias from the coarse prior

The algorithm then checks how compatible each loop node is with the coarse station/side estimate.

Nodes near the predicted station/side get a boost.
Nodes farther away get penalized.

So for each loop node:

```text
emission[k] =
  template_weight * normalized_template_score[k]
  + anchor_bias[k]
```

At this point, the algorithm has a per-node observation score for the current second.

### 5. Combine with continuity from earlier seconds

The decoder does not choose the best node from this second alone.
It combines the current emission with the previously accumulated path scores.

For each candidate destination node:

- consider staying nearby
- consider moving a small number of nodes
- penalize large jumps
- reward moves consistent with the inferred direction
- allow a reset, but only with a strong penalty

So the new score for node `dst` is the best of:

- reset into `dst`
- continue from a nearby previous node into `dst`

### 6. Backtrack the lag-5 state

After processing the current second, the decoder has better information about several earlier seconds.

It then:

- backtracks the best path through the DP table
- finalizes the node for sample `t-5`

That is why this is called `fixedlag5`:

- current evidence is used to revise the recent past
- but only up to a lag of about `5` seconds

### 7. Map the committed node to tracking output

Once the node for sample `t-5` is committed, the algorithm maps that node to:

- loop node index
- station
- side
- latitude / longitude
- local XY

That becomes the tracking output for that finalized sample.

## Old vs New In One Line

Old:

```text
current energy vector -> compare against mean template bank -> continuity decoder -> lag-5 output
```

New:

```text
current energy vector -> compare against median template bank -> same continuity decoder -> same lag-5 output
```

So the recommendation to switch to pooled median is a template-estimation change, not a decoder redesign.

It is also not a move from "simple" to "complex".

It is:

- simple summary statistic -> different simple summary statistic

inside the same structured decoder.
