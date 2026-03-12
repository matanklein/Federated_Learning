# Specification 
Run following experiments 10-fold with different seeds but with fixed datasets!
Show normalized heatmaps with the average values.
P1's Heatmap
P2's Heatmap
P11's Heatmap
P12's Heatmap
P21's Heatmap
P22's Heatmap 
Local Training
Train on [P1 Train], test on [P1 Test]
Train on [P2 Train], test on [P2 Test]
Federated Training
Train on [P1Train, P2 Train], test on [P1 Test] and [P2 Test]
Train on [P1Train, P2 Train], test on [P1 Test] and [P2 Test]
Simulation Local
Train on [P11 Train], test on [P11 Test]
Train on [P12 Train], test on [P12 Test]
Train on [P21 Train], test on [P21 Test]
Train on [P22 Train], test on [P22 Test]
Simulation Federated
Train on [P11Train, P12 Train], test on [P11 Test] and [P12 Test]
Train on [P21Train, P22 Train], test on [P21 Test] and [P22 Test]
Federated Parameters
With Privacy-Preserving methods & Parameters
Suppression
Hiding samples: [100%, 90%, 80%, 70%, 60%, 50%, 40%, 30%, 20%, 10%]
Differential Privacy
Adding noise: [0.00, 0.25, 0.50, 0.75, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]

The Core Problem: You Only Get One Shot
Imagine you (Client P1) and a stranger (Client P2) want to train a model together. You know that working together usually helps, but you also need to add Privacy Noise (Differential Privacy) or hide some data (Suppression) to feel safe.
Here is the catch:
Privacy/Accuracy Trade-off: If you add too much noise, the "Joint Model" becomes stupid—so stupid that it’s actually worse than if you had just stayed home and trained alone.
The "One-Shot" Constraint: In real life, you cannot call P2 and say, "Hey, let's train 100 times with different noise levels to see what works best." That would take forever and violate privacy repeatedly. You only get to run the Real Federated Training once.
The Question: How do you choose the perfect Privacy Parameters (Noise & Suppression) before you ever talk to P2?
The Solution: The "Dress Rehearsal" (Simulation)
Since P1 cannot test with P2, P1 decides to play a game of "make-believe" using only their own data.
Step 1: The Baseline (The "Local" Score)
First, P1 trains a model alone on all their data.
Result: Let's say P1 gets 80% accuracy.
Significance: This is the "Floor." If any collaborative model gets less than 80%, it is a failure. Why give up privacy to get a worse model?
Step 2: The Simulation (The "Split Personality")
P1 takes their own data and splits it into two smaller fake clients: P11 and P12.
P1 pretends these are two different people.
P1 runs the Federated Training on P11+P12 using the grid of parameters (changing noise and suppression) for both P11 and P12.
The Goal: Both P11 and P12 generates a Heatmap of this simulation based on their test sets.
Step 3: The Selection (Picking the Winner)
P1 looks both at P11's and P12's "Simulation Heatmap" and asks:
"At what noise level does my 'Fake Federation' (P11 + P12) beat my 'Local Score' (P11 or P12) by the biggest margin?"
 
THE SAME IS DONE BY P2, AS SHE ALSO WANT TO DETERMINE TO HERSELF WHAT IS THE BEST PARAMETER TO TRAIN TOGETHER.
 
The Research Comparison: Did the Rehearsal Work?
As researchers, we (you and I) have a "God's Eye View." We can see everything. We are running this experiment to check if P1's (and P2's) strategy actually works. 
We are comparing two heatmaps:
1. The "Predicted" Gain (Simulation)
Formula: Accuracy(P11 + P12) minus Accuracy(P11 Local)
This is what P11 thinks will happen.
Formula: Accuracy(P11 + P12) minus Accuracy(P12 Local)
This is what P12 thinks will happen.
P1 sees both!
SAME FOR P2!
 
2. The "Real" Gain (Ground Truth)
Formula: Accuracy(P1 + P2) minus Accuracy(P1 Local)
This is what actually happens when P1 and P2 train together according to P1.
Formula: Accuracy(P1 + P2) minus Accuracy(P2 Local)
This is what actually happens when P1 and P2 train together according to P2.
The Research Conclusion:
If the "Predicted Gain" heatmap looks very similar to the "Real Gain" heatmap, then P1's strategy is a success! It proves that clients can simulate federated learning locally to tune their parameters safely.
 
SO I NEED 12 HEATMAPS: 
ALL DATA IS SPLIT INTO TWO AND JOINT TRAINING COMMENCES
CREATES TWO HEATMAP AS TWO SIDE HAS TWO SEPARATE VIEW
HALF DATA IS SPLIT INTO TWO AND JOINT TRAINING COMMENCES
CREATES TWO HEATMAP AS TWO SIDE HAS TWO SEPARATE VIEW
OTHER HALF IS SPLIT INTO TWO AND JOINT TRAINING COMMENCES
CREATES TWO HEATMAP AS TWO SIDE HAS TWO SEPARATE VIEW
 
DOUBLE ALL ABOVE AS SUPPRESSION AND DIFFERENTIAL PRIVACY ARE BOTH TESTED



core question is can a client predict whether it's worth collaborating before actually collaborating?

To answer this, you run three parallel experiments:
- Real federation: P1 and P2 share their actual data and train together
- P1 self-division simulation: P1 splits its own data into P11 and P12, simulates federation internally
- P2 self-division simulation: same for P2

If the self-division heatmaps look similar to the real federation heatmap, it means a client can safely predict collaboration outcomes locally — without exposing any meaningful data to the other party.

test this under two privacy mechanisms:
- Differential Privacy (DP): adds calibrated Gaussian noise to gradients
- Suppression: hides a fraction of training samples before sharing


### Data prep — `prepare_fixed_splits`

```python
split_with_overlap(1, ecfp_tr, ic50_tr, root_dir=paths["full"], overlap=2808)
```

Splits the full ChEMBL dataset into P1 and P2 with full task overlap (`overlap=2808` = all 2808 tasks shared). 

The splits are created once and reused across all seeds.



### Model arch — `make_base_conf` + `ConfigWrapper`

The model is a trunk-and-head architecture (from SparseChem):
- Trunk: shared sparse input layer (`SparseLinear`, 32000 → 40 features) — shared component updated by all clients
- Head: dense MLP (40 → 2808 outputs) — local per client

```
Input (ECFP fingerprint, 32000 bits)
       ↓
   [Trunk]  ← shared, DP applied here
       ↓
   [Head]   ← local per client
       ↓
  Output (2808 bioactivity predictions)
```

`ConfigWrapper` is a thin wrapper around `sc.ModelConfig` that protects DP parameters from being silently reset:

```python
class ConfigWrapper:
    def __setattr__(self, name, value):
        if name in ['_base', 'dp_noise_std', 'dp_clip_norm', 'dp_scope']:
            object.__setattr__(self, name, value)  # protect these
        else:
            setattr(self._base, name, value)        # pass rest to ModelConfig
```

This is necessary because `ModelConfig.__setattr__` resets custom attributes whenever `output_size` or `batch_size` is updated inside `run_server`.



### DP clip auto-est — `estimate_dp_clip`

```python
clip = float(np.percentile(norms, 75))
```

the script runs 20 forward+backward passes on real batches and measures the actual L2 norm of trunk gradients. The 75th percentile is chosen as the clip threshold — this means roughly 75% of batches will have their gradients clipped, which is the standard DP-SGD sweet spot (strong enough to bound sensitivity, not so aggressive that it destroys the gradient signal).

This adapts automatically to the dataset size and model scale.



### DP mechanism — `apply_dp_to_gradients` (in `CollaborativeLearning.py`)

Implements Algorithm 1 from Abadi et al. (2016):

```
1. Compute gradient g on batch
2. Clip: g ← g · min(1, C / ||g||)      where C = dp_clip_norm
3. Add noise: g ← g + N(0, σ²C²I)       where σ = dp_noise_std
4. Update weights with noisy clipped gradient
```

Applied only to trunk gradients — the shared component. The head is local and never shared, so it doesn't need DP protection.

Each client gets its own noise level via per-client `ConfigWrapper`:
```python
c0 = make_wrapped_conf(base_conf, noise_c0, dp_clip)  # client 0's noise
c1 = make_wrapped_conf(base_conf, noise_c1, dp_clip)  # client 1's noise
```




### Suppression mechanism — `hide_h_percent` (in `CollaborativeLearning.py`)

```python
remaining = int((1 - h) * n_samples)
indices = random.sample(range(n_samples), remaining)
return X[indices], Y[indices]
```

Client hides `h%` of its own training rows before contributing to the shared trunk. At `h=0.9` a client only contributes 10% of its data. This simulates a client that wants to limit its data exposure without adding noise.



### Metric — `norm_improvement`

```python
(local - federated) / abs(oracle - local)
```

- `oracle` = loss before any training (random initialization)
- `local` = loss after training alone
- `federated` = loss after training with the other client

- `+1.0` = federation gave exactly as much improvement as training alone
- `0.0` = federation gave no improvement beyond local
- `-1.0` = federation was as harmful as it was possible to be



### Experiment grid — `run_scenario_dp` / `run_scenario_sup`

For each scenario and seed:

```
1. Train client 0 alone  → oracle0, local0
2. Train client 1 alone  → oracle1, local1
3. For every (noise_c0, noise_c1) combination:
      Train both clients together → federated0, federated1
      grid_c0[i,j] = norm_improvement(oracle0, local0, federated0)
      grid_c1[i,j] = norm_improvement(oracle1, local1, federated1)
```

The baselines are computed once and reused for all grid cells — this is important because it means all cells are comparable on the same scale.