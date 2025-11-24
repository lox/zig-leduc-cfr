# zig-leduc-cfr

A tiny, self-contained poker AI in Zig — built for learning.

This project solves **Leduc Hold'em** using **Counterfactual Regret Minimization (CFR)**, a foundational algorithm for computing Nash equilibria in imperfect-information games. The codebase is intentionally small (~500 lines) and heavily commented to serve as a teaching resource.

---

## Quick Start

Requires Zig 0.15+ (available via `./bin/zig`).

```bash
./bin/zig build run -Doptimize=ReleaseFast
```

Example output:

```
Leduc CFR trainer (toy, educational) - Vanilla
Sweeping iteration counts: { 100, 300, 1000, 3000, 10000, 30000, 100000 }

Base (iters 100000) avg game value: -0.08597

Iter       AvgGame    AasP0     BaseAsP0  Exploit
      100  -0.08961   -0.10125   -0.07034  0.06090
      300  -0.08667   -0.09184   -0.08034  0.02915
     1000  -0.08799   -0.08901   -0.08400  0.01392
     3000  -0.08740   -0.08734   -0.08525  0.00832
    10000  -0.08683   -0.08655   -0.08567  0.00477
    30000  -0.08598   -0.08606   -0.08578  0.00251
   100000  -0.08597   -0.08583   -0.08583  0.00142
```

The 100K-iteration strategy has exploitability of 0.00142 — a perfect opponent gains less than 0.15 cents per hand. That's essentially Nash equilibrium.

---

## What You'll Learn

This codebase teaches the core concepts of game-solving AI:

1. **Information Sets** — How to represent "what a player knows" when they can't see opponent's cards
2. **Regret Matching** — How to convert regrets into a strategy that improves over time  
3. **Counterfactual Value** — Why we weight updates by opponent reach probability
4. **Average Strategy** — Why the time-averaged strategy converges to Nash, not the current strategy
5. **Best Response** — How to compute exploitability using policy iteration

---

## Code Structure

```
src/leduc/
  game.zig         # Game rules, state transitions, payoffs
  cfr.zig          # Information sets and regret matching
  cfr_vanilla.zig  # Full-tree CFR (traverse everything)
  cfr_mccfr.zig    # Monte Carlo CFR (sample opponent actions)
  play.zig         # Evaluation: head-to-head and exploitability
main.zig           # CLI entry point
```

### Reading Order

If you're learning CFR, read the files in this order:

1. **game.zig** — Understand Leduc Hold'em: cards, betting rounds, how hands are scored. This is pure game logic with no CFR concepts.

2. **cfr.zig** — Learn about information sets and regret matching. An info set groups game states that look identical to a player. Regret matching converts cumulative regrets into action probabilities.

3. **cfr_vanilla.zig** — The core CFR algorithm. We traverse the entire game tree, computing counterfactual values and updating regrets. Pay attention to *why* we weight by opponent reach (regret) vs own reach (strategy averaging).

4. **cfr_mccfr.zig** — Monte Carlo CFR. Same algorithm, but we sample opponent actions instead of enumerating them. Compare this to vanilla CFR to see exactly what changes.

5. **play.zig** — How to evaluate strategies. Head-to-head is straightforward; best-response exploitability uses policy iteration because the best responder can't peek at opponent's cards.

---

## How CFR Works

CFR finds Nash equilibria by tracking **regret** — how much better each action would have been compared to what we actually played.

```
regret[action] += (value_if_always_played_action) - (value_of_current_strategy)
```

Then we convert regrets to a strategy via **regret matching**:
- Positive regret → play this action more
- Negative regret → avoid this action
- All non-positive → play uniformly

The **average strategy** (not the current strategy!) converges to Nash equilibrium. That's why we track `cumulative_strategy` weighted by reach probability.

### Why "Counterfactual"?

The regret update asks: "If I had *always* played action A at this decision point, how much better would I have done?" This is counterfactual because we're imagining a different history.

We weight by **opponent × chance reach** (called "counterfactual reach") because that's how often this decision point matters. Our own probability of reaching here doesn't affect the counterfactual — we're asking what happens if we deviate.

In this code, we loop over all possible deals, so chance probability is uniform and factors out. That's why you only see `reach_prob[opponent]` in the regret update.

---

## Leduc Hold'em Rules

Leduc is a toy poker game, small enough to solve exactly:

| Property | Value |
|----------|-------|
| Deck | 6 cards: J, Q, K × 2 suits |
| Players | 2 |
| Ante | 1 chip each |
| Rounds | Preflop → Flop (after board revealed) |
| Bet sizes | 2 chips (preflop), 4 chips (flop) |
| Max bets/round | 2 (bet + raise) |
| Hand ranking | Pair with board > high card |

The game has ~936 game states and ~288 information sets.

---

## Vanilla CFR vs MCCFR

Run MCCFR with:

```bash
./bin/zig build run -Doptimize=ReleaseFast -- --algo=mccfr
```

**Vanilla CFR**: Traverses the entire game tree every iteration. Low variance, high cost per iteration. Ideal for small games.

**MCCFR**: Samples opponent actions from their current strategy. Higher variance, but much faster per iteration. Scales to larger games.

For Leduc, vanilla CFR converges faster in wall-clock time. For Texas Hold'em, you'd need MCCFR (or better variants like CFR+, DCFR).

---

## Output Columns

| Column | Meaning |
|--------|---------|
| Iter | CFR iterations for this strategy |
| AvgGame | Self-play expected value for P0 (converges to ~-0.086) |
| AasP0 | This strategy's EV as P0 vs the 100K base |
| BaseAsP0 | Base strategy's EV as P0 vs this strategy |
| Exploit | Best-response exploitability (0 = Nash) |

---

## Papers

- [An Introduction to Counterfactual Regret Minimization](https://modelai.gettysburg.edu/2013/cfr/cfr.pdf) — Neller & Lanctot tutorial (start here!)
- [Regret Minimization in Games with Incomplete Information](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf) — Zinkevich et al., NIPS 2007 (original CFR)
- [Monte Carlo Sampling for Regret Minimization](https://mlanctot.info/files/papers/nips09mccfr_techreport.pdf) — Lanctot et al., NIPS 2009 (MCCFR)
- [Accelerating Best Response Calculation](http://www.cs.cmu.edu/~kwaugh/publications/johanson11.pdf) — Johanson et al., IJCAI 2011
