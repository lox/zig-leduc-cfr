# zig-leduc-cfr

A tiny, self-contained poker AI in Zig.

It plays **Leduc Hold'em**, a toy poker game with a 6-card deck (J,Q,K, two copies each), two betting rounds, fixed bets, and a single public card. We solve it with **Counterfactual Regret Minimization (CFR)**.

---

## Build & run

Requires Zig 0.15.1 (available via `./bin/zig`).

```bash
# Optimized build (recommended)
./bin/zig build run -Doptimize=ReleaseFast
```

Example output:

```text
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

**Columns:**
- **Iter**: CFR iterations for this strategy
- **AvgGame**: Self-play expected value for player 0 (converges to ~-0.086, the Nash value)
- **AasP0**: Strategy's EV as player 0 vs the 100K-iteration base
- **BaseAsP0**: Base strategy's EV as player 0 vs this strategy
- **Exploit**: Best-response exploitability (NashConv/2) - approaches 0 as strategy converges to Nash

The 100K-iteration strategy has exploitability of 0.00142, meaning a perfect best-responder gains less than 0.2 cents per hand - essentially Nash equilibrium.

---

## How it works

### CFR overview

CFR iteratively improves a strategy by tracking **regret** - how much better each action would have been compared to what was actually played. Actions with positive regret get played more often; actions with negative regret get avoided.

Over many iterations, the **average strategy** converges to a Nash equilibrium where neither player can improve by changing their strategy.

### Code structure

```
src/leduc/
  game.zig         # Game state, actions, terminal utilities
  cfr.zig          # Shared infoset helpers used by the trainer
  cfr_vanilla.zig  # CFR trainer with regret matching (full-tree)
  play.zig         # Head-to-head evaluation, best-response exploitability
main.zig           # CLI entry point
```

If you want to learn CFR, the intended reading order is:

1. **game.zig** - How a Leduc hand is represented (cards, pot, betting history)
2. **cfr_vanilla.zig** - The CFR algorithm: tree traversal, regret updates, strategy averaging
3. **play.zig** - Evaluation: head-to-head play and exploitability via best-response

### Exploitability

Exploitability measures how far a strategy is from Nash equilibrium. It's computed as:

```
exploitability = (BR_0_EV + BR_1_EV) / 2
```

Where `BR_i_EV` is the expected value of player i's **best response** against the opponent's strategy. Against a perfect Nash equilibrium, no best response can do better than the equilibrium value, so exploitability = 0.

The best-response calculation uses policy iteration to correctly handle information sets (the BR player can't peek at opponent's cards).

---

## Leduc Hold'em rules

- **Deck**: 6 cards (J, Q, K with 2 copies each)
- **Players**: 2
- **Ante**: 1 chip each
- **Rounds**: 2 betting rounds (preflop, then flop after board card is revealed)
- **Bet sizes**: Fixed-limit (2 chips preflop, 4 chips on flop)
- **Betting**: Check/bet or call/raise/fold; max 2 bets per round
- **Showdown**: Pair with board wins; otherwise high card; ties split

The game has ~936 game states and ~288 information sets, making it small enough to solve exactly.

---

## Papers

- [An Introduction to Counterfactual Regret Minimization](https://modelai.gettysburg.edu/2013/cfr/cfr.pdf) - Neller & Lanctot tutorial

- [Regret Minimization in Games with Incomplete Information](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf) - Zinkevich et al., NIPS 2007.

- [Monte Carlo Sampling for Regret Minimization in Extensive Games](https://mlanctot.info/files/papers/nips09mccfr_techreport.pdf) - Lanctot et al., NIPS 2009.

- [Efficient Nash Equilibrium Approximation through Monte Carlo CFR](http://johanson.ca/publications/poker/2012-aamas-pcs/2012-aamas-pcs.pdf) - Johanson et al., AAMAS

- [Accelerating Best Response Calculation in Large Extensive Games](http://www.cs.cmu.edu/~kwaugh/publications/johanson11.pdf) - Johanson et al., IJCAI 2011.
