// cfr_mccfr.zig — Monte Carlo CFR (External Sampling)
// ====================================================
//
// Monte Carlo CFR trades exactness for speed: instead of traversing the entire
// game tree, we *sample* trajectories through it.
//
// This implementation uses the "external sampling" variant of MCCFR (Lanctot 2009):
//   - We designate one player as the "updating player" each traversal
//   - The updating player enumerates all their actions (like vanilla CFR)
//   - The opponent's actions are *sampled* from their current strategy
//
// AVERAGE STRATEGY:
//   We use textbook reach-weighted averaging. Each time we visit an infoset I
//   for the updating player, we add:
//     cumulative_strategy[a] += reach_prob[player] * strategy[a]
//
//   The average strategy is then: cumulative_strategy[a] / sum(cumulative_strategy)
//
// Why does this work? The regret updates are *unbiased estimators* of the true
// CFR updates. Over many iterations, they average out to the same thing—just
// with more variance per iteration but much less computation.
//
// The key insight: we don't need to visit every state to learn a good strategy.
// We just need to visit states *proportionally* to how likely they are, and
// MCCFR naturally does this by sampling opponent actions.
//
// Compared to vanilla CFR (cfr_vanilla.zig):
//   - Same terminal/chance handling
//   - Same regret update formula at updating player's nodes
//   - NEW: Opponent nodes sample one action instead of enumerating all
//   - NEW: We alternate which player is "updating" each iteration
//
// CONVERGENCE NOTE:
//   External Sampling MCCFR converges at O(1/sqrt(T)), slower than vanilla CFR's
//   O(1/T) but with much less work per iteration. For Leduc, vanilla CFR is
//   typically more efficient due to the small game tree.

const std = @import("std");
const game = @import("game.zig");
const cfr = @import("cfr.zig");

const Payoffs = [game.NUM_PLAYERS]f64;

pub const MCCFRTrainer = struct {
    allocator: std.mem.Allocator,
    infosets: cfr.InfoMap,
    rng: std.Random.DefaultPrng,
    update_player: u8 = 0,

    pub fn init(allocator: std.mem.Allocator) MCCFRTrainer {
        var seed: u64 = 0;
        std.crypto.random.bytes(std.mem.asBytes(&seed));
        return .{
            .allocator = allocator,
            .infosets = cfr.InfoMap.init(allocator),
            .rng = std.Random.DefaultPrng.init(seed),
        };
    }

    pub fn deinit(self: *MCCFRTrainer) void {
        self.infosets.deinit();
    }

    /// Sample an action index from a probability distribution.
    fn sampleAction(rng: *std.Random.DefaultPrng, probs: []const f64) usize {
        const r = rng.random().float(f64);
        var cumulative: f64 = 0;
        for (probs, 0..) |p, i| {
            cumulative += p;
            if (r <= cumulative) return i;
        }
        return probs.len - 1;
    }

    /// The MCCFR traversal. Very similar to vanilla CFR, but opponent nodes
    /// sample a single action instead of enumerating all actions.
    ///
    /// `iter`: The current iteration number (1-based), used for gap weighting.
    fn traverse(self: *MCCFRTrainer, state: game.GameState, reach_prob: Payoffs) !Payoffs {

        // ===== TERMINAL NODES =====
        if (state.folded_player) |folder| {
            const payoff_p0 = state.foldPayoff(folder);
            return .{ payoff_p0, -payoff_p0 };
        }
        if (state.round == .showdown) {
            const payoff_p0 = state.showdownPayoff();
            return .{ payoff_p0, -payoff_p0 };
        }

        // ===== CHANCE NODE =====
        if (state.round == .flop and !state.board_revealed) {
            var next = state;
            next.revealBoard();
            return self.traverse(next, reach_prob);
        }

        // ===== DECISION NODE =====
        const player: usize = @intCast(state.current_player);

        var actions: [game.MAX_ACTIONS]game.Action = undefined;
        const num_actions = state.legalActions(&actions);

        const key = state.infoKey(state.current_player);
        const infoset = try cfr.getOrCreateInfoSet(&self.infosets, key, actions[0..num_actions]);
        const strategy_arr = infoset.currentStrategy();

        var strategy: [game.MAX_ACTIONS]f64 = undefined;
        for (0..num_actions) |i| strategy[i] = strategy_arr[i];

        if (player == self.update_player) {
            // ===== UPDATING PLAYER: enumerate all actions (like vanilla CFR) =====
            var action_payoffs: [game.MAX_ACTIONS]Payoffs = undefined;
            var expected_payoff: Payoffs = .{ 0, 0 };

            for (actions[0..num_actions], 0..) |action, i| {
                const child = state.apply(action);

                var child_reach = reach_prob;
                child_reach[player] *= strategy[i];

                const child_payoff = try self.traverse(child, child_reach);
                action_payoffs[i] = child_payoff;

                expected_payoff[0] += strategy[i] * child_payoff[0];
                expected_payoff[1] += strategy[i] * child_payoff[1];
            }

            // Update regrets - no weighting needed in external sampling since
            // opponent reach is implicitly 1.0 (we don't update it at sampled nodes)
            const infoset_mut = self.infosets.getPtr(key).?;
            for (0..num_actions) |i| {
                const regret = action_payoffs[i][player] - expected_payoff[player];
                infoset_mut.cumulative_regret[i] += regret;
            }

            // TEXTBOOK MCCFR AVERAGING (uniform weights)
            //
            // Each time we visit an infoset I for player `player` on an iteration,
            // we add reach_prob[player] * strategy[i] to cumulative_strategy.
            for (0..num_actions) |i| {
                infoset_mut.cumulative_strategy[i] += reach_prob[player] * strategy[i];
            }

            return expected_payoff;
        } else {
            // ===== OPPONENT: sample ONE action from their strategy =====
            // Sample opponent actions proportionally to their strategy.
            const action_idx = sampleAction(&self.rng, strategy[0..num_actions]);
            const action = actions[action_idx];
            const child = state.apply(action);

            // NOTE: We do NOT update average strategy here.
            // The average strategy is updated only by the updating player, using the gap method.

            return self.traverse(child, reach_prob);
        }
    }

    /// Train for the given number of iterations.
    /// Each iteration: sample a deal and update BOTH players.
    pub fn train(self: *MCCFRTrainer, iterations: usize) !f64 {
        const deals = game.allDeals();
        var total_payoff: f64 = 0;

        for (0..iterations) |_| {
            const deal = deals[self.rng.random().uintLessThan(usize, deals.len)];

            // Update P0
            self.update_player = 0;
            const payoff = try self.traverse(deal, .{ 1, 1 });

            // Update P1
            self.update_player = 1;
            _ = try self.traverse(deal, .{ 1, 1 });

            total_payoff += payoff[0];
        }

        return total_payoff / @as(f64, @floatFromInt(iterations));
    }
    
    /// Look up the average (Nash-converging) strategy for an information set.
    ///
    /// This is used during evaluation to see how the trained policy plays.
    /// Returns uniform distribution if this info set was never visited during training.
    pub fn getStrategy(self: *const MCCFRTrainer, key: game.InfoKey, num_actions: usize) [game.MAX_ACTIONS]f64 {
        if (self.infosets.get(key)) |infoset| {
            return infoset.averageStrategy();
        }
        // Never visited this state during training—play uniformly.
        var uniform: [game.MAX_ACTIONS]f64 = .{ 0, 0, 0 };
        for (0..num_actions) |i| uniform[i] = 1.0 / @as(f64, @floatFromInt(num_actions));
        return uniform;
    }
};
