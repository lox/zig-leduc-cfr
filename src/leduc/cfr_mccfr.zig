// cfr_mccfr.zig — Monte Carlo CFR (External Sampling)
// ====================================================
//
// Monte Carlo CFR trades exactness for speed: instead of traversing the entire
// game tree, we *sample* trajectories through it.
//
// This implementation uses "external sampling" (also called "outcome sampling"):
//   - We designate one player as the "updating player" each traversal
//   - The updating player enumerates all their actions (like vanilla CFR)
//   - The opponent's actions are *sampled* from their current strategy
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

const std = @import("std");
const game = @import("game.zig");
const cfr = @import("cfr.zig");

const Payoffs = [game.NUM_PLAYERS]f64;

/// Simple LCG random number generator. Good enough for educational Monte Carlo.
const Rng = struct {
    state: u64,

    fn init() Rng {
        var seed: u64 = 0;
        std.crypto.random.bytes(std.mem.asBytes(&seed));
        return .{ .state = seed | 1 };
    }

    fn next(self: *Rng) u64 {
        self.state = self.state *% 6364136223846793005 +% 1;
        return self.state;
    }

    fn float(self: *Rng) f64 {
        return @as(f64, @floatFromInt(self.next() >> 11)) / 9007199254740992.0;
    }

    fn uintLessThan(self: *Rng, comptime T: type, max: T) T {
        return @intCast(self.next() % @as(u64, max));
    }
};

pub const MCCFRTrainer = struct {
    allocator: std.mem.Allocator,
    infosets: cfr.InfoMap,
    rng: Rng,
    update_player: u8 = 0,

    pub fn init(allocator: std.mem.Allocator) MCCFRTrainer {
        return .{
            .allocator = allocator,
            .infosets = cfr.InfoMap.init(allocator),
            .rng = Rng.init(),
        };
    }

    pub fn deinit(self: *MCCFRTrainer) void {
        self.infosets.deinit();
    }

    /// Sample an action index from a probability distribution.
    fn sampleAction(rng: *Rng, probs: []const f64) usize {
        const r = rng.float();
        var cumulative: f64 = 0;
        for (probs, 0..) |p, i| {
            cumulative += p;
            if (r <= cumulative) return i;
        }
        return probs.len - 1;
    }

    /// The MCCFR traversal. Very similar to vanilla CFR, but opponent nodes
    /// sample a single action instead of enumerating all actions.
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
        const opponent: usize = 1 - player;

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

            // Update regrets (same formula as vanilla CFR)
            const infoset_mut = self.infosets.getPtr(key).?;
            for (0..num_actions) |i| {
                const regret = action_payoffs[i][player] - expected_payoff[player];
                infoset_mut.cumulative_regret[i] += reach_prob[opponent] * regret;
            }

            // Update average strategy
            for (0..num_actions) |i| {
                infoset_mut.cumulative_strategy[i] += reach_prob[player] * strategy[i];
            }

            return expected_payoff;
        } else {
            // ===== OPPONENT: sample ONE action from their strategy =====
            // This is the key difference from vanilla CFR!
            // We don't enumerate—we sample, making the tree traversal O(depth) not O(tree).
            const action_idx = sampleAction(&self.rng, strategy[0..num_actions]);
            const action = actions[action_idx];
            const child = state.apply(action);

            var child_reach = reach_prob;
            child_reach[player] *= strategy[action_idx];

            return self.traverse(child, child_reach);
        }
    }

    /// Train for the given number of iterations.
    /// Each iteration: sample a deal, update player 0, then update player 1.
    pub fn train(self: *MCCFRTrainer, iterations: usize) !f64 {
        const deals = game.allDeals();
        var total_payoff: f64 = 0;

        for (0..iterations) |_| {
            const deal = deals[self.rng.uintLessThan(usize, deals.len)];

            // Update player 0's regrets
            self.update_player = 0;
            const payoff = try self.traverse(deal, .{ 1, 1 });

            // Update player 1's regrets (same deal, different perspective)
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
