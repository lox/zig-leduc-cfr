// cfr_plus.zig — CFR+ (Faster Convergence)
// ========================================
//
// CFR+ is a variant of CFR that often converges much faster (order O(1/T) instead
// of O(1/sqrt(T))).
//
// Key differences from Vanilla CFR:
//   1. **Regret Matching+**: Cumulative regrets are clamped to 0 after each update.
//      R+(I, a) = max(0, R(I, a) + r(I, a))
//      This allows the algorithm to quickly "forget" bad early decisions.
//
//   2. **Linear Averaging**: The average strategy is weighted by the iteration
//      number (e.g., iteration 100 counts 100x more than iteration 1).
//      This puts more emphasis on recent, more refined strategies.
//
//   3. **Alternating Updates**: (Optional but common) Update P1, then P2, rather
//      than simultaneous updates. This implementation uses simultaneous updates
//      to match the structure of our Vanilla CFR, but the core improvements
//      (RM+ and Linear Averaging) are sufficient for speedups.
//
// References:
//   "Solving Large Imperfect Information Games Using CFR+" (Tammelin et al., 2015)

const std = @import("std");
const game = @import("game.zig");
const cfr = @import("cfr.zig");

const Payoffs = [game.NUM_PLAYERS]f64;

pub const CFRPlusTrainer = struct {
    allocator: std.mem.Allocator,
    infosets: cfr.InfoMap,
    update_player: u8 = 0,

    pub fn init(allocator: std.mem.Allocator) CFRPlusTrainer {
        return .{ .allocator = allocator, .infosets = cfr.InfoMap.init(allocator) };
    }

    pub fn deinit(self: *CFRPlusTrainer) void {
        self.infosets.deinit();
    }

    /// The CFR+ traversal.
    ///
    /// Arguments:
    ///   - weight: The weight for the average strategy update (usually iteration #).
    fn traverse(self: *CFRPlusTrainer, state: game.GameState, reach_prob: Payoffs, weight: f64) !Payoffs {

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
            return self.traverse(next, reach_prob, weight);
        }

        // ===== DECISION NODE =====
        const player: usize = @intCast(state.current_player);
        const opponent: usize = 1 - player;

        var actions: [game.MAX_ACTIONS]game.Action = undefined;
        const num_actions = state.legalActions(&actions);

        const key = state.infoKey(state.current_player);
        const infoset = try cfr.getOrCreateInfoSet(&self.infosets, key, actions[0..num_actions]);
        
        // Use Regret Matching (on R+) to get current strategy
        const strategy = infoset.currentStrategy();

        // Recurse for each action
        var action_payoffs: [game.MAX_ACTIONS]Payoffs = undefined;
        var expected_payoff: Payoffs = .{ 0, 0 };

        for (actions[0..num_actions], 0..) |action, i| {
            const child = state.apply(action);

            var child_reach = reach_prob;
            child_reach[player] *= strategy[i];

            const child_payoff = try self.traverse(child, child_reach, weight);
            action_payoffs[i] = child_payoff;

            expected_payoff[0] += strategy[i] * child_payoff[0];
            expected_payoff[1] += strategy[i] * child_payoff[1];
        }

        // ===== UPDATE REGRETS (CFR+) =====
        // Only update regrets for the current "update player" (if we are doing alternating updates).
        // However, we need to pass an "update_player" flag to this function to support that.
        // For now, let's support the alternating update structure by only updating regrets
        // if the current player matches the configured update player.

        if (player == self.update_player) {
            const infoset_mut = self.infosets.getPtr(key).?;
            for (0..num_actions) |i| {
                const regret = action_payoffs[i][player] - expected_payoff[player];
                
                // CFR+: cumulative_regret = max(0, cumulative_regret + reach_opp * regret)
                const weighted_regret = reach_prob[opponent] * regret;
                infoset_mut.cumulative_regret[i] = @max(0, infoset_mut.cumulative_regret[i] + weighted_regret);
            }

             // ===== UPDATE AVERAGE STRATEGY (Linear Averaging) =====
            for (0..num_actions) |i| {
                infoset_mut.cumulative_strategy[i] += weight * reach_prob[player] * strategy[i];
            }
        }

        return expected_payoff;
    }

    /// Train for the given number of iterations using CFR+.
    pub fn train(self: *CFRPlusTrainer, iterations: usize) !f64 {
        const deals = game.allDeals();
        var total_payoff: f64 = 0;

        // In CFR+, we use a 'burn-in' period where the average strategy accumulation is low or zero,
        // but since we use linear weighting (weight = iteration), this is implicit.
        // However, the key to CFR+ speed is updating Player 2 using Player 1's NEW strategy.

        for (0..iterations) |i| {
            // Linear weighting: iteration 1 has weight 1, iter 2 weight 2...
            // This makes the algorithm focus on recent strategies.
            // NOTE: Using i+1 for weight makes it weight=1 for first iter
            const weight: f64 = @floatFromInt(i + 1);

            // Alternating Updates:
            // 1. Update Player 0
            self.update_player = 0;
            for (deals) |deal| {
                _ = try self.traverse(deal, .{ 1, 1 }, weight);
            }

            // 2. Update Player 1
            self.update_player = 1;
            for (deals) |deal| {
                const payoff = try self.traverse(deal, .{ 1, 1 }, weight);
                // Only track payoff from one pass to avoid double counting
                total_payoff += payoff[0];
            }
        }
        
        const num_traversals: f64 = @floatFromInt(iterations * deals.len);
        return total_payoff / num_traversals;
    }

    pub fn getStrategy(self: *const CFRPlusTrainer, key: game.InfoKey, num_actions: usize) [game.MAX_ACTIONS]f64 {
        if (self.infosets.get(key)) |infoset| {
            return infoset.averageStrategy();
        }
        var uniform: [game.MAX_ACTIONS]f64 = .{ 0, 0, 0 };
        for (0..num_actions) |i| uniform[i] = 1.0 / @as(f64, @floatFromInt(num_actions));
        return uniform;
    }
};
