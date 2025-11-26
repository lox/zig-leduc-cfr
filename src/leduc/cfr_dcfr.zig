// cfr_dcfr.zig — Discounted CFR (DCFR)
// ====================================
//
// Discounted CFR improves on vanilla CFR by discounting older iterations'
// regrets and strategies. This accelerates convergence by rapidly deprecating
// early mistakes when the strategy was still poor.
//
// DCFR uses three discount parameters (Brown & Sandholm, 2019):
//   - α (alpha): Positive regret discount. Regrets are scaled by t^α / (t^α + 1)
//   - β (beta): Negative regret discount. Negative regrets scaled by t^β / (t^β + 1)
//   - γ (gamma): Strategy sum discount. Applied each iteration as (t / (t+1))^γ
//
// Common configurations:
//   - LCFR (Linear CFR): α=1, β=1, γ=1 — Simple linear weighting
//   - DCFR: α=1.5, β=0, γ=2 — Recommended by Brown & Sandholm
//   - CFR+: α=∞, β=-∞, γ=1 — Equivalent to CFR+ (immediate negative regret reset)
//
// The key insight: early iterations explore poorly because the strategy is
// still random. By discounting old regrets, we let recent (better-informed)
// iterations dominate the strategy.
//
// CONVERGENCE:
//   DCFR maintains O(1/√T) convergence like vanilla CFR, but with much better
//   constants in practice. Empirically 20-60% improvement in exploitability.

const std = @import("std");
const game = @import("game.zig");
const cfr = @import("cfr.zig");

const Payoffs = [game.NUM_PLAYERS]f64;

pub const DiscountScheme = enum {
    lcfr,
    dcfr,
};

pub const DCFRTrainer = struct {
    allocator: std.mem.Allocator,
    infosets: cfr.InfoMap,
    iteration: usize = 0,
    scheme: DiscountScheme,

    // DCFR discount parameters
    alpha: f64,
    beta: f64,
    gamma: f64,

    pub fn init(allocator: std.mem.Allocator, scheme: DiscountScheme) DCFRTrainer {
        var trainer = DCFRTrainer{
            .allocator = allocator,
            .infosets = cfr.InfoMap.init(allocator),
            .scheme = scheme,
            .alpha = 1.0,
            .beta = 1.0,
            .gamma = 1.0,
        };

        switch (scheme) {
            .lcfr => {},
            .dcfr => {
                trainer.alpha = 1.5;
                trainer.beta = 0.0;
                trainer.gamma = 2.0;
            },
        }

        return trainer;
    }

    pub fn deinit(self: *DCFRTrainer) void {
        self.infosets.deinit();
    }

    pub fn nodeCount(self: *const DCFRTrainer) usize {
        return self.infosets.count();
    }

    fn traverse(self: *DCFRTrainer, state: game.GameState, reach_prob: Payoffs) !Payoffs {
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
        const strategy = infoset.currentStrategy();

        // Recurse for each action
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

        // ===== UPDATE REGRETS =====
        const infoset_mut = self.infosets.getPtr(key).?;
        for (0..num_actions) |i| {
            const regret = action_payoffs[i][player] - expected_payoff[player];
            infoset_mut.cumulative_regret[i] += reach_prob[opponent] * regret;
        }

        // ===== UPDATE AVERAGE STRATEGY =====
        for (0..num_actions) |i| {
            infoset_mut.cumulative_strategy[i] += reach_prob[player] * strategy[i];
        }

        return expected_payoff;
    }

    /// Apply DCFR discounting to all infosets at the end of an iteration.
    fn applyDiscounting(self: *DCFRTrainer) void {
        const t: f64 = @floatFromInt(self.iteration);

        // Compute discount factors
        // Positive regret: t^α / (t^α + 1)
        const t_alpha = std.math.pow(f64, t, self.alpha);
        const pos_discount = t_alpha / (t_alpha + 1.0);

        // Negative regret: t^β / (t^β + 1)
        const t_beta = std.math.pow(f64, t, self.beta);
        const neg_discount = t_beta / (t_beta + 1.0);

        // Strategy: (t / (t+1))^γ
        const strat_discount = std.math.pow(f64, t / (t + 1.0), self.gamma);

        var iter = self.infosets.valueIterator();
        while (iter.next()) |infoset| {
            // Discount regrets
            for (0..game.MAX_ACTIONS) |i| {
                if (infoset.cumulative_regret[i] > 0) {
                    infoset.cumulative_regret[i] *= pos_discount;
                } else {
                    infoset.cumulative_regret[i] *= neg_discount;
                }
            }

            // Discount strategy sums
            for (0..game.MAX_ACTIONS) |i| {
                infoset.cumulative_strategy[i] *= strat_discount;
            }
        }
    }

    pub fn train(self: *DCFRTrainer, iterations: usize) !f64 {
        const deals = game.allDeals();
        var total_payoff: f64 = 0;

        for (0..iterations) |_| {
            self.iteration += 1;

            for (deals) |deal| {
                const payoff = try self.traverse(deal, .{ 1, 1 });
                total_payoff += payoff[0];
            }

            // Apply discounting after each iteration
            self.applyDiscounting();
        }

        const num_traversals: f64 = @floatFromInt(iterations * deals.len);
        return total_payoff / num_traversals;
    }

    pub fn getStrategy(self: *const DCFRTrainer, key: game.InfoKey, num_actions: usize) [game.MAX_ACTIONS]f64 {
        if (self.infosets.get(key)) |infoset| {
            return infoset.averageStrategy();
        }
        var uniform: [game.MAX_ACTIONS]f64 = .{ 0, 0, 0 };
        for (0..num_actions) |i| uniform[i] = 1.0 / @as(f64, @floatFromInt(num_actions));
        return uniform;
    }
};
