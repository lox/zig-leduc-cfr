// cfr.zig — Information Sets & Regret Matching
// =============================================
//
// This file implements the core abstractions shared by all CFR trainers:
//
//   1. **Information Set (InfoSet)**: Groups game states that look identical to
//      a player (same private card, same public card, same betting history).
//      We store cumulative regrets and strategy weights per action.
//
//   2. **Regret Matching**: Converts cumulative regrets into a probability
//      distribution over actions. Actions with positive regret get played
//      more often; negative regret actions are avoided.
//
//   3. **Average Strategy**: The time-averaged strategy that converges to
//      Nash equilibrium. Computed from cumulative strategy weights.
//
// The key insight of CFR is that we don't need to store strategies explicitly.
// Instead, we accumulate regrets, and the strategy emerges from regret matching.
//
// Both trainers (vanilla CFR and MCCFR) use these primitives identically—they
// only differ in how they traverse the game tree.

const std = @import("std");
const game = @import("game.zig");

pub const InfoSet = struct {
    key: game.InfoKey,

    num_actions: u8,
    actions: [game.MAX_ACTIONS]game.Action,

    // Cumulative regret R(I, a): how much better action `a` would have been
    // compared to what we actually played, summed across all iterations.
    // Positive regret = "I wish I'd played this more"
    // Negative regret = "I'm glad I didn't play this"
    cumulative_regret: [game.MAX_ACTIONS]f64,

    // Cumulative strategy weight: reach_prob(I) × strategy(I, a), summed across
    // iterations. Used to compute the average strategy that converges to Nash.
    cumulative_strategy: [game.MAX_ACTIONS]f64,

    pub fn init(key: game.InfoKey, actions_slice: []const game.Action) InfoSet {
        var info = InfoSet{
            .key = key,
            .num_actions = @intCast(actions_slice.len),
            .actions = undefined,
            .cumulative_regret = [_]f64{ 0, 0, 0 },
            .cumulative_strategy = [_]f64{ 0, 0, 0 },
        };
        for (actions_slice, 0..) |act, i| info.actions[i] = act;
        return info;
    }

    /// Regret Matching: convert cumulative regrets into action probabilities.
    ///
    /// The rule is simple:
    ///   - Take positive part of each regret: max(0, regret)
    ///   - Normalize to sum to 1
    ///   - If all regrets ≤ 0, play uniformly (no clear preference)
    ///
    /// This is the engine that drives CFR toward equilibrium: we play actions
    /// proportionally to how much we regret not playing them.
    pub fn currentStrategy(self: *const InfoSet) [game.MAX_ACTIONS]f64 {
        var strategy: [game.MAX_ACTIONS]f64 = undefined;
        var positive_regret_sum: f64 = 0;

        for (0..self.num_actions) |i| {
            const r = self.cumulative_regret[i];
            strategy[i] = if (r > 0) r else 0;
            positive_regret_sum += strategy[i];
        }

        if (positive_regret_sum > 0) {
            for (0..self.num_actions) |i| strategy[i] /= positive_regret_sum;
        } else {
            const uniform = 1.0 / @as(f64, @floatFromInt(self.num_actions));
            for (0..self.num_actions) |i| strategy[i] = uniform;
        }
        return strategy;
    }

    /// Average Strategy: the time-averaged strategy across all iterations.
    ///
    /// In two-player zero-sum games, this converges to Nash equilibrium—not
    /// the current regret-matched strategy (which can oscillate), but the
    /// average of all strategies we've played, weighted by reach probability.
    pub fn averageStrategy(self: *const InfoSet) [game.MAX_ACTIONS]f64 {
        var strategy: [game.MAX_ACTIONS]f64 = undefined;
        var total_weight: f64 = 0;

        for (0..self.num_actions) |i| total_weight += self.cumulative_strategy[i];

        if (total_weight > 0) {
            for (0..self.num_actions) |i| strategy[i] = self.cumulative_strategy[i] / total_weight;
        } else {
            const uniform = 1.0 / @as(f64, @floatFromInt(self.num_actions));
            for (0..self.num_actions) |i| strategy[i] = uniform;
        }
        return strategy;
    }
};

/// Map from information set keys to their stored regrets and strategies.
pub const InfoMap = std.AutoHashMap(game.InfoKey, InfoSet);

/// Look up or create an information set for this key.
pub fn getOrCreateInfoSet(map: *InfoMap, key: game.InfoKey, actions: []const game.Action) !*InfoSet {
    const entry = try map.getOrPut(key);
    if (!entry.found_existing) entry.value_ptr.* = InfoSet.init(key, actions);
    return entry.value_ptr;
}
