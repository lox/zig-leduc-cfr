// play.zig — Strategy Evaluation
// ===============================
//
// This file provides tools to evaluate trained CFR strategies:
//
//   - **Head-to-head**: Expected value when two strategies play each other
//   - **Exploitability**: How far a strategy is from Nash equilibrium
//
// Exploitability is the key metric: it measures how much a perfect opponent
// could win against our strategy. At Nash equilibrium, exploitability = 0.

const std = @import("std");
const game = @import("game.zig");

// =============================================================================
// Head-to-Head Evaluation
// =============================================================================

/// Expected payoff for player 0 when strategy A plays P0 and strategy B plays P1.
/// Strategies must provide: getStrategy(key, num_actions) -> [MAX_ACTIONS]f64
pub fn headToHead(comptime A: type, a: *const A, comptime B: type, b: *const B) f64 {
    const deals = game.allDeals();
    var total: f64 = 0;
    for (deals) |deal| total += evalDeal(A, a, B, b, deal);
    return total / @as(f64, @floatFromInt(deals.len));
}

fn evalDeal(comptime A: type, a: *const A, comptime B: type, b: *const B, state: game.GameState) f64 {
    // Terminal states
    if (state.folded_player) |folder| return state.foldPayoff(folder);
    if (state.round == .showdown) return state.showdownPayoff();

    // Chance node: reveal board
    if (state.round == .flop and !state.board_revealed) {
        var next = state;
        next.revealBoard();
        return evalDeal(A, a, B, b, next);
    }

    // Decision node
    const player = state.current_player;
    var actions: [game.MAX_ACTIONS]game.Action = undefined;
    const num_actions = state.legalActions(&actions);

    const key = state.infoKey(player);
    const strategy = if (player == 0)
        a.getStrategy(key, num_actions)
    else
        b.getStrategy(key, num_actions);

    var expected_value: f64 = 0;
    for (actions[0..num_actions], 0..) |action, i| {
        if (strategy[i] == 0) continue;
        const child = state.apply(action);
        expected_value += strategy[i] * evalDeal(A, a, B, b, child);
    }
    return expected_value;
}

// =============================================================================
// Exploitability
// =============================================================================
//
// Exploitability measures how far a strategy is from Nash equilibrium.
// It's computed as: (BR_0_gain + BR_1_gain) / 2
//
// Where BR_i_gain is how much player i could gain by switching to their
// best response. At Nash equilibrium, neither player can improve, so
// exploitability = 0.
//
// The tricky part is computing the best response. We can't just pick the
// best action at each game state, because the best responder doesn't know
// the opponent's cards—they must pick the same action for all states in
// an information set.
//
// We solve this with **policy iteration**:
//   1. Start with an arbitrary policy (action choice per info set)
//   2. Evaluate: compute the expected value of each action at each info set
//   3. Improve: switch to the best action at each info set
//   4. Repeat until the policy stops changing
//
// For Leduc's tiny game tree, this converges in just a few iterations.

pub fn exploitability(comptime T: type, strategy: *const T) !f64 {
    const self_play_value = headToHead(T, strategy, T, strategy);

    const br0_value = try bestResponseValue(T, strategy, 0);
    const br1_value = try bestResponseValue(T, strategy, 1);

    // Zero-sum game: if P0's value is v, P1's value is -v
    return ((br0_value - self_play_value) + (br1_value - (-self_play_value))) / 2.0;
}

// =============================================================================
// Best Response Calculation
// =============================================================================
//
// To find the best response for a player:
//
// 1. We iterate over all possible deals (all card combinations)
// 2. At each of *our* decision points, we try all actions and track their values
// 3. At *opponent* decision points, they follow their fixed strategy
// 4. We weight values by opponent reach probability (how likely they are to
//    reach this state given their strategy and the cards)
//
// Why weight by opponent reach? Because an info set groups multiple game states.
// We need to average over "which state are we actually in?" weighted by the
// probability of each state occurring.

/// Accumulated statistics for one information set during best response search.
const InfoSetStats = struct {
    num_actions: u8,
    value_sum: [game.MAX_ACTIONS]f64, // Sum of (opponent_reach × action_value)
    weight_sum: f64, // Sum of opponent_reach (for normalization)

    fn init(n: usize) InfoSetStats {
        return .{
            .num_actions = @intCast(n),
            .value_sum = .{ 0, 0, 0 },
            .weight_sum = 0,
        };
    }

    fn accumulate(self: *InfoSetStats, opponent_reach: f64, action_values: []const f64) void {
        self.weight_sum += opponent_reach;
        for (0..self.num_actions) |i| {
            self.value_sum[i] += opponent_reach * action_values[i];
        }
    }

    fn bestAction(self: *const InfoSetStats) u8 {
        var best: u8 = 0;
        var best_value: f64 = -std.math.inf(f64);
        for (0..self.num_actions) |i| {
            const avg_value = self.value_sum[i] / self.weight_sum;
            if (avg_value > best_value) {
                best_value = avg_value;
                best = @intCast(i);
            }
        }
        return best;
    }
};

/// A deterministic policy: maps each info set to a single action index.
const Policy = std.HashMap(game.InfoKey, u8, game.InfoKey.HashContext, std.hash_map.default_max_load_percentage);
const StatsMap = std.HashMap(game.InfoKey, InfoSetStats, game.InfoKey.HashContext, std.hash_map.default_max_load_percentage);

/// Compute the expected value of a player's best response against the opponent's strategy.
fn bestResponseValue(comptime T: type, opponent_strategy: *const T, br_player: u8) !f64 {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var policy = Policy.init(alloc);
    var stats = StatsMap.init(alloc);

    const deals = game.allDeals();

    // -------------------------------------------------------------------------
    // Policy Iteration
    // -------------------------------------------------------------------------
    // We alternate between:
    //   - Evaluate: given current policy, compute expected value of each action
    //   - Improve: update policy to pick the best action at each info set
    //
    // This converges quickly because Leduc has few info sets.

    var iteration: usize = 0;
    while (iteration < 10) : (iteration += 1) {
        stats.clearRetainingCapacity();

        // EVALUATE: Traverse all deals, accumulating action values
        for (deals) |deal| {
            _ = try evaluateWithPolicy(T, opponent_strategy, br_player, deal, 1.0, &policy, &stats);
        }

        // IMPROVE: Update policy to pick best action at each info set
        var changed = false;
        var it = stats.iterator();
        while (it.next()) |entry| {
            const best = entry.value_ptr.bestAction();
            const prev = policy.get(entry.key_ptr.*);

            if (prev == null or prev.? != best) {
                changed = true;
                try policy.put(entry.key_ptr.*, best);
            }
        }

        if (!changed) break; // Policy converged
    }

    // -------------------------------------------------------------------------
    // Final Evaluation
    // -------------------------------------------------------------------------
    // Now that we have the optimal policy, compute its expected value.

    var total: f64 = 0;
    for (deals) |deal| {
        total += try evalFinalPolicy(T, opponent_strategy, br_player, &policy, deal);
    }
    return total / @as(f64, @floatFromInt(deals.len));
}

/// Traverse the game tree, evaluating actions for the best responder.
///
/// - At BR player's nodes: try all actions, accumulate their values in stats
/// - At opponent's nodes: follow their strategy, weighting by reach probability
///
/// Returns the expected value under the current policy.
fn evaluateWithPolicy(
    comptime T: type,
    opponent_strategy: *const T,
    br_player: u8,
    state: game.GameState,
    opponent_reach: f64, // Probability opponent reaches this state
    policy: *Policy,
    stats: *StatsMap,
) !f64 {
    // ===== Terminal =====
    if (state.folded_player) |folder| {
        return playerPayoff(state.foldPayoff(folder), br_player);
    }
    if (state.round == .showdown) {
        return playerPayoff(state.showdownPayoff(), br_player);
    }

    // ===== Chance =====
    if (state.round == .flop and !state.board_revealed) {
        var next = state;
        next.revealBoard();
        return evaluateWithPolicy(T, opponent_strategy, br_player, next, opponent_reach, policy, stats);
    }

    // ===== Decision =====
    const player = state.current_player;
    var actions: [game.MAX_ACTIONS]game.Action = undefined;
    const num_actions = state.legalActions(&actions);

    if (player == br_player) {
        // -----------------------------------------------------------------
        // Best responder's turn: evaluate ALL actions
        // -----------------------------------------------------------------
        // We need to know the value of each action to find the best one.
        // Even though our current policy picks one action, we evaluate all
        // of them so we can improve the policy later.

        var action_values: [game.MAX_ACTIONS]f64 = undefined;
        for (actions[0..num_actions], 0..) |action, i| {
            const child = state.apply(action);
            action_values[i] = try evaluateWithPolicy(T, opponent_strategy, br_player, child, opponent_reach, policy, stats);
        }

        // Accumulate values for this info set (weighted by opponent reach)
        const key = state.infoKey(player);
        const entry = try stats.getOrPut(key);
        if (!entry.found_existing) {
            entry.value_ptr.* = InfoSetStats.init(num_actions);
        }
        entry.value_ptr.accumulate(opponent_reach, action_values[0..num_actions]);

        // Return value under current policy (or first action if no policy yet)
        const policy_action = policy.get(key) orelse blk: {
            try policy.put(key, 0);
            break :blk 0;
        };
        return action_values[policy_action];
    } else {
        // -----------------------------------------------------------------
        // Opponent's turn: follow their fixed strategy
        // -----------------------------------------------------------------
        // We weight by their action probabilities, which affects opponent_reach
        // for downstream nodes.

        const key = state.infoKey(player);
        const strategy = opponent_strategy.getStrategy(key, num_actions);

        var expected_value: f64 = 0;
        for (actions[0..num_actions], 0..) |action, i| {
            if (strategy[i] == 0) continue;

            const child = state.apply(action);
            const child_reach = opponent_reach * strategy[i];
            const child_value = try evaluateWithPolicy(T, opponent_strategy, br_player, child, child_reach, policy, stats);

            expected_value += strategy[i] * child_value;
        }
        return expected_value;
    }
}

/// Evaluate the final best-response policy (after convergence).
fn evalFinalPolicy(
    comptime T: type,
    opponent_strategy: *const T,
    br_player: u8,
    policy: *const Policy,
    state: game.GameState,
) !f64 {
    // ===== Terminal =====
    if (state.folded_player) |folder| {
        return playerPayoff(state.foldPayoff(folder), br_player);
    }
    if (state.round == .showdown) {
        return playerPayoff(state.showdownPayoff(), br_player);
    }

    // ===== Chance =====
    if (state.round == .flop and !state.board_revealed) {
        var next = state;
        next.revealBoard();
        return evalFinalPolicy(T, opponent_strategy, br_player, policy, next);
    }

    // ===== Decision =====
    const player = state.current_player;
    var actions: [game.MAX_ACTIONS]game.Action = undefined;
    const num_actions = state.legalActions(&actions);

    if (player == br_player) {
        // Best responder: use the computed optimal policy
        const key = state.infoKey(player);
        const action_idx = policy.get(key) orelse unreachable;
        const child = state.apply(actions[action_idx]);
        return evalFinalPolicy(T, opponent_strategy, br_player, policy, child);
    } else {
        // Opponent: follow their strategy
        const key = state.infoKey(player);
        const strategy = opponent_strategy.getStrategy(key, num_actions);

        var expected_value: f64 = 0;
        for (actions[0..num_actions], 0..) |action, i| {
            if (strategy[i] == 0) continue;
            const child = state.apply(action);
            expected_value += strategy[i] * try evalFinalPolicy(T, opponent_strategy, br_player, policy, child);
        }
        return expected_value;
    }
}

/// Convert P0's payoff to the specified player's payoff (zero-sum).
fn playerPayoff(p0_payoff: f64, player: u8) f64 {
    return if (player == 0) p0_payoff else -p0_payoff;
}
