const std = @import("std");
const game = @import("game.zig");

const BrStats = struct {
    action_count: u8,
    ev_sum: [game.MAX_ACTIONS]f64,
    weight_sum: f64,
};

/// Expected utility for player 0 when strategy A plays as P0 and strategy B as P1.
/// Strategies are any types that provide `averageStrategy(key, actions, n) -> ?[MAX_ACTIONS]f64`.
pub fn headToHead(comptime A: type, a: *const A, comptime B: type, b: *const B) f64 {
    const deals = game.allDeals();

    var total: f64 = 0;
    for (deals) |hand| total += evalHand(A, a, B, b, hand);
    return total / @as(f64, @floatFromInt(deals.len));
}

fn evalHand(comptime A: type, a: *const A, comptime B: type, b: *const B, state: game.GameState) f64 {
    if (state.folded_player) |fp| {
        const val0 = game.terminalFoldUtility0(&state, fp);
        return val0;
    }
    if (state.round == .showdown) {
        const val0 = game.showdownUtility0(&state);
        return val0;
    }
    if (state.round == .flop and state.public_card == null) {
        var next = state;
        game.revealBoard(&next);
        return evalHand(A, a, B, b, next);
    }

    const player = state.current_player;
    var acts: [game.MAX_ACTIONS]game.Action = undefined;
    const n = game.getAvailableActions(&state, &acts);

    const key = game.makeInfoKey(&state, player);
    const strat = if (player == 0)
        a.averageStrategy(key, acts[0..n], n)
    else
        b.averageStrategy(key, acts[0..n], n);

    var ev: f64 = 0;
    for (acts[0..n], 0..) |act, i| {
        if (strat[i] == 0) continue;
        const child = game.applyAction(&state, act);
        ev += strat[i] * evalHand(A, a, B, b, child);
    }
    return ev;
}

/// Best-response exploitability (NashConv/2) of a strategy profile where both players use `strat`'s
/// average strategy. Returns chips/hand: ((BR_0 - v0) + (BR_1 - v1)) / 2, with v0 the profile value
/// to player 0 and v1 = -v0 in this zero-sum game.
pub fn exploitability(comptime T: type, strat: *const T) !f64 {
    const v0 = headToHead(T, strat, T, strat);
    const br0 = try brValue(T, strat, 0); // payoff to player 0
    const br1 = try brValue(T, strat, 1); // payoff to player 1
    const v1 = -v0;
    const raw = ((br0 - v0) + (br1 - v1)) / 2.0;
    return raw;
}

fn brValue(comptime T: type, strat: *const T, eval_player: u8) !f64 {
    const deals = game.allDeals();

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    // Deterministic policy for the best responder (infoset -> action index).
    var policy = std.AutoHashMap(game.InfoKey, u8).init(arena.allocator());
    defer policy.deinit();

    // Temporary storage for action values during policy evaluation.
    var acc = std.AutoHashMap(game.InfoKey, BrStats).init(arena.allocator());
    defer acc.deinit();

    // Simple policy-iteration: evaluate current policy, improve greedily, repeat until stable.
    var iter: usize = 0;
    while (true) : (iter += 1) {
        acc.clearRetainingCapacity();
        // Evaluate current policy and collect counterfactual action values.
        for (deals) |hand| {
            _ = try brCollectPolicy(T, strat, eval_player, hand, 1.0, 1.0, &policy, &acc);
        }

        var changed = false;
        var it = acc.iterator();
        while (it.next()) |entry| {
            const stats = entry.value_ptr.*;
            std.debug.assert(stats.weight_sum > 0);
            var best_idx: u8 = 0;
            var best_val: f64 = -1e30;
            for (0..stats.action_count) |i| {
                const v = stats.ev_sum[i] / stats.weight_sum;
                if (v > best_val) { best_val = v; best_idx = @intCast(i); }
            }
            const prev = policy.get(entry.key_ptr.*);
            if (prev == null or prev.? != best_idx) {
                changed = true;
                try policy.put(entry.key_ptr.*, best_idx);
            }
        }

        // Extra safety to avoid pathological loops, though the game is tiny.
        if (!changed or iter >= 8) break;
    }

    var total: f64 = 0;
    for (deals) |hand| {
        total += try brEval(T, strat, eval_player, &policy, hand);
    }
    return total / @as(f64, @floatFromInt(deals.len));
}

fn brCollectPolicy(
    comptime T: type,
    strat: *const T,
    eval_player: u8,
    state: game.GameState,
    reach_opp: f64,
    chance_prob: f64,
    policy: *std.AutoHashMap(game.InfoKey, u8),
    acc: *std.AutoHashMap(game.InfoKey, BrStats),
) !f64 {
    // Terminals
    if (state.folded_player) |fp| {
        const val0 = game.terminalFoldUtility0(&state, fp);
        return if (eval_player == 0) val0 else -val0;
    }
    if (state.round == .showdown) {
        const val0 = game.showdownUtility0(&state);
        return if (eval_player == 0) val0 else -val0;
    }

    // Chance: reveal board
    if (state.round == .flop and state.public_card == null) {
        var next = state;
        game.revealBoard(&next);
        return brCollectPolicy(T, strat, eval_player, next, reach_opp, chance_prob, policy, acc);
    }

    // Decision nodes
    const player = state.current_player;
    var acts: [game.MAX_ACTIONS]game.Action = undefined;
    const n = game.getAvailableActions(&state, &acts);

    if (player == eval_player) {
        // Evaluate each action assuming we follow the current policy thereafter.
        var values: [game.MAX_ACTIONS]f64 = undefined;
        for (acts[0..n], 0..) |act, i| {
            const child = game.applyAction(&state, act);
            values[i] = try brCollectPolicy(T, strat, eval_player, child, reach_opp, chance_prob, policy, acc);
        }

        // Accumulate action values for this infoset.
        const key = game.makeInfoKey(&state, player);
        var entry = try acc.getOrPut(key);
        if (!entry.found_existing) {
            entry.value_ptr.* = BrStats{
                .action_count = @intCast(n),
                .ev_sum = [_]f64{ 0, 0, 0 },
                .weight_sum = 0,
            };
        }
        const weight = reach_opp * chance_prob;
        entry.value_ptr.weight_sum += weight;
        for (0..n) |i| entry.value_ptr.ev_sum[i] += weight * values[i];

        // Ensure the policy has a choice for this infoset; default to first action.
        const policy_idx = policy.get(key) orelse blk: {
            try policy.put(key, 0);
            break :blk 0;
        };
        return values[policy_idx];
    } else {
        // Opponent follows their average strategy.
        const key = game.makeInfoKey(&state, player);
        const strat_vec = strat.averageStrategy(key, acts[0..n], n);
        var ev: f64 = 0;
        for (acts[0..n], 0..) |act, i| {
            if (strat_vec[i] == 0) continue;
            const child = game.applyAction(&state, act);
            const v = try brCollectPolicy(T, strat, eval_player, child, reach_opp * strat_vec[i], chance_prob, policy, acc);
            ev += strat_vec[i] * v;
        }
        return ev;
    }
}

fn brEval(
    comptime T: type,
    strat: *const T,
    eval_player: u8,
    policy: *const std.AutoHashMap(game.InfoKey, u8),
    state: game.GameState,
) !f64 {
    if (state.folded_player) |fp| {
        const val0 = game.terminalFoldUtility0(&state, fp);
        return if (eval_player == 0) val0 else -val0;
    }
    if (state.round == .showdown) {
        const val0 = game.showdownUtility0(&state);
        return if (eval_player == 0) val0 else -val0;
    }
    if (state.round == .flop and state.public_card == null) {
        var next = state;
        game.revealBoard(&next);
        return brEval(T, strat, eval_player, policy, next);
    }

    const player = state.current_player;
    var acts: [game.MAX_ACTIONS]game.Action = undefined;
    const n = game.getAvailableActions(&state, &acts);

    if (player == eval_player) {
        const key = game.makeInfoKey(&state, player);
        const idx = policy.get(key) orelse std.debug.panic("missing BR policy for info set", .{});
        const act = acts[idx];
        const child = game.applyAction(&state, act);
        return brEval(T, strat, eval_player, policy, child);
    } else {
        const key = game.makeInfoKey(&state, player);
        const strat_vec = strat.averageStrategy(key, acts[0..n], n);
        var ev: f64 = 0;
        for (acts[0..n], 0..) |act, i| {
            if (strat_vec[i] == 0) continue;
            const child = game.applyAction(&state, act);
            ev += strat_vec[i] * try brEval(T, strat, eval_player, policy, child);
        }
        return ev;
    }
}
