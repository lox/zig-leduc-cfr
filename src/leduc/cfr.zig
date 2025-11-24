const std = @import("std");
const game = @import("game.zig");

/// Shared CFR utilities: information-set representation plus helpers that keep
/// both trainer implementations aligned.
pub const InfoSet = struct {
    key: game.InfoKey,
    action_count: u8,
    actions: [game.MAX_ACTIONS]game.Action,
    regret_sum: [game.MAX_ACTIONS]f64,
    strategy_sum: [game.MAX_ACTIONS]f64,

    pub fn init(key: game.InfoKey, actions_slice: []const game.Action) InfoSet {
        var info = InfoSet{
            .key = key,
            .action_count = @intCast(actions_slice.len),
            .actions = undefined,
            .regret_sum = [_]f64{ 0, 0, 0 },
            .strategy_sum = [_]f64{ 0, 0, 0 },
        };
        for (actions_slice, 0..) |act, i| info.actions[i] = act;
        return info;
    }

    /// Regret matching (or regret matching+ after clamping upstream).
    pub fn strategy(self: *const InfoSet) [game.MAX_ACTIONS]f64 {
        var s: [game.MAX_ACTIONS]f64 = undefined;
        var norm: f64 = 0;
        for (0..self.action_count) |i| {
            const r = self.regret_sum[i];
            s[i] = if (r > 0) r else 0;
            norm += s[i];
        }
        if (norm > 0) {
            for (0..self.action_count) |i| s[i] /= norm;
        } else {
            const u = 1.0 / @as(f64, @floatFromInt(self.action_count));
            for (0..self.action_count) |i| s[i] = u;
        }
        return s;
    }

    pub fn avgStrategy(self: *const InfoSet) [game.MAX_ACTIONS]f64 {
        var s: [game.MAX_ACTIONS]f64 = undefined;
        var norm: f64 = 0;
        for (0..self.action_count) |i| norm += self.strategy_sum[i];
        if (norm > 0) {
            for (0..self.action_count) |i| s[i] = self.strategy_sum[i] / norm;
        } else {
            const u = 1.0 / @as(f64, @floatFromInt(self.action_count));
            for (0..self.action_count) |i| s[i] = u;
        }
        return s;
    }
};

pub const InfoMap = std.AutoHashMap(game.InfoKey, InfoSet);

pub fn getInfo(map: *InfoMap, key: game.InfoKey, actions: []const game.Action) !*InfoSet {
    const entry = try map.getOrPut(key);
    if (!entry.found_existing) entry.value_ptr.* = InfoSet.init(key, actions);
    return entry.value_ptr;
}

pub fn avgStrategyOrUniform(map: *const InfoMap, key: game.InfoKey, num_actions: usize) [game.MAX_ACTIONS]f64 {
    if (map.get(key)) |info| {
        const s = info.avgStrategy();
        var out = [_]f64{ 0, 0, 0 };
        for (0..num_actions) |i| out[i] = s[i];
        return out;
    }
    var out = [_]f64{ 0, 0, 0 };
    const u = 1.0 / @as(f64, @floatFromInt(num_actions));
    for (0..num_actions) |i| out[i] = u;
    return out;
}
