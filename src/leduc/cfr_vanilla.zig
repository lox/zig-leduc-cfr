const std = @import("std");
const game = @import("game.zig");
const common = @import("cfr.zig");

// Vanilla full-tree CFR trainer for Leduc.

pub const VanillaCFRTrainer = struct {
    allocator: std.mem.Allocator,
    infosets: common.InfoMap,

    pub fn init(allocator: std.mem.Allocator) VanillaCFRTrainer {
        return .{ .allocator = allocator, .infosets = common.InfoMap.init(allocator) };
    }

    pub fn deinit(self: *VanillaCFRTrainer) void {
        self.infosets.deinit();
    }

    fn cfr(self: *VanillaCFRTrainer, state: game.GameState, reach: [game.NUM_PLAYERS]f64, chance: f64) ![game.NUM_PLAYERS]f64 {
        if (state.folded_player) |fp| {
            const val0 = game.terminalFoldUtility0(&state, fp);
            return [_]f64{ val0, -val0 };
        }
        if (state.round == .showdown) {
            const val0 = game.showdownUtility0(&state);
            return [_]f64{ val0, -val0 };
        }

        if (state.round == .flop and state.public_card == null) {
            var next = state;
            game.revealBoard(&next);
            return self.cfr(next, reach, chance);
        }

        const player_idx: usize = @intCast(state.current_player);
        var acts: [game.MAX_ACTIONS]game.Action = undefined;
        const n = game.getAvailableActions(&state, &acts);

        const key = game.makeInfoKey(&state, state.current_player);
        const info = try common.getInfo(&self.infosets, key, acts[0..n]);
        const strat = info.strategy();

        var action_utils: [game.MAX_ACTIONS][game.NUM_PLAYERS]f64 = undefined;
        var node_util = [_]f64{ 0, 0 };

        for (acts[0..n], 0..) |a, i| {
            const child_state = game.applyAction(&state, a);
            var next_reach = reach;
            next_reach[player_idx] *= strat[i];
            const child = try self.cfr(child_state, next_reach, chance);
            action_utils[i] = child;
            node_util[0] += strat[i] * child[0];
            node_util[1] += strat[i] * child[1];
        }

        const opp_idx: usize = 1 - player_idx;
        const cf_weight = reach[opp_idx] * chance;
        const info_mut = self.infosets.getPtr(key).?;
        for (0..n) |i| info_mut.regret_sum[i] += cf_weight * (action_utils[i][player_idx] - node_util[player_idx]);

        const strat_weight = reach[player_idx] * chance;
        for (0..n) |i| info_mut.strategy_sum[i] += strat_weight * strat[i];
        return node_util;
    }

    pub fn train(self: *VanillaCFRTrainer, iterations: usize) !f64 {
        const deals = game.allDeals();

        var running: f64 = 0;
        for (0..iterations) |_| {
            // Full-deal sweep: deterministic, low-variance updates.
            for (deals) |hand| {
                const util = try self.cfr(hand, [_]f64{ 1, 1 }, 1.0);
                running += util[0];
            }
        }
        const total_iters: f64 = @floatFromInt(iterations * deals.len);
        return running / total_iters;
    }

    pub fn averageStrategy(self: *const VanillaCFRTrainer, key: game.InfoKey, _: []const game.Action, num_actions: usize) [game.MAX_ACTIONS]f64 {
        return common.avgStrategyOrUniform(&self.infosets, key, num_actions);
    }
};
