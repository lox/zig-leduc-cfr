// cfr_vanilla.zig — Full-Tree CFR
// ================================
//
// This is the simplest CFR implementation: we traverse the *entire* game tree
// on every iteration. No sampling, no shortcuts—just pure recursive tree walk.
//
// For each deal (player 0's card, player 1's card, board card), we:
//   1. Recurse through all possible action sequences
//   2. At each decision point, compute the current strategy via regret matching
//   3. Back-propagate payoffs and update cumulative regrets
//
// The algorithm has three types of nodes:
//   - **Terminal**: Game is over (fold or showdown). Return the payoffs.
//   - **Chance**: Reveal the board card. (Deterministic for a given deal.)
//   - **Decision**: A player chooses an action. This is where CFR magic happens.
//
// At decision nodes, we update two quantities:
//   - **Cumulative Regret**: How much better each action would have been,
//     weighted by opponent's probability of reaching this state.
//   - **Cumulative Strategy**: The strategy we played, weighted by our own
//     probability of reaching this state.
//
// Why opponent reach for regret? Because counterfactual regret asks:
// "If I had always played action a here, how much better would I do?"
// The opponent's strategy (and chance) determines how often this matters.
//
// Why own reach for strategy? Because we're averaging over states we actually
// visit, weighted by how likely we are to get there under our own policy.

const std = @import("std");
const game = @import("game.zig");
const cfr = @import("cfr.zig");

const Payoffs = [game.NUM_PLAYERS]f64;

pub const VanillaCFRTrainer = struct {
    allocator: std.mem.Allocator,
    infosets: cfr.InfoMap,

    pub fn init(allocator: std.mem.Allocator) VanillaCFRTrainer {
        return .{ .allocator = allocator, .infosets = cfr.InfoMap.init(allocator) };
    }

    pub fn deinit(self: *VanillaCFRTrainer) void {
        self.infosets.deinit();
    }

    /// The core CFR recursion. Returns payoff for each player from this state.
    ///
    /// reach_prob[p] = probability of reaching this state under player p's strategy
    /// (opponent reach is used for regret weighting, own reach for strategy averaging)
    fn traverse(self: *VanillaCFRTrainer, state: game.GameState, reach_prob: Payoffs) !Payoffs {

        // ===== TERMINAL NODES =====
        // Game is over—return the payoffs.
        if (state.folded_player) |folder| {
            const payoff_p0 = state.foldPayoff(folder);
            return .{ payoff_p0, -payoff_p0 };
        }
        if (state.round == .showdown) {
            const payoff_p0 = state.showdownPayoff();
            return .{ payoff_p0, -payoff_p0 };
        }

        // ===== CHANCE NODE =====
        // Reveal the board card. For a given deal, this is deterministic.
        if (state.round == .flop and !state.board_revealed) {
            var next = state;
            next.revealBoard();
            return self.traverse(next, reach_prob);
        }

        // ===== DECISION NODE =====
        // Current player chooses among legal actions.
        const player: usize = @intCast(state.current_player);
        const opponent: usize = 1 - player;

        var actions: [game.MAX_ACTIONS]game.Action = undefined;
        const num_actions = state.legalActions(&actions);

        const key = state.infoKey(state.current_player);
        const infoset = try cfr.getOrCreateInfoSet(&self.infosets, key, actions[0..num_actions]);
        const strategy = infoset.currentStrategy();

        // Recurse for each action, tracking payoff
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
        // Regret = (value of this action) - (value of current strategy)
        // Weight by opponent reach: "how much does deviating here matter?"
        const infoset_mut = self.infosets.getPtr(key).?;
        for (0..num_actions) |i| {
            const regret = action_payoffs[i][player] - expected_payoff[player];
            infoset_mut.cumulative_regret[i] += reach_prob[opponent] * regret;
        }

        // ===== UPDATE AVERAGE STRATEGY =====
        // Weight by own reach: "how often do we actually get here?"
        for (0..num_actions) |i| {
            infoset_mut.cumulative_strategy[i] += reach_prob[player] * strategy[i];
        }

        return expected_payoff;
    }

    /// Train for the given number of iterations.
    /// Each iteration traverses all possible deals (full-tree, no sampling).
    pub fn train(self: *VanillaCFRTrainer, iterations: usize) !f64 {
        const deals = game.allDeals();
        var total_payoff: f64 = 0;

        for (0..iterations) |_| {
            for (deals) |deal| {
                const payoff = try self.traverse(deal, .{ 1, 1 });
                total_payoff += payoff[0];
            }
        }

        const num_traversals: f64 = @floatFromInt(iterations * deals.len);
        return total_payoff / num_traversals;
    }

    /// Look up the average (Nash-converging) strategy for an information set.
    ///
    /// This is used during evaluation to see how the trained policy plays.
    /// Returns uniform distribution if this info set was never visited during training.
    pub fn getStrategy(self: *const VanillaCFRTrainer, key: game.InfoKey, num_actions: usize) [game.MAX_ACTIONS]f64 {
        if (self.infosets.get(key)) |infoset| {
            return infoset.averageStrategy();
        }
        // Never visited this state during training—play uniformly.
        var uniform: [game.MAX_ACTIONS]f64 = .{ 0, 0, 0 };
        for (0..num_actions) |i| uniform[i] = 1.0 / @as(f64, @floatFromInt(num_actions));
        return uniform;
    }
};
