// cfr_vanilla_iter.zig — Iterative Vanilla CFR
// ============================================
//
// An iterative (stack-based) implementation of Vanilla CFR.
//
// Standard recursive CFR can hit stack depth limits for very deep games.
// This implementation manages its own stack on the heap, allowing for arbitrarily
// deep game trees (limited only by heap memory).
//
// It performs exactly the same logic as `cfr_vanilla.zig`.

const std = @import("std");
const game = @import("game.zig");
const cfr = @import("cfr.zig");

const Payoffs = [game.NUM_PLAYERS]f64;

const StackItem = struct {
    state: game.GameState,
    reach_prob: Payoffs,

    // Decision node state
    is_decision: bool = false,
    actions: [game.MAX_ACTIONS]game.Action = undefined,
    num_actions: usize = 0,
    strategy: [game.MAX_ACTIONS]f64 = undefined,
    action_payoffs: [game.MAX_ACTIONS]Payoffs = undefined,
    expected_payoff: Payoffs = .{ 0, 0 },

    // Child iterator
    action_idx: usize = 0,
};

pub const VanillaIterativeTrainer = struct {
    allocator: std.mem.Allocator,
    infosets: cfr.InfoMap,

    pub fn init(allocator: std.mem.Allocator) VanillaIterativeTrainer {
        return .{ .allocator = allocator, .infosets = cfr.InfoMap.init(allocator) };
    }

    pub fn deinit(self: *VanillaIterativeTrainer) void {
        self.infosets.deinit();
    }

    /// Iterative traversal using an explicit stack
    fn traverse(self: *VanillaIterativeTrainer, root_state: game.GameState, root_reach: Payoffs) !Payoffs {
        var stack = std.ArrayListUnmanaged(StackItem){};
        defer stack.deinit(self.allocator);

        try stack.append(self.allocator, .{
            .state = root_state,
            .reach_prob = root_reach,
        });

        var last_payoff: Payoffs = .{ 0, 0 };

        while (stack.items.len > 0) {
            // Always access top by index because append/pop invalidates pointers
            const top_idx = stack.items.len - 1;
            
            // We can take a pointer, but must not use it after any stack operation (push/pop)
            // until we re-fetch it.
            var top = &stack.items[top_idx];

            // ===== STATE 1: First Visit =====
            if (!top.is_decision and top.action_idx == 0) {
                // 1. Terminal checks
                if (top.state.folded_player) |folder| {
                    const p = top.state.foldPayoff(folder);
                    last_payoff = .{ p, -p };
                    _ = stack.pop();
                    continue;
                }
                if (top.state.round == .showdown) {
                    const p = top.state.showdownPayoff();
                    last_payoff = .{ p, -p };
                    _ = stack.pop();
                    continue;
                }

                // 2. Chance Node
                if (top.state.round == .flop and !top.state.board_revealed) {
                    // Optimization: Mutate state in-place and continue loop (Tail Call)
                    top.state.revealBoard();
                    continue;
                }

                // 3. Decision Node Setup
                top.is_decision = true;
                const player = top.state.current_player;
                top.num_actions = top.state.legalActions(&top.actions);

                const key = top.state.infoKey(player);
                // Note: getOrCreateInfoSet might resize map, but that's fine, we don't hold pointers to it here.
                const infoset = try cfr.getOrCreateInfoSet(&self.infosets, key, top.actions[0..top.num_actions]);
                top.strategy = infoset.currentStrategy();
            }

            // ===== STATE 2: Processing Children & Post-Order Updates =====
            // Note: We fall through from STATE 1 to here immediately if it was a decision node.
            
            if (top.is_decision) {
                // If we just returned from a child, record its payoff
                if (top.action_idx > 0) {
                    const prev_idx = top.action_idx - 1;
                    top.action_payoffs[prev_idx] = last_payoff;
                    
                    const weight = top.strategy[prev_idx];
                    top.expected_payoff[0] += weight * last_payoff[0];
                    top.expected_payoff[1] += weight * last_payoff[1];
                }

                // If there are more children to visit, push the next one
                if (top.action_idx < top.num_actions) {
                    const idx = top.action_idx;
                    const action = top.actions[idx];
                    const child_state = top.state.apply(action);

                    var child_reach = top.reach_prob;
                    child_reach[top.state.current_player] *= top.strategy[idx];

                    // Increment index for next time we return to this frame
                    stack.items[top_idx].action_idx += 1;

                    try stack.append(self.allocator, .{
                        .state = child_state,
                        .reach_prob = child_reach,
                    });
                    continue;
                }

                // All children visited. Update Regrets & Strategy.
                const player = top.state.current_player;
                const opponent = 1 - player;
                const key = top.state.infoKey(player);

                // Re-fetch pointer (map might have resized during children processing)
                const infoset_mut = self.infosets.getPtr(key).?;

                for (0..top.num_actions) |i| {
                    const regret = top.action_payoffs[i][player] - top.expected_payoff[player];
                    infoset_mut.cumulative_regret[i] += top.reach_prob[opponent] * regret;
                    infoset_mut.cumulative_strategy[i] += top.reach_prob[player] * top.strategy[i];
                }

                last_payoff = top.expected_payoff;
                _ = stack.pop();
            }
        }
        return last_payoff;
    }

    pub fn train(self: *VanillaIterativeTrainer, iterations: usize) !f64 {
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

    pub fn getStrategy(self: *const VanillaIterativeTrainer, key: game.InfoKey, num_actions: usize) [game.MAX_ACTIONS]f64 {
        if (self.infosets.get(key)) |infoset| {
            return infoset.averageStrategy();
        }
        var uniform: [game.MAX_ACTIONS]f64 = .{ 0, 0, 0 };
        for (0..num_actions) |i| uniform[i] = 1.0 / @as(f64, @floatFromInt(num_actions));
        return uniform;
    }
};
