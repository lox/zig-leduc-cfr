const std = @import("std");

// Core game definitions shared by all trainers/evaluators.

pub const NUM_PLAYERS = 2;
pub const NUM_RANKS = 3; // J, Q, K
pub const NUM_SUITS = 2; // two copies of each rank
pub const DECK_SIZE = NUM_RANKS * NUM_SUITS;
pub const TOTAL_DEALS = DECK_SIZE * (DECK_SIZE - 1) * (DECK_SIZE - 2); // ordered (p0, p1, board)

pub const MAX_ACTIONS: usize = 3; // never need more than 3 at a node
pub const MAX_STREET_HISTORY: usize = 8; // tiny per-street buffer
pub const MAX_BETS_PER_ROUND: u8 = 2; // bet + (at most) one raise
pub const ANTE_SIZE: i32 = 1;
pub const BetSizes = [_]i32{ 2, 4 }; // fixed-limit bet sizes per round
pub const INVALID_CARD: u8 = 255;

pub const Round = enum(u8) { preflop = 0, flop = 1, showdown = 2 };
pub const Action = enum { check, call, bet, raise, fold };

pub fn rank(card: u8) u8 {
    return card / @as(u8, NUM_SUITS); // 0,1 -> J; 2,3 -> Q; 4,5 -> K
}

pub fn rankChar(r: u8) u8 {
    return switch (r) {
        0 => 'J',
        1 => 'Q',
        2 => 'K',
        else => '?',
    };
}

pub fn actionChar(a: Action) u8 {
    return switch (a) {
        .check => 'k',
        .call => 'c',
        .bet => 'b',
        .raise => 'r',
        .fold => 'f',
    };
}

pub const GameState = struct {
    private_cards: [NUM_PLAYERS]u8,
    public_card: ?u8,
    community_card: u8,
    round: Round,
    current_player: u8,
    contrib: [NUM_PLAYERS]i32,
    bets_in_round: u8,
    actions_in_round: u8,
    preflop_history: [MAX_STREET_HISTORY]u8,
    preflop_len: u8,
    flop_history: [MAX_STREET_HISTORY]u8,
    flop_len: u8,
    folded_player: ?u8,

    pub fn init() GameState {
        return GameState{
            .private_cards = [_]u8{ INVALID_CARD, INVALID_CARD },
            .public_card = null,
            .community_card = INVALID_CARD,
            .round = .preflop,
            .current_player = 0,
            .contrib = [_]i32{ ANTE_SIZE, ANTE_SIZE },
            .bets_in_round = 0,
            .actions_in_round = 0,
            .preflop_history = std.mem.zeroes([MAX_STREET_HISTORY]u8),
            .preflop_len = 0,
            .flop_history = std.mem.zeroes([MAX_STREET_HISTORY]u8),
            .flop_len = 0,
            .folded_player = null,
        };
    }

    pub fn appendAction(self: *GameState, c: u8) void {
        if (self.round == .preflop) {
            if (self.preflop_len < MAX_STREET_HISTORY) {
                self.preflop_history[self.preflop_len] = c;
                self.preflop_len += 1;
            }
        } else {
            if (self.flop_len < MAX_STREET_HISTORY) {
                self.flop_history[self.flop_len] = c;
                self.flop_len += 1;
            }
        }
    }
};

pub const InfoKey = struct {
    player: u8,
    round: Round,
    priv_rank: u8,
    pub_rank_plus1: u8, // 0 = no board yet, else rank+1
    preflop_len: u8,
    preflop_hist: [MAX_STREET_HISTORY]u8,
    flop_len: u8,
    flop_hist: [MAX_STREET_HISTORY]u8,
};

pub fn makeInfoKey(state: *const GameState, player: u8) InfoKey {
    const priv = state.private_cards[@intCast(player)];
    std.debug.assert(priv != INVALID_CARD);
    const priv_rank = rank(priv);
    var pub_rank_plus1: u8 = 0;
    if (state.public_card) |board| pub_rank_plus1 = rank(board) + 1;

    return InfoKey{
        .player = player,
        .round = state.round,
        .priv_rank = priv_rank,
        .pub_rank_plus1 = pub_rank_plus1,
        .preflop_len = state.preflop_len,
        .preflop_hist = state.preflop_history,
        .flop_len = state.flop_len,
        .flop_hist = state.flop_history,
    };
}

pub fn getAvailableActions(state: *const GameState, actions_out: *[MAX_ACTIONS]Action) usize {
    const p_idx: usize = @intCast(state.current_player);
    const opp_idx: usize = 1 - p_idx;
    const to_call: i32 = state.contrib[opp_idx] - state.contrib[p_idx];
    std.debug.assert(to_call >= 0);

    var count: usize = 0;
    if (to_call == 0) {
        actions_out[count] = .check;
        count += 1;
        if (state.bets_in_round < MAX_BETS_PER_ROUND) {
            actions_out[count] = .bet;
            count += 1;
        }
    } else {
        actions_out[count] = .call;
        count += 1;
        if (state.bets_in_round < MAX_BETS_PER_ROUND) {
            actions_out[count] = .raise;
            count += 1;
        }
        actions_out[count] = .fold;
        count += 1;
    }
    return count;
}

pub fn endRound(state: GameState) GameState {
    var next = state;
    next.bets_in_round = 0;
    next.actions_in_round = 0;
    if (state.round == .preflop) {
        next.round = .flop;
        next.public_card = null;
        next.current_player = 0;
        next.flop_len = 0;
    } else if (state.round == .flop) {
        next.round = .showdown;
    }
    return next;
}

pub fn applyAction(state: *const GameState, action: Action) GameState {
    var next = state.*;
    const p_idx: usize = @intCast(state.current_player);
    const opp_idx: usize = 1 - p_idx;
    const to_call: i32 = state.contrib[opp_idx] - state.contrib[p_idx];
    std.debug.assert(to_call >= 0);
    const bet_size: i32 = BetSizes[@intFromEnum(state.round)];

    switch (action) {
        .check => {
            std.debug.assert(to_call == 0);
            next.appendAction('k');
            next.actions_in_round += 1;
            if (state.actions_in_round == 1) next = endRound(next) else next.current_player = @intCast(opp_idx);
        },
        .bet => {
            std.debug.assert(to_call == 0);
            std.debug.assert(state.bets_in_round < MAX_BETS_PER_ROUND);
            next.appendAction('b');
            next.contrib[p_idx] += bet_size;
            next.bets_in_round += 1;
            next.actions_in_round += 1;
            next.current_player = @intCast(opp_idx);
        },
        .call => {
            std.debug.assert(to_call > 0);
            next.appendAction('c');
            next.contrib[p_idx] += to_call;
            next.actions_in_round += 1;
            next = endRound(next);
        },
        .raise => {
            std.debug.assert(to_call > 0);
            std.debug.assert(state.bets_in_round < MAX_BETS_PER_ROUND);
            next.appendAction('r');
            next.contrib[p_idx] += to_call + bet_size;
            next.bets_in_round += 1;
            next.actions_in_round += 1;
            next.current_player = @intCast(opp_idx);
        },
        .fold => {
            std.debug.assert(to_call > 0);
            next.appendAction('f');
            next.actions_in_round += 1;
            next.folded_player = @intCast(p_idx);
        },
    }
    return next;
}

/// Reveal the community card and reset per-round counters for flop play.
pub fn revealBoard(state: *GameState) void {
    state.public_card = state.community_card;
    state.current_player = 0;
    state.bets_in_round = 0;
    state.actions_in_round = 0;
}

pub fn terminalFoldUtility0(state: *const GameState, folded_player: u8) f64 {
    const pot: f64 = @floatFromInt(state.contrib[0] + state.contrib[1]);
    const c0: f64 = @floatFromInt(state.contrib[0]);
    return if (folded_player == 0) -c0 else pot - c0;
}

/// Enumerate all ordered deals (p0 card, p1 card, board), returning a fixed array.
pub fn allDeals() [TOTAL_DEALS]GameState {
    var deals: [TOTAL_DEALS]GameState = undefined;
    var idx: usize = 0;
    for (0..DECK_SIZE) |c0| {
        for (0..DECK_SIZE) |c1| {
            if (c1 == c0) continue;
            for (0..DECK_SIZE) |board| {
                if (board == c0 or board == c1) continue;
                var s = GameState.init();
                s.private_cards[0] = @intCast(c0);
                s.private_cards[1] = @intCast(c1);
                s.community_card = @intCast(board);
                deals[idx] = s;
                idx += 1;
            }
        }
    }
    return deals;
}

pub fn showdownUtility0(state: *const GameState) f64 {
    std.debug.assert(state.public_card != null);
    const pot: f64 = @floatFromInt(state.contrib[0] + state.contrib[1]);
    const c0: f64 = @floatFromInt(state.contrib[0]);
    const r0 = rank(state.private_cards[0]);
    const r1 = rank(state.private_cards[1]);
    const board = rank(state.public_card.?);
    const p0_pair = r0 == board;
    const p1_pair = r1 == board;
    if (p0_pair and !p1_pair) return pot - c0;
    if (p1_pair and !p0_pair) return -c0;
    if (r0 > r1) return pot - c0;
    if (r1 > r0) return -c0;
    return pot / 2.0 - c0;
}
