// game.zig — Leduc Hold'em Game Logic
// ====================================
//
// This file defines the rules of Leduc Hold'em, a tiny poker game used to
// study game-theoretic concepts. It's small enough to solve exactly, yet
// captures the essential structure of poker: private information, betting,
// and bluffing.
//
// Leduc rules:
//   - 6-card deck: J, Q, K with 2 suits each
//   - 2 players, 1 chip ante each
//   - 2 betting rounds: preflop (hidden cards only), flop (after board revealed)
//   - Fixed-limit betting: 2 chips preflop, 4 chips on flop
//   - Max 2 bets per round (bet + raise, or check-raise)
//   - Showdown: pair with board beats high card; high card wins ties
//
// The game tree has ~936 states and ~288 information sets.

const std = @import("std");

// =============================================================================
// Constants
// =============================================================================

pub const NUM_PLAYERS = 2;
pub const NUM_RANKS = 3; // J=0, Q=1, K=2
pub const NUM_SUITS = 2; // Two copies of each rank
pub const DECK_SIZE = NUM_RANKS * NUM_SUITS; // 6 cards total

pub const MAX_ACTIONS: usize = 3; // At most: fold, call, raise (or check, bet)
pub const MAX_BETS_PER_ROUND: u8 = 2; // bet + one raise maximum
pub const MAX_HISTORY_LEN: usize = 8; // Actions per street (plenty of room)

pub const ANTE: i32 = 1;
pub const BET_SIZES = [_]i32{ 2, 4 }; // Preflop=2, flop=4

// Total possible deals: 6 choices for P0 × 5 for P1 × 4 for board = 120
pub const TOTAL_DEALS = DECK_SIZE * (DECK_SIZE - 1) * (DECK_SIZE - 2);

// =============================================================================
// Cards
// =============================================================================

pub const Card = u8;
pub const NO_CARD: Card = 255;

/// Extract rank from card. Cards 0,1 are Jacks; 2,3 are Queens; 4,5 are Kings.
pub fn cardRank(card: Card) u8 {
    return card / NUM_SUITS;
}

pub fn rankToChar(r: u8) u8 {
    return switch (r) {
        0 => 'J',
        1 => 'Q',
        2 => 'K',
        else => '?',
    };
}

// =============================================================================
// Actions
// =============================================================================

pub const Action = enum {
    check,
    bet,
    call,
    raise,
    fold,

    pub fn toChar(self: Action) u8 {
        return switch (self) {
            .check => 'k',
            .bet => 'b',
            .call => 'c',
            .raise => 'r',
            .fold => 'f',
        };
    }
};

// =============================================================================
// Betting Round
// =============================================================================

pub const Round = enum(u8) {
    preflop = 0,
    flop = 1,
    showdown = 2,

    pub fn betSize(self: Round) i32 {
        return BET_SIZES[@intFromEnum(self)];
    }
};

// =============================================================================
// Game State
// =============================================================================

pub const GameState = struct {
    // Cards
    hole_cards: [NUM_PLAYERS]Card, // Each player's private card
    board_card: Card, // The community card (revealed on flop)
    board_revealed: bool, // Has the board been shown?

    // Betting state
    round: Round,
    current_player: u8,
    pot: [NUM_PLAYERS]i32, // How much each player has contributed
    bets_this_round: u8, // Number of bets/raises so far this round
    actions_this_round: u8, // Total actions this round (for round-ending logic)

    // Action history (for information set keys)
    preflop_history: [MAX_HISTORY_LEN]u8,
    preflop_len: u8,
    flop_history: [MAX_HISTORY_LEN]u8,
    flop_len: u8,

    // Terminal state
    folded_player: ?u8,

    /// Create the initial state for a new hand.
    /// Cards must be set separately via deal().
    pub fn init() GameState {
        return .{
            .hole_cards = .{ NO_CARD, NO_CARD },
            .board_card = NO_CARD,
            .board_revealed = false,
            .round = .preflop,
            .current_player = 0,
            .pot = .{ ANTE, ANTE },
            .bets_this_round = 0,
            .actions_this_round = 0,
            .preflop_history = std.mem.zeroes([MAX_HISTORY_LEN]u8),
            .preflop_len = 0,
            .flop_history = std.mem.zeroes([MAX_HISTORY_LEN]u8),
            .flop_len = 0,
            .folded_player = null,
        };
    }

    /// Deal cards to create a complete starting state.
    pub fn deal(p0_card: Card, p1_card: Card, board: Card) GameState {
        var state = init();
        state.hole_cards[0] = p0_card;
        state.hole_cards[1] = p1_card;
        state.board_card = board;
        return state;
    }

    // -------------------------------------------------------------------------
    // Queries
    // -------------------------------------------------------------------------

    pub fn isTerminal(self: *const GameState) bool {
        return self.folded_player != null or self.round == .showdown;
    }

    pub fn amountToCall(self: *const GameState) i32 {
        const player = self.current_player;
        const opponent: usize = 1 - @as(usize, player);
        return self.pot[opponent] - self.pot[player];
    }

    pub fn canBetOrRaise(self: *const GameState) bool {
        return self.bets_this_round < MAX_BETS_PER_ROUND;
    }

    /// Get the legal actions from this state.
    /// Returns the number of actions written to the output buffer.
    pub fn legalActions(self: *const GameState, out: *[MAX_ACTIONS]Action) usize {
        const to_call = self.amountToCall();
        std.debug.assert(to_call >= 0);

        var n: usize = 0;

        if (to_call == 0) {
            // No bet to face: can check or bet
            out[n] = .check;
            n += 1;
            if (self.canBetOrRaise()) {
                out[n] = .bet;
                n += 1;
            }
        } else {
            // Facing a bet: can call, raise, or fold
            out[n] = .call;
            n += 1;
            if (self.canBetOrRaise()) {
                out[n] = .raise;
                n += 1;
            }
            out[n] = .fold;
            n += 1;
        }
        return n;
    }

    // -------------------------------------------------------------------------
    // State Transitions
    // -------------------------------------------------------------------------

    /// Apply an action and return the resulting state.
    pub fn apply(self: *const GameState, action: Action) GameState {
        var next = self.*;
        const player: usize = self.current_player;
        const opponent: usize = 1 - player;

        // Record action in history
        next.recordAction(action.toChar());
        next.actions_this_round += 1;

        switch (action) {
            .check => {
                std.debug.assert(self.amountToCall() == 0);
                // Second check ends the round
                if (self.actions_this_round == 1) {
                    next.endRound();
                } else {
                    next.current_player = @intCast(opponent);
                }
            },
            .bet => {
                std.debug.assert(self.amountToCall() == 0);
                std.debug.assert(self.canBetOrRaise());
                next.pot[player] += self.round.betSize();
                next.bets_this_round += 1;
                next.current_player = @intCast(opponent);
            },
            .call => {
                std.debug.assert(self.amountToCall() > 0);
                next.pot[player] += self.amountToCall();
                next.endRound();
            },
            .raise => {
                std.debug.assert(self.amountToCall() > 0);
                std.debug.assert(self.canBetOrRaise());
                next.pot[player] += self.amountToCall() + self.round.betSize();
                next.bets_this_round += 1;
                next.current_player = @intCast(opponent);
            },
            .fold => {
                std.debug.assert(self.amountToCall() > 0);
                next.folded_player = @intCast(player);
            },
        }
        return next;
    }

    /// Reveal the board card (transition from preflop to flop betting).
    pub fn revealBoard(self: *GameState) void {
        self.board_revealed = true;
        self.current_player = 0;
        self.bets_this_round = 0;
        self.actions_this_round = 0;
    }

    fn endRound(self: *GameState) void {
        self.bets_this_round = 0;
        self.actions_this_round = 0;

        if (self.round == .preflop) {
            self.round = .flop;
            self.board_revealed = false; // Will be revealed by caller
            self.current_player = 0;
            self.flop_len = 0;
        } else {
            self.round = .showdown;
        }
    }

    fn recordAction(self: *GameState, char: u8) void {
        if (self.round == .preflop) {
            if (self.preflop_len < MAX_HISTORY_LEN) {
                self.preflop_history[self.preflop_len] = char;
                self.preflop_len += 1;
            }
        } else {
            if (self.flop_len < MAX_HISTORY_LEN) {
                self.flop_history[self.flop_len] = char;
                self.flop_len += 1;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Payoffs
    // -------------------------------------------------------------------------

    /// Compute player 0's payoff when someone folds.
    pub fn foldPayoff(self: *const GameState, folder: u8) f64 {
        const p0_invested: f64 = @floatFromInt(self.pot[0]);
        if (folder == 0) {
            return -p0_invested; // P0 folded, loses their chips
        } else {
            const total_pot: f64 = @floatFromInt(self.pot[0] + self.pot[1]);
            return total_pot - p0_invested; // P0 wins the pot
        }
    }

    /// Compute player 0's payoff at showdown.
    /// Hand ranking: pair with board > high card > low card.
    pub fn showdownPayoff(self: *const GameState) f64 {
        std.debug.assert(self.board_revealed);

        const p0_invested: f64 = @floatFromInt(self.pot[0]);
        const total_pot: f64 = @floatFromInt(self.pot[0] + self.pot[1]);

        const p0_rank = cardRank(self.hole_cards[0]);
        const p1_rank = cardRank(self.hole_cards[1]);
        const board_rank = cardRank(self.board_card);

        const p0_paired = (p0_rank == board_rank);
        const p1_paired = (p1_rank == board_rank);

        // Pair beats no pair; otherwise high card wins
        const p0_wins = (p0_paired and !p1_paired) or
            (p0_paired == p1_paired and p0_rank > p1_rank);
        const p1_wins = (p1_paired and !p0_paired) or
            (p0_paired == p1_paired and p1_rank > p0_rank);

        if (p0_wins) return total_pot - p0_invested;
        if (p1_wins) return -p0_invested;
        return total_pot / 2.0 - p0_invested; // Split pot
    }

    // -------------------------------------------------------------------------
    // Information Set Key
    // -------------------------------------------------------------------------

    /// Build the information set key for a player at this state.
    /// This captures everything the player knows: their card, the board (if shown),
    /// and the complete betting history.
    pub fn infoKey(self: *const GameState, player: u8) InfoKey {
        const hole_card = self.hole_cards[player];
        std.debug.assert(hole_card != NO_CARD);

        return .{
            .player = player,
            .round = self.round,
            .hole_rank = cardRank(hole_card),
            .board_rank_plus1 = if (self.board_revealed) cardRank(self.board_card) + 1 else 0,
            .preflop_len = self.preflop_len,
            .flop_len = self.flop_len,
            .preflop_history = self.preflop_history,
            .flop_history = self.flop_history,
        };
    }
};

// =============================================================================
// Information Set Key
// =============================================================================

/// Everything a player knows at a decision point.
/// Two game states with the same InfoKey are indistinguishable to that player.
pub const InfoKey = struct {
    player: u8,
    round: Round,
    hole_rank: u8,
    board_rank_plus1: u8,
    preflop_len: u8,
    flop_len: u8,
    preflop_history: [MAX_HISTORY_LEN]u8,
    flop_history: [MAX_HISTORY_LEN]u8,

    /// Custom hash context for use with std.HashMap.
    ///
    /// We define explicit hash/eql functions rather than relying on AutoHashMap because:
    ///
    /// 1. AutoHashMap hashes raw struct bytes, including any padding the compiler
    ///    may insert between fields. Padding bytes are uninitialized and can contain
    ///    arbitrary values, causing the same logical key to hash differently across
    ///    calls — especially in ReleaseFast where memory isn't zero-initialized.
    ///
    /// 2. We only hash the meaningful portion of the history arrays (0..len) rather
    ///    than the full fixed-size arrays, which is both correct and more efficient.
    ///
    /// 3. Explicit hashing is self-documenting: it shows exactly which fields
    ///    contribute to key identity.
    pub const HashContext = struct {
        pub fn hash(_: HashContext, key: InfoKey) u64 {
            var h = std.hash.Wyhash.init(0);
            h.update(&.{ key.player, @intFromEnum(key.round), key.hole_rank, key.board_rank_plus1 });
            h.update(key.preflop_history[0..key.preflop_len]);
            h.update(key.flop_history[0..key.flop_len]);
            return h.final();
        }

        pub fn eql(_: HashContext, a: InfoKey, b: InfoKey) bool {
            return a.player == b.player and
                a.round == b.round and
                a.hole_rank == b.hole_rank and
                a.board_rank_plus1 == b.board_rank_plus1 and
                a.preflop_len == b.preflop_len and
                a.flop_len == b.flop_len and
                std.mem.eql(u8, a.preflop_history[0..a.preflop_len], b.preflop_history[0..b.preflop_len]) and
                std.mem.eql(u8, a.flop_history[0..a.flop_len], b.flop_history[0..b.flop_len]);
        }
    };
};

// =============================================================================
// Deal Enumeration
// =============================================================================

/// Generate all possible deals (P0 card, P1 card, board card).
/// Used for full-tree CFR which iterates over every possible deal.
pub fn allDeals() [TOTAL_DEALS]GameState {
    var deals: [TOTAL_DEALS]GameState = undefined;
    var i: usize = 0;

    for (0..DECK_SIZE) |p0| {
        for (0..DECK_SIZE) |p1| {
            if (p1 == p0) continue;
            for (0..DECK_SIZE) |board| {
                if (board == p0 or board == p1) continue;
                deals[i] = GameState.deal(@intCast(p0), @intCast(p1), @intCast(board));
                i += 1;
            }
        }
    }
    return deals;
}
