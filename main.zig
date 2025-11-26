// main.zig — CLI Entry Point
// ===========================
//
// Trains a CFR strategy on Leduc Hold'em and evaluates convergence.
//
// Usage:
//   ./zig-out/bin/leduc-cfr [--algo=vanilla|mccfr] [--max-iters=N] [--no-self-play]
//
// Example:
//   ./bin/zig build run -Doptimize=ReleaseFast -- --algo=mccfr --max-iters=50000

const std = @import("std");
const vanilla = @import("src/leduc/cfr_vanilla.zig");
const vanilla_iter = @import("src/leduc/cfr_vanilla_iter.zig");
const mccfr = @import("src/leduc/cfr_mccfr.zig");
const cfr_plus = @import("src/leduc/cfr_plus.zig");
const play = @import("src/leduc/play.zig");

const Algorithm = enum { vanilla, vanilla_iter, mccfr, cfr_plus };

const Config = struct {
    algo: Algorithm = .vanilla,
    max_iters: usize = 100_000,
    no_self_play: bool = false,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = try parseArgs(allocator);
    const sweep = generateSweep(config.max_iters);

    switch (config.algo) {
        .vanilla => try runSweep(vanilla.VanillaCFRTrainer, allocator, sweep.items(), "Vanilla", config.no_self_play),
        .vanilla_iter => try runSweep(vanilla_iter.VanillaIterativeTrainer, allocator, sweep.items(), "Vanilla (Iterative)", config.no_self_play),
        .mccfr => try runSweep(mccfr.MCCFRTrainer, allocator, sweep.items(), "MCCFR", config.no_self_play),
        .cfr_plus => try runSweep(cfr_plus.CFRPlusTrainer, allocator, sweep.items(), "CFR+", config.no_self_play),
    }
}

// Removed ArrayList dependency to work around build issue
fn parseArgs(allocator: std.mem.Allocator) !Config {
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.next(); // skip program name
    
    var config = Config{};
    
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--algo=vanilla")) {
            config.algo = .vanilla;
        } else if (std.mem.eql(u8, arg, "--algo=vanilla_iter")) {
            config.algo = .vanilla_iter;
        } else if (std.mem.eql(u8, arg, "--algo=mccfr")) {
            config.algo = .mccfr;
        } else if (std.mem.eql(u8, arg, "--algo=cfr_plus")) {
            config.algo = .cfr_plus;
        } else if (std.mem.startsWith(u8, arg, "--max-iters=")) {
            const value = arg["--max-iters=".len..];
            config.max_iters = std.fmt.parseInt(usize, value, 10) catch {
                std.debug.print("Invalid --max-iters value: {s}\n", .{value});
                return error.InvalidArgument;
            };
        } else if (std.mem.eql(u8, arg, "--no-self-play")) {
            config.no_self_play = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            std.process.exit(0);
        } else {
            std.debug.print("Unknown argument: {s}\n", .{arg});
            printUsage();
            return error.InvalidArgument;
        }
    }

    return config;
}

// Removed parseArgsSlice as we parse directly now


fn printUsage() void {
    std.debug.print(
        \\Usage: leduc-cfr [OPTIONS]
        \\
        \\Options:
        \\  --algo=vanilla|vanilla_iter|mccfr|cfr_plus  CFR variant (default: vanilla)
        \\  --max-iters=N                               Maximum iterations (default: 100000)
        \\  --no-self-play                              Skip vs-base evaluations; report exploitability only
        \\  --help, -h                                  Show this help
        \\
    , .{});
}

/// Generate iteration counts: 100, 300, 1000, 3000, ... up to max_iters.
/// Pattern: 1, 3, 10, 30, 100, ... (alternating ×3 and ×10/3)
fn generateSweep(max_iters: usize) Sweep {
    var sweep = Sweep{};
    var n: usize = 100;

    while (n <= max_iters) {
        sweep.push(n);
        if (n * 3 <= max_iters and sweep.len < Sweep.MAX) {
            n *= 3;
            sweep.push(n);
        }
        n = n / 3 * 10; // 100→1000, 300→3000, etc.
    }

    return sweep;
}

const Sweep = struct {
    const MAX = 16;
    data: [MAX]usize = undefined,
    len: usize = 0,

    fn push(self: *Sweep, val: usize) void {
        if (self.len < MAX) {
            self.data[self.len] = val;
            self.len += 1;
        }
    }

    fn items(self: *const Sweep) []const usize {
        return self.data[0..self.len];
    }
};

fn runSweep(comptime Trainer: type, allocator: std.mem.Allocator, sweep: []const usize, label: []const u8, no_self_play: bool) !void {
    std.debug.print(
        \\Leduc CFR trainer - {s}
        \\Iterations: {any}
        \\
    , .{ label, sweep });

    if (no_self_play) {
        std.debug.print("{s:>10}  {s:>8}  {s:>10}  {s:>8}\n", .{ "Iters", "Value", "Exploit", "mbb/g" });

        for (sweep) |iters| {
            var trainer = Trainer.init(allocator);
            defer trainer.deinit();

            const value = try trainer.train(iters);
            const exploit = try play.exploitability(Trainer, &trainer);
            const mbb = exploit * 500.0;

            std.debug.print("{d:>10}  {d:>8.4}  {d:>10.5}  {d:>8.2}\n", .{ iters, value, exploit, mbb });
        }

        return;
    }

    // Train a "base" strategy at max iterations for comparison
    const base_iters = sweep[sweep.len - 1];
    var base = Trainer.init(allocator);
    defer base.deinit();
    const base_value = try base.train(base_iters);

    std.debug.print("\nBase ({d} iters) game value: {d:.5}\n\n", .{ base_iters, base_value });
    std.debug.print("{s:>10}  {s:>8}  {s:>8}  {s:>8}  {s:>8}\n", .{
        "Iters", "Value", "vs Base", "Base vs", "Exploit",
    });

    for (sweep) |iters| {
        var trainer = Trainer.init(allocator);
        defer trainer.deinit();
        const value = try trainer.train(iters);

        const vs_base = play.headToHead(Trainer, &trainer, Trainer, &base);
        const base_vs = play.headToHead(Trainer, &base, Trainer, &trainer);
        const exploit = try play.exploitability(Trainer, &trainer);
        const mbb = exploit * 500.0; // 1 bb = 2 chips, so mbb = (exploit / 2) * 1000

        std.debug.print("{d:>10}  {d:>8.4}  {d:>8.4}  {d:>8.4}  {d:>8.5}  {d:>8.2}\n", .{
            iters, value, vs_base, base_vs, exploit, mbb,
        });
    }
}

// Removed parseArgsSlice test as we removed the function
// test "parseArgsSlice supports --no-self-play" {
//     const config = try parseArgsSlice(&.{"--no-self-play"});
//     try std.testing.expect(config.no_self_play);
// }

test "runSweep skips base trainer when self-play disabled" {
    const game = @import("src/leduc/game.zig");

    const TestTrainer = struct {
        pub var init_calls: usize = 0;

        pub fn init(_: std.mem.Allocator) @This() {
            init_calls += 1;
            return .{};
        }

        pub fn deinit(_: *@This()) void {}

        pub fn train(_: *@This(), _: usize) !f64 {
            return 0;
        }

        pub fn getStrategy(_: *const @This(), _: game.InfoKey, num_actions: usize) [game.MAX_ACTIONS]f64 {
            _ = num_actions;
            var strat: [game.MAX_ACTIONS]f64 = .{ 0, 0, 0 };
            strat[0] = 1;
            return strat;
        }
    };

    const sweep = [_]usize{10};
    TestTrainer.init_calls = 0;

    try runSweep(TestTrainer, std.testing.allocator, sweep[0..], "Test", true);

    try std.testing.expectEqual(@as(usize, sweep.len), TestTrainer.init_calls);
}
