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
const builtin = @import("builtin");
const vanilla = @import("src/leduc/cfr_vanilla.zig");
const mccfr = @import("src/leduc/cfr_mccfr.zig");
const cfr_plus = @import("src/leduc/cfr_plus.zig");
const dcfr = @import("src/leduc/cfr_dcfr.zig");
const play = @import("src/leduc/play.zig");

const Algorithm = enum { vanilla, mccfr, cfr_plus, lcfr, dcfr };

const Config = struct {
    algo: Algorithm = .vanilla,
    max_iters: usize = 100_000,
    no_self_play: bool = false,
    threshold_mbb: f64 = 100.0,
    seed: ?u64 = null,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = try parseArgs(allocator);
    const sweep = generateSweep(config.max_iters);

    switch (config.algo) {
        .vanilla => try runSweep(vanilla.VanillaCFRTrainer, void, allocator, {}, sweep.items(), "Vanilla", config.no_self_play, config.threshold_mbb),
        .mccfr => try runSweep(mccfr.MCCFRTrainer, u64, allocator, resolveSeed(config.seed), sweep.items(), "MCCFR", config.no_self_play, config.threshold_mbb),
        .cfr_plus => try runSweep(cfr_plus.CFRPlusTrainer, void, allocator, {}, sweep.items(), "CFR+", config.no_self_play, config.threshold_mbb),
        .lcfr => try runSweep(dcfr.DCFRTrainer, dcfr.DiscountScheme, allocator, .lcfr, sweep.items(), "LCFR", config.no_self_play, config.threshold_mbb),
        .dcfr => try runSweep(dcfr.DCFRTrainer, dcfr.DiscountScheme, allocator, .dcfr, sweep.items(), "DCFR", config.no_self_play, config.threshold_mbb),
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
        } else if (std.mem.eql(u8, arg, "--algo=mccfr")) {
            config.algo = .mccfr;
        } else if (std.mem.eql(u8, arg, "--algo=cfr_plus")) {
            config.algo = .cfr_plus;
        } else if (std.mem.eql(u8, arg, "--algo=lcfr")) {
            config.algo = .lcfr;
        } else if (std.mem.eql(u8, arg, "--algo=dcfr")) {
            config.algo = .dcfr;
        } else if (std.mem.startsWith(u8, arg, "--max-iters=")) {
            const value = arg["--max-iters=".len..];
            config.max_iters = std.fmt.parseInt(usize, value, 10) catch {
                std.debug.print("Invalid --max-iters value: {s}\n", .{value});
                return error.InvalidArgument;
            };
        } else if (std.mem.eql(u8, arg, "--no-self-play")) {
            config.no_self_play = true;
        } else if (std.mem.startsWith(u8, arg, "--threshold=")) {
            const value = arg["--threshold=".len..];
            config.threshold_mbb = std.fmt.parseFloat(f64, value) catch {
                std.debug.print("Invalid --threshold value: {s}\n", .{value});
                return error.InvalidArgument;
            };
        } else if (std.mem.startsWith(u8, arg, "--seed=")) {
            const value = arg["--seed=".len..];
            config.seed = std.fmt.parseInt(u64, value, 10) catch {
                std.debug.print("Invalid --seed value: {s}\n", .{value});
                return error.InvalidArgument;
            };
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


fn validateMonotonicity(exploits: []const f64, threshold_mbb: f64) !void {
    if (exploits.len < 2) return;

    var violations: usize = 0;
    for (1..exploits.len) |i| {
        if (exploits[i] > exploits[i - 1] * 1.1) { // Allow 10% tolerance
            violations += 1;
        }
    }

    const final_exploit = exploits[exploits.len - 1];
    const final_mbb = final_exploit * 500.0;

    if (violations > exploits.len / 2 or final_mbb > threshold_mbb) {
        std.debug.print("\nVALIDATION FAILED: exploitability not converging\n", .{});
        std.debug.print("  Monotonicity violations: {d}/{d}\n", .{ violations, exploits.len - 1 });
        std.debug.print("  Final exploitability: {d:.2} mbb/g (threshold: {d:.0})\n", .{ final_mbb, threshold_mbb });
        std.process.exit(1);
    }

    std.debug.print("Validation: OK (final {d:.2} mbb/g, threshold: {d:.0})\n", .{ final_mbb, threshold_mbb });
}

fn printUsage() void {
    std.debug.print(
        \\Usage: leduc-cfr [OPTIONS]
        \\
        \\Options:
        \\  --algo=ALGO      CFR variant (default: vanilla)
        \\                   vanilla, mccfr, cfr_plus, lcfr, dcfr
        \\  --max-iters=N    Maximum iterations (default: 100000)
        \\  --no-self-play   Skip vs-base evaluations; report exploitability only
        \\  --threshold=N    Max exploitability in mbb/g (default: 100)
        \\  --seed=N         Fixed RNG seed for MCCFR (default: random)
        \\  --help, -h       Show this help
        \\
    , .{});
}

fn resolveSeed(fixed_seed: ?u64) u64 {
    return fixed_seed orelse blk: {
        var s: u64 = 0;
        std.crypto.random.bytes(std.mem.asBytes(&s));
        break :blk s;
    };
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

fn runSweep(
    comptime Trainer: type,
    comptime Ctx: type,
    allocator: std.mem.Allocator,
    ctx: Ctx,
    sweep: []const usize,
    label: []const u8,
    no_self_play: bool,
    threshold_mbb: f64,
) !void {
    std.debug.print("Leduc CFR trainer - {s} ({s})\n", .{ label, @tagName(builtin.mode) });
    std.debug.print("Iterations: {any}\n", .{sweep});
    if (Ctx == u64) std.debug.print("Seed: {d}\n", .{ctx});

    const initTrainer = struct {
        fn init(a: std.mem.Allocator, c: Ctx) Trainer {
            return Trainer.init(a, c);
        }
    }.init;

    if (no_self_play) {
        std.debug.print("{s:>10}  {s:>6}  {s:>8}  {s:>10}  {s:>8}\n", .{ "Iters", "Nodes", "Value", "Exploit", "mbb/g" });

        var total_timer = std.time.Timer.start() catch unreachable;
        var total_iters: usize = 0;
        var exploits: [Sweep.MAX]f64 = undefined;
        var exploit_count: usize = 0;

        for (sweep) |iters| {
            var trainer = initTrainer(allocator, ctx);
            defer trainer.deinit();

            const value = try trainer.train(iters);
            const exploit = try play.exploitability(Trainer, &trainer);
            const mbb = exploit * 500.0;
            total_iters += iters;
            exploits[exploit_count] = exploit;
            exploit_count += 1;

            std.debug.print("{d:>10}  {d:>6}  {d:>8.4}  {d:>10.5}  {d:>8.2}\n", .{ iters, trainer.nodeCount(), value, exploit, mbb });
        }

        const total_ns = total_timer.read();
        const total_ms = @as(f64, @floatFromInt(total_ns)) / 1_000_000.0;
        const us_per_iter = @as(f64, @floatFromInt(total_ns)) / 1_000.0 / @as(f64, @floatFromInt(total_iters));
        std.debug.print("\nTotal: {d:.1}ms ({d} iters, {d:.2} us/iter)\n", .{ total_ms, total_iters, us_per_iter });

        try validateMonotonicity(exploits[0..exploit_count], threshold_mbb);
        return;
    }

    const base_iters = sweep[sweep.len - 1];
    var base = initTrainer(allocator, ctx);
    defer base.deinit();
    const base_value = try base.train(base_iters);

    std.debug.print("\nBase ({d} iters) game value: {d:.5}\n\n", .{ base_iters, base_value });
    std.debug.print("{s:>10}  {s:>8}  {s:>8}  {s:>8}  {s:>8}\n", .{ "Iters", "Value", "vs Base", "Base vs", "Exploit" });

    for (sweep) |iters| {
        var trainer = initTrainer(allocator, ctx);
        defer trainer.deinit();
        const value = try trainer.train(iters);

        const vs_base = play.headToHead(Trainer, &trainer, Trainer, &base);
        const base_vs = play.headToHead(Trainer, &base, Trainer, &trainer);
        const exploit = try play.exploitability(Trainer, &trainer);
        const mbb = exploit * 500.0;

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

        pub fn init(_: std.mem.Allocator, _: void) @This() {
            init_calls += 1;
            return .{};
        }

        pub fn deinit(_: *@This()) void {}

        pub fn train(_: *@This(), _: usize) !f64 {
            return 0;
        }

        pub fn nodeCount(_: *const @This()) usize {
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

    try runSweep(TestTrainer, void, std.testing.allocator, {}, sweep[0..], "Test", true, 100.0);

    try std.testing.expectEqual(@as(usize, sweep.len), TestTrainer.init_calls);
}
