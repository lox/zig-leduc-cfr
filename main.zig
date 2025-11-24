const std = @import("std");
const vanilla = @import("src/leduc/cfr_vanilla.zig");
const mccfr = @import("src/leduc/cfr_mccfr.zig");
const play = @import("src/leduc/play.zig");

const DEFAULT_SWEEP: []const usize = &.{
    100,
    300,
    1_000,
    3_000,
    10_000,
    30_000,
    100_000,
};

fn runSweep(comptime Trainer: type, allocator: std.mem.Allocator, sweep: []const usize, label: []const u8) !void {
    std.debug.print(
        \\Leduc CFR trainer (toy, educational) - {s}
        \\Sweeping iteration counts: {any}
        \\
    , .{ label, sweep });

    const base_iters = sweep[sweep.len - 1];
    var base = Trainer.init(allocator);
    defer base.deinit();
    const base_avg = try base.train(base_iters);

    std.debug.print("\nBase (iters {d}) avg game value: {d:.5}\n", .{ base_iters, base_avg });
    std.debug.print("\nIter       AvgGame    AasP0     BaseAsP0  Exploit\n", .{});

    for (sweep) |iters| {
        var trainer = Trainer.init(allocator);
        defer trainer.deinit();
        const avg = try trainer.train(iters);

        const a_vs_base = play.headToHead(Trainer, &trainer, Trainer, &base);
        const base_vs_a = play.headToHead(Trainer, &base, Trainer, &trainer);
        const exploit = try play.exploitability(Trainer, &trainer);
        std.debug.print("{d:9}  {d:.5}   {d:.5}   {d:.5}  {d:.5}\n", .{ iters, avg, a_vs_base, base_vs_a, exploit });
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var algo: enum { vanilla, mccfr } = .vanilla;
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.next(); // skip argv[0]
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--algo=vanilla")) {
            algo = .vanilla;
        } else if (std.mem.eql(u8, arg, "--algo=mccfr")) {
            algo = .mccfr;
        } else {
            std.debug.print("Unknown argument: {s}\n", .{arg});
            return error.InvalidArgument;
        }
    }

    switch (algo) {
        .vanilla => try runSweep(vanilla.VanillaCFRTrainer, allocator, DEFAULT_SWEEP, "Vanilla"),
        .mccfr => try runSweep(mccfr.MCCFRTrainer, allocator, DEFAULT_SWEEP, "MCCFR"),
    }
}
