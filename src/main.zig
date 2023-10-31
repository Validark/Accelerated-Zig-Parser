// zig fmt: off
const SKIP_OUTLIERS        = false;
const RUN_LEGACY_TOKENIZER = true;
const RUN_NEW_TOKENIZER    = true;
const RUN_LEGACY_AST       = false;
const RUN_NEW_AST          = false;
const REPORT_SPEED         = true;
const VALIDATE_UTF8        = false;
const INFIX_TEST           = false;
// zig fmt: on

// TODO: mask_for_op_cont should probably be renamed.
// I need to decide what to do with the handwritten optimizations there.
// I think the function could use a more thoughtful design, or I could potentially write a tool
// or work on the compiler so that optimal code gets generated automatically. This project made me realize
// that compilers could probably be smarter about SWAR transformations.

// I could probably reorganize the `Tag` address space and see if I can squeeze more perf out.

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const testing = std.testing;
const mem = std.mem;
const Ast = std.zig.Ast;
const Allocator = std.mem.Allocator;

/// This is the chunk size, not necessarily the native vector size.
/// If we support 64-bit operations, we want to be doing 64-bit count-trailing-zeros,
/// even if we have to combine multiple bitstrings to get to the point where we can do that.
/// For now, we use `usize` as our reasonable guess for what size of bitstring we can operate on efficiently.
/// Will probably have to be updated if we get machines with efficient 128-bit operations that still have 64-bit pointers.
const VEC_SIZE = @bitSizeOf(usize);
const VEC_INT = std.meta.Int(.unsigned, VEC_SIZE);
const VEC = @Vector(VEC_SIZE, u8);
const IVEC = @Vector(VEC_SIZE, i8);
const LOG_VEC_INT = std.math.Log2Int(VEC_INT);

const USE_SWAR = std.simd.suggestVectorSizeForCpu(u8, builtin.cpu) == null;

// This is the native vector size.
const NATIVE_VEC_INT = std.meta.Int(.unsigned, 8 * (std.simd.suggestVectorSizeForCpu(u8, builtin.cpu) orelse @sizeOf(usize)));
const NATIVE_VEC_SIZE = @sizeOf(NATIVE_VEC_INT);
const NATIVE_VEC_CHAR = @Vector(NATIVE_VEC_SIZE, u8);
const NATIVE_VEC_COND = @Vector(NATIVE_VEC_SIZE, bool);

const FRONT_SENTINELS = "\n";
const BACK_SENTINELS = "\n" ++ "\x00" ** 13 ++ "\n";
const INDEX_OF_FIRST_0_SENTINEL = std.mem.indexOfScalar(u8, BACK_SENTINELS, 0).?;
const EXTENDED_BACK_SENTINELS_LEN = BACK_SENTINELS.len - INDEX_OF_FIRST_0_SENTINEL;

/// Returns a slice that is aligned to alignment, but `len` is set to only the "valid" memory.
/// freed via: allocator.free(buffer[0..std.mem.alignForward(u64, buffer.len, alignment)])
fn readFileIntoAlignedBuffer(allocator: Allocator, file: std.fs.File, comptime alignment: u32) ![:0]align(alignment) const u8 {
    const bytes_to_allocate = std.math.cast(usize, try file.getEndPos()) orelse return error.Overflow;
    const overaligned_size = try std.math.add(usize, bytes_to_allocate, FRONT_SENTINELS.len + BACK_SENTINELS.len + (alignment - 1));
    const buffer = try allocator.alignedAlloc(u8, alignment, std.mem.alignBackward(usize, overaligned_size, alignment));

    buffer[0..FRONT_SENTINELS.len].* = FRONT_SENTINELS.*;
    var cur: []u8 = buffer[FRONT_SENTINELS.len..][0..bytes_to_allocate];
    while (true) {
        cur = cur[try file.read(cur)..];
        if (cur.len == 0) break;
    }
    cur = buffer[FRONT_SENTINELS.len + bytes_to_allocate ..];
    @memset(cur, 0);
    inline for (BACK_SENTINELS, 0..) |c, i| {
        if (c != 0) cur[i] = c;
    }
    return buffer[0 .. FRONT_SENTINELS.len + bytes_to_allocate + INDEX_OF_FIRST_0_SENTINEL :0];
}

fn readFiles(gpa: Allocator) !std.ArrayListUnmanaged([:0]const u8) {
    const directory = switch (INFIX_TEST) {
        true => "./src/beep",
        false => "./src/files_to_parse",
    };
    var parent_dir2 = try std.fs.cwd().openDirZ(directory, .{}, false);
    defer parent_dir2.close();

    var parent_dir = try std.fs.cwd().openIterableDir(directory, .{});
    defer parent_dir.close();

    var num_files: usize = 0;
    var num_bytes: usize = 0;

    var sources: std.ArrayListUnmanaged([:0]const u8) = .{};
    {
        const t1 = std.time.nanoTimestamp();
        var walker = try parent_dir.walk(gpa); // 12-14 ms just walking the tree
        defer walker.deinit();

        var total_size: usize = 0;
        _ = total_size;
        while (try walker.next()) |dir| {
            switch (dir.kind) {
                .file => if (dir.basename.len > 4 and std.mem.eql(u8, dir.basename[dir.basename.len - 4 ..][0..4], ".zig") and dir.path.len - dir.basename.len > 0) {
                    // These two are extreme outliers, omit them from our test bench
                    if (SKIP_OUTLIERS and
                        (std.mem.endsWith(u8, dir.path, "udivmodti4_test.zig") or
                        std.mem.endsWith(u8, dir.path, "udivmoddi4_test.zig")))
                        continue;

                    const file = try parent_dir2.openFile(dir.path, .{});
                    defer file.close();

                    num_files += 1;
                    const source = try readFileIntoAlignedBuffer(gpa, file, VEC_SIZE);
                    // const source = try file.readToEndAllocOptions(gpa, std.math.maxInt(u32), null, 1, 0);
                    num_bytes += source.len - 2;

                    // if (sources.items.len == 13)
                    // std.debug.print("{} {s}\n", .{ sources.items.len, dir.path });
                    (try sources.addOne(gpa)).* = source;
                },

                else => {},
            }
        }

        const t2 = std.time.nanoTimestamp();
        var lines: u64 = 0;
        for (sources.items) |source| {
            for (source[1 .. source.len - 1]) |c| {
                lines += @intFromBool(c == '\n');
            }
        }
        const elapsedNanos: u64 = @intCast(t2 - t1);
        const @"MB/s" = @as(f64, @floatFromInt(num_bytes * 1000)) / @as(f64, @floatFromInt(elapsedNanos));

        const stdout = std.io.getStdOut().writer();
        if (REPORT_SPEED)
            try stdout.print("       Read in files in {} ({d:.2} MB/s) and used {} memory with {} lines across {} files\n", .{ std.fmt.fmtDuration(elapsedNanos), @"MB/s", std.fmt.fmtIntSizeDec(num_bytes), lines, sources.items.len });
    }
    return sources;
}

const Operators = struct {
    const unpadded_ops = [_][]const u8{ ".**", "!", "|", "||", "|=", "=", "==", "=>", "!=", "(", ")", ";", "%", "%=", "{", "}", "[", "]", ".", ".*", "..", "...", "^", "^=", "+", "++", "+=", "+%", "+%=", "+|", "+|=", "-", "-=", "-%", "-%=", "-|", "-|=", "*", "*=", "**", "*%", "*%=", "*|", "*|=", "->", ":", "/", "/=", "&", "&=", "?", "<", "<=", "<<", "<<=", "<<|", "<<|=", ">", ">=", ">>", ">>=", "~", "//", "///", "//!", ".?", "\\\\", "," };

    // TODO: add assertion that this only works because the maximum support op length is currently 4

    const padded_ops: [unpadded_ops.len][4]u8 = blk: {
        var padded_ops_table: [unpadded_ops.len][4]u8 = undefined;
        inline for (unpadded_ops, 0..) |op, i| {
            padded_ops_table[i] = (op ++ ("\x00" ** (4 - op.len))).*;
        }
        break :blk padded_ops_table;
    };

    const masks = blk: {
        var bitmask = std.mem.zeroes([2]u64);

        for (unpadded_ops, padded_ops) |unpadded_op, padded_op| {
            const hash = rawHash(getOpWord(&padded_op, unpadded_op.len));
            switch (@as(u1, @truncate(bitmask[hash / 64] >> @truncate(hash)))) {
                0 => bitmask[hash / 64] |= @as(u64, 1) << @truncate(hash),
                1 => bitmask[hash / 64] &= ~(@as(u64, 1) << @truncate(hash)),
            }
        }

        if (@popCount(bitmask[0]) + @popCount(bitmask[1]) != unpadded_ops.len) {
            var err_msg: []const u8 = "Hash function failed to map operators perfectly.\nThe following collisions occurred:\n";
            for (unpadded_ops, padded_ops) |unpadded_op, padded_op| {
                const hash = rawHash(getOpWord(&padded_op, unpadded_op.len));
                switch (@as(u1, @truncate(bitmask[hash / 64] >> @truncate(hash)))) {
                    0 => err_msg = err_msg ++ std.fmt.comptimePrint("\"{s}\" => {}\n", .{ unpadded_op, hash }),
                    1 => {},
                }
            }
            @compileError(err_msg);
        }

        break :blk bitmask;
    };

    const first_mask_popcount = @popCount(masks[0]);

    const sorted_padded_ops = blk: {
        const max_hash_is_unused = (masks[masks.len - 1] >> 63) ^ 1;
        var buffer: [padded_ops.len + max_hash_is_unused][4]u8 = undefined;

        for (unpadded_ops, padded_ops) |op, padded_op| {
            const hash = hashOp(getOpWord(&padded_op, op.len));
            buffer[mapToIndex(hash)] = padded_op;
        }

        if (max_hash_is_unused == 1) {
            // Make sure we don't overflow the buffer by adding an extra element that always compares false
            buffer[padded_ops.len] = [4]u8{ 0, 0, 0, 0 };
        }

        break :blk buffer;
    };

    fn getOpWord(op: [*]const u8, len: u32) u32 {
        comptime assert(BACK_SENTINELS.len >= 3);
        const shift_amt = len * 8;
        const relevant_mask: u32 = @intCast((@as(u64, 1) << @intCast(shift_amt)) -% 1);
        return std.mem.readIntLittle(u32, op[0..4]) & relevant_mask;
    }

    fn rawHash(op_word: u32) u7 {
        return @intCast(((op_word *% 698839068) -% comptime (@as(u32, 29) << 25)) >> 25);
    }

    fn hashOp(op_word: u32) Tag {
        return @enumFromInt(rawHash(op_word));
    }

    fn mapToIndexRaw(hash_val: u7) u7 {
        const mask = (@as(u64, 1) << @truncate(hash_val)) -% 1;
        return (if (hash_val >= 64) first_mask_popcount else 0) +
            @popCount(mask & masks[hash_val / 64]);
    }

    /// Given a hash, maps it to an index in the range [0, sorted_padded_ops.len)
    fn mapToIndex(hash: Tag) u7 {
        return mapToIndexRaw(@intFromEnum(hash));
    }

    /// Given a string starting with a Zig operator, returns the tag type
    /// Assumes there are at least 4 valid bytes at the passed in `op`
    fn lookup(op: [*]const u8, len: u32) ?Tag {
        // std.debug.print("{s}\n", .{op[0..len]});
        const op_word = getOpWord(op, len);
        const hash = rawHash(op_word);
        return if (std.mem.readIntLittle(u32, &sorted_padded_ops[mapToIndexRaw(hash)]) == op_word)
            @enumFromInt(hash)
        else
            null;
    }

    const single_char_ops = [_]u8{ '~', ':', ';', '[', ']', '?', '(', ')', '{', '}', ',' };
    const multi_char_ops = [_]u8{ '.', '!', '%', '&', '*', '+', '-', '/', '<', '=', '>', '^', '|', '\\' };

    fn isSingleCharOp(c: u8) bool {
        inline for (single_char_ops) |op| if (c == op) return true;
        return false;
    }

    fn isSingleCharOpNewlineLikely(c: u8) bool {
        return switch (c) {
            ';', '{', '}', ',' => true,
            else => false,
        };
    }

    fn isSingleCharOpNewlineUnlikely(c: u8) bool {
        inline for (single_char_ops) |op| {
            comptime if (isSingleCharOpNewlineLikely(op)) continue;
            if (c == op) return true;
        }
        return false;
    }

    fn isMultiCharBeginning(c: u8) bool {
        inline for (multi_char_ops) |op| if (c == op) return true;
        return false;
    }
};

const Keywords = struct {
    const unpadded_kws = [_][]const u8{ "addrspace", "align", "allowzero", "and", "anyframe", "anytype", "asm", "async", "await", "break", "callconv", "catch", "comptime", "const", "continue", "defer", "else", "enum", "errdefer", "error", "export", "extern", "fn", "for", "if", "inline", "noalias", "noinline", "nosuspend", "opaque", "or", "orelse", "packed", "pub", "resume", "return", "linksection", "struct", "suspend", "switch", "test", "threadlocal", "try", "union", "unreachable", "usingnamespace", "var", "volatile", "while" };

    const masks = blk: {
        var bitmask = std.mem.zeroes([2]u64);
        for (unpadded_kws) |kw| {
            const hash = hashKw(kw.ptr, kw.len);

            switch (@as(u1, @truncate(bitmask[hash / 64] >> @truncate(hash)))) {
                0 => bitmask[hash / 64] |= @as(u64, 1) << @truncate(hash),
                1 => bitmask[hash / 64] &= ~(@as(u64, 1) << @truncate(hash)),
            }
        }

        if (@popCount(bitmask[0]) + @popCount(bitmask[1]) != unpadded_kws.len) {
            var err_msg: []const u8 = "Hash function failed to map operators perfectly.\nThe following collisions occurred:\n";
            for (unpadded_kws) |kw| {
                const hash = hashKw(kw.ptr, kw.len);
                switch (@as(u1, @truncate(bitmask[hash / 64] >> @truncate(hash)))) {
                    0 => err_msg = err_msg ++ std.fmt.comptimePrint("\"{s}\" => {}\n", .{ kw, hash }),
                    1 => {},
                }
            }
            @compileError(err_msg);
        }

        break :blk bitmask;
    };

    const first_mask_popcount = @popCount(masks[0]);
    const max_kw_len = blk: {
        var max = 0;
        for (unpadded_kws) |kw| max = @max(kw.len, max);
        break :blk max;
    };

    const PADDING_RIGHT = std.math.ceilPowerOfTwo(usize, max_kw_len + 1) catch unreachable;
    const padded_int = std.meta.Int(.unsigned, PADDING_RIGHT);

    const sorted_padded_kws = blk: {
        const max_hash_is_unused = (masks[masks.len - 1] >> 63) ^ 1;
        var buffer: [unpadded_kws.len + max_hash_is_unused][PADDING_RIGHT]u8 align(PADDING_RIGHT) = undefined;

        for (unpadded_kws) |kw| {
            const padded_kw =
                (kw ++
                ([1]u8{if (USE_SWAR) 0 else @intCast(kw.len)} ** (PADDING_RIGHT - 1 - kw.len)) ++
                [1]u8{@intCast(kw.len)}).*;
            assert(padded_kw.len == PADDING_RIGHT);
            const hash = hashKw(kw.ptr, kw.len);
            buffer[mapToIndex(hash)] = padded_kw;
        }

        if (max_hash_is_unused == 1) {
            // We add one extra filler item just in case we hash a value greater than the greatest hashed value
            buffer[unpadded_kws.len] = ("\xFF" ** PADDING_RIGHT).*;
        }
        break :blk buffer;
    };

    pub fn hashKw(keyword: [*]const u8, len: u32) u7 {
        comptime assert(BACK_SENTINELS.len >= 1); // Make sure it's safe to go forward a character
        const a = std.mem.readIntLittle(u16, keyword[0..2]);
        comptime assert(FRONT_SENTINELS.len >= 1); // Make sure it's safe to go back to the previous character
        const b = std.mem.readIntLittle(u16, (if (@inComptime())
            keyword[len - 2 .. len][0..2]
        else
            keyword - 2 + @as(usize, @intCast(len)))[0..2]);
        return @truncate(((a ^ (len << 14)) *% b) >> 8);
        // return @truncate(((a >> 1) *% (b >> 1) ^ (len << 14)) >> 8);
        // return @truncate((((a >> 1) *% (b >> 1)) >> 8) ^ (len << 6));
    }

    /// Given a hash, maps it to an index in the range [0, sorted_padded_kws.len)
    fn mapToIndex(hash: u7) u8 {
        const mask = (@as(u64, 1) << @truncate(hash)) -% 1;
        return (if (hash >= 64) first_mask_popcount else 0) + @popCount(mask & masks[hash / 64]);
    }

    const min_kw_len = blk: {
        var min = std.math.maxInt(u64);
        for (unpadded_kws) |kw| {
            if (kw.len < min) min = kw.len;
        }
        break :blk min;
    };

    /// Given a string starting with a Zig identifier, returns the tag type of the keyword if it's a keyword
    pub fn lookup(kw: [*]const u8, len: u32) ?Tag {
        if (BACK_SENTINELS.len < PADDING_RIGHT - 1)
            @compileError(std.fmt.comptimePrint("Keywords.lookup requires additional trailing sentinels in the source file. Expected {}, got {}\n", .{ PADDING_RIGHT - 1, BACK_SENTINELS.len }));

        if (!USE_SWAR and len > 255) return null;

        const hash = mapToIndex(hashKw(kw, len));
        const val: [PADDING_RIGHT]u8 = sorted_padded_kws[hash];
        const val_len = val[PADDING_RIGHT - 1]; // [min_kw_len, max_kw_len]

        if (USE_SWAR) {
            if (len != val_len) return null;
            var word1: u64 = @bitCast(kw[0..8].*);
            var word2: u64 = @bitCast(kw[8..16].*);

            word1 = if (std.math.cast(u6, (64 -| len * 8))) |shift1|
                (word1 << @intCast(shift1)) >> @intCast(shift1)
            else
                unreachable; // len can't be 0

            // TODO:  In-progress SWAR impl. Could probably be better...?
            // TODO: only works on little-endian

            word2 = if (std.math.cast(
                u6,
                // len can't be > max_kw_len (14 atm)
                128 - len * 8,
            )) |shift2|
                (word2 << @intCast(shift2)) >> @intCast(shift2)
            else
                0;

            // std.debug.print("\n", .{});

            // std.debug.print("\n", .{});

            const source_word1 = @as(u64, @bitCast(val[0..8].*));
            const source_word2 = @as(u64, @bitCast(val[8..16].*)) << 8 >> 8;

            if (word1 == source_word1 and word2 == source_word2) {
                // Dear reader: We can't move this higher because this might produce an invalid tag.
                // Our hash might map to a padding value in the buffer, and not be a real keyword.
                // We have to let the compiler hoist this for us.
                return @enumFromInt(~hash);
            } else {
                // std.debug.print("kw: {s}\n", .{kw[0..len]});
                // std.debug.print("vl: {s}\n", .{val});
                // printu(word1);
                // printu(word2);
                // std.debug.print("\n", .{});
                // printu(source_word1);
                // printu(source_word2);
                // std.debug.print("word1 == source_word1: {}\n", .{word1 == source_word1});
                // std.debug.print("word2 == source_word2: {}\n", .{word2 == source_word2});
                return null;
            }
        } else {
            const KW_VEC = @Vector(PADDING_RIGHT, u8);
            var vec1: KW_VEC = val;
            const vec2: KW_VEC = kw[0..PADDING_RIGHT].*;
            var other_vec: KW_VEC = @splat(@as(u8, @intCast(len)));
            const cd = @select(u8, @as(KW_VEC, @splat(val_len)) > std.simd.iota(u8, PADDING_RIGHT), vec2, other_vec);

            if (std.simd.countTrues(cd != vec1) == 0) {
                // Dear reader: We can't move this higher because this might produce an invalid tag.
                // Our hash might map to a padding value in the buffer, and not be a real keyword.
                // We have to let the compiler hoist this for us.
                return @enumFromInt(~hash);
            } else {
                return null;
            }
        }
    }
};

const Tag = blk: {
    const BitmapKinds = std.meta.fields(Parser.BitmapKind);
    var decls = [_]std.builtin.Type.Declaration{};
    var enumFields: [1 + Operators.unpadded_ops.len + Keywords.unpadded_kws.len + BitmapKinds.len]std.builtin.Type.EnumField = undefined;

    enumFields[0] = .{ .name = "invalid", .value = 0xaa };

    for (Operators.unpadded_ops, Operators.padded_ops) |op, padded_op| {
        const hash = Operators.rawHash(Operators.getOpWord(&padded_op, op.len));
        enumFields[1 + Operators.mapToIndexRaw(hash)] = .{ .name = op, .value = hash };
    }

    var i = 1 + Operators.unpadded_ops.len;
    for (Keywords.unpadded_kws) |kw| {
        const hash = Keywords.mapToIndex(Keywords.hashKw(kw.ptr, kw.len));
        enumFields[i + hash] = .{ .name = kw, .value = ~hash };
    }

    i += Keywords.unpadded_kws.len;

    for (BitmapKinds, 0..) |x, j| {
        enumFields[i + j] = x;
    }

    break :blk @Type(.{
        .Enum = .{
            .tag_type = u8,
            .fields = &enumFields,
            .decls = &decls,
            .is_exhaustive = true,
        },
    });
};

/// If we see a len of 0, go look in the next extra_long_lens slot for the true length.
const Token = extern struct { len: u8, kind: Tag };

fn printu(v: anytype) void {
    comptime var shift = @bitSizeOf(@TypeOf(v));
    inline while (true) {
        shift -= 4;
        std.debug.print("{b:0>4}{c}", .{ @as(u4, @truncate(v >> shift)), if (shift % 8 == 0) ' ' else '_' });
        if (shift == 0) break;
    }
    std.debug.print("\n", .{});
}

fn printx(v: anytype) void {
    comptime var shift = @bitSizeOf(@TypeOf(v));
    inline while (true) {
        shift -= 4;
        std.debug.print("{x}{c}", .{ @as(u4, @truncate(v >> shift)), if (shift % 8 == 0) ' ' else '_' });
        if (shift == 0) break;
    }
    std.debug.print("\n", .{});
}

test "identifier_mask should match alphanumeric characters and underscores" {
    var c: u8 = 0;
    while (true) {
        const expected: u1 = switch (c) {
            'A'...'Z', 'a'...'z', '0'...'9', '_' => 1,
            else => 0,
        };
        try std.testing.expectEqual(expected, @intCast(swarMovMask(Parser.swarNonIdentifierMask(c))));
        c +%= 1;
        if (c == 0) break;
    }
}

/// Finds the popCount of the least significant bit of each byte.
///
/// E.g. 0b.......1 .......1 .......0 .......0 .......1 .......0 .......1 .......1 -> 5
///
fn popCountLSb(v: anytype) @TypeOf(v) {
    const ones: @TypeOf(v) = @bitCast(@as(@Vector(@divExact(@bitSizeOf(@TypeOf(v)), 8), u8), @splat(0x01)));

    // This is a math trick.
    // Think about the gradeschool multiplication algorithm by a bitstring of broadcasted 1's:
    const isolate = (v & ones) *% ones;
    //   .......a .......b .......c .......d .......e .......f .......g .......h
    // * .......1 .......1 .......1 .......1 .......1 .......1 .......1 .......1
    // ---------------------------------------------------------------------------
    // + .......a .......b .......c .......d .......e .......f .......g .......h
    // + .......b .......c .......d .......e .......f .......g .......h ........
    // + .......c .......d .......e .......f .......g .......h ........ ........
    // + .......d .......e .......f .......g .......h ........ ........ ........
    // + .......e .......f .......g .......h ........ ........ ........ ........
    // + .......f .......g .......h ........ ........ ........ ........ ........
    // + .......g .......h ........ ........ ........ ........ ........ ........
    // + .......h ........ ........ ........ ........ ........ ........ ........
    // ===========================================================================

    // In the end, the bytes will contain:
    // a+b+c+d+e+f+g+h a+b+c+d+e+f+g a+b+c+d+e+f a+b+c+d+e a+b+c+d a+b+c a+b a
    // Of course, this disregards overflow between bytes, but in this case the maximum popCount is @sizeOf(@TypeOf(v)),
    // so overflowing between bytes is not possible.
    comptime assert(@sizeOf(@TypeOf(v)) < 256);

    // Then we simply shift the top byte to the lowest byte position and we have our popcount.
    return isolate >> (@bitSizeOf(@TypeOf(v)) - 8);
}

test "swar popCount" {
    inline for ([_]type{ u128, u64, u32, u16, u8 }) |T| {
        var c: std.meta.Int(.unsigned, @sizeOf(T)) = 0;
        while (true) {
            try std.testing.expectEqual(@popCount(c), @as(std.meta.Int(.unsigned, std.math.log2_int_ceil(u64, 1 + @sizeOf(T))), @intCast(popCountLSb(swarUnMovMask(c) >> 7))));
            c +%= 1;
            if (c == 0) break;
        }
    }
}

test "mask_for_op_cont should work" {
    var c: u8 = 0;
    while (true) {
        const expected: u3 = switch (c) {
            0x21, 0x25, 0x2a, 0x2b, 0x2e, 0x2f, 0x3c, 0x3d, 0x3e, 0x3f, 0x5c, 0x7c => 1,
            else => 0,
        };
        const v = std.mem.readIntLittle(u32, &[4]u8{ c, c, 0, 0 });
        const res = Parser.mask_for_op_cont(v) - 1;
        // std.debug.print("0x{x:0>2} {} vs {}\n", .{ c, res, expected });
        try std.testing.expectEqual(expected, @intCast(res));
        c +%= 1;
        if (c == 0) break;
    }
}

// test "mask_for_op_cont should work 2" {
//     std.debug.print("\n", .{});
//     {
//         const v = std.mem.readIntLittle(u32, "//.?");
//         const res = Parser.mask_for_op_cont(v);
//         std.debug.print("res: {}\n", .{res});
//     }
//     {
//         const v = std.mem.readIntLittle(u32, ".?.?");
//         const res = Parser.mask_for_op_cont(v);
//         std.debug.print("res: {}\n", .{res});
//     }
//     {
//         const v = std.mem.readIntLittle(u32, "&.??");
//         const res = Parser.mask_for_op_cont(v);
//         std.debug.print("res: {}\n", .{res});
//     }
//     {
//         const v = std.mem.readIntLittle(u32, "\\\\+=");
//         const res = Parser.mask_for_op_cont(v);
//         std.debug.print("res: {}\n", .{res});
//     }
//     {
//         const v = std.mem.readIntLittle(u32, "\\\\.?");
//         const res = Parser.mask_for_op_cont(v);
//         std.debug.print("res: {}\n", .{res});
//     }
//     {
//         const v = std.mem.readIntLittle(u32, "|*.?");
//         const res = Parser.mask_for_op_cont(v);
//         std.debug.print("res: {}\n", .{res});
//     }
// }

/// The opposite of swarMovMask. Used for tests.
fn swarUnMovMask(x: anytype) std.meta.Int(.unsigned, 8 * @bitSizeOf(@TypeOf(x))) {
    comptime assert(builtin.is_test);
    var r: std.meta.Int(.unsigned, 8 * @bitSizeOf(@TypeOf(x))) = 0;

    inline for (0..@bitSizeOf(@TypeOf(x))) |i| {
        const bit: u1 = @truncate(x >> i);
        r |= @as(@TypeOf(r), bit) << (i * 8 + 7);
    }

    return r;
}

// Performs a ctz operation using David Seal's method with a small twist.
// LLVM produces a version of this code by default, but with an explicit branch on 0.
// This code is branchless on machines that can do `@intFromBool(a == 0)` without branching.
fn ctzBranchless(a: anytype) std.meta.Int(.unsigned, std.math.ceilPowerOfTwoPromote(u64, std.math.log2_int_ceil(u64, VEC_SIZE + 1))) {
    const T = @TypeOf(a);

    // Produce a mask of just the lowest set bit if one exists, else 0.
    // There are `@bitSizeOf(T)+1` possibilities for `x`.
    // We handle the `0` case separately.
    const x = a & (~a +% 1);

    const constant = switch (@bitSizeOf(T)) {
        64 => 151050438420815295,
        32 => 125613361,
        else => return @ctz(a),
    };
    const shift = @bitSizeOf(T) - @bitSizeOf(std.math.Log2Int(T));
    const hash: std.math.Log2Int(T) = @intCast((x *% constant) >> shift);
    comptime var lookup_table: [@bitSizeOf(T)]u8 = undefined;
    comptime {
        var taken_slots: T = 0;
        for (0..@bitSizeOf(T)) |i| {
            const x_possibility = @as(T, 1) << i;
            const x_possibility_hash: std.math.Log2Int(T) = @intCast((x_possibility *% constant) >> shift);
            taken_slots |= @as(T, 1) << x_possibility_hash;
            lookup_table[x_possibility_hash] = @ctz(x_possibility);
        }
        assert(~taken_slots == 0); // proves it is a minimal perfect hash function and that we overwrote all the undefined values
        // The difference between this algorithm and the mainstream algorithm is that we set the high bit here and use a `selector` to grab the right value.
        lookup_table[0] |= @bitSizeOf(T);
    }
    const tzcnt_if_non0 = lookup_table[hash];
    const selector = @intFromBool(a == 0) +% @as(u8, @bitSizeOf(T) - 1);
    return tzcnt_if_non0 & selector;
}

test "ctzBranchless" {
    inline for ([_]type{ u64, u32 }) |T| {
        inline for (0..@bitSizeOf(T)) |i| {
            const x = @as(T, 1) << i;
            try std.testing.expectEqual(@ctz(x), @intCast(ctzBranchless(x)));
        }
    }
}

fn ctz(a: anytype) @TypeOf(ctzBranchless(a)) {
    return switch (SWAR_CTZ_IMPL) {
        .ctz => @ctz(a),
        else => ctzBranchless(a),
    };
}

fn swarCTZPlus1Generic(x: anytype, comptime impl: @TypeOf(SWAR_CTZ_PLUS_1_IMPL)) @TypeOf(x) {
    const ones: @TypeOf(x) = @bitCast(@as(@Vector(@divExact(@bitSizeOf(@TypeOf(x)), 8), u8), @splat(0x01)));
    assert(x != 0 and (x & (0x7F * ones)) == 0);
    return switch (impl) {
        .ctz => @ctz(x) / 8 +% 1,
        .popc => @divExact(@popCount(x ^ (x -% 1)), 8),
        .clz => @sizeOf(@TypeOf(x)) -% @divExact(@clz(x ^ (x -% 1)), 8),
        .swar => popCountLSb(x -% 1),
        .naive => blk: { // 7 ops, 1 constant that might not be an immediate
            var i = (x -% 1) & ones; // because the bitstring only contains bits in the highest set bit of each byte, the mask will only isolate from the trailing zeros
            inline for (0..comptime std.math.log2_int(u64, @sizeOf(@TypeOf(x)))) |y| {
                i = i +% (i >> (@bitSizeOf(@TypeOf(x)) >> (y + 1)));
            }
            break :blk @as(std.meta.Int(.unsigned, @sizeOf(@TypeOf(x))), @truncate(i));
        },
    };
}

test "swarCTZPlus1" {
    var i: u3 = 0;

    while (true) {
        var x: u32 = swarUnMovMask(@as(u4, 8) | i);

        const expected: u32 = switch (x) {
            0x80808080 => 1,
            0x80800080 => 1,
            0x80008080 => 1,
            0x80000080 => 1,
            0x80808000 => 2,
            0x80008000 => 2,
            0x80800000 => 3,
            0x80000000 => 4,
            else => unreachable,
        };

        inline for (std.meta.fields(@TypeOf(SWAR_CTZ_PLUS_1_IMPL))) |impl|
            try std.testing.expectEqual(expected, swarCTZPlus1Generic(x, @enumFromInt(impl.value)));

        i +%= 1;
        if (i == 0) break;
    }
}

const SWAR_CTZ_PLUS_1_IMPL: enum { ctz, clz, popc, swar, naive } = switch (builtin.cpu.arch) {
    .aarch64_32, .aarch64_be, .aarch64, .arm, .armeb, .thumb, .thumbeb => if (std.Target.aarch64.featureSetHas(builtin.cpu.features, .v8a) or
        (std.Target.arm.featureSetHas(builtin.cpu.features, .has_v7) and // Not sure why it's v7.
        !std.mem.eql(u8, builtin.cpu.model.name, "cortex_m0") and
        !std.mem.eql(u8, builtin.cpu.model.name, "cortex_m0plus") and
        !std.mem.eql(u8, builtin.cpu.model.name, "cortex_m1") and
        !std.mem.eql(u8, builtin.cpu.model.name, "cortex_m23"))) .ctz else .swar,
    .mips, .mips64, .mips64el, .mipsel => if (std.Target.mips.featureSetHas(builtin.cpu.features, .mips64)) .clz else .naive,
    .powerpc, .powerpc64, .powerpc64le, .powerpcle => .clz,
    .s390x => .clz,
    .ve => .ctz,
    .avr => .popc,
    .msp430 => .popc,
    .riscv32, .riscv64 => if (std.Target.riscv.featureSetHas(builtin.cpu.features, .zbb)) .ctz else if (std.Target.riscv.featureSetHas(builtin.cpu.features, .m)) .swar else .naive,
    .sparc, .sparc64, .sparcel => if (std.Target.sparc.featureSetHas(builtin.cpu.features, .popc)) .popc else .swar,
    .wasm32, .wasm64 => .ctz,
    .x86, .x86_64 => .ctz,
    else => .naive,
};

fn swarCTZPlus1(x: u32) @TypeOf(x) {
    return swarCTZPlus1Generic(x, SWAR_CTZ_PLUS_1_IMPL);
}

fn swarCTZGeneric(x: anytype, comptime impl: @TypeOf(SWAR_CTZ_IMPL)) @TypeOf(x) {
    const ones: @TypeOf(x) = @bitCast(@as(@Vector(@divExact(@bitSizeOf(@TypeOf(x)), 8), u8), @splat(1)));
    assert((x & (ones * 0x7F)) == 0);

    return switch (impl) {
        .ctz => @ctz(x) >> 3,
        .swar => popCountLSb((~x & (x -% 1)) >> 7),
        .swar_bool => popCountLSb(x -% 1) -% @intFromBool(x != 0),
        .naive => blk: {
            var i = ((~x & (x -% 1)) & (ones * 0x80)) >> 7;
            inline for (0..comptime std.math.log2_int(u64, @sizeOf(@TypeOf(x)))) |y| {
                i = i +% (i >> (@bitSizeOf(@TypeOf(x)) >> (y + 1)));
            }
            break :blk @as(std.meta.Int(.unsigned, @sizeOf(@TypeOf(x))), @truncate(i));
        },
    };
}

test "ctz popCount" {
    inline for (std.meta.fields(@TypeOf(SWAR_CTZ_IMPL))) |impl| {
        inline for ([_]type{ u64, u32, u16, u8 }) |T| {
            var c: std.meta.Int(.unsigned, @sizeOf(T)) = 0;
            while (true) {
                const y = swarCTZGeneric(swarUnMovMask(c), @enumFromInt(impl.value));
                try std.testing.expectEqual(@as(@TypeOf(y), @ctz(c)), y);
                c +%= 1;
                if (c == 0) break;
            }
        }
    }
}

const SWAR_CTZ_IMPL: enum { ctz, swar, swar_bool, naive } =
    switch (builtin.cpu.arch) {
    .riscv32, .riscv64 => if (std.Target.riscv.featureSetHas(builtin.cpu.features, .zbb)) .ctz else if (std.Target.riscv.featureSetHas(builtin.cpu.features, .m)) .swar_bool else .naive,
    .x86, .x86_64 => if (std.Target.x86.featureSetHas(builtin.cpu.features, .bmi)) .ctz else .swar_bool,
    else => switch (SWAR_CTZ_PLUS_1_IMPL) {
        .ctz, .clz, .popc => .ctz,
        .swar, .naive => .swar,
    },
};

fn swarCTZ(x: u64) @TypeOf(x) {
    return swarCTZGeneric(x, SWAR_CTZ_IMPL);
}

/// Creates a bitstring from the most significant bit of each byte in a given bitstring.
///
/// E.g. 1....... 0....... 0....... 1....... 0....... 1....... 1....... 1....... => 10010111
fn swarMovMask(v: anytype) @TypeOf(v) {
    const cpu_name = builtin.cpu.model.llvm_name orelse builtin.cpu.model.name;

    return swarMovMaskGeneric(v, switch (builtin.cpu.arch) {
        .aarch64_32, .aarch64_be, .aarch64, .arm, .armeb, .thumb, .thumbeb => .mul_lo,
        .mips, .mips64, .mips64el, .mipsel => .mul_lo,
        .powerpc, .powerpc64, .powerpc64le, .powerpcle => .mul_lo,
        .s390x => .mul_lo,
        .ve => .mul_lo,
        .avr => .naive,
        .msp430 => .naive,
        .riscv32, .riscv64 => if (std.Target.riscv.featureSetHas(builtin.cpu.features, .m)) .mul_lo else .naive,
        .sparc, .sparc64, .sparcel => .mul_lo,
        .wasm32, .wasm64 => .mul_lo,
        .x86, .x86_64 => if (!@inComptime() and std.Target.x86.featureSetHas(builtin.cpu.features, .bmi2) and
            // PEXT is microcoded (slow) on AMD architectures before Zen 3.
            (!std.mem.startsWith(u8, cpu_name, "znver") or cpu_name["znver".len] >= '3'))
            .pext
        else
            .mul_lo,
        else => .mul_lo,
    });
}

fn swarMovMaskGeneric(v: anytype, comptime impl: enum { pext, mul_hi, mul_lo, naive }) @TypeOf(v) {
    comptime assert(@divExact(@bitSizeOf(@TypeOf(v)), 8) <= 8);
    const ones: @TypeOf(v) = @bitCast(@as(@Vector(@sizeOf(@TypeOf(v)), u8), @splat(1)));
    const msb_mask = 0x80 * ones;

    // This variable is used as a multiplicand in the `mul_lo` and `mul_hi` tricks.
    // We are exploiting a multiplication as a shifter and adder, and the derivation of this number is
    // shown here as a comptime loop.
    // This trick is often generalizable to other problems too: https://stackoverflow.com/a/14547307
    comptime var mult: @TypeOf(v) = 0;
    comptime for (0..@sizeOf(@TypeOf(v))) |i| {
        mult |= @as(@TypeOf(v), 1) << (7 * i);
    };

    // Example with 32 bit integers:
    // We want to concentrate the upper bits of each byte into a single nibble.
    // Doing the gradeschool multiplication algorithm, we can see that each 1 bit
    // in the bottom multiplicand shifts the upper multiplicand, and then we add all these
    // shifted bitstrings together. (Note `.` represents a 0)
    //   a.......b.......c.......d.......
    // * ..........1......1......1......1
    // -------------------------------------------------------------------------
    //   a.......b.......c.......d.......
    //   .b.......c.......d..............
    //   ..c.......d.....................
    // + ...d............................
    // -------------------------------------------------------------------------
    //   abcd....bcd.....cd......d.......

    // Then we simply shift to the right by `32 - 4` (bitstring size minus the number of relevant bits) to isolate the desired `abcd` bits in the least significant byte!

    return switch (impl) {
        .pext => @intCast(pext(v, msb_mask)),
        // From: http://0x80.pl/articles/scalar-sse-movmask.html
        .mul_hi => @as(std.meta.Int(.unsigned, @sizeOf(@TypeOf(v))), @truncate((@shlExact(@as(std.meta.Int(.unsigned, 2 * @bitSizeOf(@TypeOf(v))), mult), @sizeOf(@TypeOf(v))) *% ((v & msb_mask))) >> @bitSizeOf(@TypeOf(v)))),
        .mul_lo => (mult *% ((v & msb_mask))) >> (@bitSizeOf(@TypeOf(v)) - @sizeOf(@TypeOf(v))),
        .naive => blk: {
            const x = (v & msb_mask) >> 7;
            var a = x;
            inline for (0..@sizeOf(@TypeOf(v))) |i| a |= x >> (7 * i);
            break :blk @as(std.meta.Int(.unsigned, @sizeOf(@TypeOf(v))), @truncate(a));
        },
    };
}

inline fn pext(v: u64, msb_mask: u64) u64 {
    return asm ("pext %[msb_mask], %[v], %[ret]"
        : [ret] "=r" (-> u64),
        : [v] "r" (v),
          [msb_mask] "r" (msb_mask),
    );
}

test "movemask swar should properly isolate the highest set bits of all bytes in a bitstring" {
    const IMPL_TYPE = @typeInfo(@TypeOf(swarMovMaskGeneric)).Fn.params[1].type.?;
    comptime var impls: []const IMPL_TYPE = &[0]IMPL_TYPE{};
    comptime for (std.meta.fields(IMPL_TYPE)) |impl| {
        // Skip pext on platforms that do not support it.
        if (!std.mem.eql(u8, "pext", impl.name) or (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .bmi2))) {
            impls = impls ++ [1]IMPL_TYPE{@enumFromInt(impl.value)};
        }
    };

    inline for ([_]type{ u64, u32, u16, u8 }) |T| {
        inline for (impls) |impl| {
            var c: std.meta.Int(.unsigned, @sizeOf(T)) = 0;
            while (true) {
                const ans = swarMovMaskGeneric(swarUnMovMask(c), impl);
                try std.testing.expectEqual(@as(@TypeOf(ans), c), ans);
                c +%= 1;
                if (c == 0) break;
            }
        }
    }
}

const Parser = struct {
    const Bitmaps = packed struct {
        non_newlines: VEC_INT,
        identifiers_or_numbers: VEC_INT,
        non_unescaped_quotes: VEC_INT,
        whitespace: VEC_INT,
        // prev_carriage: VEC_INT,
        // op_cont_chars: VEC_INT,
        // non_unescaped_char_literals: VEC_INT,
    };

    fn mask_for_op_cont(v: u32) u32 {
        // "%", "*", "+", ".", "/", "<", "=", ">", "?", "\\", "|",
        // "!", "*", ".", "/", "=", "|",
        // "="

        // const str: [4]u8 = @bitCast(v);
        // inline for (0..4) |i| {
        //     inline for (Operators.unpadded_ops, Operators.padded_ops) |op, padded_op| {
        //         if (4 - i == op.len and std.mem.eql(u8, str[0..op.len], op))
        //             return comptime Operators.lookup(&padded_op, op.len).?;
        //     }
        // }
        // return null;

        const ones: @TypeOf(v) = @bitCast(@as(@Vector(@divExact(@bitSizeOf(@TypeOf(v)), 8), u8), @splat(1)));
        const mask = comptime ones * 0x7F;
        // const dub_ones: @TypeOf(v) = @bitCast(@as(@Vector(@divExact(@bitSizeOf(@TypeOf(v)), 16), u16), @splat(1)));
        // const dub_mask = comptime dub_ones * 0x7FFF;

        // 2e26
        // 2a7c

        // std.debug.print(" " ** 1 ++ "0" ++ " " ** 3 ++ "0", .{});
        // std.debug.print(" " ** 3 ++ "{c}" ++ " " ** 3 ++ "{c}", .{ @as(u8, @truncate(v >> 8)), @as(u8, @truncate(v)) });
        // std.debug.print("\n", .{});
        const non_ascii = v >> 8;
        const low_7_bits = non_ascii & mask;
        // printu32(low_7_bits);
        const not_and_dot = (((v & 0x7FFF) ^ comptime std.mem.readIntLittle(u16, "&.")) + 0x7FFF);
        const not_bar_star = (((v & 0x7FFF) ^ comptime std.mem.readIntLittle(u16, "|*")) + 0x7FFF);

        // TODO: figure out how to do this optimization properly??
        // const dot_question = 0x8000 & ~(((v & 0x7FFF) ^ comptime std.mem.readIntLittle(u16, ".?")) + 0x7FFF);
        // const backslashes = 0x8000 & ~(((v & 0x7FFF) ^ comptime std.mem.readIntLittle(u16, "\\\\")) + 0x7FFF);
        // const slashes = 0x8000 & ~(((v & 0x7FFF) ^ comptime std.mem.readIntLittle(u16, "//")) + 0x7FFF);

        // printu32(not_and_dot);
        // printu32(not_and_dot);

        // 0x21,0x25,0x2a,0x2b,0x2e,0x2f,
        // 0x3c,0x3d,0x3e,0x3f,
        // 0x5c,0x7c

        const magic_mask = low_7_bits ^ comptime ones * 0x2e;

        // matches 0x2a,0x2b,0x2e,0x2f
        const non_a_b_e_or_f = (magic_mask & comptime ones * ~@as(u7, 5)) + mask; // TODO: make this work on first 3 bytes only

        // matches 0x21,0x3c,0x3d,0x3e,0x3f
        const is_0x0f_or_higher = magic_mask + comptime ones * (0x80 - 0x0f);
        const is_0x14_or_higher = magic_mask + comptime ones * (0x80 - 0x14);

        const not_0x25 = (low_7_bits ^ comptime ones * 0x25) + mask; // TODO: make this only work on first byte only
        const not_0x5c_and_not_0x7c = ((low_7_bits & comptime ones * ~@as(u7, 0x20)) ^ comptime ones * 0x5c) + mask; // TODO: make this work on first two bytes only

        const non_operator_cont_mask = non_ascii | (non_a_b_e_or_f & (~is_0x0f_or_higher | is_0x14_or_higher) & not_0x25 & not_0x5c_and_not_0x7c);
        const bad_cont_block_helper_mask = (~(not_and_dot & not_bar_star) >> 8) & 0x80;
        const x = (bad_cont_block_helper_mask | non_operator_cont_mask) & ~mask;

        // We can infer this because low_7_bits has the upper 8 bits set to 0.
        // This upper 0 byte will always fail our SWAR checks, i.e. it's most significant bit will be 1 no matter what.
        assert(x != 0 and x > std.math.maxInt(i32));

        // TODO: These are the operators for which we need backtracking. Let's see if we can optimize some of these to eliminate backtracking in practice.
        // 5561 "&."
        // 2311 ".?."
        // 1372 "|*"
        // 360 "!?"
        // 258 "!*"
        // 249 "\\/"
        // 243 "\\//"
        // 184 "\\<"
        // 134 ".?.*"
        // 130 ".*."
        // 91 "*?"
        // 53 "\\*"
        // 49 "\\="
        // 48 "\\=="
        // 47 "///!"
        // 45 "\\</"
        // 44 "//."
        // 42 "\\."
        // 40 "*?*"
        // 35 "!?*"
        // 18 "\\>"
        // 15 ".*.*"
        // 13 "////"
        // 13 ".*.?"
        // 10 "\\.."
        // 9 "//\\"
        // 9 "//\"
        // 9 ".?.."
        // 6 "\\<!"
        // 5 "\\<?"
        // 5 "\\/*"
        // 3 "\\|"
        // 3 "\\\\"
        // 3 "\\\"
        // 3 "\\.\"
        // 3 "\\./"
        // 3 ".*.."
        // 2 "\\%"
        // 2 "\\**"
        // 2 "//="
        // 2 "***"
        // 1 "\\+"
        // 1 "\\*/"
        // 1 "\\!"
        // 1 "//=>"
        // 1 "//=="
        // 1 "//?"
        // 1 ".*="
        // 1 ".***"
        // 1 "!??*"
        // 1 "!??"
        // 1 "!!"

        return swarCTZPlus1(x);
    }

    fn swarNonIdentifierMask(v: anytype) @TypeOf(v) {
        const ones: @TypeOf(v) = @bitCast(@as(@Vector(@divExact(@bitSizeOf(@TypeOf(v)), 8), u8), @splat(1)));
        const mask = comptime ones * 0x7F;
        const low_7_bits = v & mask;

        const non_underscore = (low_7_bits ^ comptime ones * '_') + mask;

        // alpha check:
        // Upper 3 bits must be 4 or 6. Lower 5 bits must be between 1 and 26
        const magic_mask = ((v & comptime ones * ~@as(u7, 0x20)) ^ comptime ones * 0x40);
        const non_0x40_or_0x60 = magic_mask + mask;
        const non_alpha = magic_mask + comptime ones * (0x80 - 27);

        // digit check:
        // Upper nibble must be 3. Lower nibble must be [0, 9]
        const flipped_0x30 = low_7_bits ^ comptime ones * 0x30;
        const non_digit = flipped_0x30 + comptime ones * (0x80 - 0xA);
        return ~v & (non_0x40_or_0x60 ^ (non_digit & non_alpha & non_underscore));
    }

    fn nonIdentifierMask(v: anytype) if (USE_SWAR) NATIVE_VEC_INT else std.meta.Int(.unsigned, NATIVE_VEC_SIZE) {
        if (USE_SWAR) {
            assert(@TypeOf(v) == NATIVE_VEC_INT);
            return swarNonIdentifierMask(v);
        }

        const underscores = ~maskForNonChars(v, "_");
        const upper_alpha = maskForCharRange(v, 'A', 'Z');
        const lower_alpha = maskForCharRange(v, 'a', 'z');
        const digits = maskForCharRange(v, '0', '9');
        return underscores | upper_alpha | lower_alpha | digits;
    }

    fn movMask(v: anytype) VEC_INT {
        return if (USE_SWAR) swarMovMask(v) else v;
    }

    fn maskForNonCharsGeneric(v: anytype, comptime str: []const u8, comptime use_swar: bool) if (USE_SWAR) NATIVE_VEC_INT else std.meta.Int(.unsigned, NATIVE_VEC_SIZE) {
        if (use_swar) {
            assert(@TypeOf(v) == NATIVE_VEC_INT);
            const ones: @TypeOf(v) = @bitCast(@as(@Vector(@divExact(@bitSizeOf(@TypeOf(v)), 8), u8), @splat(1)));
            const mask = comptime ones * 0x7F;
            const low_7_bits = v & mask;
            var accumulator: @TypeOf(v) = std.math.maxInt(@TypeOf(v));

            inline for (str) |c| {
                assert(c < 0x80);
                accumulator &= (low_7_bits ^ comptime ones * c) + mask;
            }

            return v | accumulator;
        } else {
            assert(@TypeOf(v) == NATIVE_VEC_CHAR);
            var accumulator = @as(std.meta.Int(.unsigned, NATIVE_VEC_SIZE), 0);
            inline for (str) |c| {
                assert(c < 0x80); // Because this would break the SWAR version, we enforce it here too.
                accumulator |= @bitCast(v == @as(NATIVE_VEC_CHAR, @splat(c)));
            }
            return ~accumulator;
        }
    }

    fn maskForNonChars(v: anytype, comptime str: []const u8) if (USE_SWAR) NATIVE_VEC_INT else std.meta.Int(.unsigned, NATIVE_VEC_SIZE) {
        return maskForNonCharsGeneric(v, str, USE_SWAR);
    }

    fn maskForCharRange(input_vec: NATIVE_VEC_CHAR, comptime char1: u8, comptime char2: u8) std.meta.Int(.unsigned, NATIVE_VEC_SIZE) {
        const VEC_T = std.meta.Int(.unsigned, NATIVE_VEC_SIZE);
        return @as(VEC_T, @bitCast(@as(NATIVE_VEC_CHAR, @splat(char1)) <= input_vec)) & @as(VEC_T, @bitCast(input_vec <= @as(NATIVE_VEC_CHAR, @splat(char2))));
    }

    // On arm and ve machines, `@ctz(x)` is implemented as `@bitReverse(@clz(x))`.
    // We can speculatively perform 4 bit reverses in nextChunk so that the main loop can use `@clz` instead.
    // This saves operations in practice because (for the codebases tested) we do an average of 10.5 ctz's/clz's
    // per 64-byte chunk, meaning we eliminate ~6.5 bit reverses per chunk.
    // Might backfire if the microarchitecture has a builtin ctz operation and the decoder automatically combines a bitreverse and clz.
    const SPECULATIVELY_REVERSED = switch (builtin.cpu.arch) {
        .aarch64_32, .aarch64_be, .aarch64, .arm, .armeb, .thumb, .thumbeb => SWAR_CTZ_PLUS_1_IMPL == .ctz,
        .ve => true,
        else => false,
    };

    fn reverseIfPreferred(b: VEC_INT) VEC_INT {
        return if (SPECULATIVELY_REVERSED) @bitReverse(b) else b;
    }

    const BitmapKind = enum(u8) {
        // zig fmt: off
        const min_bitmap_value = @intFromEnum(BitmapKind.unknown);
        const max_bitmap_value = @intFromEnum(BitmapKind.number);

        eof                   = 0,
        unknown               = 128 | @as(u8,  0), // is_quoted

        identifier            = 128 | @as(u8,  1),
        builtin               = 128 | @as(u8,  9),
        number                = 128 | @as(u8,  17),

        string                = 128 | @as(u8,  2), // is_quoted
        string_identifier     = 128 | @as(u8,  10), // is_quoted

        whitespace            = 128 | @as(u8, 19),

        char_literal          = 128 | @as(u8,  6), // is_quoted
        // zig fmt: on
    };

    pub fn isOperand(op_type: Tag) bool {
        return switch (@intFromEnum(op_type)) {
            BitmapKind.min_bitmap_value...BitmapKind.max_bitmap_value => true,
            else => false,
        };
    }
    // TODO: Maybe recover from parse_errors by switching to BitmapKind.unknown? Report errors?
    // TODO: audit usages of u32's to make sure it's impossible to ever overflow.
    // TODO: make it so quotes and character literals cannot have newlines in them?
    // TODO: audit the utf8 validator to make sure we clear the state properly when not using it
    pub fn tokenize(gpa: Allocator, source: [:0]const u8, comptime impl: u1) ![]Token {
        const FOLD_COMMENTS_INTO_ADJACENT_NODES = false;
        const end_ptr = &source.ptr[source.len];
        const extended_source_len = std.mem.alignForward(usize, source.len + EXTENDED_BACK_SENTINELS_LEN, VEC_SIZE);
        const extended_source = source.ptr[0..extended_source_len];
        const tokens = try gpa.alloc(Token, extended_source_len);
        errdefer gpa.free(tokens);

        const non_newlines_bitstrings = try gpa.alloc(VEC_INT, extended_source_len / VEC_SIZE + (@bitSizeOf(Bitmaps) - @bitSizeOf(VEC_INT)) / VEC_SIZE);
        // TODO: make this errdefer and return this data out.
        // We can use this information later to find out what line we are on.
        defer gpa.free(non_newlines_bitstrings);

        var cur_token = tokens;
        cur_token[0] = .{ .len = 0, .kind = .whitespace };
        cur_token[1..][0..2].* = @bitCast(@as(u32, 0));

        // var extra_lens = try gpa.alloc(u32, source.len / 256);
        // errdefer gpa.free(extra_lens);

        var prev = extended_source;

        comptime assert(FRONT_SENTINELS.len == 1 and FRONT_SENTINELS[0] == '\n' and BACK_SENTINELS.len >= 3);
        var cur = prev[@as(u8, @intFromBool(
            std.mem.readIntSliceNative(u32, prev) == std.mem.readIntSliceNative(u32, "\n\xEF\xBB\xBF"),
        )) * 4 ..];

        var bitmap_ptr: []VEC_INT = non_newlines_bitstrings;
        bitmap_ptr.ptr -= 1;
        bitmap_ptr.len += 1;
        var op_type: Tag = .whitespace;
        var selected_bitmap: []const VEC_INT = bitmap_ptr[@as(u2, @truncate(@intFromEnum(op_type)))..];
        var bitmap_index: usize = @intFromPtr(cur.ptr) / VEC_SIZE * VEC_SIZE -% VEC_SIZE;
        var utf8_checker: if (VALIDATE_UTF8) Utf8Checker else void = if (VALIDATE_UTF8) .{};
        var prev_escaped: VEC_INT = 0;

        outer: while (true) : (cur = cur[1..]) {
            {
                var aligned_ptr = @intFromPtr(cur.ptr) / VEC_SIZE * VEC_SIZE;
                while (true) {
                    // https://github.com/ziglang/zig/issues/8220
                    // TODO: once labeled switch continues are added, we can make this check run the first iteration only.
                    // If we loop back around, there is no need to check this.
                    // I implemented this with tail call functions but it was messy.
                    while (bitmap_index != aligned_ptr) {
                        bitmap_index +%= VEC_SIZE;
                        const base_ptr = @as([*]align(VEC_SIZE) const u8, @ptrFromInt(bitmap_index));

                        var non_newlines: VEC_INT = 0;
                        var non_quotes: VEC_INT = 0;
                        var non_backslashes: VEC_INT = 0;
                        var non_spaces: VEC_INT = 0;
                        var identifiers_or_numbers: VEC_INT = 0;

                        inline for (0..VEC_SIZE / NATIVE_VEC_SIZE) |i| {
                            const chunk = blk: {
                                const slice: *align(NATIVE_VEC_SIZE) const [NATIVE_VEC_SIZE]u8 = @alignCast(base_ptr[i * NATIVE_VEC_SIZE ..][0..NATIVE_VEC_SIZE]);
                                break :blk if (USE_SWAR) @as(*align(NATIVE_VEC_SIZE) const NATIVE_VEC_INT, @ptrCast(slice)).* else @as(@Vector(NATIVE_VEC_SIZE, u8), slice.*);
                            };

                            const shift: LOG_VEC_INT = switch (builtin.cpu.arch.endian()) {
                                .Little => @intCast(NATIVE_VEC_SIZE * i),
                                .Big => @intCast((VEC_SIZE - NATIVE_VEC_SIZE) - NATIVE_VEC_SIZE * i),
                            };

                            non_newlines |= movMask(maskForNonChars(chunk, "\n")) << shift;
                            non_quotes |= movMask(maskForNonChars(chunk, "\"")) << shift;
                            non_backslashes |= movMask(maskForNonChars(chunk, "\\")) << shift;
                            non_spaces |= movMask((maskForNonChars(chunk, " \t"))) << shift;
                            identifiers_or_numbers |= movMask(nonIdentifierMask(chunk)) << shift;

                            if (VALIDATE_UTF8) {
                                utf8_checker.check_next_input(chunk);
                                try utf8_checker.errors();
                            }
                        }

                        const backslashes = ~non_backslashes;

                        // ----------------------------------------------------------------------------
                        // This code is brought to you courtesy of simdjson and simdjzon, both licensed
                        // under the Apache 2.0 license which is included at the bottom of this file

                        // If there was overflow, pretend the first character isn't a backslash
                        const backslash: VEC_INT = backslashes & ~prev_escaped;
                        const follows_escape = (backslash << 1) | prev_escaped;

                        // Get sequences starting on even bits by clearing out the odd series using +
                        const even_bits: VEC_INT = @bitCast(@as(@Vector(@divExact(VEC_SIZE, 8), u8), @splat(0x55)));
                        const odd_sequence_starts = backslash & ~even_bits & ~follows_escape;
                        const x = @addWithOverflow(odd_sequence_starts, backslash);
                        const invert_mask: VEC_INT = x[0] << 1; // The mask we want to return is the *escaped* bits, not escapes.
                        prev_escaped = x[1];

                        // Mask every other backslashed character as an escaped character
                        // Flip the mask for sequences that start on even bits, to correct them
                        const escaped = (even_bits ^ invert_mask) & follows_escape;

                        // ----------------------------------------------------------------------------

                        bitmap_ptr = bitmap_ptr[1..];
                        bitmap_ptr[0] = reverseIfPreferred(non_newlines);
                        bitmap_ptr[1] = reverseIfPreferred(identifiers_or_numbers);
                        bitmap_ptr[2] = reverseIfPreferred((non_quotes & non_newlines) | escaped);
                        bitmap_ptr[3] = reverseIfPreferred(~(non_spaces & non_newlines));

                        selected_bitmap = selected_bitmap[1..];
                    }

                    const cur_misalignment: LOG_VEC_INT = @truncate(@intFromPtr(cur.ptr));

                    // We invert, i.e. count 1's, because 0's are shifted in by the bitshift.
                    const inverted_bitstring = ~if (SPECULATIVELY_REVERSED)
                        selected_bitmap[0] << cur_misalignment
                    else
                        selected_bitmap[0] >> cur_misalignment;

                    // Optimization: when ctz is implemented with a bitReverse+clz,
                    // we speculatively bitReverse in `nextChunk` to avoid doing so in this loop.
                    const str_len: std.meta.Int(
                        .unsigned,
                        std.math.ceilPowerOfTwoPromote(u64, std.math.log2_int_ceil(u64, VEC_SIZE + 1)),
                    ) = if (SPECULATIVELY_REVERSED) @clz(inverted_bitstring) else ctz(inverted_bitstring);

                    cur = cur[str_len..];
                    aligned_ptr = @intFromPtr(cur.ptr) / VEC_SIZE * VEC_SIZE;
                    if (bitmap_index == aligned_ptr) break;
                }
            }

            comptime assert(BACK_SENTINELS.len - 1 > std.mem.indexOf(u8, BACK_SENTINELS, "\x00").?); // there should be at least another character
            comptime assert(BACK_SENTINELS[BACK_SENTINELS.len - 1] == '\n'); // eof reads the non_newlines bitstring, therefore we need a newline at the end
            if (op_type == .eof) break :outer;

            {
                const is_quoted: u1 = @truncate((@intFromPtr(selected_bitmap.ptr) / 8) ^ (@intFromPtr(bitmap_ptr.ptr) / 8) ^ 1);
                cur = cur[is_quoted..];
            }

            while (true) {
                var len: u32 = @intCast(@intFromPtr(cur.ptr) - @intFromPtr(prev.ptr));

                comptime assert(FRONT_SENTINELS[0] == '\n');
                switch (prev[0]) {
                    'a'...'z' => if (Keywords.lookup(prev.ptr, len)) |op_kind| {
                        op_type = op_kind;
                        const is_space = @intFromBool(cur[0] == ' ');
                        cur = cur[is_space..];
                        len += is_space;
                    },
                    else => {},
                }

                advance_blk: {
                    blk: {
                        // We are fine with arbitrary lengths attached to operators and keywords, because
                        // once we know which one it is there is no loss of information in adding comments and
                        // whitespace to the length. E.g. `const` is always `const` and `+=` is `+=`, whether or
                        // not we add whitespace to either side.
                        //
                        // On the other hand, identifiers, numbers, strings, etc might become harder to deal
                        // with later if we don't know the exact length. If it turns out that we don't really
                        // need this information anyway, we can simplify this codepath and collapse comments
                        // and whitespace into those tokens too.
                        const prev_op_type = cur_token[0].kind;
                        switch (@intFromEnum(op_type)) {
                            BitmapKind.min_bitmap_value...BitmapKind.max_bitmap_value => break :blk, // identifiers, quotes, etc
                            else => {},
                        }

                        switch (@intFromEnum(prev_op_type)) {
                            BitmapKind.min_bitmap_value...BitmapKind.max_bitmap_value => break :blk, // identifiers, quotes, etc
                            else => {},
                        }

                        const op_type_c = op_type == .whitespace or (FOLD_COMMENTS_INTO_ADJACENT_NODES and op_type == .@"//");
                        const prev_op_type_c = prev_op_type == .whitespace or (FOLD_COMMENTS_INTO_ADJACENT_NODES and prev_op_type == .@"//");

                        if (op_type_c or prev_op_type_c or
                            (op_type == prev_op_type and (op_type == .@"//!" or op_type == .@"///" or (!FOLD_COMMENTS_INTO_ADJACENT_NODES and op_type == .@"//"))))
                        {
                            if (op_type_c) op_type = prev_op_type;
                            len += @bitCast(cur_token[1..][0..2].*);
                            break :advance_blk;
                        }
                    }

                    var advance_amt: u2 = if (cur_token[0].len == 0) 3 else 1;
                    cur_token = cur_token[advance_amt..];
                }

                cur_token[0] = .{ .len = if (len >= 256) 0 else @intCast(len), .kind = op_type };
                cur_token[1..][0..2].* = @bitCast(len);
                prev = cur;

                if (cur[0] == '@') {
                    cur = cur[1..];
                    op_type = switch (cur[0]) {
                        'a'...'z', 'A'...'Z' => .builtin,
                        '"' => .string_identifier,
                        else => return error.MissingQuoteOrLetterAfterAtSymbol,
                    };
                    selected_bitmap = bitmap_ptr[@as(u2, @truncate(@intFromEnum(op_type)))..];
                } else if (cur[0] == '\'') {
                    while (true) {
                        cur = cur[1..];
                        comptime assert(std.mem.indexOfAny(u8, BACK_SENTINELS, "\n") != null);
                        if (cur[0] == '\'' or cur[0] == '\n') break;
                        cur = cur[@intFromBool(cur[0] == '\\')..];
                    }

                    if (cur[0] != '\'') return error.UnterminatedCharacterLiteral;
                    cur = cur[1..];
                    const char_literal_len: u32 = @intCast(@intFromPtr(cur.ptr) - @intFromPtr(prev.ptr));
                    if (char_literal_len > "'\\u{10FFFF}'".len) return error.CharLiteralTooLong;
                    op_type = .char_literal;
                    continue;
                } else if (Operators.isSingleCharOp(cur[0])) {
                    selected_bitmap = bitmap_ptr[@as(u2, @truncate(@intFromEnum(BitmapKind.whitespace)))..];
                    op_type = Operators.hashOp(Operators.getOpWord(cur.ptr, 1));
                    // cur = cur[@intFromBool(cur[0] == ' ')..];
                } else if (Operators.isMultiCharBeginning(cur[0])) {
                    var op_len: u32 = 1;
                    if (impl == 1) {
                        // if (mask_for_op_cont(std.mem.readIntLittle(u32, cur[0..4]))) |op_data| {
                        //     op_type = op_data;
                        // } else return error.ParseError;

                        // // if (op_len == 0) return error.ParseError;
                        // // op_type = Operators.hashOp(Operators.getOpWord(prev.ptr, op_len));
                        // cur = cur[blk: {
                        //     inline for (Operators.unpadded_ops, Operators.padded_ops) |op, padded_op| {
                        //         const hash = comptime Operators.rawHash(Operators.getOpWord(&padded_op, op.len));
                        //         if (comptime switch (@as(Tag, @enumFromInt(hash))) {
                        //             .@"///", .@"//!", .@"//", .@"\\\\" => false,
                        //             else => true,
                        //         }) {
                        //             if (@intFromEnum(op_type) == hash) {
                        //                 break :blk op.len;
                        //             }
                        //         }
                        //     }
                        //     break :blk 2;
                        // }..];

                        // switch (op_type) {
                        //     .@"///", .@"//!", .@"//", .@"\\\\" => {
                        //         selected_bitmap_kind = .unknown;
                        //         break;
                        //     },
                        //     else => {},
                        // }

                        /////////////////////////////////////////////////////////////

                        // op_len = mask_for_op_cont(std.mem.readIntLittle(u32, cur[0..4]));
                        // if (op_len == 0) return error.ParseError;
                        // op_type = Operators.hashOp(Operators.getOpWord(prev.ptr, op_len));
                        // cur = cur[op_len..];
                        // switch (op_type) {
                        //     .@"///", .@"//!", .@"//", .@"\\\\" => {
                        //         selected_bitmap_kind = .unknown;
                        //         break;
                        //     },
                        //     else => {},
                        // }

                        // std.debug.print("\n{s}\n", .{cur[0..4]});
                        // printu32(@bitReverse(std.mem.readIntLittle(u32, cur[0..4])));
                        // printu32(@bitReverse(std.mem.readIntLittle(u32, cur[0..4]) >> 8));
                        op_len = mask_for_op_cont(std.mem.readIntLittle(u32, cur[0..4]));
                        const op_word4 = Operators.getOpWord(cur.ptr, 4);
                        const hash4 = Operators.rawHash(op_word4);
                        const op_word3 = Operators.getOpWord(cur.ptr, 3);
                        const hash3 = Operators.rawHash(op_word3);
                        const op_word2 = Operators.getOpWord(cur.ptr, 2);
                        const hash2 = Operators.rawHash(op_word2);
                        const op_word1 = Operators.getOpWord(cur.ptr, 1);
                        const hash1 = Operators.rawHash(op_word1);
                        assert(op_len <= 4);
                        if (op_len == 4 and std.mem.readIntLittle(u32, &Operators.sorted_padded_ops[Operators.mapToIndexRaw(hash4)]) == op_word4) {
                            op_type = @enumFromInt(hash4);
                            cur = cur[4..];
                        } else if (op_len >= 3 and std.mem.readIntLittle(u32, &Operators.sorted_padded_ops[Operators.mapToIndexRaw(hash3)]) == op_word3) {
                            op_type = @enumFromInt(hash3);
                            if (op_type == .@"///" or op_type == .@"//!") {
                                selected_bitmap = bitmap_ptr[@as(u2, @truncate(@intFromEnum(BitmapKind.unknown)))..];
                                break;
                            }
                            cur = cur[3..];
                        } else if (op_len >= 2 and std.mem.readIntLittle(u32, &Operators.sorted_padded_ops[Operators.mapToIndexRaw(hash2)]) == op_word2) {
                            op_type = @enumFromInt(hash2);
                            if (op_type == .@"//" or op_type == .@"\\\\") {
                                selected_bitmap = bitmap_ptr[@as(u2, @truncate(@intFromEnum(BitmapKind.unknown)))..];
                                break;
                            }
                            cur = cur[2..];
                        } else if (cur[0] == '\\') {
                            return error.IncompleteSingleLineStringOpener;
                        } else {
                            op_type = @enumFromInt(hash1);
                            cur = cur[1..];
                        }
                    } else {
                        comptime var op_continuation_chars align(32) = std.mem.zeroes([@divExact(256, @bitSizeOf(usize))]usize);

                        comptime for (Operators.unpadded_ops) |op| {
                            for (op[1..]) |c| {
                                op_continuation_chars[c / @bitSizeOf(usize)] |=
                                    @as(usize, 1) << @truncate(c);
                            }
                        };

                        inline for (0..3) |_| {
                            const c = cur[op_len];
                            const is_op_char: u1 = @truncate(op_continuation_chars[c / @bitSizeOf(usize)] >> @truncate(c));
                            if (is_op_char == 0) break;
                            op_len += 1;
                        }

                        op_type = blk: while (true) {
                            if (Operators.lookup(prev.ptr, op_len)) |op_kind| break :blk op_kind;
                            op_len -= 1;
                            if (op_len == 0) return error.ParseError;
                        };
                        cur = cur[op_len..];

                        switch (op_type) {
                            .@"///", .@"//!", .@"//", .@"\\\\" => {
                                selected_bitmap = bitmap_ptr[@as(u2, @truncate(@intFromEnum(BitmapKind.unknown)))..];
                                break;
                            },
                            else => {},
                        }
                    }

                    // selected_bitmap = bitmap_ptr[@as(u2, @truncate(@intFromEnum(BitmapKind.whitespace)))..];
                    // break;
                    cur = cur[@intFromBool(cur[0] == ' ')..];
                    continue;
                } else if (cur[0] == '\r') {
                    if (cur[1] != '\n') return error.UnpairedCarriageReturn;
                    op_type = .whitespace;
                    selected_bitmap = bitmap_ptr[@as(u2, @truncate(@intFromEnum(op_type)))..];
                } else {
                    op_type = switch (cur[0]) {
                        'a'...'z', 'A'...'Z', '_' => .identifier,
                        '0'...'9' => .number,
                        '"' => .string,
                        // '\'' => .char_literal,
                        ' ', '\t', '\n' => .whitespace,
                        0 => .eof,
                        else => return error.InvalidToken,
                    };

                    selected_bitmap = bitmap_ptr[@as(u2, @truncate(@intFromEnum(op_type)))..];
                }

                // cur = cur[1..];

                // On my machine, this can be a big win sometimes, other times, it has no effect.
                // According to perf stat, this supposedly increases the number of branch misses by
                // several hundred thousand. However, we decrease the total number of branches by millions.
                // Branch misses as a percentage are also higher, and yet, it's still faster.
                // Haven't looked, but maybe it's just putting the branches in a more desirable order?
                // We really need automatic profile-guided optimization in Zig so we can be certain.

                // if (selected_bitmap == &bitmap_ptr.*[@as(u2, @truncate(@intFromEnum(BitmapKind.whitespace)))] and switch (cur[0]) {
                //     ' ', '\t', '\r', '\n' => false,
                //     else => true,
                // }) continue;

                break;
            }
        }

        if (@intFromPtr(cur.ptr) < @intFromPtr(end_ptr)) return error.Found0ByteInFile;

        cur_token = cur_token[if (cur_token[0].len == 0) 3 else 1..];
        cur_token[0] = .{ .len = 1, .kind = .eof };
        cur_token = cur_token[1..];

        const new_chunks_data_len = (@intFromPtr(cur_token.ptr) - @intFromPtr(tokens.ptr)) / @sizeOf(Token);

        if (gpa.resize(tokens, new_chunks_data_len)) {
            var resized_tokens = tokens;
            resized_tokens.len = new_chunks_data_len;
            return resized_tokens;
        }

        return tokens;
    }
};

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    const gpa = std.heap.c_allocator;
    const sources = try readFiles(gpa);
    var bytes: u64 = 0;
    var lines: u64 = 0;

    for (sources.items) |source| {
        bytes += source.len - 2;
        for (source[1 .. source.len - 1]) |c| {
            lines += @intFromBool(c == '\n');
        }
    }

    // try stdout.print("-" ** 72 ++ "\n", .{});
    var num_tokens2: usize = 0;
    const legacy_token_lists: if (RUN_LEGACY_TOKENIZER) []Ast.TokenList.Slice else void = if (RUN_LEGACY_TOKENIZER) try gpa.alloc(Ast.TokenList.Slice, sources.items.len);

    const elapsedNanos2: u64 = if (!RUN_LEGACY_TOKENIZER) 0 else blk: {
        const t3 = std.time.nanoTimestamp();
        for (sources.items, legacy_token_lists) |sourcey, *legacy_token_list_slot| {
            const source = sourcey[1..];
            var tokens = Ast.TokenList{};
            defer tokens.deinit(gpa);

            // Empirically, the zig std lib has an 8:1 ratio of source bytes to token count.
            const estimated_token_count = source.len / 8;
            try tokens.ensureTotalCapacity(gpa, estimated_token_count);

            var tokenizer = std.zig.Tokenizer.init(source);
            while (true) {
                const token = tokenizer.next();
                try tokens.append(gpa, .{
                    .tag = token.tag,
                    .start = @as(u32, @intCast(token.loc.start)),
                });
                if (token.tag == .eof) break;
            }

            legacy_token_list_slot.* = tokens.toOwnedSlice();
        }
        const t4 = std.time.nanoTimestamp();

        for (legacy_token_lists) |tokens| {
            num_tokens2 += tokens.len;
        }

        const elapsedNanos2: u64 = @intCast(t4 - t3);

        if (REPORT_SPEED) {
            const throughput = @as(f64, @floatFromInt(bytes * 1000)) / @as(f64, @floatFromInt(elapsedNanos2));
            try stdout.print("\n" ** @intFromBool(RUN_LEGACY_AST or RUN_NEW_AST) ++ "Legacy Tokenizing took {: >9} ({d:.2} MB/s, {d: >5.2}M loc/s) and used {} memory\n", .{ std.fmt.fmtDuration(elapsedNanos2), throughput, @as(f64, @floatFromInt(lines)) / @as(f64, @floatFromInt(elapsedNanos2)) * 1000, std.fmt.fmtIntSizeDec(num_tokens2 * 5) });
            break :blk elapsedNanos2;
        }
    };

    if (RUN_NEW_TOKENIZER or INFIX_TEST) {
        var t1 = std.time.nanoTimestamp();

        const source_tokens = try gpa.alloc([]Token, sources.items.len);
        for (sources.items, source_tokens) |source, *source_token_slot| {
            source_token_slot.* = try Parser.tokenize(gpa, source, 1);
            // const b = try Parser.tokenize(gpa, source, 1);

            // var cur_a = a;
            // var cur_b = b;
            // var cur = source;

            // while (cur_a.len > 0 and cur_b.len > 0) {
            //     if (cur_a[0].kind != cur_b[0].kind) break;
            //     const cur_a_len = switch (cur_a[0].len) {
            //         0 => @as(u32, @bitCast(cur_a[1..3].*)),
            //         else => |l| l,
            //     };

            //     const cur_b_len = switch (cur_b[0].len) {
            //         0 => @as(u32, @bitCast(cur_b[1..3].*)),
            //         else => |l| l,
            //     };
            //     if (cur_a_len != cur_b_len) break;

            //     const adv_amt: usize = if (cur_a[0].len == 0) 3 else 1;
            //     cur_a = cur_a[adv_amt..];
            //     cur_b = cur_b[adv_amt..];
            //     cur = cur[cur_a_len..];
            // } else {
            //     continue;
            // }

            // try stdout.print("{} {s} {s}\n", .{ i, @tagName(cur_a[0].kind), @tagName(cur_b[0].kind) });
            // try stdout.print("\"{s}\"\n", .{cur[0..cur_a[0].len]});
            // {
            //     const adv_amt: usize = if (cur_a[0].len == 0) 3 else 1;
            //     cur = cur[cur_a[0].len..];
            //     cur_a = cur_a[adv_amt..];
            // }
            // try stdout.print("\"{s}\"\n", .{cur[0..cur_a[0].len]});
            // cur = cur[cur_a[0].len..];
            // cur_a = cur_a[if (cur_a[0].len == 0) 3 else 1..];
            // try stdout.print("\"{s}\"\n", .{cur[0..cur_a[0].len]});
            // cur = cur[cur_a[0].len..];
            // cur_a = cur_a[if (cur_a[0].len == 0) 3 else 1..];
            // try stdout.print("\"{s}\"\n", .{cur[0..cur_a[0].len]});
            // break;
        }

        const t2 = std.time.nanoTimestamp();
        const elapsedNanos: u64 = @intCast(t2 - t1);

        var num_tokens: usize = 0;
        for (sources.items, source_tokens, 0..) |source, tokens, i| {
            _ = source;
            _ = i;
            num_tokens += tokens.len;
        }

        // Fun fact: bytes per nanosecond is the same ratio as GB/s
        if (RUN_NEW_TOKENIZER and REPORT_SPEED) {
            var throughput = @as(f64, @floatFromInt(bytes * 1000)) / @as(f64, @floatFromInt(elapsedNanos));
            try stdout.print("       Tokenizing took {: >9} ({d:.2} MB/s, {d: >5.2}M loc/s) and used {} memory\n", .{ std.fmt.fmtDuration(elapsedNanos), throughput, @as(f64, @floatFromInt(lines)) / @as(f64, @floatFromInt(elapsedNanos)) * 1000, std.fmt.fmtIntSizeDec(num_tokens * 2) });

            if (elapsedNanos2 > 0) {
                try stdout.print("       That's {d:.2}x faster and {d:.2}x less memory!\n", .{ @as(f64, @floatFromInt(elapsedNanos2)) / @as(f64, @floatFromInt(elapsedNanos)), @as(f64, @floatFromInt(num_tokens2 * 5)) / @as(f64, @floatFromInt(num_tokens * 2)) });
            }
        }
    }

    const elapsedNanos4: u64 = if (!RUN_LEGACY_AST) 0 else blk: {
        if (!RUN_LEGACY_TOKENIZER) @compileError("Must enable legacy tokenizer to run legacy AST!");
        const legacy_asts = try gpa.alloc(Ast, legacy_token_lists.len);

        const t3 = std.time.nanoTimestamp();
        for (sources.items, legacy_token_lists, legacy_asts) |source, tokens, *ast_slot| {
            var parser: std.zig.Ast.Parse = .{
                .source = source,
                .gpa = gpa,
                .token_tags = tokens.items(.tag),
                .token_starts = tokens.items(.start),
                .errors = .{},
                .nodes = .{},
                .extra_data = .{},
                .scratch = .{},
                .tok_i = 0,
            };
            defer parser.errors.deinit(gpa);
            defer parser.nodes.deinit(gpa);
            defer parser.extra_data.deinit(gpa);
            defer parser.scratch.deinit(gpa);

            // Empirically, Zig source code has a 2:1 ratio of tokens to AST nodes.
            // Make sure at least 1 so we can use appendAssumeCapacity on the root node below.
            const estimated_node_count = (tokens.len + 2) / 2;
            try parser.nodes.ensureTotalCapacity(gpa, estimated_node_count);
            try parser.parseRoot();

            ast_slot.* = Ast{
                .source = source,
                .tokens = tokens,
                .nodes = parser.nodes.toOwnedSlice(),
                .extra_data = try parser.extra_data.toOwnedSlice(gpa),
                .errors = try parser.errors.toOwnedSlice(gpa),
            };
        }
        const t4 = std.time.nanoTimestamp();
        var memory: usize = 0;

        for (legacy_asts) |ast| {
            memory += ast.errors.len * @sizeOf(std.zig.Ast.Error) +
                ast.extra_data.len * @sizeOf(std.zig.Ast.Node.Index) +
                ast.nodes.len * @divExact(@bitSizeOf(std.zig.Ast.Node), 8);
        }

        const elapsedNanos4: u64 = @intCast(t4 - t3);

        if (REPORT_SPEED) {
            const @"GB/s 2" = @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(elapsedNanos4));
            try stdout.print("\nLegacy AST construction took {: >9} ({d:.2} GB/s, {d: >5.2}M loc/s) and used {} memory\n", .{ std.fmt.fmtDuration(elapsedNanos4), @"GB/s 2", @as(f64, @floatFromInt(lines)) / @as(f64, @floatFromInt(elapsedNanos4)) * 1000, std.fmt.fmtIntSizeDec(memory) });
            break :blk elapsedNanos4;
        }
    };
    _ = elapsedNanos4;
}

// ---------------------------------------------------------------
//
// The code below this point is licensed under the Apache License.
// Please see the License at the bottom of this file.
//
// ---------------------------------------------------------------

const Utf8Checker = struct {
    const u8x32 = @Vector(32, u8);
    const u8x64 = @Vector(64, u8);
    const u64x4 = @Vector(4, u64);
    const u32x4 = @Vector(4, u32);
    const u8x16 = @Vector(16, u8);
    const u8x8 = @Vector(8, u8);

    pub const chunk_len = switch (builtin.cpu.arch) {
        .x86_64 => 32,
        .aarch64, .aarch64_be => 16,
        else => 16,
    };
    pub const Chunk = @Vector(chunk_len, u8);
    pub const IChunk = @Vector(chunk_len, i8);
    const half_chunk_len = chunk_len / 2;
    pub const ChunkArr = [chunk_len]u8;

    err: Chunk = zeros,
    prev_input_block: Chunk = zeros,
    prev_incomplete: Chunk = zeros,

    const zeros: ChunkArr = [_]u8{0} ** chunk_len;

    // ---
    // from https://gist.github.com/sharpobject/80dc1b6f3aaeeada8c0e3a04ebc4b60a
    // ---
    // thanks to sharpobject for these implementations which make it possible to get
    // rid of old utils.c and stop linking libc.
    // ---
    fn _mm256_permute2x128_si256_0x21(a: Chunk, b: Chunk) Chunk {
        const uint = std.meta.Int(.unsigned, @bitSizeOf(Chunk) / 4);
        const V = @Vector(4, uint);
        return @bitCast(@shuffle(
            uint,
            @as(V, @bitCast(a)),
            @as(V, @bitCast(b)),
            [_]i32{ 2, 3, -1, -2 },
        ));
    }

    fn _mm256_alignr_epi8(a: Chunk, b: Chunk, comptime imm8: comptime_int) Chunk {
        var ret: Chunk = undefined;
        var i: usize = 0;
        while (i + imm8 < half_chunk_len) : (i += 1) {
            ret[i] = b[i + imm8];
        }
        while (i < half_chunk_len) : (i += 1) {
            ret[i] = a[i + imm8 - half_chunk_len];
        }
        while (i + imm8 < chunk_len) : (i += 1) {
            ret[i] = b[i + imm8];
        }
        while (i < chunk_len) : (i += 1) {
            ret[i] = a[i + imm8 - half_chunk_len];
        }
        return ret;
    }

    fn prev(comptime N: comptime_int, a: Chunk, b: Chunk) Chunk {
        assert(0 < N and N <= 3);
        return _mm256_alignr_epi8(a, _mm256_permute2x128_si256_0x21(b, a), half_chunk_len - N);
    }

    // end from https://gist.github.com/sharpobject/80dc1b6f3aaeeada8c0e3a04ebc4b60a
    pub fn mm_shuffle_epi8(x: Chunk, mask: Chunk) Chunk {
        return asm (
            \\vpshufb %[mask], %[x], %[out]
            : [out] "=x" (-> Chunk),
            : [x] "+x" (x),
              [mask] "x" (mask),
        );
    }

    // https://developer.arm.com/architectures/instruction-sets/intrinsics/vqtbl1q_s8
    pub fn lookup_16_aarch64(x: u8x16, mask: u8x16) u8x16 {
        return asm (
            \\tbl  %[out].16b, {%[mask].16b}, %[x].16b
            : [out] "=&x" (-> u8x16),
            : [x] "x" (x),
              [mask] "x" (mask),
        );
    }

    fn lookup_chunk(comptime a: [16]u8, b: Chunk) Chunk {
        switch (builtin.cpu.arch) {
            .x86_64 => return mm_shuffle_epi8(a ** (chunk_len / 16), b),
            .aarch64, .aarch64_be => return lookup_16_aarch64(b, a ** (chunk_len / 16)),
            else => {
                var r: Chunk = @splat(0);
                for (0..chunk_len) |i| {
                    const c = b[i];
                    assert(c <= 0x0F);
                    r[i] = a[c];
                }
                return r;

                // var r: Chunk = @splat(0);
                // for (0..16) |i| {
                //     inline for ([2]comptime_int{ 0, 16 }) |o| {
                //         if ((b[o + i] & 0x80) == 0) {
                //             r[o + i] = a[o + b[o + i] & 0x0F];
                //         }
                //     }
                // }
                // return r;
            },
        }
    }

    // zig fmt: off
    fn check_special_cases(input: Chunk, prev1: Chunk) Chunk {
        // Bit 0 = Too Short (lead byte/ASCII followed by lead byte/ASCII)
        // Bit 1 = Too Long (ASCII followed by continuation)
        // Bit 2 = Overlong 3-byte
        // Bit 4 = Surrogate
        // Bit 5 = Overlong 2-byte
        // Bit 7 = Two Continuations
        const TOO_SHORT:  u8 = 1 << 0;  // 11______ 0_______
                                        // 11______ 11______
        const TOO_LONG:   u8 = 1 << 1;  // 0_______ 10______
        const OVERLONG_3: u8 = 1 << 2;  // 11100000 100_____
        const SURROGATE:  u8 = 1 << 4;  // 11101101 101_____
        const OVERLONG_2: u8 = 1 << 5;  // 1100000_ 10______
        const TWO_CONTS:  u8 = 1 << 7;  // 10______ 10______
        const TOO_LARGE:  u8 = 1 << 3;  // 11110100 1001____
                                        // 11110100 101_____
                                        // 11110101 1001____
                                        // 11110101 101_____
                                        // 1111011_ 1001____
                                        // 1111011_ 101_____
                                        // 11111___ 1001____
                                        // 11111___ 101_____
        const TOO_LARGE_1000: u8 = 1 << 6;
                                        // 11110101 1000____
                                        // 1111011_ 1000____
                                        // 11111___ 1000____
        const OVERLONG_4: u8 = 1 << 6;  // 11110000 1000____

        const u3xchunk_len = @Vector(chunk_len, u3);
        const byte_1_high_0 = prev1 >> @as(u3xchunk_len, @splat(4));
        const tbl1 = [16]u8{
            // 0_______ ________ <ASCII in byte 1>
            TOO_LONG,               TOO_LONG,  TOO_LONG,                           TOO_LONG,
            TOO_LONG,               TOO_LONG,  TOO_LONG,                           TOO_LONG,
            // 10______ ________ <continuation in byte 1>
            TWO_CONTS,              TWO_CONTS, TWO_CONTS,                          TWO_CONTS,
            // 1100____ ________ <two byte lead in byte 1>
            TOO_SHORT | OVERLONG_2,
            // 1101____ ________ <two byte lead in byte 1>
            TOO_SHORT,
            // 1110____ ________ <three byte lead in byte 1>
            TOO_SHORT | OVERLONG_3 | SURROGATE,
            // 1111____ ________ <four+ byte lead in byte 1>
            TOO_SHORT | TOO_LARGE | TOO_LARGE_1000 | OVERLONG_4,
        };

        const byte_1_high = lookup_chunk(tbl1, byte_1_high_0);
        const CARRY: u8 = TOO_SHORT | TOO_LONG | TWO_CONTS; // These all have ____ in byte 1 .
        const byte_1_low0 = prev1 & @as(Chunk, @splat(0x0F));

        const tbl2 = [16]u8{
            // ____0000 ________
            CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4,
            // ____0001 ________
            CARRY | OVERLONG_2,
            // ____001_ ________
            CARRY,
            CARRY,

            // ____0100 ________
            CARRY | TOO_LARGE,
            // ____0101 ________
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            // ____011_ ________
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            CARRY | TOO_LARGE | TOO_LARGE_1000,

            // ____1___ ________
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            // ____1101 ________
            CARRY | TOO_LARGE | TOO_LARGE_1000 | SURROGATE,
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            CARRY | TOO_LARGE | TOO_LARGE_1000,
        };
        const byte_1_low = lookup_chunk(tbl2, byte_1_low0);

        const byte_2_high_0 = input >> @as(u3xchunk_len, @splat(4));
        const tbl3 = [16]u8{
            // ________ 0_______ <ASCII in byte 2>
            TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,
            TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,
            // ________ 1000____
            TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE_1000 | OVERLONG_4,
            // ________ 1001____
            TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE,
            // ________ 101_____
            TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE, TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
            // ________ 11______
            TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,
        };
        const byte_2_high = lookup_chunk(tbl3, byte_2_high_0);
        return byte_1_high & byte_1_low & byte_2_high;
    }
    // zig fmt: on

    fn check_multibyte_lengths(input: Chunk, prev_input: Chunk, sc: Chunk) Chunk {
        const prev2 = prev(2, input, prev_input);
        const prev3 = prev(3, input, prev_input);
        const must23 = must_be_2_3_continuation(prev2, prev3);
        const must23_80 = must23 & @as(Chunk, @splat(0x80));
        return must23_80 ^ sc;
    }

    fn must_be_2_3_continuation(prev2: Chunk, prev3: Chunk) Chunk {
        // do unsigned saturating subtraction, then interpret as signed so we can check if > 0 below
        // Only 111_____ will be > 0
        const is_third_byte: IChunk = @bitCast(prev2 -| @as(Chunk, @splat(0b11100000 - 1)));
        // Only 1111____ will be > 0
        const is_fourth_byte: IChunk = @bitCast(prev3 -| @as(Chunk, @splat(0b11110000 - 1)));

        // Caller requires a bool (all 1's). All values resulting from the subtraction will be <= 64, so signed comparison is fine.
        const i1xchunk_len = @Vector(chunk_len, i1);
        const result = @as(i1xchunk_len, @bitCast((is_third_byte | is_fourth_byte) > @as(@Vector(chunk_len, i8), @splat(0))));
        return @as(Chunk, @bitCast(@as(IChunk, result)));
    }

    //
    // Check whether the current bytes are valid UTF-8.
    //
    fn check_utf8_bytes(checker: *Utf8Checker, input: Chunk, prev_input: Chunk) void {
        // Flip prev1...prev3 so we can easily determine if they are 2+, 3+ or 4+ lead bytes
        // (2, 3, 4-byte leads become large positive numbers instead of small negative numbers)
        const prev1 = prev(1, input, prev_input);
        const sc = check_special_cases(input, prev1);
        checker.err |= check_multibyte_lengths(input, prev_input, sc);
    }

    // The only problem that can happen at EOF is that a multibyte character is too short
    // or a byte value too large in the last bytes: check_special_cases only checks for bytes
    // too large in the first of two bytes.
    fn check_eof(checker: *Utf8Checker) void {
        // If the previous block had incomplete UTF-8 characters at the end, an ASCII block can't
        // possibly finish them.
        checker.err |= checker.prev_incomplete;
    }

    fn is_ascii(input: u8x64) bool {
        //--- TODO: We can remove this optimization once it is integrated into LLVM
        // https://github.com/llvm/llvm-project/issues/66159
        const bytes: [64]i8 = @bitCast(input);
        var chunks_or: IChunk = @splat(0);
        inline for (0..64 / chunk_len) |i| chunks_or |= bytes[chunk_len * i .. chunk_len * (i + 1)].*;
        return 0 == std.simd.countTrues(chunks_or < @as(IChunk, @splat(0)));
        //---
        // return 0 == @as(u64, @bitCast(input >= @as(u8x64, @splat(0x80))));
    }

    fn check_next_input(checker: *Utf8Checker, input: u8x64) void {
        if (is_ascii(input)) {
            checker.err |= checker.prev_incomplete;
            return;
        }

        assert(chunk_len <= 64);
        const NUM_CHUNKS = 64 / chunk_len;
        const chunks = @as([NUM_CHUNKS][chunk_len]u8, @bitCast(input));
        checker.check_utf8_bytes(chunks[0], checker.prev_input_block);
        inline for (1..NUM_CHUNKS) |i| {
            checker.check_utf8_bytes(chunks[i], chunks[i - 1]);
        }
        checker.prev_incomplete = is_incomplete(chunks[NUM_CHUNKS - 1]);
        checker.prev_input_block = chunks[NUM_CHUNKS - 1];
    }

    // do not forget to call check_eof!
    fn errors(checker: Utf8Checker) error{InvalidUtf8}!void {
        if (@reduce(.Or, checker.err) != 0) return error.InvalidUtf8;
    }

    //
    // Return nonzero if there are incomplete multibyte characters at the end of the block:
    // e.g. if there is a 4-byte character, but it's 3 bytes from the end.
    //
    fn is_incomplete(input: Chunk) Chunk {
        // If the previous input's last 3 bytes match this, they're too short (they ended at EOF):
        // ... 1111____ 111_____ 11______

        const max_value = comptime max_value: {
            var max_array: Chunk = @splat(255);
            max_array[chunk_len - 3] = 0b11110000 - 1;
            max_array[chunk_len - 2] = 0b11100000 - 1;
            max_array[chunk_len - 1] = 0b11000000 - 1;
            break :max_value max_array;
        };
        return input -| max_value;
    }
};

//                                  Apache License
//                            Version 2.0, January 2004
//                         http://www.apache.org/licenses/

//    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

//    1. Definitions.

//       "License" shall mean the terms and conditions for use, reproduction,
//       and distribution as defined by Sections 1 through 9 of this document.

//       "Licensor" shall mean the copyright owner or entity authorized by
//       the copyright owner that is granting the License.

//       "Legal Entity" shall mean the union of the acting entity and all
//       other entities that control, are controlled by, or are under common
//       control with that entity. For the purposes of this definition,
//       "control" means (i) the power, direct or indirect, to cause the
//       direction or management of such entity, whether by contract or
//       otherwise, or (ii) ownership of fifty percent (50%) or more of the
//       outstanding shares, or (iii) beneficial ownership of such entity.

//       "You" (or "Your") shall mean an individual or Legal Entity
//       exercising permissions granted by this License.

//       "Source" form shall mean the preferred form for making modifications,
//       including but not limited to software source code, documentation
//       source, and configuration files.

//       "Object" form shall mean any form resulting from mechanical
//       transformation or translation of a Source form, including but
//       not limited to compiled object code, generated documentation,
//       and conversions to other media types.

//       "Work" shall mean the work of authorship, whether in Source or
//       Object form, made available under the License, as indicated by a
//       copyright notice that is included in or attached to the work
//       (an example is provided in the Appendix below).

//       "Derivative Works" shall mean any work, whether in Source or Object
//       form, that is based on (or derived from) the Work and for which the
//       editorial revisions, annotations, elaborations, or other modifications
//       represent, as a whole, an original work of authorship. For the purposes
//       of this License, Derivative Works shall not include works that remain
//       separable from, or merely link (or bind by name) to the interfaces of,
//       the Work and Derivative Works thereof.

//       "Contribution" shall mean any work of authorship, including
//       the original version of the Work and any modifications or additions
//       to that Work or Derivative Works thereof, that is intentionally
//       submitted to Licensor for inclusion in the Work by the copyright owner
//       or by an individual or Legal Entity authorized to submit on behalf of
//       the copyright owner. For the purposes of this definition, "submitted"
//       means any form of electronic, verbal, or written communication sent
//       to the Licensor or its representatives, including but not limited to
//       communication on electronic mailing lists, source code control systems,
//       and issue tracking systems that are managed by, or on behalf of, the
//       Licensor for the purpose of discussing and improving the Work, but
//       excluding communication that is conspicuously marked or otherwise
//       designated in writing by the copyright owner as "Not a Contribution."

//       "Contributor" shall mean Licensor and any individual or Legal Entity
//       on behalf of whom a Contribution has been received by Licensor and
//       subsequently incorporated within the Work.

//    2. Grant of Copyright License. Subject to the terms and conditions of
//       this License, each Contributor hereby grants to You a perpetual,
//       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
//       copyright license to reproduce, prepare Derivative Works of,
//       publicly display, publicly perform, sublicense, and distribute the
//       Work and such Derivative Works in Source or Object form.

//    3. Grant of Patent License. Subject to the terms and conditions of
//       this License, each Contributor hereby grants to You a perpetual,
//       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
//       (except as stated in this section) patent license to make, have made,
//       use, offer to sell, sell, import, and otherwise transfer the Work,
//       where such license applies only to those patent claims licensable
//       by such Contributor that are necessarily infringed by their
//       Contribution(s) alone or by combination of their Contribution(s)
//       with the Work to which such Contribution(s) was submitted. If You
//       institute patent litigation against any entity (including a
//       cross-claim or counterclaim in a lawsuit) alleging that the Work
//       or a Contribution incorporated within the Work constitutes direct
//       or contributory patent infringement, then any patent licenses
//       granted to You under this License for that Work shall terminate
//       as of the date such litigation is filed.

//    4. Redistribution. You may reproduce and distribute copies of the
//       Work or Derivative Works thereof in any medium, with or without
//       modifications, and in Source or Object form, provided that You
//       meet the following conditions:

//       (a) You must give any other recipients of the Work or
//           Derivative Works a copy of this License; and

//       (b) You must cause any modified files to carry prominent notices
//           stating that You changed the files; and

//       (c) You must retain, in the Source form of any Derivative Works
//           that You distribute, all copyright, patent, trademark, and
//           attribution notices from the Source form of the Work,
//           excluding those notices that do not pertain to any part of
//           the Derivative Works; and

//       (d) If the Work includes a "NOTICE" text file as part of its
//           distribution, then any Derivative Works that You distribute must
//           include a readable copy of the attribution notices contained
//           within such NOTICE file, excluding those notices that do not
//           pertain to any part of the Derivative Works, in at least one
//           of the following places: within a NOTICE text file distributed
//           as part of the Derivative Works; within the Source form or
//           documentation, if provided along with the Derivative Works; or,
//           within a display generated by the Derivative Works, if and
//           wherever such third-party notices normally appear. The contents
//           of the NOTICE file are for informational purposes only and
//           do not modify the License. You may add Your own attribution
//           notices within Derivative Works that You distribute, alongside
//           or as an addendum to the NOTICE text from the Work, provided
//           that such additional attribution notices cannot be construed
//           as modifying the License.

//       You may add Your own copyright statement to Your modifications and
//       may provide additional or different license terms and conditions
//       for use, reproduction, or distribution of Your modifications, or
//       for any such Derivative Works as a whole, provided Your use,
//       reproduction, and distribution of the Work otherwise complies with
//       the conditions stated in this License.

//    5. Submission of Contributions. Unless You explicitly state otherwise,
//       any Contribution intentionally submitted for inclusion in the Work
//       by You to the Licensor shall be under the terms and conditions of
//       this License, without any additional terms or conditions.
//       Notwithstanding the above, nothing herein shall supersede or modify
//       the terms of any separate license agreement you may have executed
//       with Licensor regarding such Contributions.

//    6. Trademarks. This License does not grant permission to use the trade
//       names, trademarks, service marks, or product names of the Licensor,
//       except as required for reasonable and customary use in describing the
//       origin of the Work and reproducing the content of the NOTICE file.

//    7. Disclaimer of Warranty. Unless required by applicable law or
//       agreed to in writing, Licensor provides the Work (and each
//       Contributor provides its Contributions) on an "AS IS" BASIS,
//       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
//       implied, including, without limitation, any warranties or conditions
//       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
//       PARTICULAR PURPOSE. You are solely responsible for determining the
//       appropriateness of using or redistributing the Work and assume any
//       risks associated with Your exercise of permissions under this License.

//    8. Limitation of Liability. In no event and under no legal theory,
//       whether in tort (including negligence), contract, or otherwise,
//       unless required by applicable law (such as deliberate and grossly
//       negligent acts) or agreed to in writing, shall any Contributor be
//       liable to You for damages, including any direct, indirect, special,
//       incidental, or consequential damages of any character arising as a
//       result of this License or out of the use or inability to use the
//       Work (including but not limited to damages for loss of goodwill,
//       work stoppage, computer failure or malfunction, or any and all
//       other commercial damages or losses), even if such Contributor
//       has been advised of the possibility of such damages.

//    9. Accepting Warranty or Additional Liability. While redistributing
//       the Work or Derivative Works thereof, You may choose to offer,
//       and charge a fee for, acceptance of support, warranty, indemnity,
//       or other liability obligations and/or rights consistent with this
//       License. However, in accepting such obligations, You may act only
//       on Your own behalf and on Your sole responsibility, not on behalf
//       of any other Contributor, and only if You agree to indemnify,
//       defend, and hold each Contributor harmless for any liability
//       incurred by, or claims asserted against, such Contributor by reason
//       of your accepting any such warranty or additional liability.

//    END OF TERMS AND CONDITIONS

//    APPENDIX: How to apply the Apache License to your work.

//       To apply the Apache License to your work, attach the following
//       boilerplate notice, with the fields enclosed by brackets "[]"
//       replaced with your own identifying information. (Don't include
//       the brackets!)  The text should be enclosed in the appropriate
//       comment syntax for the file format. We also recommend that a
//       file or class name and description of purpose be included on the
//       same "printed page" as the copyright notice for easier
//       identification within third-party archives.

//    Copyright [yyyy] [name of copyright owner]

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
