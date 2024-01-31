// zig fmt: off
const SKIP_OUTLIERS        = false;
const RUN_LEGACY_TOKENIZER = false;
const RUN_NEW_TOKENIZER    = false;
const RUN_LEGACY_AST       = false;
const RUN_NEW_AST          = true;
const REPORT_SPEED         = true;
const INFIX_TEST           = true;
// zig fmt: on

// TODO: move the unary stuff so it is in mostly one location
// TODO: figure out what behavior we should have for invalid vs unknown
// TODO: 64-bit popcounts could probably be refactored to be nicer on 32-bit machines.
// TODO: the quote algo only works for little-endian bit orders
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

/// This is the "efficient builtin operand" size.
/// E.g. if we support 64-bit operations, we want to be doing 64-bit count-trailing-zeros.
/// For now, we use `usize` as our reasonable guess for what size of bitstring we can operate on efficiently.
const uword = std.meta.Int(.unsigned, if (std.mem.endsWith(u8, @tagName(builtin.cpu.arch), "64_32")) 64 else @bitSizeOf(usize));

const IS_VECTORIZER_BROKEN = builtin.cpu.arch.isSPARC() or builtin.cpu.arch.isPPC() or builtin.cpu.arch.isPPC64();
const SUGGESTED_VEC_SIZE = if (IS_VECTORIZER_BROKEN) null else std.simd.suggestVectorLengthForCpu(u8, builtin.cpu);
const USE_SWAR = SUGGESTED_VEC_SIZE == null;

// This is the native vector size.
const NATIVE_VEC_INT = std.meta.Int(.unsigned, 8 * (SUGGESTED_VEC_SIZE orelse @sizeOf(uword)));
const NATIVE_VEC_SIZE = @sizeOf(NATIVE_VEC_INT);
const NATIVE_CHAR_VEC = @Vector(NATIVE_VEC_SIZE, u8);

const Chunk = if (USE_SWAR) NATIVE_VEC_INT else @Vector(NATIVE_VEC_SIZE, u8);

const FRONT_SENTINELS = "\n";
// const BACK_SENTINELS_OLD = "\n" ++ "\x00" ** 13 ++ " ";
const BACK_SENTINELS = "\n" ++ "\x00" ** 62 ++ " ";
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

fn readFiles(gpa: Allocator) !std.ArrayListUnmanaged([:0]align(@bitSizeOf(uword)) const u8) {
    if (SKIP_OUTLIERS)
        std.debug.print("Skipping outliers!\n", .{});
    std.debug.print("v0.7\n", .{});
    const directory = switch (INFIX_TEST) {
        true => "./src/beep",
        false => "./src/files_to_parse",
    };
    var parent_dir2 = try std.fs.cwd().openDirZ(directory, .{ .iterate = false });
    defer parent_dir2.close();

    var parent_dir = try std.fs.cwd().openDirZ(directory, .{ .iterate = true });
    defer parent_dir.close();

    var num_files: usize = 0;
    var num_bytes: usize = 0;

    var sources: std.ArrayListUnmanaged([:0]align(@bitSizeOf(uword)) const u8) = .{};
    {
        const t1 = std.time.nanoTimestamp();
        var walker = try parent_dir.walk(gpa); // 12-14 ms just walking the tree
        defer walker.deinit();

        const total_size: usize = 0;
        _ = total_size;
        while (try walker.next()) |dir| {
            switch (dir.kind) {
                .file => if (dir.basename.len > 4 and std.mem.eql(u8, dir.basename[dir.basename.len - 4 ..][0..4], ".zig") and dir.path.len - dir.basename.len > 0) {
                    // These two are extreme outliers, omit them from our test bench
                    if (std.mem.eql(u8, dir.basename, "normal_string_with_newline.zig") or (SKIP_OUTLIERS and
                        (std.mem.eql(u8, dir.basename, "udivmodti4_test.zig") or
                        std.mem.eql(u8, dir.basename, "udivmoddi4_test.zig"))))
                        continue;

                    const file = try parent_dir2.openFile(dir.path, .{});
                    defer file.close();

                    num_files += 1;
                    const source = try readFileIntoAlignedBuffer(gpa, file, @bitSizeOf(uword));
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

fn vec_cmp(a: anytype, comptime cmp_type: enum { @"<", @"<=", @"==", @"!=", @">", @">=" }, x: anytype) @TypeOf(a) {
    const child_type = @typeInfo(@TypeOf(a)).Vector.child;
    const true_vec = @as(@TypeOf(a), @splat(std.math.maxInt(child_type)));
    const false_vec = @as(@TypeOf(a), @splat(0));

    const b: @TypeOf(a) = switch (@TypeOf(x)) {
        @TypeOf(a) => x,
        child_type => @splat(x),
        else => unreachable,
    };

    return @select(child_type, switch (cmp_type) {
        .@"<" => a < b,
        .@"<=" => a <= b,
        .@"==" => a == b,
        .@"!=" => a != b,
        .@">" => a > b,
        .@">=" => a >= b,
    }, true_vec, false_vec);
}

const Operators = struct {
    const unpadded_ops = [_][]const u8{ ".**", "!", "|", "||", "|=", "=", "==", "=>", "!=", "(", ")", ";", "%", "%=", "{", "}", "[", "]", ".", ".*", "..", "...", "^", "^=", "+", "++", "+=", "+%", "+%=", "+|", "+|=", "-", "-=", "-%", "-%=", "-|", "-|=", "*", "*=", "**", "*%", "*%=", "*|", "*|=", "->", ":", "/", "/=", "&", "&=", "?", "<", "<=", "<<", "<<=", "<<|", "<<|=", ">", ">=", ">>", ">>=", "~", "//", "///", "//!", ".?", "\\\\", "," };

    const potentially_unary = [_][]const u8{ "&", "-", "-%", "*", "**", ".", ".." };

    fn unarifyBinaryOperatorRaw(hash: u8) u8 {
        return pext(hash *% 29, 0b11000100) +% 150;
    }

    fn unarifyBinaryOperator(tag: Tag) Tag {
        return @enumFromInt(unarifyBinaryOperatorRaw(@intFromEnum(tag)));
    }

    fn postifyOperatorRaw(hash: u8) u8 {
        return hash + 3;
    }

    fn postifyOperator(tag: Tag) Tag {
        return @enumFromInt(postifyOperatorRaw(@intFromEnum(tag)));
    }

    // We do `| 1` if the context is such that
    const TagClassification = enum(u8) {
        binary_op = 0,
        pre_unary_op = 1,
        ambiguous_pre_unary_or_binary = 2,
        post_unary_op = 3,
        operand = 4,
        post_unary_ctx_reset_op = 5,
        something = 6,
        ambiguous_pre_unary_or_post_unary = 7,
    };

    fn classify(op_type: Tag) TagClassification {
        const OperatorClassifications = comptime blk: {
            var table = [1]TagClassification{.something} ** 256;

            for (table[Parser.BitmapKind.min_bitmap_value .. Parser.BitmapKind.max_bitmap_value + 1]) |*slot|
                slot.* = TagClassification.operand;

            for ([_]struct { TagClassification, []const Tag }{
                .{ .ambiguous_pre_unary_or_binary, &[_]Tag{ .@".", .@"-", .@"-%", .@"*", .@"**", .@"&" } },
                .{ .pre_unary_op, &[_]Tag{ .@"!", .@"const", .@"fn" } },
                .{ .ambiguous_pre_unary_or_post_unary, &[_]Tag{.@"("} },
                .{ .post_unary_op, &[_]Tag{.@"call ("} },
                .{ .pre_unary_op, &[_]Tag{ .@"unary &", .@"unary -", .@"unary -%", .@"unary *", .@"unary **", .@"unary .", .@"unary .." } },
                .{ .binary_op, &[_]Tag{ .@"or", .@"and", .@"orelse", .@"catch" } },
                .{ .binary_op, &[_]Tag{ .@"||", .@"|", .@"=", .@"==", .@"=>", .@"|=", .@"!=", .@"%", .@"%=", .@"..", .@"^", .@"^=", .@"+", .@"++", .@"+=", .@"+%", .@"+%=", .@"+|", .@"+|=", .@"-=", .@"-%=", .@"-|", .@"-|=", .@"*=", .@"*%", .@"*%=", .@"*|", .@"*|=", .@"/", .@"/=", .@"&=", .@"<", .@"<=", .@"<<", .@"<<=", .@"<<|", .@"<<|=", .@">", .@">=", .@">>", .@">>=", .@"...", .@":" } },
                .{ .post_unary_op, &[_]Tag{ .@".*", .@".?" } },
                .{ .post_unary_ctx_reset_op, &[_]Tag{ .@")", .@";" } },
                .{ .something, &[_]Tag{ .@".**", .@"{", .@"}", .@"[", .@"]", .@"->", .@"?", .@"~", .@"//", .@"///", .@"//!", .@"\\\\", .@"," } },
            }) |data| {
                for (data[1]) |tag| {
                    assert(table[@intFromEnum(tag)] == .something);
                    table[@intFromEnum(tag)] = data[0];
                }
            }
            break :blk table;
        };

        return OperatorClassifications[@intFromEnum(op_type)];
    }

    fn isUnary(op_type: Tag) bool {
        comptime for (std.meta.fields(TagClassification)) |field| {
            if (1 == (1 & field.value)) {
                switch (@as(TagClassification, @enumFromInt(field.value))) {
                    .pre_unary_op,
                    .post_unary_op,
                    .post_unary_ctx_reset_op,
                    .ambiguous_pre_unary_or_post_unary,
                    => {},
                    else => {
                        var alternative = field.value + 1;
                        while (true) : (alternative += 2)
                            std.meta.intToEnum(TagClassification, alternative) catch
                                @compileError(std.fmt.comptimePrint(
                                "Only unary operators should have the least significant bit set in their value. Please change {s} to {}\n",
                                .{ field.name, alternative },
                            ));
                    },
                }
            }
        };

        return 1 == (1 & @intFromEnum(classify(op_type)));
    }

    fn getPrecedence(tag: Tag) u8 {
        comptime var lookup_table = std.mem.zeroes([256]u8);
        comptime {
            for ([_][]const Tag{
                &[_]Tag{.eof},
                &[_]Tag{.sentinel_operator},
                &[_]Tag{.@";"},
                &[_]Tag{ .@"(", .@"call (" },
                &[_]Tag{.@")"},
                &[_]Tag{.@","},
                &[_]Tag{.@"="},
                &[_]Tag{.@":"},

                // Binary operators
                &[_]Tag{.@"or"},
                &[_]Tag{.@"and"},
                &[_]Tag{ .@"==", .@"!=", .@"<", .@">", .@"<=", .@">=" },
                &[_]Tag{ .@"&", .@"^", .@"|", .@"orelse", .@"catch" },
                &[_]Tag{ .@"<<", .@"<<|", .@">>" },
                &[_]Tag{ .@"+", .@"-", .@"++", .@"+%", .@"-%", .@"+|", .@"-|" },
                &[_]Tag{ .@"||", .@"*", .@"/", .@"%", .@"**", .@"*%", .@"*|" },

                &[_]Tag{ .@"!", .@"unary &", .@"unary *", .@"unary **", .@"unary -", .@"unary -%", .@"unary .", .@"unary .." },
                &[_]Tag{ .@"const", .@"fn" },
                &[_]Tag{ .@".*", .@".?" },
                &[_]Tag{.@"."},
            }, 1..) |ops, i| {
                for (ops) |op| {
                    assert(lookup_table[@intFromEnum(op)] == 0);
                    lookup_table[@intFromEnum(op)] = i;
                }
            }

            assert(lookup_table[@intFromEnum(Tag.@":")] > lookup_table[@intFromEnum(Tag.@"call (")]);
            assert(lookup_table[@intFromEnum(Tag.@")")] <= lookup_table[@intFromEnum(Tag.@",")]);
        }

        const result = lookup_table[@intFromEnum(tag)];
        assert(result != 0);
        return result;
    }

    // TODO: add assertion that this only works because the maximum support op length is currently 4

    const padded_ops: [unpadded_ops.len][4]u8 = blk: {
        var padded_ops_table: [unpadded_ops.len][4]u8 = undefined;
        for (unpadded_ops, 0..) |op, i| {
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

    fn getOpWord(op: [*]const u8, len: u32) u32 { // TODO: make this work with big-endian
        comptime assert(BACK_SENTINELS.len >= 3);
        const shift_amt = len * 8;
        const relevant_mask: u32 = @intCast((@as(u64, 1) << @intCast(shift_amt)) -% 1);
        return std.mem.readInt(u32, op[0..4], .little) & relevant_mask;
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

        return if (std.mem.readInt(u32, &sorted_padded_ops[mapToIndexRaw(hash)], .little) == op_word)
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
    const LOOKUP_IMPL: u1 = @intFromBool(builtin.mode == .ReleaseSmall);
    const SWAR_LOOKUP_IMPL: u1 = @intFromBool(builtin.mode == .ReleaseSmall);

    // We could halve the number of cache lines by using two-byte slices into a dense superstring:
    const kw_buffer: []const u8 = "addrspacerrdeferrorelsenumswitchunreachablepackedforeturnunionwhilecontinueconstructcomptimevolatileifnpubreakawaitestryasyncatchlinksectionosuspendanytypeanyframeandallowzeropaquexporthreadlocallconvaresumexternoinlinealignoaliasmusingnamespace_\x00";

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

    const PADDING_RIGHT = std.math.ceilPowerOfTwo(u64, max_kw_len + 1) catch unreachable;
    const padded_int = std.meta.Int(.unsigned, PADDING_RIGHT);

    const sorted_padded_kws align(PADDING_RIGHT) = blk: {
        const max_hash_is_unused = (masks[masks.len - 1] >> 63) ^ 1;
        var buffer: [unpadded_kws.len + max_hash_is_unused][PADDING_RIGHT]u8 = undefined;

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

    const kw_slice = struct {
        const uint = std.meta.Int(
            .unsigned,
            std.math.ceilPowerOfTwoPromote(u64, std.math.log2_int_ceil(u64, kw_buffer.len + 1)),
        );
        start_index: uint,
        len: uint,
    };

    const kw_slices = blk: {
        @setEvalBranchQuota(std.math.maxInt(u16));

        const max_hash_is_unused = (masks[masks.len - 1] >> 63) ^ 1;
        var buffer: [unpadded_kws.len + max_hash_is_unused]kw_slice = undefined;

        for (unpadded_kws) |kw|
            buffer[mapToIndex(hashKw(kw.ptr, kw.len))] = .{
                .start_index = std.mem.indexOf(u8, kw_buffer, kw) orelse @compileError(std.fmt.comptimePrint("Please include the keyword \"{s}\" inside of `kw_buffer`.\n\n`kw_buffer` is currently:\n{s}\n", .{ kw, kw_buffer })),
                .len = kw.len,
            };

        if (max_hash_is_unused == 1) {
            // We add one extra filler item just in case we hash a value greater than the greatest hashed value
            buffer[unpadded_kws.len] = std.mem.zeroes(kw_slice);
        }

        break :blk buffer;
    };

    const kw_slices_raw = blk: {
        @setEvalBranchQuota(std.math.maxInt(u16));

        const max_hash_is_unused = (masks[masks.len - 1] >> 63) ^ 1;
        var buffer = std.mem.zeroes([std.math.maxInt(u7) + 1]kw_slice);

        for (unpadded_kws) |kw|
            buffer[hashKw(kw.ptr, kw.len)] = .{
                .start_index = std.mem.indexOf(u8, kw_buffer, kw) orelse @compileError(std.fmt.comptimePrint("Please include the keyword \"{s}\" inside of `kw_buffer`.\n\n`kw_buffer` is currently:\n{s}\n", .{ kw, kw_buffer })),
                .len = kw.len,
            };

        if (max_hash_is_unused == 1) {
            // We add one extra filler item just in case we hash a value greater than the greatest hashed value
            buffer[unpadded_kws.len] = std.mem.zeroes(kw_slice);
        }

        break :blk buffer;
    };

    pub fn hashKw(keyword: [*]const u8, len: u32) u7 {
        assert(len != 0);
        comptime assert(BACK_SENTINELS.len >= 1); // Make sure it's safe to go forward a character when len=1
        const a = std.mem.readInt(u16, keyword[0..2], .little);
        comptime assert(FRONT_SENTINELS.len >= 1); // Make sure it's safe to go back to the previous character when len=1
        const b = std.mem.readInt(u16, (if (@inComptime())
            keyword[len - 2 .. len][0..2]
        else
            keyword - 2 + @as(usize, @intCast(len)))[0..2], .little);
        return @truncate(((a ^ (len << 14)) *% b) >> 8);
        // return @truncate(((a >> 1) *% (b >> 1) ^ (len << 14)) >> 8);
        // return @truncate((((a >> 1) *% (b >> 1)) >> 8) ^ (len << 6));
    }

    fn hasIndex(hash: u7) bool {
        return 1 == @as(u1, @truncate(masks[hash / 64] >> @truncate(hash)));
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

        assert(len != 0);
        const len_u8: u8 = @truncate(len); // std.math.cast(u8, len) orelse return null;
        const raw_hash = hashKw(kw, len);
        const hash = mapToIndex(raw_hash);

        if (USE_SWAR) { // TODO: make sure this works with big-endian
            switch (SWAR_LOOKUP_IMPL) {
                0 => {
                    const val_len = sorted_padded_kws[hash][PADDING_RIGHT - 1]; // [min_kw_len, max_kw_len]
                    if (len != val_len) return null;

                    const Q = u64;
                    const native_endian = comptime builtin.cpu.arch.endian();
                    const log_int_t = std.math.Log2Int(std.meta.Int(.unsigned, @sizeOf(Q)));
                    const misalignment_kernel = @as(log_int_t, @truncate(@intFromPtr(kw)));
                    const misalignment = @as(std.meta.Int(.unsigned, @bitSizeOf(log_int_t) + 3), misalignment_kernel) * 8;
                    const chunk1: [*]align(@sizeOf(Q)) const u8 = @ptrFromInt(std.mem.alignBackward(usize, @intFromPtr(kw), @sizeOf(Q)));
                    const int1 = std.mem.readInt(Q, chunk1[0..@sizeOf(Q)], native_endian);

                    const final1 = switch (native_endian) {
                        .little => int1 >> misalignment,
                        .big => int1 << misalignment,
                    };
                    const chunk2: [*]align(@sizeOf(Q)) const u8 = chunk1 + @sizeOf(Q);
                    const int2 = std.mem.readInt(Q, chunk2[0..@sizeOf(Q)], native_endian);

                    const other = ~misalignment +% 1;
                    const selector = @as(Q, @intFromBool(misalignment == 0)) -% 1;

                    const final2 = switch (native_endian) {
                        .little => int2 << other,
                        .big => int2 >> other,
                    };

                    const final3 = switch (native_endian) {
                        .little => int2 >> misalignment,
                        .big => int2 << misalignment,
                    };

                    const chunk4: [*]align(@sizeOf(Q)) const u8 = chunk2 + @sizeOf(Q);
                    const int4 = std.mem.readInt(Q, chunk4[0..@sizeOf(Q)], native_endian);

                    const final4 = switch (native_endian) {
                        .little => int4 << other,
                        .big => int4 >> other,
                    };

                    var word1: Q = final1 | (final2 & selector);
                    var word2: Q = final3 | (final4 & selector);

                    const byte_len = len * 8;

                    if (byte_len < 64) {
                        assert(byte_len != 0);
                        const shift1: u6 = @intCast(64 - byte_len);
                        word1 = word1 << shift1 >> shift1;
                    }

                    const shift2 = 128 - byte_len; // TODO: can we use wraparound here?
                    word2 = if (shift2 >= 64) 0 else word2 << @intCast(shift2) >> @intCast(shift2);

                    switch (native_endian) {
                        .little => word2 |= @as(@TypeOf(word2), len) << (@bitSizeOf(@TypeOf(word2)) - 8),
                        .big => word2 |= @as(@TypeOf(word2), len),
                    }

                    const source_word1 = @as(*align(8) const u64, @alignCast(@ptrCast(sorted_padded_kws[hash][0..8].ptr))).*;
                    const source_word2 = @as(*align(8) const u64, @alignCast(@ptrCast(sorted_padded_kws[hash][8..16].ptr))).*; // << 8 >> 8;

                    if (word1 == source_word1 and word2 == source_word2) {
                        return @enumFromInt(~hash);
                    } else {
                        return null;
                    }
                },
                1 => {
                    const str = kw_slices[hash];
                    if (len != str.len) return null;
                    for (kw[0..len], kw_buffer[str.start_index..][0..str.len]) |c1, c2| {
                        if (c1 != c2) return null;
                    }
                    return @enumFromInt(~hash);
                },
            }
        } else {
            switch (LOOKUP_IMPL) {
                0 => {
                    // This algorithm compares two vectors of the form `<keyword><len...>`
                    // Where:
                    //   <keyword> is the identifier we found of the form [a-z][a-zA-Z0-9_]*
                    //   <len..> is the len byte duplicated until the end of the vector
                    // Here are real examples of a keyword comparisons:
                    // vec1: enum444444444444
                    // cd  : allocator9999999

                    // vec1: anyframe88888888
                    // cd  : appendAssumeCapa

                    // How is it guaranteed that the length cannot be erroneously matched as a character?
                    // Since the keywords we are matching both can only have characters in the set [a-zA-Z0-9_],
                    // in order for the length to potentially match a character, the length must be at least '0'/0x30/48.
                    // If a token did have such a length, it would not make it into the 16-byte vector, the vector would be
                    // filled entirely with bytes in [a-zA-Z0-9_], and the check against the target keyword length [0-14] would fail.

                    // The algorithm relies on the fact that a length of '0'/0x30/48 will not make it into the vector
                    comptime assert(max_kw_len < '0' and '0' > PADDING_RIGHT - 1);

                    // const val_len = sorted_padded_kws[hash][PADDING_RIGHT - 1]; // [min_kw_len, max_kw_len]
                    const KW_VEC = @Vector(PADDING_RIGHT, u8);
                    const vec1: KW_VEC = sorted_padded_kws[hash];
                    const vec2: KW_VEC = kw[0..PADDING_RIGHT].*; // the safety of this operation is validated at the top of this function
                    const len_vec: KW_VEC = @splat(len_u8);
                    const cd = @select(u8, len_vec > std.simd.iota(u8, PADDING_RIGHT), vec2, len_vec);

                    if (std.simd.countTrues(cd != vec1) == 0) {
                        return @enumFromInt(~hash);
                    } else {
                        return null;
                    }
                },
                1 => {
                    const KW_VEC = @Vector(PADDING_RIGHT, u8);

                    // This version utilizes `kw_slices` and `kw_buffer`.
                    // `kw_slices` has a start-index and a length into a `kw_buffer`,
                    // where the kw_buffer is an approximate minimal superstring of all the keywords.
                    // This can greatly reduce the size of the lookup table, at the cost of adding another layer of indirection

                    const str = kw_slices[hash];
                    // For performance reasons, instead of branching on a length mismatch here, we incorporate it into the final check. (See comments below)

                    // check at compile-time that it's safe to grab a full vector at any start_index
                    comptime for (kw_slices) |kw_s|
                        if (kw_s.start_index + PADDING_RIGHT > kw_buffer.len)
                            @compileError(
                                std.fmt.comptimePrint(
                                    "Loading {} bytes starting at \"{s}\" in `kw_buffer` causes a buffer overrun. Please add {} more \\x00 byte(s) to the end." ++
                                        blk: {
                                        const kw_buffer_trimmed = std.mem.trimRight(u8, kw_buffer, "\x00");
                                        var longest_kw: []const u8 = "";
                                        for (unpadded_kws) |unpadded_kw| {
                                            if (unpadded_kw.len == max_kw_len) {
                                                if (longest_kw.len != 0) longest_kw = longest_kw ++ " or ";
                                                longest_kw = longest_kw ++ "\"" ++ unpadded_kw ++ "\"";
                                                if (std.mem.endsWith(u8, kw_buffer_trimmed, unpadded_kw)) break :blk "";
                                            }
                                        }
                                        break :blk std.fmt.comptimePrint(" (To reduce the number of required trailing \\x00 bytes to {}, it is recommended to move {s} to the end of `kw_buffer`)", .{ PADDING_RIGHT - max_kw_len, longest_kw });
                                    },
                                    .{
                                        PADDING_RIGHT,
                                        kw_buffer[kw_s.start_index..][0..kw_s.len],
                                        kw_s.start_index + PADDING_RIGHT - kw_buffer.len,
                                    },
                                ),
                            );

                    const vec1: KW_VEC = kw_buffer[str.start_index..][0..PADDING_RIGHT].*;
                    const vec2: KW_VEC = kw[0..PADDING_RIGHT].*; // the safety of this operation is validated at the top of this function

                    // Prove all keyword lengths fit in a byte.
                    comptime for (kw_slices) |kw_s|
                        if (std.math.cast(u8, kw_s.len) == null)
                            @compileError(std.fmt.comptimePrint("\"{s}\" is too long.", .{kw_buffer[kw_s.start_index..][kw_s.len]}));

                    const len_vec1: KW_VEC = @splat(@as(u8, @intCast(str.len)));
                    const len_vec2: KW_VEC = @splat(len_u8);

                    const cc = @select(u8, len_vec1 > std.simd.iota(u8, PADDING_RIGHT), vec1, len_vec1);
                    const cd = @select(u8, len_vec2 > std.simd.iota(u8, PADDING_RIGHT), vec2, len_vec2);

                    // This algorithm compares two vectors of the form `<keyword><len...>`
                    // Where:
                    //   <keyword> is the identifier we found of the form [a-z][a-zA-Z0-9_]*
                    //   <len..> is the len byte duplicated until the end of the vector
                    // Here are real examples of a keyword comparisons:
                    // cc: enum444444444444
                    // cd: allocator9999999

                    // cc: anyframe88888888
                    // cd: appendAssumeCapa

                    // How is it guaranteed that the length cannot be erroneously matched as a character?
                    // Since the keywords we are matching both can only have characters in the set [a-zA-Z0-9_],
                    // in order for the length to potentially match a character, the length must be at least '0'/0x30/48.
                    // 1. For cc, we know that all kw_buffer keywords have a length that is less than '0'
                    //   Therefore, the length in the kw_buffer keyword cannot match any byte in cd
                    comptime assert(max_kw_len < '0');

                    // 2. For kw keywords, we know that lengths above '0' exceed the length of the vector
                    //   E.g. if kw had a length of 0x41 ('A'), so long as that is larger than PADDING_RIGHT - 1, it won't get in the vector.
                    comptime assert('0' > PADDING_RIGHT - 1);

                    if (std.simd.countTrues(cd != cc) == 0) {
                        return @enumFromInt(~hash);
                    } else {
                        return null;
                    }
                },
            }
        }
    }
};

// TODO: clean this up a bit
fn pext(src: anytype, comptime mask: @TypeOf(src)) @TypeOf(src) {
    if (mask == 0) return 0;

    const num_one_groups = @popCount(mask & ~(mask << 1));

    const cpu_name = builtin.cpu.model.llvm_name orelse builtin.cpu.model.name;
    if (!@inComptime() and comptime num_one_groups >= 3 and @bitSizeOf(@TypeOf(src)) <= 64 and builtin.cpu.arch == .x86_64 and
        std.Target.x86.featureSetHas(builtin.cpu.features, .bmi2) and

        // PEXT is microcoded (slow) on AMD architectures before Zen 3.
        (!std.mem.startsWith(u8, cpu_name, "znver") or cpu_name["znver".len] >= '3'))
    {
        return switch (@TypeOf(src)) {
            u64, u32 => asm ("pext %[mask], %[src], %[ret]"
                : [ret] "=r" (-> @TypeOf(src)),
                : [src] "r" (src),
                  [mask] "r" (mask),
            ),
            else => @intCast(pext(@as(if (@bitSizeOf(@TypeOf(src)) <= 32) u32 else u64, src), mask)),
        };
    } else if (num_one_groups >= 4) blk: {
        // Attempt to produce a `global_shift` value such that
        // the return statement at the end of this block moves the desired bits into the least significant
        // bit position.

        comptime var global_shift: @TypeOf(src) = 0;
        comptime {
            var x = mask;
            var target = @as(@TypeOf(src), 1) << (@bitSizeOf(@TypeOf(src)) - 1);
            for (0..@popCount(x) - 1) |_| target |= target >> 1;

            // The maximum sum of the garbage data. If this overflows into the target bits,
            // we can't use the global_shift.
            var left_overs: @TypeOf(src) = 0;
            var cur_pos: @TypeOf(src) = 0;

            while (true) {
                const shift = (@clz(x) - cur_pos);
                global_shift |= @as(@TypeOf(src), 1) << shift;
                var shifted_mask = x << shift;
                cur_pos = @clz(shifted_mask);
                cur_pos += @clz(~(shifted_mask << cur_pos));
                shifted_mask = shifted_mask << cur_pos >> cur_pos;
                left_overs += shifted_mask;
                if ((target & left_overs) != 0) break :blk;
                if ((shifted_mask & target) != 0) break :blk;
                x = shifted_mask >> shift;
                if (x == 0) break;
            }
        }

        return ((src & mask) *% global_shift) >> (@bitSizeOf(@TypeOf(src)) - @popCount(mask));
    }

    {
        var ans: @TypeOf(src) = 0;
        comptime var cur_pos = 0;
        comptime var x = mask;
        inline while (x != 0) {
            const mask_ctz = @ctz(x);
            const num_ones = @ctz(~(x >> mask_ctz));
            comptime var ones = 1;
            inline for (0..num_ones) |_| ones <<= 1;
            ones -%= 1;
            // @compileLog(std.fmt.comptimePrint("ans |= (src >> {}) & 0b{b}", .{ mask_ctz - cur_pos, (ones << cur_pos) }));
            ans |= (src >> (mask_ctz - cur_pos)) & (ones << cur_pos);
            cur_pos += num_ones;
            inline for (0..num_ones) |_| x &= x - 1;
        }
        return ans;
    }
}

const Tag = blk: {
    @setEvalBranchQuota(std.math.maxInt(u32));
    const BitmapKinds = std.meta.fields(Parser.BitmapKind);
    var decls = [_]std.builtin.Type.Declaration{};
    var enumFields: [2 + Operators.unpadded_ops.len + Keywords.unpadded_kws.len + BitmapKinds.len + Operators.potentially_unary.len]std.builtin.Type.EnumField = undefined;

    enumFields[0] = .{ .name = "invalid", .value = 0xaa };

    for (Operators.unpadded_ops, Operators.padded_ops) |op, padded_op| {
        const hash = Operators.rawHash(Operators.getOpWord(&padded_op, op.len));
        enumFields[1 + Operators.mapToIndexRaw(hash)] = .{ .name = op ++ "\x00", .value = hash };
    }

    var i = 1 + Operators.unpadded_ops.len;
    for (Keywords.unpadded_kws) |kw| {
        const hash = Keywords.mapToIndex(Keywords.hashKw(kw.ptr, kw.len));
        enumFields[i + hash] = .{ .name = kw ++ "\x00", .value = ~hash };
    }

    i += Keywords.unpadded_kws.len;

    for (Operators.potentially_unary) |op| {
        const padded_op = (op ++ ("\x00" ** (4 - op.len))).*;
        const hash = Operators.rawHash(Operators.getOpWord(&padded_op, op.len));
        const modified_hash = Operators.unarifyBinaryOperatorRaw(hash);
        enumFields[i] = .{ .name = "unary " ++ op, .value = modified_hash };
        i += 1;
    }

    enumFields[i] = .{ .name = "call (", .value = Operators.postifyOperatorRaw(Operators.rawHash(Operators.getOpWord("(" ++ "\x00" ** 3, 1))) };
    i += 1;

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

test "maskIdentifiersSWAR should match alphanumeric characters and underscores" {
    var c: u8 = 0;
    while (true) {
        const expected: u1 = switch (c) {
            'A'...'Z', 'a'...'z', '0'...'9', '_' => 1,
            else => 0,
        };
        try std.testing.expectEqual(expected, @as(@TypeOf(expected), @intCast(swarMovMask(Parser.maskIdentifiersSWAR(c)))));
        c +%= 1;
        if (c == 0) break;
    }
}

test "swarControlCharMask should match alphanumeric characters and underscores" {
    var c: u8 = 0;
    while (true) {
        const expected: u1 = switch (c) {
            0x7F, 0x0...'\t' - 1, '\t' + 1...0x1F => 1,
            else => 0,
        };
        const got: u1 = @intCast(swarMovMask(Parser.swarControlCharMask(c)));
        // std.debug.print("c: {x}, expected: {}, got: {}\n", .{ c, expected, got });
        try std.testing.expectEqual(expected, got);
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
        const v = std.mem.readInt(u32, &[4]u8{ c, c, 0, 0 }, .little);
        const res = Parser.mask_for_op_cont(v) - 1;
        // std.debug.print("0x{x:0>2} {} vs {}\n", .{ c, res, expected });
        try std.testing.expectEqual(expected, @as(@TypeOf(expected), @intCast(res)));
        c +%= 1;
        if (c == 0) break;
    }
}

// test "mask_for_op_cont should work 2" {
//     std.debug.print("\n", .{});
//     {
//         const v = std.mem.readInt(u32, "//.?", .little);
//         const res = Parser.mask_for_op_cont(v);
//         std.debug.print("res: {}\n", .{res});
//     }
//     {
//         const v = std.mem.readInt(u32, ".?.?", .little);
//         const res = Parser.mask_for_op_cont(v);
//         std.debug.print("res: {}\n", .{res});
//     }
//     {
//         const v = std.mem.readInt(u32, "&.??", .little);
//         const res = Parser.mask_for_op_cont(v);
//         std.debug.print("res: {}\n", .{res});
//     }
//     {
//         const v = std.mem.readInt(u32, "\\\\+=", .little);
//         const res = Parser.mask_for_op_cont(v);
//         std.debug.print("res: {}\n", .{res});
//     }
//     {
//         const v = std.mem.readInt(u32, "\\\\.?", .little);
//         const res = Parser.mask_for_op_cont(v);
//         std.debug.print("res: {}\n", .{res});
//     }
//     {
//         const v = std.mem.readInt(u32, "|*.?", .little);
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

fn swarCTZPlus1Generic(x: anytype, comptime impl: @TypeOf(SWAR_CTZ_PLUS_1_IMPL)) @TypeOf(x) {
    const ones: @TypeOf(x) = @bitCast(@as(@Vector(@divExact(@bitSizeOf(@TypeOf(x)), 8), u8), @splat(0x01)));
    assert(x != 0 and (x & (0x7F * ones)) == 0);
    return switch (impl) {
        .ctz => @ctz(x) / 8 +% 1,
        .popc => @divExact(@popCount(x ^ (x -% 1)), 8),
        .clz => @sizeOf(@TypeOf(x)) -% @divExact(@clz(x ^ (x -% 1)), 8),
        .swar => popCountLSb(x -% 1),
        // .naive => blk: { // 7 ops, 1 constant that might not be an immediate
        //     var i = (x -% 1) & ones; // because the bitstring only contains bits in the highest set bit of each byte, the mask will only isolate from the trailing zeros
        //     inline for (0..comptime std.math.log2_int(u64, @sizeOf(@TypeOf(x)))) |y| {
        //         i = i +% (i >> (@bitSizeOf(@TypeOf(x)) >> (y + 1)));
        //     }
        //     break :blk @as(std.meta.Int(.unsigned, @sizeOf(@TypeOf(x))), @truncate(i));
        // },
    };
}

fn swarCTZGeneric(x: anytype, comptime impl: @TypeOf(SWAR_CTZ_IMPL)) @TypeOf(x) {
    const ones: @TypeOf(x) = @bitCast(@as(@Vector(@divExact(@bitSizeOf(@TypeOf(x)), 8), u8), @splat(1)));
    assert((x & (ones * 0x7F)) == 0);

    return switch (impl) {
        .ctz => @ctz(x) >> 3,
        .swar => popCountLSb((~x & (x -% 1)) >> 7),
        .swar_bool => popCountLSb(x -% 1) -% @intFromBool(x != 0),
        // .naive => blk: {
        //     var i = ((~x & (x -% 1)) & (ones * 0x80)) >> 7;
        //     inline for (0..comptime std.math.log2_int(u64, @sizeOf(@TypeOf(x)))) |y| {
        //         i = i +% (i >> (@bitSizeOf(@TypeOf(x)) >> (y + 1)));
        //     }
        //     break :blk @as(std.meta.Int(.unsigned, @sizeOf(@TypeOf(x))), @truncate(i));
        // },
    };
}

fn swarCTZ(x: uword) @TypeOf(x) {
    return swarCTZGeneric(x, SWAR_CTZ_IMPL);
}

test "swarCTZPlus1" {
    var i: u3 = 0;

    while (true) {
        const x: u32 = swarUnMovMask(@as(u4, 8) | i);

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

const HAS_FAST_VECTOR_REVERSE: bool = switch (builtin.cpu.arch) {
    .powerpc, .powerpc64, .powerpc64le, .powerpcle => std.Target.powerpc.featureSetHas(builtin.cpu.features, .isa_v30_instructions),
    else => false,
};

const HAS_FAST_BYTE_SWAP = switch (builtin.cpu.arch) {
    .mips, .mips64, .mips64el, .mipsel => std.Target.mips.featureSetHas(builtin.cpu.features, .mips64r2),
    .x86, .x86_64 => true, // we could probably exclude ancient hardware that lacks a bswap, i.e. everything before the 80486. Not sure the LLVM flags for that.
    .riscv32, .riscv64 => std.Target.riscv.featureSetHas(builtin.cpu.features, .zbb),
    else => false,
};

const SWAR_CTZ_PLUS_1_IMPL: enum { ctz, clz, popc, swar } = switch (builtin.cpu.arch) {
    .aarch64_32, .aarch64_be, .aarch64, .arm, .armeb, .thumb, .thumbeb => if (std.Target.aarch64.featureSetHas(builtin.cpu.features, .v8a) or
        (std.Target.arm.featureSetHas(builtin.cpu.features, .has_v7) and // rbit is available since v7. TODO: what about armv6t2?
        !std.mem.eql(u8, builtin.cpu.model.name, "cortex_m0") and
        !std.mem.eql(u8, builtin.cpu.model.name, "cortex_m0plus") and
        !std.mem.eql(u8, builtin.cpu.model.name, "cortex_m1") and
        !std.mem.eql(u8, builtin.cpu.model.name, "cortex_m23"))) .ctz else .swar,
    .mips, .mips64, .mips64el, .mipsel => if (std.Target.mips.featureSetHas(builtin.cpu.features, .mips64)) .clz else .swar,
    .powerpc, .powerpc64, .powerpc64le, .powerpcle => .clz,
    .s390x => .clz,
    .ve => .ctz,
    .avr => .popc,
    .msp430 => .popc,
    .riscv32, .riscv64 => if (std.Target.riscv.featureSetHas(builtin.cpu.features, .zbb)) .ctz else .swar,
    .sparc, .sparc64, .sparcel => if (std.Target.sparc.featureSetHas(builtin.cpu.features, .popc)) .popc else .swar,
    .wasm32, .wasm64 => .ctz,
    .x86, .x86_64 => if (std.Target.x86.featureSetHas(builtin.cpu.features, .bmi)) .ctz else .swar,
    else => .swar,
};

fn swarCTZPlus1(x: u32) @TypeOf(x) {
    return swarCTZPlus1Generic(x, SWAR_CTZ_PLUS_1_IMPL);
}

// fn swarCTZGeneric(x: anytype, comptime impl: @TypeOf(SWAR_CTZ_IMPL)) @TypeOf(x) {
//     const ones: @TypeOf(x) = @bitCast(@as(@Vector(@divExact(@bitSizeOf(@TypeOf(x)), 8), u8), @splat(1)));
//     assert((x & (ones * 0x7F)) == 0);

//     return switch (impl) {
//         .ctz => @ctz(x) >> 3,
//         .swar => popCountLSb((~x & (x -% 1)) >> 7),
//         .swar_bool => popCountLSb(x -% 1) -% @intFromBool(x != 0),
//         .naive => blk: {
//             var i = ((~x & (x -% 1)) & (ones * 0x80)) >> 7;
//             inline for (0..comptime std.math.log2_int(u64, @sizeOf(@TypeOf(x)))) |y| {
//                 i = i +% (i >> (@bitSizeOf(@TypeOf(x)) >> (y + 1)));
//             }
//             break :blk @as(std.meta.Int(.unsigned, @sizeOf(@TypeOf(x))), @truncate(i));
//         },
//     };
// }

// test "ctz popCount" {
//     inline for (std.meta.fields(@TypeOf(SWAR_CTZ_IMPL))) |impl| {
//         inline for ([_]type{ u64, u32, u16, u8 }) |T| {
//             var c: std.meta.Int(.unsigned, @sizeOf(T)) = 0;
//             while (true) {
//                 const y = swarCTZGeneric(swarUnMovMask(c), @enumFromInt(impl.value));
//                 try std.testing.expectEqual(@as(@TypeOf(y), @ctz(c)), y);
//                 c +%= 1;
//                 if (c == 0) break;
//             }
//         }
//     }
// }

const SWAR_CTZ_IMPL: enum { ctz, swar, swar_bool } = switch (builtin.cpu.arch) {
    else => switch (SWAR_CTZ_PLUS_1_IMPL) {
        .ctz, .clz, .popc => .ctz,
        .swar => switch (builtin.cpu.arch) {
            .aarch64_32, .aarch64_be, .aarch64, .arm, .armeb, .thumb, .thumbeb => .swar,
            .mips, .mips64, .mips64el, .mipsel => .swar_bool,
            .riscv32, .riscv64 => .swar_bool,
            .sparc, .sparc64, .sparcel => .swar,
            .x86, .x86_64 => .swar_bool,
            else => .swar_bool,
        },
    },
};

// fn swarCTZ(x: u64) @TypeOf(x) {
//     return swarCTZGeneric(x, SWAR_CTZ_IMPL);
// }

/// Creates a bitstring from the most significant bit of each byte in a given bitstring.
///
/// E.g. 1....... 0....... 0....... 1....... 0....... 1....... 1....... 1....... => 10010111
fn swarMovMask(v: anytype) @TypeOf(v) {
    comptime assert(@divExact(@bitSizeOf(@TypeOf(v)), 8) <= 8);
    const ones: @TypeOf(v) = @bitCast(@as(@Vector(@sizeOf(@TypeOf(v)), u8), @splat(1)));
    const msb_mask = 0x80 * ones;

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

    // Inspired by: http://0x80.pl/articles/scalar-sse-movmask.html
    return (mult *% (v & msb_mask)) >> (@bitSizeOf(@TypeOf(v)) - @sizeOf(@TypeOf(v)));
}

test "movemask swar should properly isolate the highest set bits of all bytes in a bitstring" {
    inline for ([_]type{ u64, u32, u16, u8 }) |T| {
        var c: std.meta.Int(.unsigned, @sizeOf(T)) = 0;
        while (true) {
            const ans = swarMovMask(swarUnMovMask(c));
            try std.testing.expectEqual(@as(@TypeOf(ans), c), ans);
            c +%= 1;
            if (c == 0) break;
        }
    }
}

fn swarMovMaskReversed(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    comptime assert(@divExact(@bitSizeOf(T), 8) <= 8);
    const ones: T = @bitCast(@as(@Vector(@sizeOf(T), u8), @splat(1)));
    const msb_mask = 0x80 * ones;

    // We are exploiting a multiplication as a shifter and adder, and the derivation of this number is
    // shown here as a comptime loop.
    // This trick is often generalizable to other problems too: https://stackoverflow.com/a/14547307
    comptime var mult: T = 0;
    comptime for (0..@sizeOf(T)) |i| {
        mult |= @as(T, 1) << (9 * i);
    };

    //   .......a.......b.......c.......d.......e.......f.......g.......h
    // * ..........................................1......1......1......1
    // -------------------------------------------------------------------------
    //   .......a.......b.......c.......d.......e.......f.......g.......h << 0
    //   ......b.......c.......d.......e.......f.......g.......h......... << 9
    //   .....c.......d.......e.......f.......g.......h.................. << 18
    //   ....d.......e.......f.......g.......h........................... << 27
    //   ...e.......f.......g.......h.................................... << 36
    //   ..f.......g.......h............................................. << 45
    //   .g.......h...................................................... << 54
    // + h............................................................... << 63

    return (((v & msb_mask) >> (@sizeOf(T) - 1)) *% mult) >> (@bitSizeOf(T) - @sizeOf(T));

    // We could save an instruction by using mulhi.... but is it worth it? I think not
    // return @as(std.meta.Int(.unsigned, @sizeOf(T)), @truncate((((v & msb_mask)) *% @as(std.meta.Int(.unsigned, @bitSizeOf(T) * 2), (mult << (@sizeOf(T) - 1)))) >> (@bitSizeOf(T))));
}

test "movemask reversed swar should properly isolate the highest set bits of all bytes in a bitstring" {
    inline for ([_]type{ u64, u32, u16, u8 }) |T| {
        var c: std.meta.Int(.unsigned, @sizeOf(T)) = 0;
        while (true) {
            const ans = swarMovMaskReversed(swarUnMovMask(c));
            try std.testing.expectEqual(@as(@TypeOf(ans), @bitReverse(c)), ans);
            c +%= 1;
            if (c == 0) break;
        }
    }
}

fn _lookup_chunk(a: @Vector(16, u8), b: @Vector(16, u8)) @Vector(16, u8) {
    if (!@inComptime()) {
        switch (builtin.cpu.arch) {
            .aarch64, .aarch64_32, .aarch64_be => return asm (
                \\tbl  %[out].16b, {%[a].16b}, %[b].16b
                : [out] "=&x" (-> @Vector(16, u8)),
                : [b] "x" (b),
                  [a] "x" (a),
            ),
            else => {},
        }
    }

    var r: @Vector(16, u8) = @splat(0);
    for (0..16) |i| {
        const c = b[i];
        assert(c <= 0x0F);
        r[i] = a[c];
    }
    return r;
}

fn vector_shuffle(comptime T: type, a: T, b: T) T {
    const type_info = @typeInfo(T).Vector;
    var r: T = @splat(0);

    for (0..type_info.len) |i| {
        const c = b[i];
        assert(c < type_info.len);
        r[i] = a[c];
    }
    return r;
}

const HAS_FAST_PDEP_AND_PEXT = blk: {
    const cpu_name = builtin.cpu.model.llvm_name orelse builtin.cpu.model.name;
    break :blk builtin.cpu.arch == .x86_64 and
        std.Target.x86.featureSetHas(builtin.cpu.features, .bmi2) and
        // pdep is microcoded (slow) on AMD architectures before Zen 3.
        !std.mem.startsWith(u8, cpu_name, "bdver") and
        (!std.mem.startsWith(u8, cpu_name, "znver") or cpu_name["znver".len] >= '3');
};

inline fn pdep(src: u64, mask: u64) u64 {
    return asm ("pdep %[mask], %[src], %[ret]"
        : [ret] "=r" (-> u64),
        : [src] "r" (src),
          [mask] "r" (mask),
    );
}

const SHUFFLE_VECTOR_IMPL: enum { x86_pdep, aarch_bdep, riscv_iota, swar, default_vec } =
    if (HAS_FAST_PDEP_AND_PEXT)
    .x86_pdep
else if (std.Target.aarch64.featureSetHas(builtin.cpu.features, .sve2_bitperm))
    .default_vec // TODO: aarch_bdep, should be similar to the pdep implementation
else if (std.Target.riscv.featureSetHas(builtin.cpu.features, .v))
    .default_vec // TODO: .riscv_iota
else
    .default_vec;

fn getShuffleVectorForByte(x: u8) @Vector(16, u8) {
    if (builtin.mode == .ReleaseSmall)
        return produceShuffleVectorForByteSpecifyImpl(x, if (@inComptime()) .swar else SHUFFLE_VECTOR_IMPL);

    comptime var lookup_table: [1 << 8][16]u8 = undefined;
    inline for (&lookup_table, 0..) |*slot, i| {
        slot.* = produceShuffleVectorForByteSpecifyImpl(@intCast(i), .swar);
    }

    return lookup_table[x];
}

fn expand8xu8To16xu4AsByteVector(vec: @Vector(8, u8)) @Vector(16, u8) {
    if (builtin.cpu.arch == .x86_64) {
        return std.simd.interlace(.{ vec, vec >> @splat(4) }) & @as(@Vector(16, u8), @splat(0xF));
    } else {
        return std.simd.interlace(.{ vec & @as(@Vector(8, u8), @splat(0xF)), vec >> @splat(4) });
    }
}

fn produceShuffleVectorForByteSpecifyImpl(x: u8, impl: @TypeOf(SHUFFLE_VECTOR_IMPL)) @Vector(16, u8) {
    switch (impl) {
        .x86_pdep => { // Expand each bit of `x` into a byte. 1 -> 11111111, 0 -> 00000000
            const vec = 255 * pdep(x, 0x0101010101010101);
            const nibble_indices: u64 = @bitCast(std.simd.iota(u4, 16));
            // Selects the nibble_indices, two at a time.
            const interleaved_shuffle_vector = pdep(nibble_indices, vec) | pdep(nibble_indices, ~vec);
            // Workaround until https://github.com/ziglang/zig/issues/18631 is solved.
            return expand8xu8To16xu4AsByteVector(@bitCast(interleaved_shuffle_vector));
            // return @as(@Vector(16, u4), @bitCast(interleaved_shuffle_vector));
        },

        .aarch_bdep => unreachable,
        .riscv_iota => unreachable, // This might not end up in here

        .swar => {
            const ones: u64 = 0x0101010101010101;
            const byte_indices: u64 = 0x8040_2010_0804_0201;
            const unique_bytes_diff_from_msb = (ones * 0x80) - byte_indices;
            const y = (((x *% ones) & byte_indices) + unique_bytes_diff_from_msb) & 0x8080808080808080;
            const vec = (y >> 3) | (y >> 7);

            const selector = vec *% 0xF;
            const prefix_sums1 = (vec *% 0x1111111111111111) << 4;
            const prefix_sums2 = ((vec ^ 0x1111111111111111) *% 0x1111111111111111) << 4;
            const interleaved_shuffle_vector = ((prefix_sums1 & selector) | (prefix_sums2 & ~selector));
            // Workaround until https://github.com/ziglang/zig/issues/18631 is solved.
            return expand8xu8To16xu4AsByteVector(@bitCast(interleaved_shuffle_vector));
            // return @as(@Vector(16, u4), @bitCast(interleaved_shuffle_vector));

        },

        .default_vec => {
            const byte_indices = @as(@Vector(8, u8), @splat(1)) << std.simd.iota(u3, 8);
            const splatted = @as(@Vector(8, u8), @splat(x));
            const selector = (splatted & byte_indices) != byte_indices;
            const vec: u64 = @bitCast(@select(u8, selector, @as(@Vector(8, u8), @splat(0b00010001)), @as(@Vector(8, u8), @splat(0))));

            const prefix_sums1: @Vector(8, u8) = @bitCast((vec *% 0x1111111111111111) << 4);
            const prefix_sums2: @Vector(8, u8) = @bitCast(((vec ^ 0x1111111111111111) *% 0x1111111111111111) << 4);

            const interleaved_shuffle_vector = @select(u8, selector, prefix_sums1, prefix_sums2);
            // Workaround until https://github.com/ziglang/zig/issues/18631 is solved.
            return expand8xu8To16xu4AsByteVector(interleaved_shuffle_vector);
            // return @as(@Vector(16, u4), @bitCast(interleaved_shuffle_vector));
        },
    }
}

test "produceShuffleVectorForByteSpecifyImpl" {
    var x: u8 = 0;
    comptime var impls: []const @TypeOf(SHUFFLE_VECTOR_IMPL) = &[_]@TypeOf(SHUFFLE_VECTOR_IMPL){ .swar, .default_vec };
    if (comptime std.mem.indexOfScalar(@TypeOf(SHUFFLE_VECTOR_IMPL), impls, SHUFFLE_VECTOR_IMPL) == null) {
        impls = impls ++ [_]@TypeOf(SHUFFLE_VECTOR_IMPL){SHUFFLE_VECTOR_IMPL};
    }

    var results: [impls.len]@Vector(16, u8) = undefined;

    while (true) {
        inline for (impls, &results) |impl, *result_slot| {
            result_slot.* = produceShuffleVectorForByteSpecifyImpl(x, impl);
        }
        const result_0 = results[0];
        inline for (results[1..]) |result| {
            try std.testing.expectEqual(result_0, result);
        }
        x +%= 1;
        if (x == 0) break;
    }
}

const Parser = struct {
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
        const not_and_dot = (((v & 0x7FFF) ^ comptime std.mem.readInt(u16, "&.", .little)) + 0x7FFF);
        const not_bar_star = (((v & 0x7FFF) ^ comptime std.mem.readInt(u16, "|*", .little)) + 0x7FFF);

        // TODO: figure out how to do this optimization properly??
        // const dot_question = 0x8000 & ~(((v & 0x7FFF) ^ comptime std.mem.readInt(u16, ".?", .little)) + 0x7FFF);
        // const backslashes = 0x8000 & ~(((v & 0x7FFF) ^ comptime std.mem.readInt(u16, "\\\\", .little)) + 0x7FFF);
        // const slashes = 0x8000 & ~(((v & 0x7FFF) ^ comptime std.mem.readInt(u16, "//", .little)) + 0x7FFF);

        // printu32(not_and_dot);
        // printu32(not_and_dot);

        // 0x21,0x25,0x2a,0x2b,0x2e,0x2f,
        // 0x3c,0x3d,0x3e,0x3f,
        // 0x5c,0x7c
        // maskNonChars(v, "\x21\x25\x2a\x2b\x2e\x2f\x3c\x3d\x3e\x3f\x5c\x7c");

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

    fn maskIdentifiersSWAR(v: anytype) @TypeOf(v) {
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

    test "maskIdentifiersSWAR" {
        var x: u8 = 0;
        while (true) {
            const a: u1 = switch (x) {
                'a'...'z', 'A'...'Z', '0'...'9', '_' => 1,
                else => 0,
            };

            const b: u1 = @truncate(maskIdentifiersSWAR(x) >> 7);
            try std.testing.expectEqual(a, b);

            x +%= 1;
            if (x == 0) break;
        }
    }

    fn maskIdentifiers(v: anytype) if (USE_SWAR) NATIVE_VEC_INT else std.meta.Int(.unsigned, @sizeOf(@TypeOf(v))) {
        if (USE_SWAR) {
            assert(@TypeOf(v) == NATIVE_VEC_INT);
            return maskIdentifiersSWAR(v);
        }

        const underscores = maskChars(v, "_");
        const upper_alpha = maskCharRange(v, 'A', 'Z');
        const lower_alpha = maskCharRange(v, 'a', 'z');
        const digits = maskCharRange(v, '0', '9');
        return underscores | upper_alpha | lower_alpha | digits;
    }

    fn swarControlCharMaskInverse(v: anytype) @TypeOf(v) {
        const ones: @TypeOf(v) = @bitCast(@as(@Vector(@divExact(@bitSizeOf(@TypeOf(v)), 8), u8), @splat(1)));
        const mask = comptime ones * 0x7F;
        const low_7_bits = v & mask;
        const non_del = ~ones - low_7_bits;
        const non_ctrl = low_7_bits + comptime ones * (0x80 - 0x20);
        const non_tabs = (low_7_bits ^ comptime ones * '\t') + mask;
        return v | (non_del & non_ctrl) | ~non_tabs;
    }

    fn swarControlCharMask(v: anytype) @TypeOf(v) {
        const ones: @TypeOf(v) = @bitCast(@as(@Vector(@divExact(@bitSizeOf(@TypeOf(v)), 8), u8), @splat(1)));
        const mask = comptime ones * 0x7F;
        const low_7_bits = v & mask;
        const del = ones + low_7_bits;
        const non_tabs = (low_7_bits ^ comptime ones * '\t') + mask;
        const ctrl = (comptime ~(ones * (0x80 - 0x20))) - low_7_bits;
        return (~v & (del | ctrl) & non_tabs);
    }

    test "swarControlCharMask and swarControlCharMaskInverse" {
        var x: u8 = 0;
        while (true) {
            const a: u1 = switch (x) {
                0x7F => 1,
                0...'\t' - 1, '\t' + 1...' ' - 1 => 1,
                else => 0,
            };

            const b: u1 = @truncate(swarControlCharMask(x) >> 7);
            const c: u1 = @truncate(~swarControlCharMaskInverse(x) >> 7);
            try std.testing.expectEqual(a, b);
            try std.testing.expectEqual(a, c);

            x +%= 1;
            if (x == 0) break;
        }
    }

    fn maskControls(v: anytype) if (USE_SWAR) NATIVE_VEC_INT else std.meta.Int(.unsigned, @sizeOf(@TypeOf(v))) {
        if (USE_SWAR) {
            return swarControlCharMask(v);
        }

        const delete_code = maskChars(v, "\x7F");
        const other_controls = maskCharRange(v, 0, 31) ^ maskChars(v, "\t");
        return delete_code | other_controls;
    }

    fn maskNonControls(v: anytype) if (USE_SWAR) NATIVE_VEC_INT else std.meta.Int(.unsigned, @sizeOf(@TypeOf(v))) {
        if (USE_SWAR) {
            return swarControlCharMaskInverse(v);
        }

        const delete_code = maskNonChars(v, "\x7F");
        const other_controls = ~maskCharRange(v, 0, 31) | maskChars(v, "\t");
        return delete_code & other_controls;
    }

    fn maskNonCharsGeneric(v: anytype, comptime str: []const u8, comptime use_swar: bool) if (USE_SWAR) NATIVE_VEC_INT else std.meta.Int(.unsigned, @sizeOf(@TypeOf(v))) {
        if (use_swar) {
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
            return ~maskCharsGeneric(v, str, use_swar);
        }
    }

    fn maskNonChars(v: anytype, comptime str: []const u8) if (USE_SWAR) NATIVE_VEC_INT else std.meta.Int(.unsigned, @sizeOf(@TypeOf(v))) {
        return maskNonCharsGeneric(v, str, USE_SWAR);
    }

    fn maskCharsGeneric(v: anytype, comptime str: []const u8, comptime use_swar: bool) if (USE_SWAR) NATIVE_VEC_INT else std.meta.Int(.unsigned, @sizeOf(@TypeOf(v))) {
        if (use_swar) {
            const ones: @TypeOf(v) = @bitCast(@as(@Vector(@divExact(@bitSizeOf(@TypeOf(v)), 8), u8), @splat(1)));
            const mask = comptime ones * 0x7F;
            const low_7_bits = v & mask;
            var accumulator: @TypeOf(v) = 0;

            inline for (str) |c| {
                assert(c < 0x80);
                accumulator |= ~mask - (low_7_bits ^ comptime ones * c);
            }

            return ~v & accumulator;
        } else {
            var accumulator: @TypeOf(v) = @splat(0);
            inline for (str) |c| {
                assert(c < 0x80); // Because this would break the SWAR version, we enforce it here too.
                accumulator |= vec_cmp(v, .@"==", c);
            }
            return @bitCast(accumulator != @as(@TypeOf(v), @splat(0)));
        }
    }

    fn maskChars(v: anytype, comptime str: []const u8) if (USE_SWAR) NATIVE_VEC_INT else std.meta.Int(.unsigned, @sizeOf(@TypeOf(v))) {
        return maskCharsGeneric(v, str, USE_SWAR);
    }

    fn maskCharRange(input_vec: anytype, comptime char1: u8, comptime char2: u8) std.meta.Int(.unsigned, @sizeOf(@TypeOf(input_vec))) {
        const VEC_T = std.meta.Int(.unsigned, @sizeOf(@TypeOf(input_vec)));
        return @as(VEC_T, @bitCast(@as(@TypeOf(input_vec), @splat(char1)) <= input_vec)) & @as(VEC_T, @bitCast(input_vec <= @as(@TypeOf(input_vec), @splat(char2))));
    }

    // On arm and ve machines, `@ctz(x)` is implemented as `@bitReverse(@clz(x))`.
    // We can speculatively perform 3 bit reverses in the producer loop so that the consumer loop can use `@clz` instead.
    // This saves operations in practice because (for the codebases tested) we do an average of 10.5 ctz's/clz's
    // per 64-byte chunk, meaning we eliminate ~6.5 bit reverses per chunk.
    // Might backfire if the microarchitecture has a builtin ctz operation and the decoder automatically combines a bitreverse and clz.
    const DO_BIT_REVERSE = switch (builtin.cpu.arch) {
        .aarch64_32, .aarch64_be, .aarch64, .arm, .armeb, .thumb, .thumbeb, .ve => SWAR_CTZ_PLUS_1_IMPL == .ctz and builtin.cpu.arch.endian() == .little,
        else => false,
    };

    // On some machines, it is more efficient to use clz over ctz, and vice versa.
    const DO_MASK_REVERSE = 0 == 1 and !DO_BIT_REVERSE and USE_SWAR and switch (builtin.cpu.arch.endian()) {
        .little => (builtin.target.cpu.arch.isArmOrThumb() or SWAR_CTZ_PLUS_1_IMPL == .clz),
        // I think it should usually be faster to emulate ctz rather than clz
        .big => SWAR_CTZ_PLUS_1_IMPL == .popc or SWAR_CTZ_PLUS_1_IMPL == .swar,
    };

    const USE_REVERSED_BITSTRINGS = (DO_BIT_REVERSE or DO_MASK_REVERSE) == (builtin.cpu.arch.endian() == .little);
    const ASSEMBLE_BITSTRINGS_BACKWARDS = !DO_BIT_REVERSE and USE_REVERSED_BITSTRINGS;

    fn reverseIfCheap(b: uword) uword {
        return if (DO_BIT_REVERSE) @bitReverse(b) else b;
    }

    fn movMask(v: anytype) uword {
        return if (USE_SWAR) (if (DO_MASK_REVERSE) swarMovMaskReversed(v) else swarMovMask(v)) else v;
    }

    const BitmapKind = enum(u8) {
        // zig fmt: off
        const min_bitmap_value = @intFromEnum(BitmapKind.unknown);
        const max_bitmap_value = @intFromEnum(BitmapKind.number);

        eof                   = 0,
        sentinel_operator     = 128 | @as(u8, 20),
        unknown               = 128 | @as(u8,  0),

        identifier            = 128 | @as(u8,  1),
        builtin               = 128 | @as(u8,  9),
        number                = 128 | @as(u8,  17),

        whitespace            = 128 | @as(u8, 2),

        string                = 128 | @as(u8,  10),
        string_identifier     = 128 | @as(u8,  19),


        char_literal          = 128 | @as(u8,  4),
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
    pub fn tokenize(gpa: Allocator, source: [:0]align(@bitSizeOf(uword)) const u8, comptime impl: u1) ![]Token {
        _ = impl;
        const ON_DEMAND_IMPL: u1 = @intFromBool(!USE_SWAR);
        const FOLD_COMMENTS_INTO_ADJACENT_NODES = true;
        const end_ptr = &source.ptr[source.len];
        const extended_source_len = std.mem.alignForward(usize, source.len + EXTENDED_BACK_SENTINELS_LEN, @bitSizeOf(uword));
        const extended_source = source.ptr[0..extended_source_len];
        const tokens = try gpa.alloc(Token, extended_source_len);
        errdefer gpa.free(tokens);

        // TODO: add dynamic math here based on DO_CHAR_LITERAL_IN_SIMD and DO_QUOTE_IN_SIMD

        // We write our 3 bitstrings to consecutive slots in a buffer.
        // |a|b|c|
        //   |a|b|c| <- Each time, we move one slot forward
        //     |a|b|c|
        //       |a|b|c|
        //         |a|b|c|
        //           |a|b|c|
        //             |a|b|c|
        // |a|a|a|a|a|a|a|b|c| <- In the end, we are left with this
        // In this way, we can preserve one of the bitstrings, in this case the control-chars mask
        // inside this buffer with minimal overhead. If tokenizing went properly the control-chars mask
        // will contain only newlines. We can use this information later to find out what line we are on.
        const non_newlines_bitstrings = try gpa.alloc(uword, extended_source_len / @bitSizeOf(uword) + 2);

        // TODO: make this errdefer and return this data out.
        defer gpa.free(non_newlines_bitstrings);

        var cur_token = tokens;
        cur_token[0] = .{ .len = 0, .kind = .whitespace };
        cur_token[1..][0..2].* = @bitCast(@as(u32, 0));

        var prev: []const u8 = extended_source;

        comptime assert(FRONT_SENTINELS.len == 1 and FRONT_SENTINELS[0] == '\n' and BACK_SENTINELS.len >= 3);
        var cur = prev[@as(u8, @intFromBool(
            std.mem.readInt(u32, prev[0..4], comptime builtin.cpu.arch.endian()) == std.mem.readInt(u32, "\n\xEF\xBB\xBF", comptime builtin.cpu.arch.endian()),
        )) * 4 ..];

        // errdefer var loc: u32 = @intCast(@intFromPtr(cur.ptr) - @intFromPtr(source.ptr));

        var bitmap_ptr: []uword = non_newlines_bitstrings;
        bitmap_ptr.ptr -= 1;
        bitmap_ptr.len += 1;
        var op_type: Tag = .whitespace;
        var selected_bitmap: []const uword = bitmap_ptr[@as(u3, @truncate(@intFromEnum(op_type)))..];
        var bitmap_index = @intFromPtr(cur.ptr) / @bitSizeOf(uword) * @bitSizeOf(uword) -% @bitSizeOf(uword);
        var utf8_checker: Utf8Checker = .{};

        outer: while (true) : (cur = cur[1..]) {
            {
                var aligned_ptr = @intFromPtr(cur.ptr) / @bitSizeOf(uword) * @bitSizeOf(uword);
                while (true) {
                    // https://github.com/ziglang/zig/issues/8220
                    // TODO: once labeled switch continues are added, we can make this check run the first iteration only.
                    // If we loop back around, there is no need to check this.
                    // I implemented this with tail call functions but it was messy.

                    // Note: we could technically make a separate path that only does utf8/bad character checking, but
                    // the case where we have to run this loop multiple times in a row is not something that really happens in practice.
                    // We have this loop here just for correctness in extreme edge cases, since our character-by-character loop
                    // could, in theory, skip over an entire chunk that has bad characters in it.
                    while (bitmap_index != aligned_ptr) {
                        bitmap_index +%= @bitSizeOf(uword);
                        const base_ptr = @as([*]align(@bitSizeOf(uword)) const u8, @ptrFromInt(bitmap_index));

                        // tabs are allowed in multiline strings, comments, and whitespace (start-state)
                        // carriage returns are allowed before newlines only
                        // other control characters are not allowed ever
                        // no control characters are allowed in strings or character literals, neither of which are handled in this loop.

                        // Control characters besides tab. (used by multiline strings, comments, eof)
                        var ctrls: uword = 0;
                        // Anything besides space, newline, or tab. Used for skipping over whitespace.
                        var non_spaces: uword = 0;
                        // Anything in the set [A-Za-z0-9_]. Used for identifier (including keywords) and number matching.
                        var identifiers_or_numbers: uword = 0;

                        // Overall, this code layout allows us to get pretty decent emit on x86-64 and SWAR-targets.
                        // A LOT of optimization opportunities are still left for aarch64
                        // E.g. using LD4 and doing an interleaved movmask rather than individual movmasks
                        // and OR'ing them together. It also might be possible to do a reversed movmask on aarch64.
                        // However, this is a problem that can only be solved in the compiler backend.
                        inline for (0..comptime @bitSizeOf(uword) / NATIVE_VEC_SIZE) |i| {
                            const chunk = blk: {
                                const slice: *align(NATIVE_VEC_SIZE) const [NATIVE_VEC_SIZE]u8 = @alignCast(base_ptr[i * NATIVE_VEC_SIZE ..][0..NATIVE_VEC_SIZE]);

                                break :blk if (USE_SWAR)
                                    @as(*align(NATIVE_VEC_SIZE) const NATIVE_VEC_INT, @ptrCast(slice)).*
                                else
                                    @as(@Vector(NATIVE_VEC_SIZE, u8), slice.*);
                            };

                            const shift: std.math.Log2Int(uword) = if (ASSEMBLE_BITSTRINGS_BACKWARDS)
                                @intCast((@bitSizeOf(uword) - NATIVE_VEC_SIZE) - NATIVE_VEC_SIZE * i)
                            else
                                @intCast(NATIVE_VEC_SIZE * i);

                            ctrls |= movMask(if (USE_SWAR) maskControls(chunk) else maskNonControls(chunk)) << shift;
                            non_spaces |= movMask(if (USE_SWAR) maskNonChars(chunk, " \t\n") else maskChars(chunk, " \t\n")) << shift;
                            identifiers_or_numbers |= movMask(maskIdentifiers(chunk)) << shift;
                            try utf8_checker.validateChunk(chunk);
                        }

                        bitmap_ptr = bitmap_ptr[1..];
                        selected_bitmap = selected_bitmap[1..];

                        const inverter: uword = if (USE_SWAR) std.math.maxInt(uword) else 0;

                        // Optimization: when ctz is implemented with a bitReverse+clz,
                        // we speculatively bitReverse in the producer loop to avoid doing so in this loop.
                        bitmap_ptr[0] = reverseIfCheap(ctrls ^ inverter);
                        bitmap_ptr[1] = reverseIfCheap(identifiers_or_numbers);
                        bitmap_ptr[2] = reverseIfCheap(non_spaces ^ inverter);
                    }

                    const cur_misalignment: std.math.Log2Int(uword) = @truncate(@intFromPtr(cur.ptr));

                    const bitstring = if (USE_REVERSED_BITSTRINGS)
                        selected_bitmap[0] << cur_misalignment
                    else
                        selected_bitmap[0] >> cur_misalignment;

                    // We invert, i.e. count 1's, because 0's are shifted in by the bitshift.
                    // The reason we increase the type size is so LLVM does not insert an unnecessary AND
                    const str_len: std.meta.Int(
                        .unsigned,
                        std.math.ceilPowerOfTwoPromote(u64, std.math.log2_int_ceil(u64, @bitSizeOf(uword) + 1)),
                    ) = if (USE_REVERSED_BITSTRINGS)
                        @clz(~bitstring)
                    else
                        @ctz(~bitstring);

                    cur = cur[str_len..];
                    aligned_ptr = @intFromPtr(cur.ptr) / @bitSizeOf(uword) * @bitSizeOf(uword);
                    if (bitmap_index == aligned_ptr) break;
                }
            }

            comptime assert(BACK_SENTINELS.len - 1 > std.mem.indexOf(u8, BACK_SENTINELS, "\x00").?); // there should be at least another character
            comptime assert(BACK_SENTINELS[BACK_SENTINELS.len - 1] == ' '); // eof reads the non_newlines bitstring, therefore we need a newline at the end
            if (op_type == .eof) break :outer;

            while (true) {
                var len: u32 = @intCast(@intFromPtr(cur.ptr) - @intFromPtr(prev.ptr));
                assert(len != 0);

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

                    const advance_amt: u2 = if (cur_token[0].len == 0) 3 else 1;
                    cur_token = cur_token[advance_amt..];
                }

                // If `len` does not fit in a u8, store a 0 and store the true length in the next 4 bytes.
                cur_token[0] = .{ .len = if (len >= 256) 0 else @intCast(len), .kind = op_type };
                // We write this unconditionally, but we will probably overwrite this later.
                cur_token[1..][0..2].* = @bitCast(len);
                prev = cur;

                if (cur[0] == '@') {
                    cur = cur[1..];
                    op_type = switch (cur[0]) {
                        'a'...'z', 'A'...'Z' => .builtin,
                        '"' => .string_identifier,
                        else => return error.MissingQuoteOrLetterAfterAtSymbol,
                    };

                    selected_bitmap = bitmap_ptr[@as(u3, @truncate(@intFromEnum(op_type)))..];
                } else if (Operators.isSingleCharOp(cur[0])) {
                    selected_bitmap = bitmap_ptr[@as(u3, @truncate(@intFromEnum(BitmapKind.whitespace)))..];
                    op_type = Operators.hashOp(Operators.getOpWord(cur.ptr, 1));
                    // cur = cur[@intFromBool(cur[0] == ' ')..];
                } else if (Operators.isMultiCharBeginning(cur[0])) {
                    var op_len: u32 = 1;

                    op_len = mask_for_op_cont(std.mem.readInt(u32, cur[0..4], .little));
                    assert(0 < op_len and op_len <= 4);

                    const op_word4 = Operators.getOpWord(cur.ptr, 4);
                    const hash4 = Operators.rawHash(op_word4);
                    const op_word3 = Operators.getOpWord(cur.ptr, 3);
                    const hash3 = Operators.rawHash(op_word3);
                    const op_word2 = Operators.getOpWord(cur.ptr, 2);
                    const hash2 = Operators.rawHash(op_word2);
                    const op_word1 = Operators.getOpWord(cur.ptr, 1);
                    const hash1 = Operators.rawHash(op_word1);

                    if (op_len == 4 and std.mem.readInt(u32, &Operators.sorted_padded_ops[Operators.mapToIndexRaw(hash4)], .little) == op_word4) {
                        op_type = @enumFromInt(hash4);
                        cur = cur[4..];
                    } else if (op_len >= 3 and std.mem.readInt(u32, &Operators.sorted_padded_ops[Operators.mapToIndexRaw(hash3)], .little) == op_word3) {
                        op_type = @enumFromInt(hash3);
                        if (op_type == .@"///" or op_type == .@"//!") {
                            selected_bitmap = bitmap_ptr[@as(u3, @truncate(@intFromEnum(BitmapKind.unknown)))..];
                            break;
                        }
                        cur = cur[3..];
                    } else if (op_len >= 2 and std.mem.readInt(u32, &Operators.sorted_padded_ops[Operators.mapToIndexRaw(hash2)], .little) == op_word2) {
                        op_type = @enumFromInt(hash2);
                        if (op_type == .@"//" or op_type == .@"\\\\") {
                            selected_bitmap = bitmap_ptr[@as(u3, @truncate(@intFromEnum(BitmapKind.unknown)))..];
                            break;
                        }
                        cur = cur[2..];
                    } else {
                        op_type = if (cur[0] == '\\') .invalid else @enumFromInt(hash1);
                        cur = cur[1..];
                    }

                    // op_type = bitmap_ptr[@as(u3, @truncate(@intFromEnum(BitmapKind.whitespace)))..];
                    // break;
                    cur = cur[@intFromBool(cur[0] == ' ')..];
                    continue;
                } else if (cur[0] == '\r') {
                    // TODO: unset the corresponding bit in the control-chars mask, since it should contain only newlines at the end.
                    if (cur[1] != '\n') {
                        return error.UnpairedCarriageReturn;
                    }
                    op_type = .whitespace;
                    selected_bitmap = bitmap_ptr[@as(u3, @truncate(@intFromEnum(op_type)))..];
                } else {
                    op_type = switch (cur[0]) {
                        'a'...'z', 'A'...'Z', '_' => .identifier,
                        '0'...'9' => .number,
                        '\'' => .char_literal,
                        '"' => .string,
                        ' ', '\t', '\n' => .whitespace,
                        0 => .eof,
                        else => .unknown,
                    };

                    selected_bitmap = bitmap_ptr[@as(u3, @truncate(@intFromEnum(op_type)))..];
                }

                if (cur[0] == '\'' or cur[0] == '"') {
                    const chr = cur[0];

                    switch (ON_DEMAND_IMPL) {
                        // 0 => while (true) {
                        //     cur = cur[1..];
                        //     comptime assert(std.mem.indexOfAny(u8, BACK_SENTINELS, "\n") != null);
                        //     if (cur[0] == chr or cur[0] < ' ') break;
                        //     std.mem.doNotOptimizeAway(cur[0]); // disable LLVM's default unroll.
                        //     cur = cur[@intFromBool(cur[0] == '\\')..];
                        //     if (cur[0] < ' ') break;
                        // },

                        0 => while (true) {
                            cur = cur[1..];
                            const is_escaped = cur[0] == '\\';
                            cur = cur[@intFromBool(is_escaped)..];

                            // Guarantee that this loop will terminate because '\n' < ' '
                            comptime assert(std.mem.indexOfAny(u8, BACK_SENTINELS, "\n") != null);
                            if ((cur[0] == chr and !is_escaped) or cur[0] < ' ') break;
                            std.mem.doNotOptimizeAway(cur[0]); // disable LLVM's default unroll.
                        },

                        // 0 => while (true) {
                        //     comptime assert(std.mem.indexOfAny(u8, BACK_SENTINELS, "\n") != null);
                        //     cur = cur[1..];

                        //     if (cur[0] == '\\') {
                        //         cur = cur[1..];
                        //     } else if (cur[0] == chr) {
                        //         break;
                        //     }

                        //     if (cur[0] < ' ') break;
                        // },

                        1 => {
                            cur = cur[1..];
                            var next_is_escaped_on_demand: uword = 0;

                            const QuoteChunk = @Vector(64, u8);
                            const QuoteMask = std.meta.Int(.unsigned, @sizeOf(QuoteChunk));

                            while (true) {
                                const chunk = blk: {
                                    const vec: QuoteChunk = cur[0..@sizeOf(QuoteChunk)].*;
                                    break :blk switch (comptime builtin.cpu.arch.endian()) {
                                        .little => vec,
                                        .big => std.simd.reverseOrder(vec),
                                    };
                                };

                                const ctrls: QuoteMask = @bitCast(chunk < @as(QuoteChunk, @splat(' ')));
                                const quotes: QuoteMask = @bitCast(chunk == @as(QuoteChunk, @splat(chr)));
                                const backslash: QuoteMask = @bitCast(chunk == @as(QuoteChunk, @splat('\\')));

                                // ----------------------------------------------------------------------------
                                // This code is brought to you courtesy of simdjson, licensed
                                // under the Apache 2.0 license which is included at the bottom of this file
                                const ODD_BITS: uword = @bitCast(@as(@Vector(@divExact(@bitSizeOf(uword), 8), u8), @splat(0xaa)));

                                // |                                | Mask (shows characters instead of 1's) | Depth | Instructions        |
                                // |--------------------------------|----------------------------------------|-------|---------------------|
                                // | string                         | `\\n_\\\n___\\\n___\\\\___\\\\__\\\`   |       |                     |
                                // |                                | `    even   odd    even   odd   odd`   |       |                     |
                                // | potential_escape               | ` \  \\\    \\\    \\\\   \\\\  \\\`   | 1     | 1 (backslash & ~first_is_escaped)
                                // | escape_and_terminal_code       | ` \n \ \n   \ \n   \ \    \ \   \ \`   | 5     | 5 (next_escape_and_terminal_code())
                                // | escaped                        | `\    \ n    \ n    \ \    \ \   \ ` X | 6     | 7 (escape_and_terminal_code ^ (potential_escape | first_is_escaped))
                                // | escape                         | `    \ \    \ \    \ \    \ \   \ \`   | 6     | 8 (escape_and_terminal_code & backslash)
                                // | first_is_escaped               | `\                                 `   | 7 (*) | 9 (escape >> 63) ()
                                //                                                                               (*) this is not needed until the next iteration
                                const potential_escape = backslash & ~next_is_escaped_on_demand;

                                // If we were to just shift and mask out any odd bits, we'd actually get a *half* right answer:
                                // any even-aligned backslash runs would be correct! Odd-aligned backslash runs would be
                                // inverted (\\\ would be 010 instead of 101).
                                //
                                // ```
                                // string:              | ____\\\\_\\\\_____ |
                                // maybe_escaped | ODD  |     \ \   \ \      |
                                //               even-aligned ^^^  ^^^^ odd-aligned
                                // ```
                                //
                                // Taking that into account, our basic strategy is:
                                //
                                // 1. Use subtraction to produce a mask with 1's for even-aligned runs and 0's for
                                //    odd-aligned runs.
                                // 2. XOR all odd bits, which masks out the odd bits in even-aligned runs, and brings IN the
                                //    odd bits in odd-aligned runs.
                                // 3. & with backslash to clean up any stray bits.
                                // runs are set to 0, and then XORing with "odd":
                                //
                                // |                                | Mask (shows characters instead of 1's) | Instructions        |
                                // |--------------------------------|----------------------------------------|---------------------|
                                // | string                         | `\\n_\\\n___\\\n___\\\\___\\\\__\\\`   |
                                // |                                | `    even   odd    even   odd   odd`   |
                                // | maybe_escaped                  | `  n  \\n    \\n    \\\_   \\\_  \\` X | 1 (potential_escape << 1)
                                // | maybe_escaped_and_odd          | ` \n_ \\n _ \\\n_ _ \\\__ _\\\_ \\\`   | 1 (maybe_escaped | odd)
                                // | even_series_codes_and_odd      | `  n_\\\  _    n_ _\\\\ _     _    `   | 1 (maybe_escaped_and_odd - potential_escape)
                                // | escape_and_terminal_code       | ` \n \ \n   \ \n   \ \    \ \   \ \`   | 1 (^ odd)
                                //

                                // Escaped characters are characters following an escape.
                                const maybe_escaped = potential_escape << 1;

                                // To distinguish odd from even escape sequences, therefore, we turn on any *starting*
                                // escapes that are on an odd byte. (We actually bring in all odd bits, for speed.)
                                // - Odd runs of backslashes are 0000, and the code at the end ("n" in \n or \\n) is 1.
                                // - Odd runs of backslashes are 1111, and the code at the end ("n" in \n or \\n) is 0.
                                // - All other odd bytes are 1, and even bytes are 0.
                                const maybe_escaped_and_odd_bits = maybe_escaped | ODD_BITS;
                                const even_series_codes_and_odd_bits = maybe_escaped_and_odd_bits -% potential_escape;

                                // Now we flip all odd bytes back with xor. This:
                                // - Makes odd runs of backslashes go from 0000 to 1010
                                // - Makes even runs of backslashes go from 1111 to 1010
                                // - Sets actually-escaped codes to 1 (the n in \n and \\n: \n = 11, \\n = 100)
                                // - Resets all other bytes to 0
                                const escape_and_terminal_code = even_series_codes_and_odd_bits ^ ODD_BITS;

                                const escaped = escape_and_terminal_code ^ (backslash | next_is_escaped_on_demand);
                                const escape = escape_and_terminal_code & backslash;
                                next_is_escaped_on_demand = escape >> (@bitSizeOf(uword) - 1);

                                const bitstring = (quotes | ctrls) & ~escaped;
                                cur = cur[@ctz(bitstring)..];
                                if (bitstring != 0) break;
                            }
                        },
                    }

                    if (cur[0] != chr) {
                        // TODO: remove this, handle invalid characters some-place else.
                    }

                    cur = cur[1..];
                    continue;
                }

                // cur = cur[1..];

                // On my machine, this can be a big win sometimes, other times, it has no effect.
                // According to perf stat, this supposedly increases the number of branch misses by
                // several hundred thousand. However, we decrease the total number of branches by millions.
                // Branch misses as a percentage are also higher, and yet, it's still faster.
                // Haven't looked, but maybe it's just putting the branches in a more desirable order?
                // We really need automatic profile-guided optimization in Zig so we can be certain.

                // if (selected_bitmap == &bitmap_ptr.*[@as(u3, @truncate(@intFromEnum(BitmapKind.whitespace)))] and switch (cur[0]) {
                //     ' ', '\t', '\r', '\n' => false,
                //     else => true,
                // }) continue;

                break;
            }
        }

        if (@intFromPtr(cur.ptr) < @intFromPtr(end_ptr)) return error.Found0ByteInFile;

        // TODO: think about whether we need this...?
        // utf8_checker.check_eof();
        // try utf8_checker.errors();

        cur_token = cur_token[if (cur_token[0].len == 0) 3 else 1..];
        cur_token[0] = .{ .len = 1, .kind = .eof };
        cur_token = cur_token[1..];

        // TODO: we do this because of cur_token[0..4].*
        // Prove at compiletime that 3 cannot be too much for cur_token to hold
        const new_chunks_data_len = 3 + (@intFromPtr(cur_token.ptr) - @intFromPtr(tokens.ptr)) / @sizeOf(Token);

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
        const t1 = std.time.nanoTimestamp();

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
            const throughput = @as(f64, @floatFromInt(bytes * 1000)) / @as(f64, @floatFromInt(elapsedNanos));
            try stdout.print("       Tokenizing took {: >9} ({d:.2} MB/s, {d: >5.2}M loc/s) and used {} memory\n", .{ std.fmt.fmtDuration(elapsedNanos), throughput, @as(f64, @floatFromInt(lines)) / @as(f64, @floatFromInt(elapsedNanos)) * 1000, std.fmt.fmtIntSizeDec(num_tokens * 2) });

            if (elapsedNanos2 > 0) {
                try stdout.print("       That's {d:.2}x faster and {d:.2}x less memory!\n", .{ @as(f64, @floatFromInt(elapsedNanos2)) / @as(f64, @floatFromInt(elapsedNanos)), @as(f64, @floatFromInt(num_tokens2 * 5)) / @as(f64, @floatFromInt(num_tokens * 2)) });
            }
        }

        if (INFIX_TEST) {
            for (sources.items, source_tokens) |source, tokens| {
                _ = try infixToPrefix(gpa, source, tokens);
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

fn printStack(buffer: *[16]Token, end_slot: [*]Token) void {
    var cur: [*]Token = buffer;

    std.debug.print("[", .{});
    while (cur != end_slot) : (cur += 1) {
        std.debug.print(" {s}", .{@tagName(cur[0].kind)});
    }
    std.debug.print(" ]\n", .{});
}

// fn priority(kind: Tag) u8 {
//     comptime var i: struct {
//         val: comptime_int = 0,

//         fn incr(comptime self: *@This()) comptime_int {
//             const v = self.val;
//             self.val = v + 1;
//             return v;
//         }
//     } = .{};

//     return switch (kind) {
//         .@"//!" => i.incr(),
//         .@"///", .@";" => i.incr(),
//         .@"=" => i.incr(),
//         .@"const" => i.incr(),
//         .@"+", .@"-" => i.incr(),
//         .@"*", .@"/", .@"%" => i.incr(),
//         .@"^" => i.incr(),
//         else => {
//             std.debug.print("No priority found for '{s}'\n", .{@tagName(kind)});
//             unreachable;
//         },
//     };
// }

const ast_formatter = struct {
    // Based on https://github.com/geoffleyland/lua-heaps/blob/18c5397762110aeaa8eae3237e04be677df56895/lua/binary_heap.lua#L138-L170
    fn formatHelperRecursive(
        comptime fmt: []const u8,
        items: anytype,
        writer: anytype,
        level_to_print: usize,
        i: usize,
        level: usize,
        end_padding: usize,
        nodeToString: anytype,
    ) !usize {
        if (i >= items.len) return 0;
        const element_len = std.math.lossyCast(usize, std.fmt.count(fmt, .{nodeToString.toString(items[i])}));
        const left_child_index = (std.math.mul(usize, i, 2) catch std.math.maxInt(usize)) | 1;
        const left_padding = try formatHelperRecursive(fmt, items, writer, level_to_print, left_child_index, level + 1, element_len, nodeToString);
        const right_padding = try formatHelperRecursive(fmt, items, writer, level_to_print, left_child_index +| 1, level + 1, end_padding, nodeToString);
        const added_len = left_padding +| element_len +| right_padding;

        if (level_to_print == level) {
            try writer.writeByteNTimes(' ', left_padding);
            try writer.print(fmt, .{nodeToString.toString(items[i])});
            try writer.writeByteNTimes(' ', right_padding +| end_padding);
        }
        return added_len;
    }

    pub fn format(
        self: anytype,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
        nodeToString: anytype,
    ) !void {
        _ = options;
        if (self.items.len == 0) return;
        for (0..@as(usize, std.math.log2_int(usize, self.items.len)) + 1) |level_to_print| {
            _ = try formatHelperRecursive(fmt, self.items[0..self.items.len], writer, level_to_print, 0, 0, 0, nodeToString);
            try writer.writeAll("\n");
        }
    }
};

// function foo(a: string, b: number)
// function foo ( : a string , : b number )

// , is now a postfix unary operator
// `fn` is now a prefix unary operator

// parenthesis

// a.b.c(d, e, f) => ( , d e f  )

fn recurse(nodes: std.zig.Ast.NodeList.Slice, index: usize, list: *std.ArrayList(std.zig.Ast.Node)) !void {
    if (index == 0) return;
    if (index < nodes.len) {
        const node = nodes.get(index);

        try list.append(node);
        try recurse(nodes, node.data.lhs, list);
        if (node.tag != .unwrap_optional and node.tag != .grouped_expression and node.tag != .field_access and node.data.lhs != node.data.rhs) try recurse(nodes, node.data.rhs, list);
    }
}

/// This OpString class is a specialized string data structure,
/// created specifically for the problem of converting infix
/// expressions to prefix.
///
/// The canonical algorithm to convert an infix expression to something more amenable is called
/// "Shunting Yard", which is a stack-based algorithm that can also
/// be implemented implicitly using a "Pratt Parser". Unfortunately,
/// any algorithm that exclusively pushes and pops from a stack can only convert to postfix.
///
/// This is an inherent problem that can be theoretically grounded in parser types too.
/// See: https://youtu.be/ZI198eFghJk?t=3234
///
/// Of course, we can still convert from infix to prefix, it is just a bit harder.
/// Basically, you need something akin to concatenation somewhere in the pipeline.
/// See the adapted Shunting Yard algorithm given here: https://stackoverflow.com/a/2034863
/// (Ctrl+F for `operator + LeftOperand + RightOperand`, that's the concatenation
/// in that answer.)
///
/// Given this conundrum, this has been solved in the following ways:
/// 1. We can simply let the post-ordering be the AST we want and let consumers deal with that
///     (per Chandler Carruth in the previous YouTube link).
/// 2. We can produce the AST nodes in (in-order and) post-order, but link them in pre-order fashion,
///     so we can still consume them in pre-order. This is sort of equivalent to deferring concatenation
///     by using a linked list of strings rather than allocating a new buffer and copying both of the old buffers over.
///     Basically, we are still doing the same work that is theoretically called for.
///     Even if we don't perceive the cost, we are still paying it.
/// 3. We can parse the tokens in reverse order. This makes the Shunting-Yard-equivalent algorithm produce pre-order
///     easily and post-order becomes the one that would require something akin to concatenation. I think this could work if we:
///         (a) allowed invalid AST's to be produced, which we would then check later (something which ASTGen does anyway atm)
///         (b) use some shift-reduce parsing techniques. Going backwards, return types might be hard to distinguish from struct literals.
///     I have seen people talk about this technique but am unaware of any practioners. Please note that my previous points are "spit-balling",
///     I do not actually know how it would go to try to implement this technique, as I have not done it.
/// 4. Skip the AST and go right to a bytecode IR. (a la PUC Lua).
///
/// This is an attempt to optimize the second option, by getting closer to the stack overflow algorithm.
/// Instead of just making a linked list of nodes to consume in the proper order later,
/// I want to see some more classic concatenation, when it's fast. Otherwise, we can fall back to a linked list.
///
/// The other optimization we can do is eliminate operands from our concatenation algorithm.
///     Since in-order to pre-order conversion ends with the operands in the same order relative to each other, we simply
///     need a bitstring which tracks where the operands should be inserted, but we do not need to care which operand until the end.
const OpString = extern struct {
    pub const BUF_SIZE = 8; // std.simd.suggestVectorSize(Token) orelse 8;
    pub const buf_idx_int = std.meta.Int(.unsigned, std.math.log2_int(u64, BUF_SIZE));
    // const BUF_SIZE_MASK = std.math.maxInt(buf_idx_int);

    ops: [BUF_SIZE]Token,
    bitstr: u64,
    next: u32,
    ops_len: u8,
    bitstr_len: u8,

    // fn append(self: *@This(), other: *const @This()) void {
    //     append_small: {
    //         const new_ops_len = std.math.add(buf_idx_int, self.ops_len, other.ops_len) catch break :append_small;
    //         const new_bitstr_len = std.math.add(
    //             @TypeOf(self.bitstr_len),
    //             self.bitstr_len,
    //             other.bitstr_len,
    //         ) catch break :append_small;

    //         const iota = std.simd.iota(u8, BUF_SIZE);
    //         const ops_len_splatted = @as(@Vector(BUF_SIZE, u8), @splat(self.ops_len));
    //         const shifted_op_indicies = iota -| ops_len_splatted;
    //         const shifted_other_ops = _lookup_chunk(other.ops, shifted_op_indicies);
    //         self.ops = @select(u8, iota < ops_len_splatted, self.ops, shifted_other_ops);
    //         self.ops_len = new_ops_len;

    //         const shifted_other_bitstr = other.bitstr << self.bitstr_len;
    //         self.bitstr = shifted_other_bitstr | self.bitstr;
    //         self.bitstr_len = new_bitstr_len;

    //         return;
    //     }

    //     // append_large
    //     unreachable;
    //     // self.next =
    // }

    fn prepend(self: *@This(), other: *@This()) void {
        small_enough: {
            const new_ops_len = std.math.add(buf_idx_int, @intCast(self.ops_len), @intCast(other.ops_len)) catch break :small_enough;
            const new_bitstr_len = std.math.add(
                std.math.Log2Int(@TypeOf(self.bitstr)),
                @intCast(self.bitstr_len),
                @intCast(other.bitstr_len),
            ) catch break :small_enough;

            const CONCAT_IMPL: u2 = 2;

            switch (CONCAT_IMPL) {
                0 => {
                    const iota = std.simd.iota(u8, BUF_SIZE);
                    const ops_len_splatted = @as(@Vector(BUF_SIZE, u8), @splat(other.ops_len));

                    const shifted_op_indicies = iota -| ops_len_splatted;
                    const shifted_self_ops = _lookup_chunk(self.ops, shifted_op_indicies);
                    self.ops = @select(u8, iota < ops_len_splatted, other.ops, shifted_self_ops);
                    self.ops_len = new_ops_len;

                    const shifted_self_bitstr = self.bitstr << @intCast(self.bitstr_len);
                    self.bitstr = shifted_self_bitstr | other.bitstr;
                    self.bitstr_len = new_bitstr_len;

                    return;
                },
                1 => {
                    const shifted_self_bitstr = self.bitstr << @intCast(self.bitstr_len);
                    const old_ops: @Vector(16, u8) = self.ops;

                    self.ops = other.ops;
                    self.ops[other.ops_len..].ptr[0..16].* = old_ops;

                    self.ops_len = new_ops_len;
                    self.bitstr = shifted_self_bitstr | other.bitstr;
                    self.bitstr_len = new_bitstr_len;
                    // self.next = ; // TODO: set self.next
                    return;
                },

                2 => {
                    const shifted_self_bitstr = self.bitstr << @intCast(self.bitstr_len);
                    const old_ops: @Vector(16, u8) = self.ops;

                    self.ops_len = new_ops_len;
                    self.bitstr = shifted_self_bitstr | other.bitstr;
                    self.bitstr_len = new_bitstr_len;

                    other.ops[other.ops_len..].ptr[0..16].* = old_ops;
                    self.ops = other.ops;
                    // self.next = ; // TODO: set self.next
                    return;
                },

                3 => {},
            }
        }

        // append_large
        unreachable;
        // self.next =
    }

    fn print(self: @This(), index: u32) void {
        std.debug.print("{b:0>32} | bitstring [{}]\n", .{ self.bitstr, self.bitstr_len });
        for (self.ops[0..self.ops_len]) |op| {
            std.debug.print("{s} ", .{@tagName(op.kind)});
        }
        std.debug.print("| ops [{}]\n", .{self.ops_len});
        std.debug.print("next: {}, i: {}, last: {}\n", .{ self.next, index, self.last });
    }
};

// test "OpString append" {
//     std.debug.print("\n", .{});
//     var a = std.mem.zeroInit(OpString, .{});
//     var b = std.mem.zeroInit(OpString, .{});

//     a.ops[0] = 1;
//     a.ops[1] = 2;
//     a.ops_len = 2;

//     a.bitstr = 0b110;
//     a.bitstr_len = 3;

//     b.ops[0] = 3;
//     b.ops[1] = 4;
//     b.ops_len = 2;

//     b.bitstr = 0b101;
//     b.bitstr_len = 3;

//     a.prepend(&b);
//     a.print();
// }

// If one is a sentinel, both are sentinels
const LinkedOpStringHead = struct { first: u32, last: u32 };

const OpStringBuffer = struct {
    const SENTINELS = [_]OpString{
        .{
            .ops = std.mem.zeroes([OpString.BUF_SIZE]Token),
            .ops_len = 0,
            .bitstr = 0b1,
            .bitstr_len = 1,
            .next = 0,
        },
        .{
            .ops = std.mem.zeroes([OpString.BUF_SIZE]Token),
            .ops_len = 0,
            .bitstr = 0b111,
            .bitstr_len = 3,
            .next = 1,
        },
    };
    const BUFFER_START_SIZE = 56;

    buffer: []OpString,
    next_available: u32 = if (BUFFER_START_SIZE > SENTINELS.len) SENTINELS.len else 0,

    fn init(gpa: Allocator) !@This() {
        const buffer = try gpa.alloc(OpString, BUFFER_START_SIZE);
        var self: @This() = .{ .buffer = buffer };

        // These are sentinel values we can use to initialize operands consisting of 1 token.
        comptime assert(BUFFER_START_SIZE >= SENTINELS.len);
        buffer[0..SENTINELS.len].* = SENTINELS;

        if (BUFFER_START_SIZE > SENTINELS.len) self.constructFreeList();
        return self;
    }

    fn deinit(self: @This(), gpa: Allocator) void {
        gpa.free(self.buffer);
    }

    fn isSentinel(a: LinkedOpStringHead) bool {
        return a.first < SENTINELS.len;
    }

    // fn deSentinelize(self: *@This(), gpa: Allocator, op_str_head: *LinkedOpStringHead) !void {
    //     if (isSentinel(op_str_head.first)) {
    //         assert(op_str_head.first == op_str_head.last);
    //         if (self.next_available == 0) try self.realloc(gpa);
    //         const available_slot = self.next_available;
    //         self.next_available = self.buffer[available_slot].next;
    //         op_str_head.first = available_slot;
    //         op_str_head.last = available_slot;
    //         self.buffer[available_slot] = self.buffer[op_str_head.first];
    //     }
    // }

    fn getNewSlot(self: *@This(), gpa: Allocator) !u32 {
        if (self.next_available == 0) try self.realloc(gpa);
        const available_slot = self.next_available;
        self.next_available = self.buffer[available_slot].next;
        return available_slot;
    }

    inline fn appendRightParen(self: *@This(), gpa: Allocator, right: LinkedOpStringHead, cur_token: [4]Token) !LinkedOpStringHead {
        assert(!isSentinel(right));
        // right: abc -> def -> ghi
        const ops_len: u8 = if (cur_token[0].len == 0) 3 else 1;
        const operand = &self.buffer[right.last];
        const bitstr_len = ops_len;
        const new_ops_len = operand.ops_len + ops_len;
        const new_bitstr_len = operand.bitstr_len + bitstr_len;
        const can_operator_fit_in_right_last = new_ops_len <= OpString.BUF_SIZE and new_bitstr_len <= @bitSizeOf(uword);

        if (can_operator_fit_in_right_last) {
            const dest_slot = right.last;
            const dest = &self.buffer[dest_slot];
            const shifted_self_bitstr = operand.bitstr;
            const self_next = operand.next;
            comptime assert(@sizeOf(OpString) >= OpString.BUF_SIZE + 3);
            dest.ops[operand.ops_len..].ptr[0..4].* = cur_token;
            dest.ops_len = new_ops_len;
            dest.bitstr = shifted_self_bitstr;
            dest.bitstr_len = new_bitstr_len;
            dest.next = self_next;

            return .{
                .first = right.first,
                .last = dest_slot,
            };
        } else {
            const dest_slot = try self.getNewSlot(gpa);
            const dest = &self.buffer[dest_slot];
            dest.ops[0..4].* = cur_token;
            dest.bitstr = 0;
            dest.bitstr_len = bitstr_len;
            dest.ops_len = ops_len;
            dest.next = 0;
            operand.next = dest_slot;

            return .{
                .first = right.first,
                .last = dest_slot,
            };
        }
    }

    /// Appends right to the left OpString. I.e. the ending order should be left followed by right.
    fn prependOperatorFromStack(self: *@This(), gpa: Allocator, operator_stack: *std.ArrayListUnmanaged(Token), right: LinkedOpStringHead) !LinkedOpStringHead {
        // right: abc -> def -> ghi
        const top_operator = operator_stack.pop();
        std.debug.print("top_op: {any}\n", .{top_operator});
        var full_operator = [4]Token{ top_operator, undefined, undefined, undefined };
        var ops_len: u8 = 1;

        if (top_operator.len == 0) {
            full_operator[1..][0..2].* = operator_stack.items[operator_stack.items.len - 2 ..][0..2].*;
            operator_stack.items.len -= 2;
            ops_len = 3;
        }

        const operand = self.buffer[right.first];
        const bitstr_len = ops_len;
        const new_ops_len = operand.ops_len + ops_len;
        const new_bitstr_len = operand.bitstr_len + bitstr_len;
        const can_operator_fit_in_right_first = new_ops_len <= OpString.BUF_SIZE and new_bitstr_len <= @bitSizeOf(uword);

        const is_right_sentinel = isSentinel(right);
        const dest_slot = if (is_right_sentinel or !can_operator_fit_in_right_first) try self.getNewSlot(gpa) else right.first;
        const dest = &self.buffer[dest_slot];
        const old_ops = operand.ops;
        dest.ops[0..4].* = full_operator;

        if (can_operator_fit_in_right_first) {
            const shifted_self_bitstr = operand.bitstr << @intCast(bitstr_len); // 0's are shifted in implicitly
            const self_next = operand.next;
            comptime assert(@sizeOf(OpString) >= OpString.BUF_SIZE + 3);
            dest.ops[ops_len..].ptr[0..OpString.BUF_SIZE].* = old_ops;
            dest.ops_len = new_ops_len;
            dest.bitstr = shifted_self_bitstr;
            dest.bitstr_len = new_bitstr_len;
            dest.next = self_next;
        } else {
            // We would not be here if `right` was a sentinel
            assert(!is_right_sentinel);

            dest.bitstr = 0;
            dest.bitstr_len = bitstr_len;
            dest.next = right.first;
            dest.ops_len = ops_len;
        }

        return .{
            .first = dest_slot,
            .last = if (is_right_sentinel) dest_slot else right.last,
        };
    }

    fn print(self: @This(), op_str_head: LinkedOpStringHead) void {
        std.debug.print("\n", .{});
        var cur = self.buffer[op_str_head.first];

        while (true) {
            var i = cur.bitstr_len;
            while (true) {
                i -= 1;
                std.debug.print("{b}", .{@as(u1, @truncate(cur.bitstr >> @intCast(i)))});
                if (i == 0) break;
            }
            std.debug.print(" | bitstring [{}]\n", .{cur.bitstr_len});
            for (cur.ops[0..cur.ops_len]) |op| {
                std.debug.print("{s} ", .{@tagName(op.kind)});
            }
            std.debug.print("| ops [{}]\n", .{cur.ops_len});
            std.debug.print("next: {}\n\n", .{cur.next});
            if (cur.next < SENTINELS.len) break;
            cur = self.buffer[cur.next];
        }
    }

    /// Appends right to the left OpString. I.e. the ending order should be left followed by right.
    fn joinOperands(self: *@This(), gpa: Allocator, left: LinkedOpStringHead, right: LinkedOpStringHead) !LinkedOpStringHead {
        //[1 9 0 4 5 8 a 6  7  3  b f 2 e g d h    ] 68
        // - 1 + & - ! b .? .* * -% 3 - ! 4 / & 5
        // 0 1 2 3 4 5 6 7  8  9  a b c d e f g h
        //
        // first : 12
        // - + unary - 1 * unary & unary - ! .* .? 2 unary -% 3 / ! 4 unary & 5
        //  left: abcd -> efgh -> ijkl
        // right: mnop -> qrst -> uvxy

        // if right.first can fit inside left.last -> append it in-place in left.last, set left.last to right.last, set buffer[left.last].next = right.first.next
        // otherwise -> left.last.next = right.first, left.last = right.last,
        //

        const is_left_a_sentinel = isSentinel(left);
        const is_right_a_sentinel = isSentinel(right);

        const new_ops_len = self.buffer[left.last].ops_len + self.buffer[right.first].ops_len;
        const new_bitstr_len = self.buffer[left.last].bitstr_len + self.buffer[right.first].bitstr_len;

        if (new_ops_len <= OpString.BUF_SIZE and new_bitstr_len <= @bitSizeOf(uword)) {
            const dest_slot = if (!is_left_a_sentinel)
                left.last
            else if (!is_right_a_sentinel)
                right.first // This is safe because left is a sentinel, and our sentinels have no operators in their buffer.
            else
                try self.getNewSlot(gpa);

            const a = self.buffer[left.last];
            const b = &self.buffer[right.first];

            const dest = &self.buffer[dest_slot];
            const new_bitstr = a.bitstr | (b.bitstr << @intCast(a.bitstr_len));
            const b_next = b.next;
            const b_ops = b.ops;
            comptime assert(@sizeOf(OpString) >= OpString.BUF_SIZE * 2);
            dest.ops[a.ops_len..].ptr[0..OpString.BUF_SIZE].* = b_ops;
            dest.ops_len = new_ops_len;
            dest.bitstr = new_bitstr;
            dest.bitstr_len = new_bitstr_len;
            dest.next = b_next;

            if (!is_left_a_sentinel and !is_right_a_sentinel) {
                // kill right.first, insert it back into the free list
                b.next = self.next_available;
                self.next_available = right.first;
            }

            // is_sentinel table
            // T T => .{ .first = dest_slot, .last = dest_slot }
            // T F => .{ .first = right.first, .last = right.last }
            // F T => .{ .first = left.first, .last = left.last }
            // F F => .{ .first = left.first, .last = right.last }

            const first = if (is_left_a_sentinel) dest_slot else left.first;
            return .{
                .first = first,
                .last = if (is_right_a_sentinel) dest_slot else if (right.first == right.last) first else right.last,
            };
        } else {
            // It is trivial to see we would not be in this else branch if both left and right were sentinels.
            assert(!is_left_a_sentinel or !is_right_a_sentinel);

            const new_slot = if (is_left_a_sentinel or is_right_a_sentinel) try self.getNewSlot(gpa) else 0;
            const old_slot = if (is_left_a_sentinel) left.last else if (is_right_a_sentinel) right.first else 0;

            const next_slot = if (is_left_a_sentinel) new_slot else left.last;
            const new_next = if (is_right_a_sentinel) new_slot else right.first;

            self.buffer[new_slot] = self.buffer[old_slot];
            self.buffer[next_slot].next = new_next;

            return .{
                .first = if (is_left_a_sentinel) new_slot else left.first,
                .last = if (is_right_a_sentinel) new_slot else right.last,
            };
        }
    }

    fn realloc(self: *@This(), gpa: Allocator) !void {
        // Safe because the number of operators cannot exceed the number of bytes in a source file (u32)
        // TODO: is this true?
        self.next_available = @intCast(self.buffer.len);
        const new_op_str_buffer_len = self.buffer.len * 2;
        const new_op_str_buffer = try gpa.alloc(OpString, new_op_str_buffer_len);
        @memcpy(new_op_str_buffer[0..self.buffer.len], self.buffer);
        gpa.free(self.buffer);
        self.buffer = new_op_str_buffer;
        self.constructFreeList();
    }

    fn constructFreeList(self: @This()) void {
        const last_slot_idx = self.buffer.len - 1;
        var i = self.next_available;

        while (i < last_slot_idx) : (i += 1)
            self.buffer[i].next = i + 1;

        self.buffer[last_slot_idx].next = 0;
    }
};

test "OpStringBuffer" {
    std.debug.print("\n", .{});
    const gpa = std.testing.allocator;
    var op_str_buffer = try OpStringBuffer.init(gpa);
    defer op_str_buffer.deinit(gpa);

    const left: LinkedOpStringHead = .{
        .first = 0,
        .last = 0,
    };
    const right: LinkedOpStringHead = .{
        .first = 1,
        .last = 1,
    };
    var res = left;
    inline for (0..20) |_| {
        res = try op_str_buffer.joinOperands(gpa, res, right);
        res = try op_str_buffer.joinOperands(gpa, left, res);
        res = try op_str_buffer.joinOperands(gpa, res, left);
        res = try op_str_buffer.joinOperands(gpa, right, res);
    }

    const new_slot = try op_str_buffer.getNewSlot(gpa);
    const new_head: LinkedOpStringHead = .{ .first = new_slot, .last = new_slot };

    const slot = &op_str_buffer.buffer[new_slot];
    slot.bitstr = 0b0110;
    slot.bitstr_len = 4;
    slot.next = 0;
    slot.ops_len = 2;
    slot.ops[0..2].* = [2]Token{ .{ .kind = .number, .len = 1 }, .{ .kind = .@";", .len = 1 } };

    res = try op_str_buffer.joinOperands(gpa, new_head, left); // TODO: step through this
    std.debug.print("res: {any}\n", .{res});

    // res = try op_str_buffer.joinOperands(gpa, right, res);
    op_str_buffer.print(res);
}

fn infixToPrefix(gpa: Allocator, source: [:0]const u8, tokens: []const Token) ![]Token {
    try infixToPrefixPrinter(gpa, source, tokens);

    const my_token: Token = .{ .kind = .@".", .len = 1 };
    _ = my_token;
    var op_str_buffer = try OpStringBuffer.init(gpa);
    defer op_str_buffer.deinit(gpa);

    // This is now the head of our free list :)

    const output: std.ArrayListUnmanaged(Token) = .{};
    _ = output;

    var operand_list_strs = if (builtin.mode == .Debug) try std.ArrayListUnmanaged([]const u8).initCapacity(gpa, 1000);
    var operand_list = try std.ArrayListUnmanaged(Token).initCapacity(gpa, 1000);

    // Each operand element u32 points into `op_str_buffer`
    // Each element is a linked-list data structure, where each node in `op_str_buffer` has a `next` reference to some other node in `op_str_buffer`,
    // where the data here can tell us where the `first` and `last` elements are.
    var operand_stack: std.ArrayListUnmanaged(LinkedOpStringHead) = .{};
    var operator_stack: std.ArrayListUnmanaged(Token) = .{};
    try operator_stack.append(gpa, Token{ .len = 0, .kind = .sentinel_operator });

    var cur_token = tokens[0..];
    var cur = source;

    if (tokens[0].kind == .whitespace) {
        cur_token = cur_token[1..];
        cur = cur[tokens[0].len..];
    }

    // Algorithm ConvertInfixtoPrefix

    // Purpose: Convert an infix expression into a prefix expression. Begin
    // // Create operand and operator stacks as empty stacks.
    // Create OperandStack
    // Create OperatorStack

    // // While input expression still remains, read and process the next token.

    // while( not an empty input expression ) read next token from the input expression

    //     // Test if token is an operand or operator
    //     if ( token is an operand )
    //     // Push operand onto the operand stack.
    //         OperandStack.Push (token)
    //     endif

    //     // If it is a left parentheses or operator of higher precedence than the last, or the stack is empty,
    //     else if ( token is '(' or OperatorStack.IsEmpty() or OperatorHierarchy(token) > OperatorHierarchy(OperatorStack.Top()) )
    //     // push it to the operator stack
    //         OperatorStack.Push ( token )
    //     endif

    //     else if( token is ')' )
    //     // Continue to pop operator and operand stacks, building
    //     // prefix expressions until left parentheses is found.
    //     // Each prefix expression is push back onto the operand
    //     // stack as either a left or right operand for the next operator.
    //         while( OperatorStack.Top() not equal '(' )
    //             OperatorStack.Pop(operator)
    //             OperandStack.Pop(RightOperand)
    //             OperandStack.Pop(LeftOperand)
    //             operand = operator + LeftOperand + RightOperand
    //             OperandStack.Push(operand)
    //         endwhile

    //     // Pop the left parthenses from the operator stack.
    //     OperatorStack.Pop(operator)
    //     endif

    //     else if( operator hierarchy of token is less than or equal to hierarchy of top of the operator stack )
    //     // Continue to pop operator and operand stack, building prefix
    //     // expressions until the stack is empty or until an operator at
    //     // the top of the operator stack has a lower hierarchy than that
    //     // of the token.
    //         while( !OperatorStack.IsEmpty() and OperatorHierarchy(token) lessThen Or Equal to OperatorHierarchy(OperatorStack.Top()) )
    //             OperatorStack.Pop(operator)
    //             OperandStack.Pop(RightOperand)
    //             OperandStack.Pop(LeftOperand)
    //             operand = operator + LeftOperand + RightOperand
    //             OperandStack.Push(operand)
    //         endwhile
    //         // Push the lower precedence operator onto the stack
    //         OperatorStack.Push(token)
    //     endif
    // endwhile
    // // If the stack is not empty, continue to pop operator and operand stacks building
    // // prefix expressions until the operator stack is empty.
    // while( !OperatorStack.IsEmpty() ) OperatorStack.Pop(operator)
    //     OperandStack.Pop(RightOperand)
    //     OperandStack.Pop(LeftOperand)
    //     operand = operator + LeftOperand + RightOperand

    //     OperandStack.Push(operand)
    // endwhile

    // // Save the prefix expression at the top of the operand stack followed by popping // the operand stack.

    // print OperandStack.Top()

    // OperandStack.Pop()

    // End

    var un_ctx = true;

    while (true) : (cur_token = cur_token[1..]) {
        if (!Parser.isOperand(cur_token[0].kind)) {
            std.debug.print("operator_stack: ", .{});
            for (operator_stack.items) |operator| std.debug.print("{s} ", .{@tagName(operator.kind)});
            std.debug.print("\n", .{});
        }

        const cur_token_str = cur[0..cur_token[0].len];
        std.debug.print("cur_token: \"{s}\"\n", .{cur_token_str});
        cur = cur[if (cur_token[0].len == 0) @as(u32, @bitCast(cur_token[1..][0..2].*)) else cur_token[0].len..];
        const raw_class = Operators.classify(cur_token[0].kind);
        const cur_token_kind: Tag = switch (raw_class) {
            .ambiguous_pre_unary_or_binary => if (un_ctx) Operators.unarifyBinaryOperator(cur_token[0].kind) else cur_token[0].kind,
            .ambiguous_pre_unary_or_post_unary => if (un_ctx) cur_token[0].kind else Operators.postifyOperator(cur_token[0].kind),
            else => cur_token[0].kind,
        };

        const cur_class: @TypeOf(raw_class) = switch (raw_class) {
            .ambiguous_pre_unary_or_binary => if (un_ctx) .pre_unary_op else .binary_op,
            .ambiguous_pre_unary_or_post_unary => if (un_ctx) .pre_unary_op else .post_unary_op,
            else => |r| r,
        };

        un_ctx = raw_class != .operand and raw_class != .post_unary_op;

        if (cur_class == .operand) {
            try operand_list.append(gpa, cur_token[0]);
            if (builtin.mode == .Debug) try operand_list_strs.append(gpa, cur_token_str);
            const sentinel_index = @intFromBool(cur_token[0].len == 0);
            try operand_stack.append(gpa, .{ .first = sentinel_index, .last = sentinel_index });
        } else {
            if (cur_class != .pre_unary_op and cur_token_kind != .@"call (") {
                // std.debug.print("::isunary::{}\n", .{Parser.isUnary(cur_token_kind)});
                std.debug.print("::{s}\n", .{@tagName(cur_class)});

                // Continue to pop operator and operand stack, building prefix
                // expressions until the stack is empty or until an operator at
                // the top of the operator stack has a lower hierarchy than that
                // of the token.

                var top_operator = operator_stack.getLast();

                if (Operators.getPrecedence(cur_token_kind) <= Operators.getPrecedence(top_operator.kind)) {
                    // new_operand = operator_stack.pop() + left + right
                    var new_operand: LinkedOpStringHead = operand_stack.pop();

                    while (true) {
                        op_str_buffer.print(new_operand);
                        if (!Operators.isUnary(top_operator.kind)) {
                            new_operand = try op_str_buffer.joinOperands(gpa, operand_stack.pop(), new_operand);
                        }
                        op_str_buffer.print(new_operand);
                        new_operand = try op_str_buffer.prependOperatorFromStack(gpa, &operator_stack, new_operand);
                        op_str_buffer.print(new_operand);
                        top_operator = operator_stack.getLast();
                        if (Operators.getPrecedence(cur_token_kind) > Operators.getPrecedence(top_operator.kind)) break;
                    }

                    try operand_stack.append(gpa, new_operand);

                    if (cur_token_kind == .@")") {
                        // TODO: can we carefully craft the precedence table to do this implicitly?
                        const last_slot = &operand_stack.items[operand_stack.items.len - 1];
                        last_slot.* = try op_str_buffer.prependOperatorFromStack(gpa, &operator_stack, last_slot.*);
                        op_str_buffer.print(last_slot.*);
                        last_slot.* = try op_str_buffer.appendRightParen(gpa, last_slot.*, cur_token[0..4].*);
                        op_str_buffer.print(last_slot.*);
                        continue;
                    }
                    // std.debug.print("{any}\n", .{operand_stack.items});
                }
            }
            // Push the current operator into the operator stack.
            // If the len field is 0, write the actual length BEFORE the current token in the stack.
            // That way, when we pop the top token, we unambiguously know whether to pop extra bytes to get the length.
            const current_token = Token{ .len = cur_token[0].len, .kind = cur_token_kind };
            if (current_token.len == 0) {
                try operator_stack.appendSlice(gpa, cur_token[1..][0..2]);
                cur_token = cur_token[2..];
            }
            try operator_stack.append(gpa, current_token);
        }

        if (cur_token_kind == .@";") {
            const last_slot = &operand_stack.items[operand_stack.items.len - 1];
            last_slot.* = try op_str_buffer.prependOperatorFromStack(gpa, &operator_stack, last_slot.*);
        }

        if (cur_token_kind == .@";") break;
    }

    // 0110100101011011 =>
    // 0's: [0, _, _, 1, _, 2, 3, _, 4, _, 5, _, _, 6, _, _]
    // 1's: [_, 0, 1, _, 2, _, _, 3, _, 4, _, 5, 6, _, 7, 8]

    // 01101001, 10011001 =>
    // 0's: [0, _, _, 1, _, 2, 3, _]
    // 1's: [_, 0, 1, _, 2, _, _, 3]

    // ^12*345^67*890
    // +^12-*34+/5^67-*890

    var top: *OpString = &op_str_buffer.buffer[operand_stack.pop().first];
    var cur_ops = [2][*]const Token{ undefined, operand_list.items.ptr };

    var total_buf_len: u32 = 0;
    {
        var t = top;
        while (true) {
            total_buf_len += t.bitstr_len;
            if (t.next < OpStringBuffer.SENTINELS.len) break;
            t = &op_str_buffer.buffer[t.next];
        }
    }
    const ret_buffer = try gpa.alloc(Token, total_buf_len);
    errdefer gpa.free(ret_buffer);
    var ret_buf_cur = ret_buffer;
    var k: usize = 0;

    while (true) : (top = &op_str_buffer.buffer[top.next]) {
        // top.printFinal(&operand_list);
        // top.print();

        const operator_tokens = @as([OpString.BUF_SIZE]Token, @bitCast(top.ops))[0..top.ops_len];
        cur_ops[0] = operator_tokens.ptr;
        // std.debug.print("{any: >4}\n", .{PRECOMPUTED_TABLE[0b10110100]});
        // std.debug.print("{any: >4}\n", .{top.ops});
        // std.debug.print("{any: >4}\n", .{@as([*]u8, @ptrCast(operand_list.items.ptr))[0..16].*});
        // const precomputed_shuffle = PRECOMPUTED_TABLE[0b10110100];
        // const SHUFFLE_VEC_TYPE = if (USE_BYTE_SHUFFLE) @Vector(16, u8) else @Vector(8, u16);
        // const vec1 = vector_shuffle(SHUFFLE_VEC_TYPE, top.ops, precomputed_shuffle);
        // const vec2 = vector_shuffle(SHUFFLE_VEC_TYPE, @as([*]u8, @ptrCast(operand_list.items.ptr))[0..16].*, precomputed_shuffle);

        // std.debug.print("{any: >4}\n", .{vec1});
        // std.debug.print("{any: >4}\n", .{vec2});
        // const answer = @select(u8, @as(@Vector(16, bool), @bitCast(@as(u16, 0b1100111100110000))), vec2, vec1);

        // std.debug.print("{any: >4}\n", .{answer});
        // { 0, 1, 0, 2, 1, 2, 3, 3 }
        // { 0, 1, 2, 3, 0, 1, 4, 5, 2, 3, 4, 5, 6, 7, 6, 7 }
        {
            var j: usize = 0;

            for (0..top.bitstr_len) |i| {
                //  - + 1 * 2 3 / 4 5
                switch (@as(u1, @truncate(top.bitstr >> @intCast(i)))) {
                    0 => {
                        const ret = operator_tokens[j];
                        j += 1;
                        std.debug.print("{s} ", .{@tagName(ret.kind)});
                    },
                    1 => {
                        // const ret = operand_list.items[k];
                        k += 1;
                        if (builtin.mode == .Debug) {
                            std.debug.print("{s} ", .{operand_list_strs.items[k - 1]});
                        } else {
                            std.debug.print("{} ", .{k});
                        }
                    },
                }
            }
        }

        for (0..top.bitstr_len) |i| {
            //  - + 1 * 2 3 / 4 5
            const bit = @as(u1, @truncate(top.bitstr >> @intCast(i)));
            const ptr = &cur_ops[bit];
            ret_buf_cur[0] = ptr.*[0];
            ret_buf_cur = ret_buf_cur[1..];
            ptr.* = ptr.*[1..];
        }

        if (top.next < OpStringBuffer.SENTINELS.len) break;
    }
    std.debug.print("\n", .{});
    return ret_buffer;
}

fn infixToPrefixPrinter(gpa: Allocator, source: [:0]const u8, tokens: []const Token) !void {
    const TokenList = std.MultiArrayList(struct {
        tag: std.zig.Token.Tag,
        start: Ast.ByteOffset,
        end: Ast.ByteOffset,
    });
    var legacy_tokens = TokenList{};
    defer legacy_tokens.deinit(gpa);

    // Empirically, the zig std lib has an 8:1 ratio of source bytes to token count.
    const estimated_token_count = source.len / 8;
    try legacy_tokens.ensureTotalCapacity(gpa, estimated_token_count);

    var tokenizer = std.zig.Tokenizer.init(source);
    while (true) {
        const token = tokenizer.next();
        try legacy_tokens.append(gpa, .{
            .tag = token.tag,
            .start = @as(u32, @intCast(token.loc.start)),
            .end = @as(u32, @intCast(token.loc.end)),
        });
        if (token.tag == .eof) break;
    }

    const ast = try std.zig.Ast.parse(gpa, source, .zig);

    if (ast.errors.len > 0) {
        std.debug.print("{} error{s} occurred:\n", .{ ast.errors.len, if (ast.errors.len == 1) "" else "s" });
        for (ast.errors) |error_item| {
            std.debug.print("\t{}\n", .{error_item});
        }
        std.debug.print("\n", .{});
    }

    for (0.., ast.nodes.items(.tag), ast.nodes.items(.main_token), ast.nodes.items(.data)) |i, tag, main_token, data| {
        std.debug.print("{: <2} {s: ^16} {{ main_tok = {: >2}, lhs = {}, rhs = {: >2} }}\n", .{ i, @tagName(tag), main_token, data.lhs, data.rhs });
    }

    var ast_list = std.ArrayList(std.zig.Ast.Node).init(gpa);
    if (ast.nodes.len > 1) {
        const stdout = std.io.getStdOut().writer();
        var ws = std.json.writeStream(stdout, .{ .whitespace = .indent_2 });
        defer ws.deinit();

        std.debug.print("\n", .{});

        const root = ast.nodes.get(1).data;

        try recurse(ast.nodes, root.lhs, &ast_list);
        try recurse(ast.nodes, root.rhs, &ast_list);

        const nodeToString = struct {
            source: @TypeOf(source),
            ast: @TypeOf(ast),
            legacy_tokens: @TypeOf(legacy_tokens),

            pub fn toString(this: *const @This(), node: std.zig.Ast.Node) []const u8 {
                const data = this.legacy_tokens.get(node.main_token);
                return if (node.tag == .unwrap_optional)
                    ".?"
                else if (node.tag == .root)
                    "_"
                else
                    std.zig.Token.Tag.lexeme(data.tag) orelse this.source[data.start..data.end];
            }
        }{
            .source = source,
            .ast = ast,
            .legacy_tokens = legacy_tokens,
        };

        for (ast_list.items) |node| {
            std.debug.print("{s} ", .{nodeToString.toString(node)});
        }
        std.debug.print("\n", .{});
    }

    // const c = blk: {
    //     const a = 1;
    //     const b = 2;
    //     break :blk a + b;
    // } + blk: {
    //     const a = 2;
    //     const b = 3;
    //     break :blk a + b;
    // } * blk: {
    //     const a = 3;
    //     const c = 4;
    //     break :blk a + c;
    // };
    // _ = c;

    // std.zig.Ast.parse(gpa: Allocator, source: [:0]const u8, mode: Mode)

    // const c4 = 1.e4;
    // @compileLog(c4);

    // Print first
    var pos: u32 = 0;

    for (tokens) |token| {
        if (Parser.isOperand(token.kind)) {
            std.debug.print("{s} ", .{source[pos..][0..token.len]});
        } else std.debug.print("{s} ", .{@tagName(token.kind)});
        pos += token.len;
    }
    std.debug.print("\n", .{});
}

// ---------------------------------------------------------------
//
// The code below this point is licensed under the Apache License.
// Please see the License at the bottom of this file.
//
// ---------------------------------------------------------------

const Utf8Checker = struct {
    is_invalid_place_to_end: bool = false,
    prev_input_block: Chunk = std.mem.zeroes(Chunk),

    fn prev(comptime N: comptime_int, a: Chunk, b: Chunk) Chunk {
        comptime assert(0 < N and N < @sizeOf(Chunk));
        return if (USE_SWAR)
            (b >> ((@sizeOf(Chunk) - N) * 8)) |
                (a << (N * 8))
        else
            std.simd.mergeShift(b, a, @sizeOf(Chunk) - N);
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
    pub fn lookup_16_aarch64(x: @Vector(16, u8), mask: @Vector(16, u8)) @Vector(16, u8) {
        return asm (
            \\tbl  %[out].16b, {%[mask].16b}, %[x].16b
            : [out] "=&x" (-> @Vector(16, u8)),
            : [x] "x" (x),
              [mask] "x" (mask),
        );
    }

    fn lookup_chunk(comptime a: [16]u8, b: Chunk) Chunk {
        switch (builtin.cpu.arch) {
            .x86_64 => return mm_shuffle_epi8(a ** (@sizeOf(Chunk) / 16), b),
            .aarch64, .aarch64_be => return lookup_16_aarch64(b, a ** (@sizeOf(Chunk) / 16)),
            else => {
                var r: Chunk = @splat(0);
                for (0..@sizeOf(Chunk)) |i| {
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
        const CARRY: u8 = TOO_SHORT | TOO_LONG | TWO_CONTS; // These all have ____ in byte 1 .

        const byte_1_high_tbl = [16]u8{
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

        const byte_1_low_tbl = [16]u8{
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

        const byte_2_high_tbl = [16]u8{
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


        // zig fmt: on
        if (USE_SWAR) {
            // Merge two of the 16-byte lookup tables for the SWAR implementation, that way we "only" do
            // number-of-bytes*2 table lookups, rather than number-of-bytes*3
            comptime var byte_1_tbl: [256]u8 = undefined;
            inline for (&byte_1_tbl, 0..) |*slot, i| {
                slot.* = byte_1_low_tbl[i & 0xF] & byte_1_high_tbl[i >> 4];
            }

            var result: @TypeOf(input) = 0;
            inline for (0..8) |i| {
                const j = i * 8;
                const prev_ans = if (builtin.mode == .ReleaseSmall)
                    byte_1_low_tbl[@as(u4, @truncate(prev1 >> j))] & byte_1_high_tbl[@as(u4, @truncate(prev1 >> (j + 4)))]
                else
                    byte_1_tbl[@as(u8, @truncate(prev1 >> j))];
                result |= @as(Chunk, byte_2_high_tbl[@as(u4, @truncate(input >> (j + 4)))] & prev_ans) << j;
            }
            return result;
        } else {
            return lookup_chunk(byte_1_low_tbl, prev1 & @as(@TypeOf(prev1), @splat(0x0F))) &
                lookup_chunk(byte_1_high_tbl, prev1 >> @splat(4)) &
                lookup_chunk(byte_2_high_tbl, input >> @splat(4));
        }
    }

    fn must_be_2_3_continuation(prev2: Chunk, prev3: Chunk) Chunk {
        if (USE_SWAR) {
            const ones: Chunk = @bitCast(@as(@Vector(@sizeOf(Chunk), u8), @splat(1)));
            const msbs: Chunk = ones * 0x80;
            const is_3rd_byte = prev2 & ((prev2 | msbs) - (0b11100000 - 0x80) * ones);
            const is_4th_byte = prev3 & ((prev3 | msbs) - (0b11110000 - 0x80) * ones);
            return (is_3rd_byte | is_4th_byte) & msbs;
        } else {
            const is_3rd_byte = prev2 -| @as(Chunk, @splat(0b11100000 - 0x80));
            const is_4th_byte = prev3 -| @as(Chunk, @splat(0b11110000 - 0x80));
            return (is_3rd_byte | is_4th_byte) & @as(Chunk, @splat(0x80));
        }
    }

    fn isASCII(input: Chunk) bool {
        return 0 == if (USE_SWAR)
            (input & @as(NATIVE_VEC_INT, @bitCast(@as(@Vector(@sizeOf(uword), u8), @splat(0x80)))))
        else
            @as(std.meta.Int(.unsigned, NATIVE_VEC_SIZE), @bitCast(input >= @as(@Vector(NATIVE_VEC_SIZE, u8), @splat(0x80))));
    }

    fn validateChunk(utf8_checker: *Utf8Checker, input: Chunk) !void {
        if (isASCII(input)) {
            // Fast path, don't do so much work if we found an ascii chunk, because most chunks are ascii.
            try utf8_checker.errors();
            return;
        }

        // Check whether the current bytes are valid UTF-8.
        // Flip prev1...prev3 so we can easily determine if they are 2+, 3+ or 4+ lead bytes
        // (2, 3, 4-byte leads become large positive numbers instead of small negative numbers)

        const prev_input = utf8_checker.prev_input_block;
        utf8_checker.prev_input_block = input;
        const prev1 = prev(1, input, prev_input);
        const sc = check_special_cases(input, prev1);
        const prev2 = prev(2, input, prev_input);
        const prev3 = prev(3, input, prev_input);
        const must23_80 = must_be_2_3_continuation(prev2, prev3);

        const ones: Chunk = @bitCast(@as(@Vector(@sizeOf(Chunk), u8), @splat(1)));
        const msbs: Chunk = @bitCast(@as(@Vector(@sizeOf(Chunk), u8), @splat(0x80)));

        const err = (must23_80 ^ sc) | if (USE_SWAR) blk: {
            // Will have a zero byte in `y` if there was a '\u{2028}' or '\u{2029}'
            const x = (input ^ (0b1010_1000 * ones));
            const y = (prev2 ^ (0b1110_0010 * ones)) |
                (prev1 ^ (0b1000_0000 * ones)) |
                (x & (x ^ ones));

            // Will have a zero byte in `z` if there was a '\u{85}'
            const z =
                (prev1 ^ (0b1100_0010 * ones)) |
                (input ^ (0b1000_0101 * ones));

            const is0x2028or0x2029 = (y -% ones) & ~y;
            const is0x85 = (z -% ones) & ~z;
            break :blk (is0x2028or0x2029 | is0x85) & msbs;
        } else blk: {
            const true_vec = @as(Chunk, @splat(0xFF));
            const false_vec = @as(Chunk, @splat(0));

            const is0x2028or0x2029 =
                @select(u8, prev2 == @as(Chunk, @splat(0b1110_0010)), true_vec, false_vec) &
                @select(u8, prev1 == @as(Chunk, @splat(0b1000_0000)), true_vec, false_vec) &
                (@select(u8, input == @as(Chunk, @splat(0b1010_1000)), true_vec, false_vec) |
                @select(u8, input == @as(Chunk, @splat(0b1010_1001)), true_vec, false_vec));

            const is0x85 =
                @select(u8, prev1 == @as(Chunk, @splat(0b1100_0010)), true_vec, false_vec) &
                @select(u8, input == @as(Chunk, @splat(0b1000_0101)), true_vec, false_vec);
            break :blk is0x2028or0x2029 | is0x85;
        };

        if (hasError(err))
            return error.InvalidUtf8;

        // If there are incomplete multibyte characters at the end of the block,
        // write that data into `self.err`.
        // e.g. if there is a 4-byte character, but it's 3 bytes from the end.
        //
        // If the previous input's last 3 bytes match this, they're too short (if they ended at EOF):
        // ... 1111____ 111_____ 11______
        utf8_checker.is_invalid_place_to_end = 0 !=
            if (USE_SWAR)
            msbs & input & ((input | msbs) - comptime max_value: {
                var max_array = [1]u8{0} ** @sizeOf(Chunk);
                max_array[@sizeOf(Chunk) - 3] = 0b11110000 - 0x80;
                max_array[@sizeOf(Chunk) - 2] = 0b11100000 - 0x80;
                max_array[@sizeOf(Chunk) - 1] = 0b11000000 - 0x80;
                break :max_value @as(Chunk, @bitCast(max_array));
            })
        else
            @reduce(if (builtin.cpu.arch == .x86_64) .Or else .Max, input -| comptime max_value: {
                var max_array: Chunk = @splat(0xFF);
                max_array[@sizeOf(Chunk) - 3] = 0b11110000 - 1;
                max_array[@sizeOf(Chunk) - 2] = 0b11100000 - 1;
                max_array[@sizeOf(Chunk) - 1] = 0b11000000 - 1;
                break :max_value max_array;
            });
    }

    fn hasError(err: Chunk) bool {
        return (0 != if (USE_SWAR) err else @reduce(if (builtin.cpu.arch == .x86_64) .Or else .Max, err));
    }

    fn errors(checker: Utf8Checker) !void {
        if (checker.is_invalid_place_to_end)
            return error.InvalidUtf8;
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
