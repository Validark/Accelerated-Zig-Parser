const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const testing = std.testing;
const mem = std.mem;
const Ast = std.zig.Ast;
const Allocator = std.mem.Allocator;

const VEC_SIZE = 64;
const VEC_INT = std.meta.Int(.unsigned, VEC_SIZE);
const VEC = @Vector(VEC_SIZE, u8);
const IVEC = @Vector(VEC_SIZE, i8);
const LOG_VEC_INT = std.math.Log2Int(VEC_INT);

const FRONT_SENTINELS = "\n";
const BACK_SENTINELS = "\n" ++ "\x00'\"" ++ "\x00" ** 10;

/// Returns a slice that is aligned to alignment, but `len` is set to only the "valid" memory.
/// freed via: allocator.free(buffer[0..std.mem.alignForward(u64, buffer.len, alignment)])
fn readFileIntoAlignedBuffer(allocator: Allocator, file: std.fs.File, comptime alignment: u32) ![:0]align(alignment) u8 {
    const bytes_to_allocate = try file.getEndPos();
    const overaligned_size = try std.math.add(u64, bytes_to_allocate, FRONT_SENTINELS.len + BACK_SENTINELS.len + (alignment - 1));
    const buffer = try allocator.alignedAlloc(u8, alignment, std.mem.alignBackward(u64, overaligned_size, alignment));

    buffer[0..FRONT_SENTINELS.len].* = FRONT_SENTINELS.*;
    var cur: []u8 = buffer[FRONT_SENTINELS.len..][0..bytes_to_allocate];
    while (true) {
        cur = cur[try file.read(cur)..];
        if (cur.len == 0) break;
    }
    buffer[FRONT_SENTINELS.len + bytes_to_allocate ..][0..BACK_SENTINELS.len].* = BACK_SENTINELS.*;
    @memset(buffer[FRONT_SENTINELS.len + bytes_to_allocate + BACK_SENTINELS.len ..], 0);
    return buffer[0 .. FRONT_SENTINELS.len + bytes_to_allocate + std.mem.indexOfScalar(u8, BACK_SENTINELS, 0).? :0];
}

fn readFiles(gpa: Allocator) !std.ArrayListUnmanaged([:0]const u8) {
    var parent_dir2 = try std.fs.cwd().openDirZ("./src", .{}, false);
    defer parent_dir2.close();

    var parent_dir = try std.fs.cwd().openIterableDir("./src", .{});
    defer parent_dir.close();

    // var list: std.SegmentedList(Token.Tag, 0) = .{};
    // defer list.deinit(gpa);

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
                    const file = try parent_dir2.openFile(dir.path, .{});
                    defer file.close();
                    num_files += 1;
                    const source = try readFileIntoAlignedBuffer(gpa, file, VEC_SIZE);
                    // const source = try file.readToEndAllocOptions(gpa, std.math.maxInt(u32), null, 1, 0);
                    num_bytes += source.len;
                    // std.debug.print("{} {s}\n", .{ sources.items.len, dir.path });
                    (try sources.addOne(gpa)).* = source;
                },

                else => {},
            }
        }

        const t2 = std.time.nanoTimestamp();
        std.debug.print("Read {} files in {} ({})\n", .{ num_files, std.fmt.fmtDuration(@intCast(t2 - t1)), std.fmt.fmtIntSizeBin(num_bytes) });
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
            bitmask[hash / 64] |= @as(u64, 1) << @truncate(hash);
        }

        if (@popCount(bitmask[0]) + @popCount(bitmask[1]) != unpadded_ops.len)
            @compileError("Hash function failed to map operators perfectly");

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

    fn getOpWord(op: [*]const u8, len: u3) u32 {
        const shift_amt = @as(u6, len) * 8;
        const relevant_mask: u32 = @intCast((@as(u64, 1) << @intCast(shift_amt)) -% 1);
        return std.mem.readIntLittle(u32, op[0..4]) & @as(u32, @intCast(relevant_mask));
    }

    fn rawHash(op_word: u32) u7 {
        return @intCast((op_word *% 698839068) >> 25);
    }

    fn hashOp(op_word: u32) Tag {
        return @enumFromInt(rawHash(op_word));
    }

    fn mapToIndexRaw(hash_val: u8) u7 {
        const mask = (@as(u64, 1) << @truncate(hash_val)) - 1;
        return (if (hash_val >= 64) first_mask_popcount else 0) + @popCount(mask & masks[hash_val / 64]);
    }

    /// Given a hash, maps it to an index in the range [0, sorted_padded_ops.len)
    fn mapToIndex(hash: Tag) u7 {
        return mapToIndexRaw(@intFromEnum(hash));
    }

    /// Given a string starting with a Zig operator, returns the tag type
    /// Assumes there are at least 4 valid bytes at the passed in `op`
    fn lookup(op: [*]const u8, len: u3) ?Tag {
        // std.debug.print("{s}\n", .{op[0..len]});
        const op_word = getOpWord(op, len);
        const hash = rawHash(op_word);
        return if (std.mem.readIntLittle(u32, &sorted_padded_ops[mapToIndexRaw(hash)]) == op_word)
            @enumFromInt(hash)
        else
            null;
    }

    const single_char_ops = [_]u8{ '~', ':', ';', '[', ']', '?', '(', ')', '{', '}', ',' };
    const multi_char_ops = [_]u8{ '\\', '.', '!', '%', '&', '*', '+', '-', '/', '<', '=', '>', '^', '|' };

    fn isSingleCharOp(c: u8) bool {
        inline for (single_char_ops) |op| if (c == op) return true;
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
            bitmask[hash / 64] |= @as(u64, 1) << @truncate(hash);
        }

        if (@popCount(bitmask[0]) + @popCount(bitmask[1]) != unpadded_kws.len)
            @compileError("Hash function failed to map operators perfectly");

        break :blk bitmask;
    };

    const first_mask_popcount = @popCount(masks[0]);
    const PADDING_RIGHT = blk: {
        var max_len = 0;
        for (unpadded_kws) |kw| max_len = @max(kw.len, max_len);
        break :blk std.math.ceilPowerOfTwo(usize, max_len + 1) catch unreachable;
    };
    const padded_int = std.meta.Int(.unsigned, PADDING_RIGHT);

    const sorted_padded_kws = blk: {
        const max_hash_is_unused = (masks[masks.len - 1] >> 63) ^ 1;
        var buffer: [unpadded_kws.len + max_hash_is_unused][PADDING_RIGHT]u8 = undefined;

        for (unpadded_kws) |kw| {
            const padded_kw =
                (kw ++
                ("\x00" ** (PADDING_RIGHT - 1 - kw.len)) ++
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
            keyword - 2 + len)[0..2]);
        return @truncate(((a ^ (len << 14)) *% b) >> 8);
        // return @truncate(((a >> 1) *% (b >> 1) ^ (len << 14)) >> 8);
        // return @truncate((((a >> 1) *% (b >> 1)) >> 8) ^ (len << 6));
    }

    /// Given a hash, maps it to an index in the range [0, sorted_padded_kws.len)
    fn mapToIndex(hash: u7) u8 {
        const mask = (@as(u64, 1) << @truncate(hash)) - 1;
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
        const hash = mapToIndex(hashKw(kw, len));
        const val = sorted_padded_kws[hash];
        const val_len = val[PADDING_RIGHT - 1];
        if (len != val_len) return null;
        const vec1: @Vector(PADDING_RIGHT, u8) = val;
        const tag: Tag = @enumFromInt(~hash);

        comptime assert(BACK_SENTINELS.len >= PADDING_RIGHT - min_kw_len);
        const vec2: @Vector(PADDING_RIGHT, u8) = kw[0..PADDING_RIGHT].*;

        if (@ctz(@as(padded_int, @bitCast(vec1 != vec2))) >= val_len) {
            return tag;
        } else {
            return null;
        }
    }
};

const Tag = blk: {
    const BitmapKinds = std.meta.fields(Parser.BitmapKind);
    var decls = [_]std.builtin.Type.Declaration{};
    var enumFields: [Operators.unpadded_ops.len + Keywords.unpadded_kws.len + BitmapKinds.len]std.builtin.Type.EnumField = undefined;

    for (Operators.unpadded_ops, Operators.padded_ops) |op, padded_op| {
        const hash = Operators.rawHash(Operators.getOpWord(&padded_op, op.len));
        enumFields[Operators.mapToIndexRaw(hash)] = .{ .name = op, .value = hash };
    }

    var i = Operators.unpadded_ops.len;
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
const Token = struct { len: u8, kind: Tag };

const Parser = struct {
    const Bitmaps = packed struct {
        whitespace: VEC_INT,
        non_linebreaks: VEC_INT,
        identifiers_or_numbers: VEC_INT,
        non_unescaped_quotes: VEC_INT,
        prev_escaped: VEC_INT = 0,
    };

    fn nextChunk(utf8_checker: *Utf8Checker, source: [*]align(VEC_SIZE) const u8, prev_escaped: VEC_INT) Bitmaps {
        const input_vec: VEC = source[0..VEC_SIZE].*;
        // zig fmt: off
        const quotes     : VEC_INT = @bitCast(input_vec == @as(VEC, @splat(@as(u8, '"'))));
        // const apostrophes: VEC_INT = @bitCast(input_vec == @as(VEC, @splat(@as(u8, '\''))));
        const backslashes: VEC_INT = @bitCast(input_vec == @as(VEC, @splat(@as(u8, '\\'))));
        const tabs       : VEC_INT = @bitCast(input_vec == @as(VEC, @splat(@as(u8, '\t'))));
        const newlines   : VEC_INT = @bitCast(input_vec == @as(VEC, @splat(@as(u8, '\n'))));
        const carriages  : VEC_INT = @bitCast(input_vec == @as(VEC, @splat(@as(u8, '\r'))));
        const spaces     : VEC_INT = @bitCast(input_vec == @as(VEC, @splat(@as(u8, ' '))));
        const underscores: VEC_INT = @bitCast(input_vec == @as(VEC, @splat(@as(u8, '_'))));
        const upper_alpha: VEC_INT = @as(VEC_INT, @bitCast(@as(VEC, @splat(@as(u8, 'A'))) <= input_vec)) & @as(VEC_INT, @bitCast(input_vec <= @as(VEC, @splat(@as(u8, 'Z')))));
        const lower_alpha: VEC_INT = @as(VEC_INT, @bitCast(@as(VEC, @splat(@as(u8, 'a'))) <= input_vec)) & @as(VEC_INT, @bitCast(input_vec <= @as(VEC, @splat(@as(u8, 'z')))));
        const digits     : VEC_INT = @as(VEC_INT, @bitCast(@as(VEC, @splat(@as(u8, '0'))) <= input_vec)) & @as(VEC_INT, @bitCast(input_vec <= @as(VEC, @splat(@as(u8, '9')))));
        // zig fmt: on

        // ----------------------------------------------------------------------------
        // This code is brought to you courtesy of simdjson and simdjzon, both licensed
        // under the Apache 2.0 license which is included at the bottom of this file

        // If there was overflow, pretend the first character isn't a backslash
        const backslash: VEC_INT = backslashes & ~prev_escaped;
        const follows_escape = (backslash << 1) | prev_escaped;

        // Get sequences starting on even bits by clearing out the odd series using +
        const even_bits: VEC_INT = @bitCast(@as(@Vector(@divExact(VEC_SIZE, 8), u8), @splat(@as(u8, 0x55))));
        const odd_sequence_starts = backslash & ~even_bits & ~follows_escape;
        const x = @addWithOverflow(odd_sequence_starts, backslash);
        const invert_mask: VEC_INT = x[0] << 1; // The mask we want to return is the *escaped* bits, not escapes.;

        // Mask every other backslashed character as an escaped character
        // Flip the mask for sequences that start on even bits, to correct them
        const escaped = (even_bits ^ invert_mask) & follows_escape;

        utf8_checker.check_next_input(input_vec);
        // ----------------------------------------------------------------------------

        return .{
            .whitespace = tabs | newlines | carriages | spaces,
            .non_linebreaks = ~(newlines | carriages),
            .identifiers_or_numbers = underscores | upper_alpha | lower_alpha | digits,
            .non_unescaped_quotes = ~(quotes & ~escaped),
            // .empty = 0,
            .prev_escaped = x[1],
            // .non_unescaped_apostrophes = ~(apostrophes & ~escaped),
            // .operators = operators,
        };
    }

    const BitmapKind = enum(u8) {
        // zig fmt: off
        start_of_file         = 123, // TODO: probe for something less than 128, doesn't matter what
        whitespace            = 128 | @as(u8, 0),

        unknown               = 128 | @as(u8,  1), // is_quoted

        identifier            = 128 | @as(u8,  2),
        builtin               = 128 | @as(u8,  6),
        number                = 128 | @as(u8,  10),

        string                = 128 | @as(u8,  3), // is_quoted
        string_identifier     = 128 | @as(u8,  7), // is_quoted

        char_literal          = 128 | @as(u8,  5), // is_quoted

        eof                   = 128 | @as(u8,  4),

        // zig fmt: on
    };

    // TODO: recover from parse_errors by switching to BitmapKind.unknown? Report errors?
    // TODO: audit usages of u32's to make sure it's impossible to ever overflow.
    // TODO: make it so quotes and character literals cannot have newlines in them.
    // TODO: audit the utf8 validator to make sure we clear the state properly when not using it
    pub fn tokenize(gpa: Allocator, source: [:0]const u8) ![]Token {
        const end_ptr = &source.ptr[source.len];

        // const BACK_SENTINELS = "\n" ++ "\x00'\"" ++ "\x00" ** 10;
        // TODO: make 13 based on comptime derivation
        const extended_source_len = std.mem.alignForward(u64, source.len + 13, VEC_SIZE);
        const extended_source = source.ptr[0..extended_source_len];

        var tokens = try gpa.alloc(Token, extended_source_len);
        errdefer gpa.free(tokens);

        var cur_token = tokens;
        cur_token[0] = .{ .len = 0, .kind = .start_of_file };
        cur_token[1..][0..2].* = @bitCast(@as(u32, 0));

        comptime assert(FRONT_SENTINELS.len == 1 and FRONT_SENTINELS[0] == '\n');
        comptime assert(BACK_SENTINELS.len >= 3);

        var cur = extended_source[@as(u3, @intFromBool(std.mem.readIntSliceNative(u32, extended_source) == std.mem.readIntSliceNative(u32, "\n\xEF\xBB\xBF"))) << 2 ..];
        var prev = cur;

        var bitmaps = std.mem.zeroes(Bitmaps);
        const bitmaps_len = @divExact(@bitSizeOf(Bitmaps), @bitSizeOf(VEC_INT));
        const bitmap_ptr: *[bitmaps_len]VEC_INT = @ptrCast(&bitmaps);
        var selected_bitmap_kind: BitmapKind = .whitespace;
        var op_type: Tag = @enumFromInt(@intFromEnum(selected_bitmap_kind));
        var bitmap_index: usize = std.math.maxInt(usize);
        var utf8_checker: Utf8Checker = .{};

        outer: while (true) {
            while (true) {
                const bitmask_i: u2 = @truncate(@intFromEnum(selected_bitmap_kind));
                const cur_misalignment: LOG_VEC_INT = @truncate(@intFromPtr(cur.ptr));
                const cur_bitmap_index = @intFromPtr(cur.ptr) / VEC_SIZE;
                if (bitmap_index != cur_bitmap_index) {
                    bitmap_index = cur_bitmap_index;
                    bitmaps = nextChunk(&utf8_checker, @alignCast(cur.ptr - cur_misalignment), bitmaps.prev_escaped);
                    utf8_checker.errors() catch return error.ParseError;
                }
                const bitmask = bitmap_ptr.*[bitmask_i] >> cur_misalignment;
                const str_len = @ctz(~bitmask);
                cur = cur[str_len..];
                if (VEC_SIZE - str_len != @as(u8, cur_misalignment)) break;
            }

            // Catch if we had an unpaired `"`
            if (@intFromPtr(cur.ptr) > @intFromPtr(end_ptr)) return error.ParseError;
            const is_quoted: u1 = @truncate(@intFromEnum(selected_bitmap_kind));
            cur = cur[is_quoted..];

            while (true) {
                var len: u32 = @intCast(@intFromPtr(cur.ptr) - @intFromPtr(prev.ptr));

                switch (prev[0]) {
                    'a'...'z' => if (Keywords.lookup(prev.ptr, len)) |op_kind| {
                        op_type = op_kind;
                    },
                    else => {},
                }

                {
                    const is_collapsible = switch (op_type) {
                        .@"//", .@"//!", .@"///", .whitespace => true,
                        else => false,
                    };

                    comptime var min_bitmap_value = 128;
                    comptime var max_bitmap_value = std.math.minInt(@typeInfo(BitmapKind).Enum.tag_type);

                    comptime for (std.meta.fields(BitmapKind)) |field| {
                        max_bitmap_value = @max(field.value, max_bitmap_value);
                    };

                    // We are fine with arbitrary lengths attached to operators and keywords, because
                    // once we know which one it is there is no loss of information in adding comments and
                    // whitespace to the length. E.g. `const` is always `const` and `+=` is `+=`, whether or
                    // not we add whitespace to either side.
                    //
                    // On the other hand, identifiers, numbers, strings, etc might become harder to deal
                    // with later if we don't know the exact length. If it turns out that we don't really
                    // need this information anyway, we can simplify this codepath and collapse comments
                    // and whitespace into those tokens too.
                    const prev_is_collapse_amenable = switch (@intFromEnum(cur_token[0].kind)) {
                        min_bitmap_value...max_bitmap_value => false,
                        else => true,
                    };

                    var advance_amt: u2 = switch (cur_token[0].len) {
                        0 => 3,
                        else => 1,
                    };

                    if (is_collapsible and prev_is_collapse_amenable) {
                        advance_amt = 0;
                        op_type = cur_token[0].kind;
                        len += switch (cur_token[0].len) {
                            0 => @bitCast(cur_token[1..][0..2].*),
                            else => |l| l,
                        };
                    }

                    cur_token = cur_token[advance_amt..];
                    cur_token[0] = .{ .len = if (len >= 256) 0 else @intCast(len), .kind = op_type };
                    cur_token[1..][0..2].* = @bitCast(len);
                }

                prev = cur;

                if (cur[0] == 0) {
                    break :outer;
                } else if (cur[0] == '@') {
                    cur = cur[1..];
                    selected_bitmap_kind = switch (cur[0]) {
                        'a'...'z', 'A'...'Z' => .builtin,
                        '"' => .string_identifier,
                        else => return error.ParseError,
                    };
                    op_type = @enumFromInt(@intFromEnum(selected_bitmap_kind));
                } else if (cur[0] == '\'') {
                    while (true) {
                        cur = cur[1..];
                        if (cur[0] == '\'') break;
                        cur = cur[@intFromBool(cur[0] == '\\')..];
                    }

                    // Catch if we had an unpaired `'`
                    if (@intFromPtr(cur.ptr) > @intFromPtr(end_ptr)) return error.ParseError;
                    cur = cur[1..];
                    op_type = .char_literal;
                    continue;
                } else if (Operators.isSingleCharOp(cur[0])) {
                    selected_bitmap_kind = .whitespace;
                    op_type = Operators.hashOp(Operators.getOpWord(prev.ptr, 1));
                } else if (Operators.isMultiCharBeginning(cur[0])) {
                    var op_len: u3 = 1;

                    comptime var op_continuation_chars = std.mem.zeroes([@divExact(256, @bitSizeOf(usize))]usize);
                    comptime for (Operators.unpadded_ops) |op| {
                        var i = 1;

                        while (i < op.len) : (i += 1) {
                            const c = op[i];
                            op_continuation_chars[c / @bitSizeOf(usize)] |= @as(usize, 1) << @truncate(c);
                        }
                    };

                    inline for (0..3) |_| {
                        const c = cur[op_len];
                        const is_op_char: u1 = @truncate(op_continuation_chars[c / @bitSizeOf(usize)] >> @truncate(c));
                        if (is_op_char == 0) break;
                        op_len += 1;
                    }

                    while (true) : (op_len -= 1) {
                        if (Operators.lookup(prev.ptr, op_len)) |op_kind| {
                            op_type = op_kind;
                            cur = cur[op_len..];
                            break;
                        }

                        // TODO: insert comptime assertions that this is the only case we have to handle
                        if (op_len == 1 and prev[0] == '\\') return error.ParseError;
                    }

                    switch (op_type) {
                        .@"///", .@"//!", .@"//", .@"\\\\" => {
                            selected_bitmap_kind = .unknown;
                            break;
                        },

                        else => {},
                    }

                    cur = cur[@intFromBool(cur[0] == ' ')..];
                    continue;
                } else {
                    selected_bitmap_kind = switch (cur[0]) {
                        'a'...'z', 'A'...'Z', '_' => .identifier,
                        '0'...'9' => .number,
                        '"' => .string,
                        ' ', '\t', '\r', '\n' => .whitespace,
                        else => return error.ParseError,
                    };
                    op_type = @enumFromInt(@intFromEnum(selected_bitmap_kind));
                }

                cur = cur[1..];
                break;
            }
        }

        cur_token = cur_token[switch (cur_token[0].len) {
            0 => 3,
            else => 1,
        }..];
        cur_token[0] = .{ .len = 0, .kind = .eof };
        cur_token = cur_token[1..];
        const new_chunks_data_len = (@intFromPtr(cur_token.ptr) - @intFromPtr(tokens.ptr)) / @sizeOf(Token);
        if (gpa.resize(tokens, new_chunks_data_len)) {
            tokens.len = new_chunks_data_len;
        }
        return tokens;
    }
};

pub fn main() !void {
    const gpa = std.heap.c_allocator;
    const sources = try readFiles(gpa);
    var t1 = std.time.nanoTimestamp();

    // mine:                85ms
    // original tokenizer: 192ms
    // mine:               18.36490249633789MiB
    // original tokenizer: 43.44332695007324MiB

    // mine:
    // original ast: 332ms

    var num_tokens: usize = 0;
    var source_tokens = try std.ArrayListUnmanaged([]Token).initCapacity(gpa, sources.items.len);

    for (sources.items) |source| {
        const tokens = try Parser.tokenize(gpa, source);
        num_tokens += tokens.len;
        source_tokens.addOneAssumeCapacity().* = tokens;
    }

    const t2 = std.time.nanoTimestamp();

    const duration: u64 = @intCast(t2 - t1);

    var total_len: u64 = 0;
    var lines: u64 = 0;
    for (sources.items) |source| {
        total_len += source.len;
        for (source) |c| {
            lines += @intFromBool(c == '\n');
        }
        lines -= 1;
    }

    std.debug.print("Tokenized in {} ({})\n", .{ std.fmt.fmtDuration(duration), std.fmt.fmtIntSizeBin(num_tokens * 2) });
    std.debug.print("Totals:\n", .{});
    std.debug.print("{: >20} bytes\n", .{total_len});
    std.debug.print("{: >20} lines\n", .{lines});
    std.debug.print("{: >20} num_tokens\n", .{num_tokens});

    // for (source_tokens.items) |tokens| {
    // for (tokens, 0..) |token, i| {
    // if (1870 <= i or i < 5) std.debug.print("{} {}\n", .{ i, token });
    // }
    // break;
    // }
}

// switch (op_type) {
//     inline else => |t| {
//         @setEvalBranchQuota(1000000);
//         const fields = @typeInfo(@TypeOf(op_type)).Enum.fields;
//         inline for (fields) |field| {
//             if (field.value == @intFromEnum(t)) {
//                 std.debug.print("{s}\n", .{field.name});
//             }
//         }
//     },
// }

// ---------------------------------------------------------------
//
// The code below this point is licensed under the Apache License.
// Please see the License at the bottom of this file.
//
// ---------------------------------------------------------------

pub const Chunk = @Vector(32, u8);
pub const IChunk = @Vector(32, i8);
pub const chunk_len = @sizeOf(Chunk);
const half_chunk_len = chunk_len / 2;
pub const ChunkArr = [chunk_len]u8;

const u3x32 = @Vector(32, u3);
const u8x32 = @Vector(32, u8);
const u8x64 = @Vector(64, u8);
const u64x4 = @Vector(4, u64);
const u32x4 = @Vector(4, u32);
const u8x16 = @Vector(16, u8);

// ---
// from https://gist.github.com/sharpobject/80dc1b6f3aaeeada8c0e3a04ebc4b60a
// ---
// thanks to sharpobject for these implementations which make it possible to get
// rid of old utils.c and stop linking libc.
// ---
fn __mm256_permute2x128_si256_0x21(comptime V: type, a: V, b: V) V {
    var ret: V = undefined;
    ret[0] = a[2];
    ret[1] = a[3];
    ret[2] = b[0];
    ret[3] = b[1];
    return ret;
}

fn _mm256_permute2x128_si256_0x21(a: Chunk, b: Chunk) Chunk {
    const V = if (chunk_len == 32) u64x4 else u32x4;
    return @bitCast(__mm256_permute2x128_si256_0x21(V, @as(V, @bitCast(a)), @as(V, @bitCast(b))));
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

// ---
// --- end from https://gist.github.com/sharpobject/80dc1b6f3aaeeada8c0e3a04ebc4b60a
// ---

pub fn mm256_shuffle_epi8(x: u8x32, mask: u8x32) u8x32 {
    return asm (
        \\ vpshufb %[mask], %[x], %[out]
        : [out] "=x" (-> u8x32),
        : [x] "+x" (x),
          [mask] "x" (mask),
    );
}

// https://developer.arm.com/architectures/instruction-sets/intrinsics/vqtbl1q_s8
pub fn lookup_16_aarch64(x: u8x16, mask: u8x16) u8x16 {
    // tbl     v0.16b, { v0.16b }, v1.16b
    return asm (
        \\tbl  %[out].16b, {%[mask].16b}, %[x].16b
        : [out] "=&x" (-> u8x16),
        : [x] "x" (x),
          [mask] "x" (mask),
    );
}

const Utf8Checker = struct {
    err: Chunk = zeros,
    prev_input_block: Chunk = zeros,
    prev_incomplete: Chunk = zeros,

    const zeros: ChunkArr = [1]u8{0} ** chunk_len;

    fn prev(comptime N: u8, a: Chunk, b: Chunk) Chunk {
        assert(0 < N and N <= 3);
        return _mm256_alignr_epi8(a, _mm256_permute2x128_si256_0x21(b, a), half_chunk_len - N);
    }
    const check_special_cases = switch (builtin.cpu.arch) {
        .aarch64 => check_special_cases_arm64,
        .x86_64 => check_special_cases_x86,
        else => unreachable,
    };
    // zig fmt: off
    fn check_special_cases_x86(input: Chunk, prev1: Chunk) Chunk {
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

        const byte_1_high_0 = prev1 >> @as(u3x32, @splat(4));
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
        } ** 2;
        const byte_1_high = mm256_shuffle_epi8(tbl1, byte_1_high_0);
        const CARRY: u8 = TOO_SHORT | TOO_LONG | TWO_CONTS; // These all have ____ in byte 1 .
        const byte_1_low0 = prev1 & @as(u8x32, @splat(0x0F));

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
        } ** 2;
        const byte_1_low = mm256_shuffle_epi8(tbl2, byte_1_low0);

        const byte_2_high_0 = input >> @as(u3x32, @splat(4));
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
        } ** 2;
        const byte_2_high = mm256_shuffle_epi8(tbl3, byte_2_high_0);
        return (byte_1_high & byte_1_low & byte_2_high);
    }

    fn check_special_cases_arm64(input: Chunk, prev1: Chunk) Chunk {
        // Bit 0 = Too Short (lead byte/ASCII followed by lead byte/ASCII)
        // Bit 1 = Too Long (ASCII followed by continuation)
        // Bit 2 = Overlong 3-byte
        // Bit 4 = Surrogate
        // Bit 5 = Overlong 2-byte
        // Bit 7 = Two Continuations

        const TOO_SHORT: u8 = 1 << 0;   // 11______ 0_______
                                        // 11______ 11______
        const TOO_LONG: u8 = 1 << 1;    // 0_______ 10______
        const OVERLONG_3: u8 = 1 << 2;  // 11100000 100_____
        const SURROGATE: u8 = 1 << 4;   // 11101101 101_____
        const OVERLONG_2: u8 = 1 << 5;  // 1100000_ 10______
        const TWO_CONTS: u8 = 1 << 7;   // 10______ 10______
        const TOO_LARGE: u8 = 1 << 3;   // 11110100 1001____
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

        const u3xchunk_len = @Vector(u3, chunk_len);
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

        const byte_1_high = lookup_16_aarch64(byte_1_high_0, tbl1);
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
        const byte_1_low = lookup_16_aarch64(byte_1_low0, tbl2);

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
        const byte_2_high = lookup_16_aarch64(byte_2_high_0, tbl3);
        return (byte_1_high & byte_1_low & byte_2_high);
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
        const bytes: [64]u8 = input;
        const a: u8x32 = bytes[0..32].*;
        const b: u8x32 = bytes[32..64].*;
        const non_ascii_mask: u8x32 = @splat(0x80);
        const mask: u32 = @bitCast((a | b) >= non_ascii_mask);
        return mask == 0;
    }

    fn check_next_input(checker: *Utf8Checker, input: u8x64) void {
        // const NUM_CHUNKS = cmn.STEP_SIZE / 32;

        if (is_ascii(input)) {
            checker.err |= checker.prev_incomplete;
        } else {
            // you might think that a for-loop would work, but under Visual Studio, it is not good enough.
            // static_assert((simd8x64<uint8_t>::NUM_CHUNKS == 2) || (simd8x64<uint8_t>::NUM_CHUNKS == 4),
            // "We support either two or four chunks per 64-byte block.");
            switch (builtin.cpu.arch) {
                .x86_64 => {
                    const NUM_CHUNKS = 2;
                    const chunks = @as([NUM_CHUNKS][32]u8, @bitCast(input));
                    checker.check_utf8_bytes(chunks[0], checker.prev_input_block);
                    checker.check_utf8_bytes(chunks[1], chunks[0]);
                    checker.prev_incomplete = is_incomplete(chunks[NUM_CHUNKS - 1]);
                    checker.prev_input_block = chunks[NUM_CHUNKS - 1];
                },
                .aarch64 => {
                    const NUM_CHUNKS = 4;
                    const chunks = @as([NUM_CHUNKS][16]u8, @bitCast(input));
                    checker.check_utf8_bytes(chunks[0], checker.prev_input_block);
                    checker.check_utf8_bytes(chunks[1], chunks[0]);
                    checker.check_utf8_bytes(chunks[2], chunks[1]);
                    checker.check_utf8_bytes(chunks[3], chunks[2]);
                    checker.prev_incomplete = is_incomplete(chunks[NUM_CHUNKS - 1]);
                    checker.prev_input_block = chunks[NUM_CHUNKS - 1];
                },
                else => unreachable,
            }
        }
    }
    // do not forget to call check_eof!
    fn errors(checker: Utf8Checker) !void {
        const err = @reduce(.Or, checker.err);
        if (err != 0) return error.UTF8_ERROR;
    }

    //
    // Return nonzero if there are incomplete multibyte characters at the end of the block:
    // e.g. if there is a 4-byte character, but it's 3 bytes from the end.
    //
    fn is_incomplete(input: Chunk) Chunk {
        // If the previous input's last 3 bytes match this, they're too short (they ended at EOF):
        // ... 1111____ 111_____ 11______
        const max_array: [32]u8 = .{
            255, 255, 255, 255, 255, 255,            255,            255,
            255, 255, 255, 255, 255, 255,            255,            255,
            255, 255, 255, 255, 255, 255,            255,            255,
            255, 255, 255, 255, 255, 0b11110000 - 1, 0b11100000 - 1, 0b11000000 - 1,
        };
        const max_value = @as(Chunk, @splat(max_array[@sizeOf(@TypeOf(max_array)) - @sizeOf(u8x32)]));
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
