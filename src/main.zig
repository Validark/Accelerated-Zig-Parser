// zig fmt: off
const WRITE_OUT_DATA       = false;
const SKIP_OUTLIERS        = true;
const RUN_LEGACY_TOKENIZER = true;
const RUN_NEW_TOKENIZER    = true;
const RUN_COMPRESS_TOKENIZER = true;
const RUN_LEGACY_AST       = false;
const RUN_NEW_AST          = false;
const REPORT_SPEED         = true;
const INFIX_TEST           = false;
// zig fmt: on

// const rpmalloc = @import("rpmalloc");
// const zimalloc = @import("zimalloc");
const jdz_allocator = @import("jdz_allocator");

// const jemalloc = @import("jemalloc");
const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const testing = std.testing;
const mem = std.mem;
const Ast = std.zig.Ast;
const Allocator = std.mem.Allocator;

const HAS_ARM_NEON = switch (builtin.cpu.arch) {
    .aarch64, .aarch64_be => std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon),
    .arm, .armeb => std.Target.arm.featureSetHas(builtin.cpu.features, .neon),
    else => false,
};

const HAS_ARM32_FAST_16_BYTE_VECTORS = switch (builtin.cpu.arch) {
    .arm, .armeb => HAS_ARM_NEON and
        (!std.mem.startsWith(u8, builtin.cpu.model.name, "cortex_a") or
        builtin.cpu.model.name.len != 9 or
        std.mem.indexOfAnyPos(u8, builtin.cpu.model.name, 8, "5789") == null),
    else => false,
};

const HAS_FAST_PDEP_AND_PEXT = blk: {
    const cpu_name = builtin.cpu.model.llvm_name orelse builtin.cpu.model.name;
    break :blk (builtin.cpu.arch.isPowerPC() and std.Target.powerpc.featureSetHas(builtin.cpu.features, .isa_v31_instructions)) or
        builtin.cpu.arch.isX86() and
        std.Target.x86.featureSetHas(builtin.cpu.features, .bmi2) and
        // pdep is microcoded (slow) on AMD architectures before Zen 3.
        !std.mem.startsWith(u8, cpu_name, "bdver") and
        (!std.mem.startsWith(u8, cpu_name, "znver") or !(cpu_name["znver".len] < '3' and cpu_name["znver".len..].len == 1));
};

/// This is the "efficient builtin operand" size.
/// E.g. if we support 64-bit operations, we want to be doing 64-bit count-trailing-zeros.
/// For now, we use `usize` as our reasonable guess for what size of bitstring we can operate on efficiently.
const uword = std.meta.Int(.unsigned, if (std.mem.endsWith(u8, @tagName(builtin.cpu.arch), "64_32")) 64 else @bitSizeOf(usize));

const IS_VECTORIZER_BROKEN = builtin.cpu.arch.isSPARC() or builtin.cpu.arch.isPowerPC();
const SUGGESTED_VEC_SIZE: ?comptime_int = if (IS_VECTORIZER_BROKEN)
    null
else if (HAS_ARM_NEON)
    if (HAS_ARM32_FAST_16_BYTE_VECTORS) 16 else @sizeOf(uword) * 2
else
    std.simd.suggestVectorLengthForCpu(u8, builtin.cpu);

// https://github.com/llvm/llvm-project/issues/76812
const USE_SWAR = SUGGESTED_VEC_SIZE == null;

// This is the native vector size.
const NATIVE_VEC_INT = std.meta.Int(.unsigned, 8 * (SUGGESTED_VEC_SIZE orelse @sizeOf(uword)));
const NATIVE_VEC_SIZE = @sizeOf(NATIVE_VEC_INT);
const NATIVE_CHAR_VEC = @Vector(NATIVE_VEC_SIZE, u8);

/// The number of chunks to process at once
/// We can experiment on different machines to see what is most beneficial
const BATCH_SIZE = if (HAS_ARM32_FAST_16_BYTE_VECTORS) 2 else if (HAS_FAST_PDEP_AND_PEXT) 1 else 1;
const CHUNK_ALIGNMENT = BATCH_SIZE * @bitSizeOf(uword);

const NUM_CHUNKS = if (HAS_ARM32_FAST_16_BYTE_VECTORS) 4 else @bitSizeOf(uword) / NATIVE_VEC_SIZE;
const Chunk = if (USE_SWAR) NATIVE_VEC_INT else @Vector(NATIVE_VEC_SIZE, u8);

/// A buffer with one or more zeroed values appended to align to an aligned length.
/// The pointer is aligned too.
fn LenAlignedBuffer(comptime T: type, comptime options: struct {
    alignment: u29 = 64,
    front_sentinels: []const T = &.{},
    back_sentinels: []const T = &.{},
    constant: bool = false,
}) type {
    return packed struct {
        const alignment = options.alignment;
        const front_sentinels = options.front_sentinels;
        const back_sentinels = options.back_sentinels;
        const constant = options.constant;

        ptr: if (constant) [*]align(alignment) const T else [*]align(alignment) T,
        len: usize,

        fn slice(self: @This()) if (constant) []align(alignment) const T else []align(alignment) T {
            return self.ptr[0..self.len];
        }

        fn allocatedSlice(self: @This()) if (constant) []align(alignment) const T else []align(alignment) T {
            return (self.ptr - front_sentinels.len)[0..std.mem.alignForward(usize, self.len + front_sentinels.len + back_sentinels.len, @divExact(alignment, @sizeOf(T)))];
        }

        fn allocatedSliceAsVectors(self: @This(), V: type) if (constant) []align(alignment) const V else []align(alignment) V {
            return @ptrCast(self.allocatedSlice());
        }

        fn deinit(self: @This(), gpa: Allocator) void {
            gpa.free(self.allocatedSlice());
        }

        fn initReadFile(allocator: Allocator, file: std.fs.File) !@This() {
            const things_to_allocate: u64 = @divExact(try file.getEndPos(), @sizeOf(T));
            // Overflow is undefined behavior because I assume we can trust `getEndPos()` to not report a file size of ~18 exabytes (quintillion bytes)

            const overaligned_size = std.mem.alignForward(
                u64,
                things_to_allocate + front_sentinels.len + back_sentinels.len,
                @divExact(alignment, @sizeOf(T)),
            );
            const buffer = try allocator.alignedAlloc(T, alignment, overaligned_size);
            errdefer allocator.free(buffer);

            buffer[0..front_sentinels.len].* = front_sentinels[0..front_sentinels.len].*;
            var cur: []u8 = @ptrCast(buffer[front_sentinels.len..][0..things_to_allocate]);
            while (true) {
                cur = cur[try file.read(cur)..];
                if (cur.len == 0) break;
            }

            const cur2 = buffer[front_sentinels.len + things_to_allocate ..];

            @memset(cur2, std.mem.zeroes(T));

            inline for (back_sentinels, 0..) |c, i| {
                if (c != 0) cur2[i] = c;
            }

            return .{
                .ptr = buffer.ptr + front_sentinels.len,
                .len = things_to_allocate,
            };
        }

        // Aligns `n` forward and memsets the overallocated memory to zeroes.
        // All other memory is still undefined.
        fn alloc(allocator: Allocator, n: usize) !@This() {
            const buf = try allocator.alignedAlloc(
                T,
                alignment,
                std.mem.alignForward(usize, n, alignment),
            );
            @memset(buf[n..], std.mem.zeroes(T));
            return .{
                .ptr = buf.ptr,
                .len = n,
            };
        }
    };
}

/// Reads in the files specifically for this test-bench.
const SourceData = struct {
    const LenAlignedBuf = LenAlignedBuffer(u8, .{
        .alignment = CHUNK_ALIGNMENT,
        .back_sentinels = "\x00",
        .constant = true,
    });

    const SourceList = std.MultiArrayList(struct {
        file_contents: LenAlignedBuf,
        path: [:0]const u8,
    });

    source_list: SourceList.Slice,
    path_buffer: []const u8,
    num_bytes: u64,
    num_lines: u64,

    const empty: @This() = .{
        .source_list = undefined,
        .path_buffer = undefined,
        .num_bytes = 0,
        .num_lines = 0,
    };

    fn deinit(self: *const @This(), gpa: Allocator) void {
        for (self.source_list.items(.file_contents)) |source| source.deinit(gpa);
        var other = self.source_list.toMultiArrayList();
        other.deinit(gpa);
        gpa.free(self.path_buffer);
    }

    fn readFiles(gpa: Allocator) !@This() {
        if (SKIP_OUTLIERS)
            std.debug.print("Skipping outliers!\n", .{});
        std.debug.print("v0.8\n", .{});
        const directory = switch (INFIX_TEST) {
            true => "./src/bee",
            false => "./src/files_to_parse/",
        };

        var parent_dir2 = try std.fs.cwd().openDirZ(directory, .{ .iterate = false });
        defer parent_dir2.close();

        var parent_dir = try std.fs.cwd().openDirZ(directory, .{ .iterate = true });
        defer parent_dir.close();

        const t1 = std.time.nanoTimestamp();

        const STRIDE = 8 * 2;
        var source_list: SourceList = .empty;
        try source_list.setCapacity(gpa, STRIDE);
        var path_buffer: std.ArrayListUnmanaged(u8) = .empty;
        var sources: @This() = .empty;
        errdefer sources.deinit(gpa); // Will free both `source_list` and `path_buffer`.

        {
            defer {
                sources.source_list = source_list.toOwnedSlice();
                sources.path_buffer = path_buffer.allocatedSlice();
            }

            var walker = try parent_dir.walk(gpa); // 12-14 ms just walking the tree
            defer walker.deinit();

            while (try walker.next()) |dir| {
                switch (dir.kind) {
                    .file => if (std.mem.endsWith(u8, dir.basename, ".zig") and dir.path.len - dir.basename.len > 0) {

                        // These two are extreme outliers, omit them from our test bench
                        // if (!std.mem.endsWith(u8, dir.path, "zig/test/behavior/bugs/11162.zig")) continue;

                        if (SKIP_OUTLIERS and (std.mem.eql(u8, dir.basename, "udivmodti4_test.zig") or std.mem.eql(u8, dir.basename, "udivmoddi4_test.zig")))
                            continue;

                        const needed_source_list_capacity = source_list.len + 1;
                        if (source_list.capacity < needed_source_list_capacity) {
                            try source_list.setCapacity(gpa, 2 * source_list.capacity);
                        }

                        var path_ptr: [:0]const u8 = undefined;
                        // Initially, we store just the index as the pointer, and later, we add the base pointer to all indices
                        @as(*usize, @ptrCast(&path_ptr.ptr)).* = path_buffer.items.len;
                        path_ptr.len = dir.path.len;

                        try path_buffer.appendSlice(gpa, dir.path[0 .. dir.path.len + 1]);
                        const file = try parent_dir2.openFile(dir.path, .{});
                        defer file.close();

                        const file_contents: LenAlignedBuf = try .initReadFile(gpa, file);
                        errdefer file_contents.deinit(gpa); // The compiler should be able to deduce that no errors can occur to trigger this logic?

                        source_list.appendAssumeCapacity(.{ .file_contents = file_contents, .path = path_ptr });

                        sources.num_bytes += file_contents.len;

                        var line_vecs: @Vector(16, u64) = @splat(0);

                        for (file_contents.allocatedSliceAsVectors(@Vector(@typeInfo(@TypeOf(line_vecs)).vector.len, u8))) |v| {
                            line_vecs += v;
                        }

                        const extra_lines = comptime std.mem.count(u8, SourceData.LenAlignedBuf.front_sentinels, "\n") +
                            std.mem.count(u8, SourceData.LenAlignedBuf.back_sentinels, "\n");

                        sources.num_lines += @reduce(.Add, line_vecs) - extra_lines * sources.source_list.len;
                    },

                    else => {},
                }
            }
        }

        const paths_ptr_update_impl: enum { llvm, custom } = .custom;

        // Update indices in `paths` to be actual pointers!
        var paths = sources.source_list.items(.path);

        switch (paths_ptr_update_impl) {
            .llvm => for (paths) |*path| {
                path.ptr = @ptrCast(sources.path_buffer.ptr[@as(*usize, @ptrCast(&path.ptr)).*..]);
            },
            .custom => {
                const add_vec = std.simd.repeat(STRIDE * 2, @shuffle(
                    u64,
                    @as(@Vector(1, u64), .{@intFromPtr(sources.path_buffer.ptr)}),
                    @as(@Vector(1, u64), .{0}),
                    [_]i32{ 0, -1 },
                ));

                var cur = paths.ptr[0..std.mem.alignForward(u64, paths.len, STRIDE)];

                while (cur.len != 0) {
                    const cur_vec: @Vector(STRIDE * 2, u64) = @bitCast(cur[0..STRIDE].*);
                    cur[0..STRIDE].* = @bitCast(add_vec + cur_vec);
                    cur = cur[STRIDE..];
                }
            },
        }

        const t2 = std.time.nanoTimestamp();

        const elapsedNanos: u64 = @intCast(t2 - t1);
        const @"MB/s" = @as(f64, @floatFromInt(sources.num_bytes * 1000)) / @as(f64, @floatFromInt(elapsedNanos));

        const stdout = std.io.getStdOut().writer();

        if (REPORT_SPEED)
            try stdout.print("       Read in files in {} ({d:.2} MB/s) and used {} memory with {} lines across {} files\n", .{ std.fmt.fmtDuration(elapsedNanos), @"MB/s", std.fmt.fmtIntSizeDec(sources.num_bytes), sources.num_lines, sources.source_list.len });

        return sources;
    }
};

fn Tokenizer(
    /// A set of compile-time switches that allow for modifications of behavior or optimization strategies.
    comptime options: struct {
        /// This allows us to validate the correctness of various optimization strategies that we normally wouldn't use on the target hardware.
        optimization_strategies: struct {
            classifier_strategy: enum { global, on_demand, none } = if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi))
                .global
            else
                .on_demand,
        } = .{},

        emitted_token_data: packed struct {
            tag: bool = true,

            /// This will be emitted separately from the `tag` and `len`, if present. Will share a buffer with `start` fields if present.
            start: bool = false,

            /// This will be emitted separately from the `tag` and `len`, if present. Will share a buffer with `start` fields if present.
            end: bool = false,

            /// Len's are emitted as bytes. If we need more than 1 byte, we store a 0 and the next 4 bytes are the true length.
            len: bool = true,
        } = .{},

        exclude_newlines_from_comments_and_string_literals_and_invalids: bool = false,

        emit_whitespace_and_comments: enum {
            yes,
            no,
            fold_whitespace_and_comments_into_adjacent,
        } = .fold_whitespace_and_comments_into_adjacent,

        const LegacyTokenizerReplacement = @This(){
            .emitted_token_data = .{
                .tag = true,
                .start = true,
            },
            .exclude_newlines_from_comments_and_string_literals_and_invalids = true,
            .emit_whitespace_and_comments = .no,
        };
    },
) type {
    if (options.emitted_token_data != @TypeOf(options.emitted_token_data){ .tag = true, .len = true, .start = false, .end = false })
        @compileError("Needs to be implemented");

    return struct {
        const EXCLUDE_NEWLINES_FROM_COMMENTS_AND_STRING_LITERALS_AND_INVALIDS = options.exclude_newlines_from_comments_and_string_literals_and_invalids;
        const EMIT_WHITESPACE_AND_COMMENTS = options.emit_whitespace_and_comments;
        const CLASSIFIER_STRATEGY = options.optimization_strategies.classifier_strategy;
        const EMITTED_TOKEN_DATA = options.emitted_token_data;

        const Operators = struct {
            // zig fmt: off
            const unpadded_ops = [_][:0]const u8{
                ".**", "!", "|", "||", "|=", "=", "==", "=>", "!=", "(", ")", ";", "%", "%=", "{", "}", "[", "]", ".", ".*", "..", "...", "^",
                "^=", "+", "++", "+=", "+%", "+%=", "+|", "+|=", "-", "-=", "-%", "-%=", "-|", "-|=", "*", "*=", "**", "*%", "*%=", "*|", "*|=",
                "->", ":", "/", "/=", "&", "&=", "?", "<", "<=", "<<", "<<=", "<<|", "<<|=", ">", ">=", ">>", ">>=", "~", "//", "///", "//!",
                ".?", "\\\\", ",", "*^"
            };
            // zig fmt: on

            const potentially_unary = [_][]const u8{ "&", "-", "-%", "*", "**", ".", ".." };

            fn unarifyBinaryOperatorRaw(hash: u8) u8 {
                return pextComptime(hash *% 29, 0b11000100) +% 150;
            }

            fn unarifyBinaryOperator(tag: Tag) Tag {
                return @enumFromInt(unarifyBinaryOperatorRaw(@intFromEnum(tag)));
            }

            const padded_ops: [unpadded_ops.len][4]u8 = blk: {
                var padded_ops_table: [unpadded_ops.len][4]u8 = undefined;
                for (unpadded_ops, 0..) |op, i| {
                    padded_ops_table[i] = (op ++ ("\x00" ** (4 - op.len))).*;
                }
                break :blk padded_ops_table;
            };

            const masks = blk: {
                var bitmask = std.mem.zeroes([2]u64);
                var cnt = 0;

                for (unpadded_ops, padded_ops) |unpadded_op, padded_op| {
                    if (unpadded_op.len == 1) continue;
                    cnt += 1;
                    const hash = rawHash(getOpWord(&padded_op, unpadded_op.len));
                    switch (@as(u1, @truncate(bitmask[hash / 64] >> @truncate(hash)))) {
                        0 => bitmask[hash / 64] |= @as(u64, 1) << @truncate(hash),
                        1 => bitmask[hash / 64] &= ~(@as(u64, 1) << @truncate(hash)),
                    }
                }

                if (@popCount(bitmask[0]) + @popCount(bitmask[1]) != cnt) {
                    var err_msg: []const u8 = "Hash function failed to map operators perfectly.\nThe following collisions occurred:\n";
                    for (unpadded_ops, padded_ops) |unpadded_op, padded_op| {
                        if (unpadded_op.len == 1) continue;
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

            fn getOpWord(op: [*]const u8, len: u32) u32 { // TODO: make this work with big-endian
                // comptime assert(back_sentinels.len >= 3);
                assert(@inComptime());
                const shift_amt = len * 8;
                const relevant_mask: u32 = @intCast((@as(u64, 1) << @intCast(shift_amt)) -% 1);
                return std.mem.readInt(u32, op[0..4], .little) & relevant_mask;
            }

            fn rawHash(op_word: u32) u7 {
                var hash: u7 = @intCast(((op_word *% 698839068) -% comptime (@as(u32, 29) << 25)) >> 25);
                if (hash < 16) hash +%= 18;
                return hash;
            }

            fn hashOp(op_word: u32) Tag {
                return @enumFromInt(rawHash(op_word));
            }

            fn mapToIndexRaw(hash_val: u7) u7 {
                const mask = (@as(u64, 1) << @truncate(hash_val)) -% 1;
                return (if (hash_val >= 64) (comptime @popCount(masks[0])) else 0) +
                    @popCount(mask & masks[hash_val / 64]);
            }

            /// Given a hash, maps it to an index in the range [0, sorted_padded_ops.len)
            fn mapToIndex(hash: Tag) u7 {
                return mapToIndexRaw(@intFromEnum(hash));
            }

            const single_char_ops = [_]u8{ '~', ':', ';', '[', ']', '(', ')', '{', '}', ',' };

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
        };

        const Keywords = struct {
            const LOOKUP_IMPL: u1 = @intFromBool(builtin.mode == .ReleaseSmall);
            const SWAR_LOOKUP_IMPL: u1 = @intFromBool(builtin.mode == .ReleaseSmall);

            // We can halve the number of cache lines by using two-byte slices into a dense superstring:
            const kw_buffer: []const u8 =
                "addrspacerrdeferrorelsenumswitchunreachablepackedforeturnunionwhilecontinueconstructcomptimevolatileifnpubreakawaitestryasyncatch" ++ "linksectionosuspendanytypeanyframeandallowzeropaquexporthreadlocallconvaresumexternoinlinealignoaliasmusingnamespace_\x00";

            // zig fmt: off
            const unpadded_kws = [_][:0]const u8{
                "addrspace", "align", "allowzero", "and", "anyframe", "anytype", "asm", "async", "await", "break", "callconv", "catch",
                "comptime", "const", "continue", "defer", "else", "enum", "errdefer", "error", "export", "extern", "fn", "for", "if", "inline",
                "noalias", "noinline", "nosuspend", "opaque", "or", "orelse", "packed", "pub", "resume", "return", "linksection", "struct",
                "suspend", "switch", "test", "threadlocal", "try", "union", "unreachable", "usingnamespace", "var", "volatile", "while"
            };
            // zig fmt: on

            const masks = blk: {
                var bitmask = std.mem.zeroes([2]u64);
                for (unpadded_kws) |kw| {
                    const hash = hashKw(kw.ptr, kw.len);
                    if (hash == 65) @compileError("Oh no!");
                    bitmask[hash / 64] |= @as(u64, 1) << @truncate(hash);
                }

                if (@popCount(bitmask[0]) + @popCount(bitmask[1]) != unpadded_kws.len) { //TODO: fix error message
                    const err_msg: []const u8 = "Hash function failed to map operators perfectly.\nThe following collisions occurred:\n";
                    // for (unpadded_kws) |kw| {
                    //     const hash = hashKw(kw.ptr, kw.len);
                    //     switch (@as(u1, @truncate(bitmask[hash / 64] >> @truncate(hash)))) {
                    //         0 => err_msg = err_msg ++ std.fmt.comptimePrint("\"{s}\" => {}\n", .{ kw, hash }),
                    //         1 => {},
                    //     }
                    // }
                    @compileError(err_msg);
                }

                break :blk bitmask;
            };

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
                var buffer = std.mem.zeroes([std.math.maxInt(u7) + 1]kw_slice);

                for (unpadded_kws) |kw|
                    buffer[hashKw(kw.ptr, kw.len)] = .{
                        .start_index = std.mem.indexOf(u8, kw_buffer, kw) orelse @compileError(std.fmt.comptimePrint("Please include the keyword \"{s}\" inside of `kw_buffer`.\n\n`kw_buffer` is currently:\n{s}\n", .{ kw, kw_buffer })),
                        .len = kw.len,
                    };

                break :blk buffer;
            };

            pub fn hashKw(keyword: [*]const u8, len: u32) u7 {
                assert(len != 0);
                assert(@inComptime());
                // comptime assert(back_sentinels.len >= 1); // Make sure it's safe to go forward a character when len=1
                const a = std.mem.readInt(u16, keyword[0..2], .little);
                // comptime assert(front_sentinels.len >= 1); // Make sure it's safe to go back to the previous character when len=1
                const b = std.mem.readInt(u16, (if (@inComptime())
                    keyword[len - 2 .. len][0..2]
                else
                    keyword - 2 + @as(usize, @intCast(len)))[0..2], .little);
                return @truncate((((len << 14) ^ a) *% b) >> 8);
                // return @truncate(((a >> 1) *% (b >> 1) ^ (len << 14)) >> 8);
                // return @truncate((((a >> 1) *% (b >> 1)) >> 8) ^ (len << 6));
            }

            fn hasIndex(hash: u7) bool {
                return 1 == @as(u1, @truncate(masks[hash / 64] >> @truncate(hash)));
            }

            /// Given a hash, maps it to an index in the range [0, sorted_padded_kws.len)
            fn mapToIndex(hash: u7) u8 {
                const mask = (@as(u64, 1) << @truncate(hash)) -% 1;
                return @as(u8, if (hash >= 64) (comptime @popCount(masks[0])) else 0) + @popCount(mask & masks[hash / 64]);
            }
        };

        const MultiCharSymbolTokenizer = struct {
            prev_chunk: @Vector(64, u8) = @splat(0),
            prev_singleCharEnds: @Vector(64, u8) = @splat(0),
            prev_doubleCharEnds: @Vector(64, u8) = @splat(0),
            delete_triple_char_pos_carry: uword = 0,

            // TODO: we could separate `chunk` and `classified_chunk` as separate here, for code clarity.
            fn getMultiCharEndPositions(self: *@This(), carry: *Carry, chunk: @Vector(64, u8)) struct { uword, uword, uword } {
                const prelim_ends = self.getMultiCharMasks(chunk);
                const refined_ends = self.refineMultiCharEndsMasks(carry, prelim_ends.doubleCharEnds, prelim_ends.tripleCharEnds);

                return .{
                    // Whatever remains outside of a double or triple char symbol can remain a single char symbol
                    prelim_ends.singleCharEnds &
                        ~(refined_ends.double_char_ends | (refined_ends.double_char_ends >> 1)) &
                        ~(refined_ends.triple_char_ends | (refined_ends.triple_char_ends >> 1) | (refined_ends.triple_char_ends >> 2)),
                    refined_ends.double_char_ends,
                    refined_ends.triple_char_ends,
                };
            }

            fn hashOpChars(b: anytype) @TypeOf(b) {
                const e = if (@TypeOf(b) == u8) @as(@Vector(1, u8), .{b}) else b;

                const ret = @as(@Vector(@sizeOf(@TypeOf(e)), u4), @truncate(blk: {
                    var r = e >> @splat(5);
                    r -%= e >> @splat(1);
                    r ^= e << @splat(1);
                    r +%= e >> @splat(5);
                    break :blk r +% @as(@TypeOf(e), @splat(8));
                }));

                return if (@TypeOf(b) == u8) ret[0] else ret;
            }

            const hashed_op_chars = blk: {
                @setEvalBranchQuota(100000);
                var hashed_chars: []const u8 = "?";
                for (Operators.unpadded_ops) |op| {
                    if (op.len > 1) {
                        for (op) |c| {
                            const char = &[1]u8{c};
                            if (std.mem.indexOf(u8, hashed_chars, char) == null)
                                hashed_chars = hashed_chars ++ char;
                        }
                    }
                }
                break :blk hashed_chars;
            };

            const perfectly_hashed_op_chars = blk: {
                var buffer: @Vector(16, u8) = @splat(0);
                for (hashed_op_chars) |c| {
                    const slot = hashOpChars(c);
                    if (slot == 0) @compileError(std.fmt.comptimePrint("`{c}` hashed to 0", .{c}));
                    if (buffer[slot] != 0) @compileError(std.fmt.comptimePrint("Collision between `{c}` and `{c}`", .{ buffer[slot], c }));
                    buffer[slot] = c;
                }
                break :blk buffer;
            };

            fn hash_shuffle(v: anytype, comptime table: @Vector(16, u8)) @TypeOf(v) {
                if (CLASSIFIER_STRATEGY == .global) {
                    // See `classifier` definition. `v` has only the characters we care about in the range [1, 15]
                    return vpshufb(table, v);
                } else {
                    const indices = hashOpChars(v);
                    return @select(
                        u8,
                        vpshufb(perfectly_hashed_op_chars, indices) == v,
                        vpshufb(table, indices),
                        @as(@TypeOf(v), @splat(0)),
                    );
                }
            }

            fn getMultiCharMasks(self: *@This(), chunk: @Vector(64, u8)) struct {
                singleCharEnds: u64,
                doubleCharEnds: u64,
                tripleCharEnds: u64,
            } {
                // We produce 3 shuffle vectors: The i-th shuffle vector, after looking up the textual characters (as indices into it),
                // tells us whether that character is a valid i-th character of a multi-character symbol.
                // E.g. the 1st shuffle vector would tell us that `+` is a valid first character of some multi-char symbol.
                // The 2nd would tell us `|` is valid in the second position, and the 3rd would tell us that `=` is valid
                // in the third position. By taking the mask of each of the 3 shuffles and ANDing them together, we
                // thus know that `+|=` is valid. However, we don't want to match any arbitrary combination, so we extend this idea:
                // Because we get 8 bits as the result of a lookup, we can make each bit a separate "channel".
                // When we AND the results of the shuffles together, the first bit tells us whether there was a match in the first channel,
                // the second bit tells us whether there was a match in the second channel, etc.
                // Within a "channel", you can have any combination of the first two sets of chars, and optionally any char
                // from the set of 3rd characters. E.g. the `bar_mod` set { "-+*", "%|", "=" } matches any char from the first string,
                // followed by any char from the second string, optionally followed by any char from the third string.
                //
                // We can augment this trick, as we do for the `self` matcher. For this one, we insert an extra piece of logic not
                // applicable to the rest of the "channels". We want the characters that match on the `self` channel to all be the same,
                // and enforce that by zeroing out the bit corresponding to `self` if they are not the same.
                //
                // Note: by default, we assume that for any 3-character multi-char symbol, the first 2 characters are also a valid symbol.
                // If this is not the case, you will need to FIXME

                // These should always work as 16-byte lookup tables, because we shouldn't ever have more than 15 or 16 possible characters
                // in a multi-character symbol due to the keys that are on a typical keyboard. We might want to use more than one vector shuffle
                // though if we need more than 8 channels. E.g., since we get 8 bits per character, we could use two to get 16 bits per character.
                comptime var first_char_data: @Vector(16, u8) = @splat(0);
                comptime var second_char_data: @Vector(16, u8) = @splat(0);
                comptime var third_char_data: @Vector(16, u8) = @splat(0);

                const Channels = packed struct {
                    bar_mod: u1 = 0,
                    eql: u1 = 0,
                    dot: u1 = 0,
                    arrow: u1 = 0,
                    @"<": u1 = 0,
                    @">": u1 = 0,
                    dot_question: u1 = 0,
                    self: u1 = 0, // TODO: move this field to whichever slot allows us to reuse a vector
                };

                comptime for (std.meta.fields(Channels)) |o| {
                    // Note: this assumes that the prefix string of all operators defined here are also an operator.
                    // If you want a new operator that has a prefix string which is not a valid operator,
                    // you need to either check it yourself without using the vector shuffle,
                    // or just write a line that augments the logic done by the vector shuffle.
                    // The augmentation technique would look similar to what we do below for the "self" matcher.

                    for (@as([3][]const u8, switch (@field(std.meta.FieldEnum(Channels), o.name)) {
                        // zig fmt: off
                        .bar_mod      => .{ "-+*",             "%|",            "="  }, // Matches [-+*][%|]=?
                        .eql          => .{ "|=!%^+*/&<>-",    "=",             ""   }, // Matches [|=!%^+*/&<>-]=
                        .self         => .{ "|.+*",          "|.+*",            "."  }, // Matches ([|.+*])(\1)|...
                        .dot          => .{ ".",               "*",             "*"  }, // Matches .**|.*
                        .arrow        => .{ "-=",              ">",             ""   }, // Matches [-=]>
                        .@"<"         => .{ "<",               "<",             "|=" }, // Matches <<[|=]
                        .@">"         => .{ ">",               ">",             "="  }, // Matches >>=
                        .dot_question => .{ ".",               "?",             ""   }, // Matches .?
                        // zig fmt: on
                    }), .{ &first_char_data, &second_char_data, &third_char_data }) |string, dest_char_data| {
                        var state: Channels = .{};
                        @field(state, o.name) = 1;
                        // @compileLog(hashOpChars(@as(@Vector(1, u8), .{'/'})));
                        for (string) |c| {
                            dest_char_data[hashOpChars(c)] |= @bitCast(state);

                            // if (@field(std.meta.FieldEnum(Channels), o.name) == .self) {
                            //     @compileLog(std.fmt.comptimePrint("i: {}, c: {c}, hashOpChars(c): {}, state: {b:0>8}", .{ i, c, hashOpChars(c), @as(u8, @bitCast(state)) }));
                            // }
                        }
                    }
                };

                comptime var state_without_self: Channels = @bitCast(~@as(@typeInfo(Channels).@"struct".backing_integer.?, 0));
                state_without_self.self = 0;

                const prev1 = shift_in_prev(1, chunk, self.prev_chunk);
                const prev2 = shift_in_prev(2, chunk, self.prev_chunk);

                const singleCharEnds = hash_shuffle(chunk, first_char_data);
                defer self.prev_singleCharEnds = singleCharEnds;

                const doubleCharEnds = hash_shuffle(chunk, second_char_data) &
                    shift_in_prev(1, singleCharEnds, self.prev_singleCharEnds) &
                    @select(u8, prev1 == chunk, @as(@Vector(64, u8), @splat(0b11111111)), @as(@Vector(64, u8), @splat(@bitCast(state_without_self))));
                defer self.prev_doubleCharEnds = doubleCharEnds;

                const tripleCharEnds = hash_shuffle(chunk, third_char_data) &
                    shift_in_prev(1, doubleCharEnds, self.prev_doubleCharEnds) &
                    @select(u8, prev2 == prev1, @as(@Vector(64, u8), @splat(0b11111111)), @as(@Vector(64, u8), @splat(@bitCast(state_without_self))));

                defer self.prev_chunk = chunk;

                const singleCharEndsBitstr = @as(u64, @bitCast(singleCharEnds != @as(@Vector(64, u8), @splat(0)))) |
                    @as(u64, @bitCast(chunk == @as(@TypeOf(chunk), @splat(classifier['?']))));

                const doubleCharEndsBitstr = @as(u64, @bitCast(doubleCharEnds != @as(@Vector(64, u8), @splat(0))))
                // add *^
                | (@as(u64, @bitCast(prev1 == @as(@TypeOf(prev1), @splat(classifier['*'])))) &
                    @as(u64, @bitCast(chunk == @as(@TypeOf(chunk), @splat(classifier['^'])))));

                const tripleCharEndsBitstr = @as(u64, @bitCast(tripleCharEnds != @as(@Vector(64, u8), @splat(0))));

                return .{
                    .singleCharEnds = singleCharEndsBitstr,
                    .doubleCharEnds = doubleCharEndsBitstr,
                    .tripleCharEnds = tripleCharEndsBitstr,
                };
            }

            fn refineMultiCharEndsMasks(self: *@This(), carry: *Carry, doubleCharEnd_: uword, tripleCharEnd: uword) struct {
                double_char_ends: uword,
                triple_char_ends: uword,
            } {
                // The rule for multi-char symbol matching is that we always want to match the longest possible symbol that we can.
                // That means that if there is a 3-char-end in the next position from a two-char-end, we can unset the two-char-end bit.
                // E.g.              a +|= b
                //  doubleCharEnd <- 0001000 (we unset the 1 bit)
                //  tripleCharEnd <- 0000100
                const doubleCharEnd = doubleCharEnd_ & ~(tripleCharEnd >> 1);
                //                                   ^ We could use XOR, instead of ANDN, so long as all 2-char prefixes of 3-char symbols are valid.

                // The idea is to iterate over a bitstring which is the bitwise-OR of `doubleCharEnd` and `tripleCharEnd`.
                // We iteratively find the lowest 1 bit, unset that bit, then unset one, possibly two bits after that bit, and repeat.
                // We would unset two bits after if that bit corresponds to the end of a triple-char symbol. E.g.
                //
                //                                   a =>>= b
                //  doubleCharEnd | tripleCharEnd <- 00011100
                //                                       ^ always unset this bit, because we shouldn't match `=>` and `>>`
                //                                        ^ unset this bit if there's a potential 3-char-symbol here. We shouldn't match `=>` and `>>=`.
                var s = doubleCharEnd | tripleCharEnd;

                // Will be a subset of `doubleCharEnd` | `tripleCharEnd`, but with the boundaries properly figured out.
                var ends: uword = 0;

                // If a triple-char-symbol-end is at the beginning of the next chunk, we unset it if the last chunk ended with a multi-char symbol within two chars. E.g.:
                //                                                                     a =>>= b
                // ends <- 0000000000000000000000000000000000000000000000000000000000000001
                //                                                           next chunk -> 1100000000000000000000000000000000000000000000000000000000000000
                //                                                                          ^ unset this one since it's a triple-char-end within two positions of the previous symbol-end
                //                                                                    a =>>= b
                // ends <- 0000000000000000000000000000000000000000000000000000000000000010
                //                                                           next chunk -> 1000000000000000000000000000000000000000000000000000000000000000
                //                                                                         ^ unset this one since it's a triple-char-end within two positions of the previous symbol-end

                s &= ~(self.delete_triple_char_pos_carry & ~doubleCharEnd);
                defer self.delete_triple_char_pos_carry = ends >> 62;

                // This handles the case where a chunk ends on a double-char symbol, only to find in the next chunk that it's a triple-char symbol.
                //                                                                     a +%= b
                // ends <- 0000000000000000000000000000000000000000000000000000000000000001
                //                                                           next chunk -> 100000000000000000000000000000000000000000000000000000000000000
                //                                                                         ^ this is valid because it's a triple-char-end
                //                                                                     a ++= b
                // ends <- 0000000000000000000000000000000000000000000000000000000000000001
                //                                                           next chunk -> 100000000000000000000000000000000000000000000000000000000000000
                //                                                                         ^ unset because it's a double-char-end
                s &= ~andn(carry.get(.ended_on_double_char_carry), tripleCharEnd);
                defer carry.set(.ended_on_double_char_carry, ends & doubleCharEnd);

                while (true) {
                    // Optimization: Instead of iterating over a single position at once, we iterate over multiple positions at once
                    // that have at least two "spaces" (not-possible-to-be-multi-char-symbol chars) in between them. (Or at the start.)
                    // For a "pathological" input of randomly assorted 2 and 3-char symbols that overlap a lot, we have to iterate from
                    // start to finish, in order, because we cannot tell in parallel where all the symbols start.
                    // However, in real Zig code, you typically have "spaces" in between the symbols, sufficient that you can guarantee that
                    // the first character in a group is the start of a multi-char symbol.
                    const iter = s & ~(s << 1) & ~(s << 2);

                    ends |= iter;

                    // We want to unset the current bit(s) pointed to by `iter`, as well as the next bit, given by `iter << 1`,
                    // because you can't have a 2 or 3 character symbol end in the next byte after the end of a 2 or 3 character symbol.
                    // E.g.         a  b
                    //      iter <- 00010000
                    //      s    <- 00011100
                    //                  ^ always unset this bit, because we can't match both `=>` and `>>` since they overlap

                    // We also want to zero two bits after, if that corresponds to a non-2-char symbol `(iter << 2) & ~doubleCharEnd`.
                    // E.g.                 .....
                    //              iter <- 00100
                    //              s    <- 00111
                    //     doubleCharEnd <- 00001
                    //     tripleCharEnd <- 00111
                    //                         ^ always unset
                    //                          ^ not unset because this happens to be a valid 2-char-symbol
                    // E.g.                 ......
                    //              iter <- 001000
                    //              s    <- 001111
                    //     doubleCharEnd <- 000001
                    //     tripleCharEnd <- 001111
                    //                         ^ always unset
                    //                          ^ unset because this is a valid 3-char-symbol but not a valid 2-char-symbol (due to the next char making a 3-char sequence, see `doubleCharEnd &= ...` above)
                    // E.g.         a =>>= b
                    //      iter <- 00010000
                    //      s    <- 00011100
                    //                   ^ unset this bit because there's a potential 3-char-symbol-end within two chars of one of the `iter` bits.
                    //                     I.e., we can't match both `=>` and `>>=` because they overlap.

                    s &= ~(disjoint_or(iter, iter << 1) | andn(iter << 2, doubleCharEnd));
                    //           ^ All 1's in `iter` must have at least two 0's between them. The compiler gives us the shift+add in a single `lea` instruction :)

                    if (s == 0) {
                        @branchHint(.likely);
                        break;
                    }

                    // We could advance each 1 bit in `iter` to the next 1 bit in `s` using `iter = s & ~(s -% iter);`,
                    // but it's pretty much impossible for this loop to branch in practice, so this optimization would be pointless.
                }

                // 3-char sequences take precedence over two-char sequences, except when they are just two characters apart.
                const double_char_ends_final = (ends & ~tripleCharEnd) | (ends & (ends << 2));
                const triple_char_ends_final = ends ^ double_char_ends_final;
                // Alternative implementation:
                //const triple_char_ends_final = ((ends & tripleCharEnd) & ~(ends & (ends << 2)));
                //const double_char_ends_final = ends ^ triple_char_ends_final;
                return .{
                    .double_char_ends = double_char_ends_final,
                    .triple_char_ends = triple_char_ends_final,
                };
            }
        };

        // TODO: I believe the best idea is to take advantage of movemask facilities.
        // By moving each of these over to a vector and using the hardware's builtin functionality of taking the uppermost bit,
        // we might be able to avoid some of our interactions with the stack

        const Carry = struct {
            quote_starts: u64 = 0,
            char_lit_starts: u64 = 0,
            comment_starts: u64 = 0,
            line_string_starts_incl_carry: u64 = 0,

            slashes: u64 = 0,
            backslashes: u64 = 0,
            non_newlines: u64 = 0,
            carriages: u64 = 0,
            ats: u64 = 0,

            inside_strings_and_comments_including_start: u64 = 0,
            next_is_escaped: u64 = 0,
            ended_on_double_char_carry: u64 = 0,

            alpha_numeric_underscores: u64 = 0,
            identifier_ends: u64 = 0,
            identifier_or_number_or_builtin_ends: u64 = 0,
            number_or_builtin_ends: u64 = 0,

            inside_quotes_incl_start_and_carry: u64 = 0,
            inside_apostrophes_incl_start_and_carry: u64 = 0,
            inside_comments: u64 = 0,
            inside_line_strings: u64 = 0,
            // _: u54 = 0,

            fn print(self: @This()) void {
                if (comptime builtin.mode == .Debug) {
                    comptime var fmtStr: []const u8 = "";
                    const field_names = comptime std.meta.fieldNames(@This());
                    inline for (field_names, 0..) |field_name, i| {
                        if (comptime std.mem.eql(u8, field_name, "_")) continue;
                        fmtStr = fmtStr ++ field_name ++ ": {}".* ++ [1]u8{(if (i != field_names.len - 1) ',' else '\n')};
                        if (i != field_names.len - 1) fmtStr = fmtStr ++ " ".*;
                    }

                    var tuple: GetArgsStruct() = undefined;
                    inline for (field_names) |field_name| {
                        if (comptime std.mem.eql(u8, field_name, "_")) continue;
                        @field(tuple, field_name) = @field(self, field_name);
                    }

                    std.debug.print(fmtStr, @as(GetArgsStruct(), tuple));
                }
            }

            fn GetArgsStruct() type {
                comptime var fields: []const std.builtin.Type.StructField = &[0]std.builtin.Type.StructField{};

                inline for (std.meta.fieldNames(@This())) |field_name| {
                    if (std.mem.eql(u8, field_name, "_")) continue;
                    fields = fields ++ [1]std.builtin.Type.StructField{.{
                        .name = field_name,
                        .type = u64,
                        .default_value_ptr = null,
                        .is_comptime = false,
                        .alignment = 8,
                    }};
                }

                return @Type(.{
                    .@"struct" = .{
                        .layout = .auto,
                        .fields = fields,
                        .decls = &.{},
                        .is_tuple = false,
                    },
                });
            }

            fn get(self: *const @This(), comptime field: std.meta.FieldEnum(GetArgsStruct())) u64 {
                return @field(self, @tagName(field)) >> 63;
            }

            fn set(self: *@This(), comptime field: std.meta.FieldEnum(GetArgsStruct()), value: u64) void {
                @field(self, @tagName(field)) = value;
            }

            // Takes the top bit of each bitstring passed in as a u64.
            // Reduces visual clutter at the call site.
            fn updateViaTopBits(self: *@This(), in: GetArgsStruct()) void {
                inline for (comptime std.meta.fieldNames(@TypeOf(in))) |field_name| {
                    @field(self, field_name) = @truncate(@field(in, field_name) >> 63);
                }
            }
        };

        const TokenizerCarry2 = struct {
            buffer: @Vector(8, u64),
        };

        fn getEscapedPositions(carry: *Carry, backslashes: uword) uword {
            // ----------------------------------------------------------------------------
            // This code is brought to you courtesy of simdjson, licensed
            // under the Apache 2.0 license which is included at the bottom of this file
            // Credit to John Keiser (@jkeiser) for designing this algorithm.
            // See https://github.com/simdjson/simdjson/pull/2042

            const ODD_BITS: uword = @bitCast([_]u8{0xaa} ** @divExact(@bitSizeOf(uword), 8));
            const next_is_escaped = carry.get(.next_is_escaped);
            // |                                | Mask (shows characters instead of 1's) | Depth | Instructions        |
            // |--------------------------------|----------------------------------------|-------|---------------------|
            // | string                         | `\\n_\\\n___\\\n___\\\\___\\\\__\\\`   |       |                     |
            // |                                | `    even   odd    even   odd   odd`   |       |                     |
            // | potential_escape               | ` \  \\\    \\\    \\\\   \\\\  \\\`   | 1     | 1 (backslashes & ~first_is_escaped)
            // | escape_and_terminal_code       | ` \n \ \n   \ \n   \ \    \ \   \ \`   | 5     | 5 (next_escape_and_terminal_code())
            // | escaped                        | `\    \ n    \ n    \ \    \ \   \ ` X | 6     | 7 (escape_and_terminal_code ^ (potential_escape | first_is_escaped))
            // | escape                         | `    \ \    \ \    \ \    \ \   \ \`   | 6     | 8 (escape_and_terminal_code & backslashes)
            // | first_is_escaped               | `\                                 `   | 7 (*) | 9 (escape >> 63) ()
            //                                                                               (*) this is not needed until the next iteration
            const potential_escape = backslashes & ~next_is_escaped;

            // If we were to just shift and mask out any odd bits, we'd actually get a *half* right answer:
            // any even-aligned backslashes runs would be correct! Odd-aligned backslashes runs would be
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
            // 3. & with backslashes to clean up any stray bits.
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
            const escaped = escape_and_terminal_code ^ (backslashes | next_is_escaped);
            carry.set(.next_is_escaped, escape_and_terminal_code & backslashes);
            return escaped;
            // ----------------------------------------------------------------------------
        }

        const in_multi_char_symbol: u128 = blk: {
            @setEvalBranchQuota(100000);
            var x: u128 = 0;
            for (0..128) |c| {
                for (Operators.unpadded_ops) |op| {
                    if (op.len == 1) continue;
                    if (std.mem.eql(u8, op, "//") or std.mem.eql(u8, op, "\\\\")) continue;

                    for (op) |op_chr| {
                        if (c == op_chr)
                            x |= @as(u128, 1) << c;
                    }
                }
            }
            break :blk x;
        };

        const in_single_char_symbol: u128 = blk: {
            @setEvalBranchQuota(100000);
            var x: u128 = 0;
            for (0..128) |c| {
                for (Operators.unpadded_ops) |op| {
                    if (op.len == 1 and op[0] == c)
                        x |= @as(u128, 1) << c;
                }
            }
            break :blk x;
        };

        const exclusively_in_single_char_symbol = in_single_char_symbol & ~in_multi_char_symbol;

        const CharClasses = blk: {
            var char_classes = @typeInfo(enum(u8) {
                eof = 0,
                _nada = 6, // I was using this before, but now it could be used for some other char in a multi-char identifier
                backslash = 159,
                alpha_underscore = 0x81, // identifier
                digit = 0x91, // number
                bad_control = 0xBB, // invalid

                whitespace = 128 | @as(u8, 34), // 162
                quote = 128 | @as(u8, 44), // string
                string_identifier = 128 | @as(u8, 20),
                apostrophe = 128 | @as(u8, 51), // char_literal

                at = 128 | @as(u8, 21),
                unknown = 128 + 32,
            });

            var kw_and_unary_hashes: u128 = 0;

            for (Keywords.unpadded_kws) |kw| {
                const hash = Keywords.hashKw(kw.ptr, kw.len);
                kw_and_unary_hashes |= @as(@TypeOf(kw_and_unary_hashes), 1) << hash;
            }

            for (Operators.potentially_unary) |op| { // For future use
                const padded_op = (op ++ ("\x00" ** (4 - op.len))).*;
                const hash = Operators.rawHash(Operators.getOpWord(&padded_op, op.len));
                const modified_hash = Operators.unarifyBinaryOperatorRaw(hash);
                kw_and_unary_hashes |= @as(@TypeOf(kw_and_unary_hashes), 1) << @truncate(modified_hash);
            }

            // Fill in single_char_symbols from 0xFF downward
            var b = exclusively_in_single_char_symbol;
            var value = 0xFF;
            while (b != 0) : ({
                b &= b -% 1;
                value -= 1;
            }) {
                while (((kw_and_unary_hashes >> @truncate(value)) & 1) == 1) value -= 1;

                char_classes.@"enum".fields = char_classes.@"enum".fields ++ [1]std.builtin.Type.EnumField{.{
                    .name = std.fmt.comptimePrint("{c}", .{@ctz(b)}),
                    .value = value,
                }};
            }

            // Fill in multi_char_symbols in [0, 15]
            b = in_multi_char_symbol;

            while (b != 0) : (b &= b -% 1) {
                const char = @ctz(b);
                const new_field = [1]std.builtin.Type.EnumField{.{
                    .name = std.fmt.comptimePrint("{c}", .{char}),
                    .value = MultiCharSymbolTokenizer.hashOpChars(char),
                }};
                char_classes.@"enum".fields = char_classes.@"enum".fields ++ new_field;
            }

            break :blk @Type(char_classes);
        };

        const last_exclusively_in_single_char_symbol = @intFromEnum(std.meta.stringToEnum(
            CharClasses,
            std.fmt.comptimePrint("{c}", .{(127 - @clz(exclusively_in_single_char_symbol))}),
        ).?);

        const Tag = blk: {
            @setEvalBranchQuota(std.math.maxInt(u32));
            var decls = [_]std.builtin.Type.Declaration{};
            var enumFields: []const std.builtin.Type.EnumField = &.{};

            for (std.meta.fields(CharClasses)) |field| {
                const enumField = [1]std.builtin.Type.EnumField{.{
                    .value = field.value,
                    .name = switch (@as(Tokenizer(.{}).CharClasses, @enumFromInt(field.value))) {
                        .digit => "number",
                        .alpha_underscore => "identifier",
                        .bad_control => "invalid",
                        .quote => "string",
                        .apostrophe => "char_literal",
                        .at => "builtin",
                        .backslash => "\\\\",
                        else => field.name,
                    },
                }};
                enumFields = enumFields ++ enumField;
            }

            for (Operators.unpadded_ops, Operators.padded_ops) |op, padded_op| {
                if (op.len == 1) {
                    assert(std.meta.stringToEnum(Tokenizer(.{}).CharClasses, op) != null);
                    continue;
                }
                if (std.mem.eql(u8, op, "\\\\")) continue;

                const hash: u8 = Operators.rawHash(Operators.getOpWord(&padded_op, op.len));
                enumFields = enumFields ++ [1]std.builtin.Type.EnumField{.{ .name = op, .value = hash }};
            }

            for (Keywords.unpadded_kws) |kw| {
                const hash: u8 = Keywords.hashKw(kw.ptr, kw.len);
                enumFields = enumFields ++ [1]std.builtin.Type.EnumField{.{ .name = kw, .value = 128 | hash }};
            }

            // @as(*const [12:0]u8, "unary -% 42") 204
            // @as(*const [12:0]u8, "unary ** 247-204")
            // 44
            // @as(*const [11:0]u8, "unary - 249")
            // @as(*const [11:0]u8, "unary * 250")
            // @as(*const [11:0]u8, "unary . 251")
            // @as(*const [11:0]u8, "unary & 252")
            // @as(*const [12:0]u8, "unary .. 253")

            // for (Operators.potentially_unary) |op| {
            //     const padded_op = (op ++ ("\x00" ** (4 - op.len))).*;
            //     const hash = Operators.rawHash(Operators.getOpWord(&padded_op, op.len));
            //     const modified_hash = Operators.unarifyBinaryOperatorRaw(hash);

            //     const field = [1]std.builtin.Type.EnumField{.{ .name = "unary " ++ op, .value = modified_hash }};
            //     enumFields = enumFields ++ field;
            // }

            var enumFieldsSorted: [enumFields.len]std.builtin.Type.EnumField = enumFields[0..enumFields.len].*;
            std.sort.heap(std.builtin.Type.EnumField, &enumFieldsSorted, {}, struct {
                fn lessThanFn(_: void, a: std.builtin.Type.EnumField, b: std.builtin.Type.EnumField) bool {
                    return a.value < b.value;
                }
            }.lessThanFn);

            // for (enumFieldsSorted) |field|
            //     @compileLog(std.fmt.comptimePrint("{s} {}", .{ field.name, field.value }));

            // enumFields = enumFields ++ [1]std.builtin.Type.EnumField{.{ .name = "call (", .value = Operators.postifyOperatorRaw(Operators.rawHash(Operators.getOpWord("(" ++ "\x00" ** 3, 1))) }};

            break :blk @Type(.{
                .@"enum" = .{
                    .tag_type = u8,
                    .fields = enumFields,
                    .decls = &decls,
                    .is_exhaustive = true,
                },
            });
        };

        const Token = packed struct { len: u8, kind: Tag };

        const TokenInfo = struct { is_large_token: bool, kind: Tag, source: []const u8 };

        const TokenInfoIterator = struct {
            cursor: [*]const u8,
            cur_token: [*]const Token,
            // cur_token: [*:Token{ .len = 0, .kind = .eof }]Token,

            pub fn init(source: []const u8, tokens: []const Token) @This() {
                return .{ .cursor = source.ptr, .cur_token = tokens.ptr };
            }

            pub fn current(self: *const @This()) TokenInfo {
                const is_large_token = self.cur_token[0].len == 0;
                const large_len: u32 = @bitCast(self.cur_token[1..3].*);
                const len: u32 = if (is_large_token) large_len else self.cur_token[0].len;
                const kind = self.cur_token[0].kind;
                const source = self.cursor[0..len];
                return .{ .is_large_token = is_large_token, .kind = kind, .source = source };
            }

            pub fn advance(self: *@This()) void {
                const info = self.current();
                self.cursor = self.cursor[info.source.len..];
                self.cur_token = self.cur_token[if (info.is_large_token) 3 else 1..];
            }

            pub fn advanceToken(self: *@This()) void {
                const info = self.current();
                self.cur_token = self.cur_token[if (info.is_large_token) 3 else 1..];
            }

            pub fn advanceCursor(self: *@This()) void {
                const info = self.current();
                self.cursor = self.cursor[info.source.len..];
            }
        };

        // This lookup table:
        //  1. Is used to map all characters than can be in multi-character operators to the range [1,15].
        //     Once the relevant characters are in the range [1, 15], it can be fed into a `vpshufb` to
        //     determine where the multi-char operator are. Everything not mapped to the range [1, 15] is
        //     mapped to the range [0x80, 0xFF] instead, that way, when passed into `vpshufb`, the result will be zeroed.
        //  2. Simplifies the checks we have to do to produce bitstrings such that only a single instruction
        //     is necessary to produce any bitstring that contains some interesting set of characters.
        //     We can take advantage of signed and unsigned comparisons by making our interesting clusters
        //     in the range [0x80, 0x80+c] and/or [0xFF-k, 0xFF] for some value of `c` or `k`
        //  3. Is used as the base of the `kinds` vector, which holds the uncompressed token kinds. Things
        //     like '"' and '\'' will be prefilled the kind for a quote or a character-literal.
        //     That way, we don't have to put such things in the vector before compressing it.
        const classifier: @Vector(128, u8) = blk: {
            @setEvalBranchQuota(100000);
            var classifier_vec: @Vector(128, u8) = @splat(@intFromEnum(@as(CharClasses, .unknown)));

            var multi_sym_bitmap: u16 = 0; // TODO: make this dynamically adjust according to u4

            var c: u8 = 0;
            while (c < 128) : (c += 1) {
                const is_in_multi_char_symbol: bool = ((in_multi_char_symbol >> @intCast(c)) & 1) == 1;
                const is_exclusively_single_symbol: bool = ((exclusively_in_single_char_symbol >> @intCast(c)) & 1) == 1;

                classifier_vec[c] = if (is_exclusively_single_symbol or is_in_multi_char_symbol)
                    @intFromEnum(std.meta.stringToEnum(CharClasses, &[1]u8{c}).?)
                else if (is_in_multi_char_symbol) classified_byte: {
                    const classified_byte = MultiCharSymbolTokenizer.hashOpChars(c);
                    // TODO: move this into enum declaration
                    const bit_slot = @as(@TypeOf(multi_sym_bitmap), 1) << @intCast(classified_byte);
                    var str: []const u8 = std.fmt.comptimePrint("`{c}`", .{c});
                    var str2: []const u8 = std.fmt.comptimePrint("'{c}'", .{c});

                    if (0 != (multi_sym_bitmap & bit_slot)) {
                        var c2: u8 = 0;
                        while (c2 < 128) : (c2 += 1) {
                            if (c != c2 and ((in_multi_char_symbol >> @intCast(c2)) & 1) == 1) {
                                const classified_byte2 = MultiCharSymbolTokenizer.hashOpChars(c2);
                                if (classified_byte == classified_byte2) {
                                    str = str ++ std.fmt.comptimePrint(" and `{c}`", .{c2});
                                    str2 = str2 ++ std.fmt.comptimePrint(" OR '{c}'", .{c2});
                                }
                                multi_sym_bitmap |= @as(@TypeOf(multi_sym_bitmap), 1) << @intCast(classified_byte2);
                            }
                        }

                        // TODO: make it so the code has an easier path to adjust
                        @compileError(std.fmt.comptimePrint(
                            "Duplicate hash in perfect hash function `hashOpChars`. {s} each hash to `{}`.\n{s}\n",
                            .{ str, classified_byte, if (~multi_sym_bitmap == 0)
                                std.fmt.comptimePrint("To have {s} in multi-character symbols, you are going to have to change the mapping from a u4 to a u5.", .{str})
                            else
                                std.fmt.comptimePrint("It might be time for a new perfect hash function.\nBut, for a quick fix, you could go to `hashOpChars` and add the following line:\n    ret = @select(u8, e == @as(@TypeOf(e), @splat(<{s}>)), @as(@TypeOf(ret), @splat({})), ret);", .{ str2, @ctz(~multi_sym_bitmap) }) },
                        ));
                    }

                    multi_sym_bitmap |= bit_slot;
                    break :classified_byte classified_byte;
                } else @intFromEnum(@as(CharClasses, switch (c) {
                    0 => .bad_control,
                    1...'\t' - 1, '\n' + 1...'\r' - 1, '\r' + 1...' ' - 1, 0x7F => .bad_control,
                    '\n', '\r', '\t', ' ' => .whitespace,
                    '0'...'9' => .digit,
                    'a'...'z', 'A'...'Z', '_' => .alpha_underscore,

                    '\'' => .apostrophe,
                    '"' => .quote,
                    '/' => .slash,
                    '\\' => .backslash,
                    '@' => .at,

                    else => .unknown,
                }));
            }

            break :blk classifier_vec;
        };

        fn getKeywordExtents(
            prev_chunk: @Vector(64, u8),
            contextless_chunk: @Vector(64, u8),
            lens: *@Vector(64, u8),
            kinds: *@Vector(64, u8),
            identifier_starts: u64,
            identifier_ends: u64,
            prev_carried_len: u4,
            potential_kw_started_before_chunk: u1,
        ) void {
            // TODO: Maybe don't consider identifiers that run off the end of the buffer.
            //        these wouldn't work if their data is split across two chunks, because we would not hash them properly.
            //        I think the solution here is just to do an individual hash for those that cross the boundary.
            //        This also prevents errors where there could theoretically be two keywords where one is a prefix of the other,
            //        and we match the prefix that occurs before the chunk boundary:
            //            |
            //          or|else
            //            |

            // We could prune identifiers based on length. Not worth it though:
            // 0123456789abcdef
            // ..11111111.1..1.

            // Data carries over IF (((identifier_ends >> 63) AND identifier_starts) & 1)
            // The length from the last token...?
            // alphanum_starts and ends should be modified to include a potentially-carried over value
            //

            printb(identifier_starts, "identifier_starts");
            printb(identifier_ends, "identifier_ends");

            // There are at most 32 identifiers in a chunk.
            // There are at most 22 multi-character identifiers in a chunk, including a carry-over from the previous chunk
            // i|f(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if
            //  ^ break between previous chunk and current chunk
            const alphanum_starts_larger_than_1 = identifier_starts & (~identifier_ends | potential_kw_started_before_chunk);
            // printb(alphanum_starts_larger_than_1, "alphanum_starts_larger_than_1");
            const alphanum_ends_larger_than_1 = (identifier_ends & ~identifier_starts) | ((potential_kw_started_before_chunk & identifier_ends) << 1);
            // printb(alphanum_ends_larger_than_1, "alphanum_ends_larger_than_1");

            const first_two_chars_mask = disjoint_or(alphanum_starts_larger_than_1, alphanum_starts_larger_than_1 << 1);
            const last_two_chars_mask = disjoint_or(alphanum_ends_larger_than_1, alphanum_ends_larger_than_1 >> 1);

            // const first_two_chars: @Vector(32, u16) = @bitCast(compress(contextless_chunk, first_two_chars_mask));
            // const last_two_chars: @Vector(32, u16) = @bitCast(compress(contextless_chunk, last_two_chars_mask));
            // std.debug.print("prev_carried_len:: {}\n", .{prev_carried_len});
            var start_indices = bitsToIndices(first_two_chars_mask, 0) -% ([2]u8{ prev_carried_len, prev_carried_len } ++ [1]u8{0} ** 62);
            const should_end_indices_decrement: u1 = @truncate(potential_kw_started_before_chunk & identifier_ends);
            const end_indices = bitsToIndices(last_two_chars_mask, 0) -% ([2]u8{ should_end_indices_decrement, should_end_indices_decrement } ++ [1]u8{0} ** 62);
            var identifier_lens = end_indices -% start_indices +% @as(@Vector(64, u8), @splat(2));

            const first_two_chars: @Vector(32, u16) = @bitCast(vperm2(std.simd.join(contextless_chunk, prev_chunk), start_indices));
            const last_two_chars: @Vector(32, u16) = @bitCast(vperm2(std.simd.join(contextless_chunk, prev_chunk), end_indices));

            lens.* = @select(u8, @as(@Vector(64, bool), @bitCast(@as(u64, potential_kw_started_before_chunk))), identifier_lens, lens.*);

            // std.debug.print("start_indices  : {d: >2}\n", .{start_indices});
            // std.debug.print("end_indices  : {d: >2}\n", .{end_indices});
            // std.debug.print("identifier_lens: {d: >2}\n", .{identifier_lens});
            // std.debug.print("first_two_chars: {c}\n", .{@as(@Vector(64, u8), @bitCast(first_two_chars))});
            // std.debug.print("last_two_chars: {c}\n", .{@as(@Vector(64, u8), @bitCast(last_two_chars))});

            // A SIMD version of the hash function found in `Keywords.hashKw`
            // Hash is just the upper byte of the u16's, ignoring the uppermost bit.
            // We do a `>> splat(8)` to move it to the lower byte, that way we can reuse `first8Interlacedbroadcast` below
            const hash: @Vector(64, u8) = @bitCast(
                (((@as(@Vector(32, u16), @bitCast(identifier_lens)) << @splat(14)) ^ first_two_chars) *% last_two_chars) >> @splat(8),
            );

            @setEvalBranchQuota(100000);
            const super_string_starts, const super_string_lens =
                comptime std.simd.deinterlace(2, @as([256]u8, @bitCast(Keywords.kw_slices_raw)));

            var super_looked_up_starts = vperm2(super_string_starts, hash);
            var super_looked_up_lens = vperm2(super_string_lens, hash);

            // Written in a weird way because of https://github.com/llvm/llvm-project/issues/111431
            var matched1: u8 = undefined;
            var matched2: u16 = undefined;
            var matched: u32 = undefined;

            // We can match 8 keywords at a time, because that's how many subdivisions we get in a 64-byte vector.
            // I.e. the maximum granularity of a compare-equal is 8 bytes, and we can fit 8 of those in a 64-byte vector.
            // There are at most 22 multi-character identifiers in a chunk, including a carry-over from the previous chunk
            // i|f(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if(if
            //  ^ break between previous chunk and current chunk
            const REQUIRED_ITERATIONS = divCeil(22, 8);

            outer: inline for (0..REQUIRED_ITERATIONS) |i| {
                const first8Interlacedbroadcast = (std.simd.iota(u8, 64) >> @splat(3) << @splat(1));

                const lookup_starts_group = @shuffle(u8, super_looked_up_starts, undefined, first8Interlacedbroadcast);
                const lookup_lens_group = @shuffle(u8, super_looked_up_lens, undefined, first8Interlacedbroadcast);

                const starts_group = @shuffle(u8, start_indices, undefined, first8Interlacedbroadcast);
                const lens_group = @shuffle(u8, identifier_lens, undefined, first8Interlacedbroadcast);

                var cur_matched: u8 = std.math.maxInt(u8);

                inline for ([2]@Vector(64, u8){
                    comptime std.simd.repeat(64, std.simd.iota(u8, 8)),
                    comptime std.simd.repeat(64, std.simd.iota(u8, 8) + @as(@Vector(8, u8), @splat(8))),
                }) |indices| {
                    const keyword_candidates_from_chunk_half = @select(
                        u8,
                        indices < lens_group,
                        vperm2(std.simd.join(contextless_chunk, prev_chunk), starts_group +% indices),
                        lens_group,
                    );

                    // std.debug.print("keyword_candidates_from_chunk_half: {c}\n", .{@select(
                    //     u8,
                    //     keyword_candidates_from_chunk_half < @as(@TypeOf(keyword_candidates_from_chunk_half), @splat(16)),
                    //     keyword_candidates_from_chunk_half + @as(@TypeOf(keyword_candidates_from_chunk_half), @splat('0')),
                    //     keyword_candidates_from_chunk_half,
                    // )});

                    const keywords_from_table2_half = @select(
                        u8,
                        indices < lookup_lens_group,
                        vperm4(
                            comptime Keywords.kw_buffer[0..Keywords.kw_buffer.len].* ++ [1]u8{0} ** (256 - Keywords.kw_buffer.len),
                            lookup_starts_group +% indices,
                        ),
                        lookup_lens_group,
                    );

                    // std.debug.print("keywords_from_table2_half: {c}\n", .{@select(
                    //     u8,
                    //     keywords_from_table2_half < @as(@TypeOf(keywords_from_table2_half), @splat(16)),
                    //     keywords_from_table2_half + @as(@TypeOf(keywords_from_table2_half), @splat('0')),
                    //     keywords_from_table2_half,
                    // )});

                    cur_matched &= @bitCast(
                        @as(@Vector(8, u64), @bitCast(keyword_candidates_from_chunk_half)) ==
                            @as(@Vector(8, u64), @bitCast(keywords_from_table2_half)),
                    );
                }

                if (i == 0) {
                    if (@popCount(alphanum_starts_larger_than_1) <= 8) {
                        @branchHint(.likely);
                        matched = cur_matched;
                        break :outer;
                    }
                }

                // Written in a weird way because of https://github.com/llvm/llvm-project/issues/111431
                switch (i) {
                    0 => matched1 = cur_matched,
                    1 => matched2 = @as(u16, @bitCast(std.simd.join(@as(@Vector(8, bool), @bitCast(matched1)), @as(@Vector(8, bool), @bitCast(cur_matched))))),
                    2 => matched = @as(u32, @bitCast(std.simd.join(@as(@Vector(16, bool), @bitCast(matched2)), std.simd.join(@as(@Vector(8, bool), @bitCast(cur_matched)), @as(@Vector(8, bool), @splat(false)))))),
                    else => unreachable,
                }

                start_indices = std.simd.shiftElementsLeft(start_indices, 16, 0);
                super_looked_up_starts = std.simd.shiftElementsLeft(super_looked_up_starts, 16, 0);
                identifier_lens = std.simd.shiftElementsLeft(identifier_lens, 16, 0);
                super_looked_up_lens = std.simd.shiftElementsLeft(super_looked_up_lens, 16, 0);
            }

            // The hash is 7 bits, but in our vector each one still occupies 2 bytes of space.
            // Reduce it to a single byte.
            // Set the highest bit of each byte because that is how we fit the keywords into the `Tag` enum.
            const potential_kinds = @shuffle(u8, hash, undefined, std.simd.iota(u8, 32) << @splat(1)) | @as(@Vector(32, u8), @splat(128));

            // inline for (0..32) |i|
            //     std.debug.print("{s} ", .{if (std.meta.intToEnum(Tag, potential_kinds[i])) |t| @tagName(t) else |_| "_"});
            // std.debug.print("\n", .{});

            const matched_kinds = @select(
                u8,
                unmovemask32(matched),
                potential_kinds,
                @as(@Vector(32, u8), @splat(@intFromEnum(Tag.identifier))),
            );

            kinds.* = expand(matched_kinds, kinds.*, alphanum_starts_larger_than_1);

            // const keyword_starts = pdep(matched, alphanum_starts_larger_than_1);
            // const keyword_ends = pdep(matched, alphanum_ends_larger_than_1);

            // return .{ keyword_starts, keyword_ends };
        }

        fn getMultiCharKinds(
            prev_chunk: @Vector(64, u8),
            chunk: @Vector(64, u8),
            multi_char_end_indices: @Vector(16, u8),
            which_multi_char_symbols_are_triple_chars_compacted: @Vector(16, bool),
        ) @Vector(16, u8) {
            // std.debug.print("multi_char_end_indices: {}\n", .{multi_char_end_indices});

            const adjustment_vec: @Vector(64, u8) = @bitCast(@select(
                u32,
                which_multi_char_symbols_are_triple_chars_compacted,
                [_]u32{@bitCast([4]u8{ 2, 1, 0, 0x80 })} ** 16, // 3 characters, zero the uppermost byte
                [_]u32{@bitCast([4]u8{ 1, 0, 0x80, 0x80 })} ** 16, // 2 characters, zero the 2 upper bytes
            ));

            const successive_indices = @shuffle(u8, multi_char_end_indices, undefined, [_]i32{
                0,  0,  0,  0,
                1,  1,  1,  1,
                2,  2,  2,  2,
                3,  3,  3,  3,

                4,  4,  4,  4,
                5,  5,  5,  5,
                6,  6,  6,  6,
                7,  7,  7,  7,

                8,  8,  8,  8,
                9,  9,  9,  9,
                10, 10, 10, 10,
                11, 11, 11, 11,

                12, 12, 12, 12,
                13, 13, 13, 13,
                14, 14, 14, 14,
                15, 15, 15, 15,
            }) -% adjustment_vec;

            // std.debug.print("successive_indices: {}\n", .{successive_indices & @as(@TypeOf(successive_indices), @splat(0b0111_1111))});
            // std.debug.print("adjustment_vec: {}\n", .{adjustment_vec});

            // std.debug.print("prev_chunk:\n", .{});
            // printStr(@as([64]u8, @bitCast(prev_chunk))[0..64]);
            // std.debug.print("chunk:\n", .{});
            // printStr(@as([64]u8, @bitCast(chunk))[0..64]);

            const multi_char_symbol_strs_zero_padded = vperm2_zmask(
                std.simd.join(chunk, prev_chunk),
                successive_indices,
                @bitCast(adjustment_vec < @as(@TypeOf(adjustment_vec), @splat(0x80))),
            );

            // std.debug.print("multi_char_symbol_strs_zero_padded: {c}\n", .{multi_char_symbol_strs_zero_padded});

            var hash: @Vector(16, u7) = @truncate(
                @as(@Vector(16, u32), @bitCast(multi_char_symbol_strs_zero_padded)) *% ([1]u32{698839068} ** 16) >> @splat(25),
            );
            // It would be cool if we changed our hash function that didn't require these post-processing steps.
            // Everytime we do a 7-bit addtion or subtraction, there is implicitly a bitwise AND that must be inserted.
            // This hash function is from the old version, which had more inputs because it had to accept single-character symbols.
            // We could probably find a better one
            hash -%= @as(@TypeOf(hash), @splat(29));
            hash = @select(u7, hash < @as(@TypeOf(hash), @splat(16)), hash +% @as(@TypeOf(hash), @splat(18)), hash);
            return hash;
        }

        // Caller owns memory
        fn tokenize(
            gpa: Allocator,
            /// This is align(64) because we always want it to be safe (not crossing a page boundary) when loading a 64 byte chunk that
            /// may be past the last byte (according to `source.len`) on the last chunk.
            /// The other way to provide safety would be to guarantee that loading all 64 byte chunks
            /// is safe by allocating at least `std.mem.alignForward(usize, source.len, 64)` bytes.
            source: []align(64) const u8,
        ) ![]Token {
            // Allocate enough space such that:
            //  1. Each character could become a token, and
            //  2. that we can write out a full 64 tokens safely at any point
            const tokens = try gpa.alloc(Token, source.len + 64);
            errdefer gpa.free(tokens);

            var cur_token = tokens;

            var multi_char_symbol_parser: MultiCharSymbolTokenizer = .{};
            var carry: Carry = .{};
            var utf8_checker: Utf8Checker(.{
                .USE_ARM_NEON = HAS_ARM_NEON,
                .V = Chunk,
                .native_int = NATIVE_VEC_INT,
            }) = .empty;

            // if (comptime builtin.mode == .Debug) {
            //     const str =
            //         \\ "" // hello
            //     ++ " " ++
            //         \\
            //         \\
            //         \\
            //         \\ // hello
            //         \\  comptime  var  "/'"  ++  if
            //         \\  ++// good
            //         \\ "me"  // m
            //         \\
            //         \\  //m
            //         \\
            //         \\  // m
            //         \\  ++
            //         \\ e  // soup
            //         \\
            //         \\
            //         \\"\\\"\\; //\"'\\\"\"\\\"///'\"  \\ \"\\\"\\"; "'\\"
            //         \\
            //         \\//=///' %" "\\\"\\";
            //         \\
            //         \\//"'\\""\\"///'" +$ "\\\"\\";//"'\\""\\"
            //         \\/
            //     ;
            //     @constCast(source.ptr)[0..str.len].* = str.*;
            // }

            // if (comptime builtin.mode == .Debug) {
            //     const str =
            //         \\ "" // hello
            //     ++ " " ++
            //         \\
            //         \\
            //         \\
            //         \\ // hello
            //         \\  comptime  var // hello
            //         \\
            //         \\
            //         \\ "/'"  ++  if
            //         \\  ++// good
            //         \\ "me"  // m
            //         \\
            //         \\  //m
            //         \\
            //         \\  // m
            //         \\  ++
            //         \\ e  // soup
            //         \\
            //         \\
            //         \\"\\\"\\; //\"'\\\"\"\\\"///'\"  \\ \"\\\"\\"; "'\\"
            //         \\
            //         \\//=///' %" "\\\"\\";
            //         \\
            //         \\//"'\\""\\"///'" +$ "\\\"\\";//"'\\""\\"
            //         \\/
            //     ;
            //     @constCast(source.ptr)[0..str.len].* = str.*;
            // }

            // if (comptime builtin.mode == .Debug) {
            //     const str =
            //         // \\var prefix_sums = comptime std.simd.shiftElementsRight(std.simd.
            //         \\  comptime  var  "/'"  ++  if
            //         \\  ++// good
            //         \\ "me" 1e14 // m
            //         \\
            //         \\
            //         \\
            //         \\
            //         \\
            //         \\
            //         \\
            //         \\  +
            //         \\
            //         \\
            //         \\ // m
            //         \\  ++
            //         \\ e  // soup
            //         \\
            //         \\
            //         \\"\\\"\\; //\"'\\\"\"\\\"///'\"  \\ \"\\\"\\"; "'\\"
            //         \\
            //         \\//=///' %" "\\\"\\";
            //         \\
            //         \\//"'\\""\\"///'" +$ "\\\"\\";//"'\\""\\"
            //         \\/
            //     ;
            //     @constCast(source.ptr)[0..str.len].* = str.*;
            // }

            // var super_duper_pls_delete_me_iter: u64 = 0;
            // Preserve //! and /// comments. "////" counts as a regular comment
            // Merge // comments and whitespace together.
            // Merge operators and keywords with nearby whitespace and // comments.
            // Deal with @ symbol
            // Make sure things work properly across chunks
            // @constCast(sourcey.allocatedSlice().ptr)[23..][0..41].* = "defghij\"lmnopqrs\" vwxyzABCDEFGHIJKLMNOPQR".*;

            var cur = source;

            // var chunk_ptr = source.ptr[0..];
            // const final_chunk_ptr = source.ptr + source.len;

            var prev_carried_len: u32 = 0;
            var prev_chunk: @Vector(64, u8) = @splat(0);

            while (true) {
                // Tell our cache to evict the chunk next time room is needed
                // We can do this because we do not need to access data from this chunk again.
                // This helps us not clutter up the cache unnecessarily.
                @prefetch(cur.ptr[0..64], .{ .locality = 0 });

                const V = @Vector(64, u8);

                const before_eofs = before_eofs: {
                    // https://github.com/llvm/llvm-project/issues/132714
                    const chunk_mask = bzhi(~@as(u64, 0), cur.len);
                    break :before_eofs if (cur.len >= 64) ~@as(u64, 0) else chunk_mask;
                };

                const chunk: V = @select(u8, @as(@Vector(64, bool), @bitCast(before_eofs)), cur.ptr[0..64].*, [1]u8{0} ** 64);
                defer prev_chunk = chunk;

                printb(before_eofs, "before_eofs");
                printStr(@as([64]u8, @bitCast(chunk))[0..64]);

                // Bitstrings that mark where each token begins and ends
                var all_starts: u64 = 0;
                var all_ends: u64 = 0;

                // zig fmt: off
                const carriages  : u64 = @bitCast(chunk == @as(V, @splat('\r')));
                const newlines   : u64 = @bitCast(chunk == @as(V, @splat('\n')));
                const slashes    : u64 = @bitCast(chunk == @as(V, @splat('/')));
                const backslashes: u64 = @bitCast(chunk == @as(V, @splat('\\')));
                const nonASCII   : u64 = @bitCast(chunk >= @as(V, @splat(0x80)));
                const quotes     : u64 = @bitCast(chunk == @as(V, @splat('"')));
                const apostrophes: u64 = @bitCast(chunk == @as(V, @splat('\'')));
                // zig fmt: on

                defer carry.set(.slashes, slashes);
                defer carry.set(.backslashes, backslashes);
                defer carry.set(.carriages, carriages);

                // '\r' is only valid when there is a '\n' immediately after.
                const bad_carriage_returns = ~newlines & ((carriages << 1) | carry.get(.carriages));

                if (bad_carriage_returns != 0) {
                    @branchHint(.unlikely);
                    return error.BadCarriageReturn;
                }

                // We intentionally shift right here (i.e. backwards in the source file) because
                // we want these to start on the first character instead of the second.
                const double_slash_starts = slashes & ((slashes >> 1) | carry.get(.slashes));
                const double_backslash_starts = backslashes & ((backslashes >> 1) | carry.get(.backslashes));

                const comment_bounds_incl_carry = double_slash_starts | carry.get(.inside_comments);
                const line_string_bounds_incl_carry = double_backslash_starts | carry.get(.inside_line_strings);

                const escaped_positions = getEscapedPositions(&carry, backslashes);

                const unescaped_quotes = quotes & ~escaped_positions;
                const unescaped_apostrophes = apostrophes & ~escaped_positions;

                const quote_bounds_incl_carry = unescaped_quotes | carry.get(.inside_quotes_incl_start_and_carry);
                const apostrophe_bounds_incl_carry = unescaped_apostrophes | carry.get(.inside_apostrophes_incl_start_and_carry);

                // Figure out which characters are inside a quote, comment, line-string, or, to a lesser extent, a character literal.
                // The fundamental problem is that any of these could be arbitrarily nested inside of each other, meaning that a
                // 100% parallel tokenize is probably not going to be developed.
                // We could, however, operate in parallel on comments+line-strings because they both end on a '\n' (using pext & pdep).

                // Algorithm: Create an `iter` bitstring that has 1 bits as cursors at the first non-newline in each group of non-newlines (henceforth called a "line")
                // Take the OR of all the starting positions of { quotes, apostrophes, comments, line_strings }, call it `all_bounds_incl_carry`
                // For each bit in OR, let it heat-seek/advance to the next bit in `all_bounds_incl_carry` which comes after its position, stopping at newlines too.
                // This will gives us a bitstring that tells us, for each line, where the first comment/quote/char-literal/line-string begins.
                // Then we want to advance to the end of whatever we found in each position. If we found a string, we should stop at the next quote
                // character. If we found a character literal, we stop at this next apostrophe. Everything else, of course, stops at a newline, and it
                // is an error if the quotes/apostrophes terminate at a newline.
                // To accomplish this, we replace the bits from the starting position to the end of the line with whatever we hoped to find.
                // Then we advance the cursors forward, hopefully matching the right characters, hitting newlines otherwise.
                // By tracking all of our starting and ending positions, we have computed which pieces are nested within each other.
                // E.g.              `"// is this a comment?"` would result in:
                //       all_starts:  1......................
                //         all_ends:  ......................1
                // With this information, we can tell that `// is this a comment?` is, in fact, not a comment.
                // Note: We do not throw an error for a quote or character literal that is missing a terminator. That should happen in Sema.zig
                // Note: We assume there is a newline at the end of the file.

                // printb(newlines_or_eof, "newlines_or_eof");

                // 1111111111..........
                // x & ~(x + 1)

                var iter = (~newlines & ~(~newlines << 1)); // | is_first_char_inside_string_or_comment;
                var all_bounds_incl_carry = quote_bounds_incl_carry | apostrophe_bounds_incl_carry | comment_bounds_incl_carry | line_string_bounds_incl_carry;
                printb(quote_bounds_incl_carry, "quote_bounds_incl_carry");
                printb(apostrophe_bounds_incl_carry, "apostrophe_bounds_incl_carry");
                printb(comment_bounds_incl_carry, "comment_bounds_incl_carry");
                printb(line_string_bounds_incl_carry, "line_string_bounds_incl_carry");
                const newlines_or_eof = newlines | ~before_eofs;

                // We might have the end of some string/char_literal at the beginning of the chunk,
                // and in this case we need the (pseudo) start and end character to be the same.
                // That's why we handle this separately from the loop that follows, which only permits
                // starts to be different from ends.
                const first_char_ends_quote_or_apostrophe =
                    (unescaped_quotes & carry.get(.inside_quotes_incl_start_and_carry)) |
                    (unescaped_apostrophes & carry.get(.inside_apostrophes_incl_start_and_carry));

                all_starts |= first_char_ends_quote_or_apostrophe;
                all_ends |= first_char_ends_quote_or_apostrophe;
                all_bounds_incl_carry ^= first_char_ends_quote_or_apostrophe;

                while (true) {
                    printb(iter, "iter");
                    printb(all_bounds_incl_carry, "all_bounds_incl_carry");
                    // Advance each cursor in `iter` to next 1 bit in `all_bounds_incl_carry | newlines`, but don't count `newlines` as a valid start
                    const starts = (all_bounds_incl_carry & ~((all_bounds_incl_carry | newlines_or_eof) -% iter));
                    printb(starts, "starts");
                    all_starts |= starts;

                    // From each bit in `starts` to the next newline, fold in the pieces of the target bitstring
                    // that contains the end-characters we want to match against.
                    // E.g. "...\n
                    //      ^ we start a quote here
                    // E.g. "...\n
                    //       ^^^ so we put the characters from the `quote_bounds_incl_carry` bitstring in there
                    var interleaved = newlines_or_eof;
                    interleaved |= quote_bounds_incl_carry & (newlines_or_eof -% (starts & quote_bounds_incl_carry));
                    interleaved |= apostrophe_bounds_incl_carry & (newlines_or_eof -% (starts & apostrophe_bounds_incl_carry));
                    interleaved &= ~starts;
                    // interleaved |= (unescaped & ~(newlines_or_eof -% andn(starts, unescaped)));

                    printb(interleaved, "interleaved");

                    // Advance cursor within interleaved bitstring
                    const cur_ends = interleaved & ~(interleaved -% starts);
                    all_ends |= cur_ends;
                    printb(cur_ends, "cur_ends");

                    // Make cur_ends the new iters, but if we ended on a newline, delete that bit.
                    iter = cur_ends & ~newlines_or_eof;
                    assert((all_bounds_incl_carry & iter) == iter);
                    all_bounds_incl_carry ^= iter;

                    printb(iter, "iter");
                    printb(all_bounds_incl_carry, "all_bounds_incl_carry");

                    if (iter == 0) {
                        @branchHint(.likely);
                        break;
                    }
                }

                const inside_strings_and_comments_including_start = all_ends -% all_starts;
                defer carry.set(.inside_strings_and_comments_including_start, inside_strings_and_comments_including_start);
                printb(all_starts, "all_starts");
                printb(all_ends, "all_ends");
                printb(inside_strings_and_comments_including_start, "inside_strings_and_comments_including_start");

                const is_first_char_inside_string_or_comment = carry.get(.inside_strings_and_comments_including_start) & ~newlines & ~first_char_ends_quote_or_apostrophe;

                // Find out which characters are inside a string/char-literal/comment.
                // We need the carry because all_starts might include the first character as a pseudo-start.
                const inside_strings_or_comments = (inside_strings_and_comments_including_start & ~all_starts) | is_first_char_inside_string_or_comment;
                printb(inside_strings_or_comments, "inside_strings_or_comments");
                // defer carry.set(.inside_strings_and_comments_including_start, inside_strings_or_comments);

                // carry.get(.inside_quotes_incl_start_and_carry);
                // carry.get(.inside_apostrophes_incl_start_and_carry);

                // alternative idea:
                // const inside_quotes_incl_start_and_carry = inside_strings_or_comments & ~(inside_strings_and_comments_including_start +% quote_starts_incl_carry);

                printb(all_ends -% (all_starts & apostrophe_bounds_incl_carry), "all_ends -% (all_starts & apostrophe_bounds_incl_carry)");
                carry.set(.inside_quotes_incl_start_and_carry, (all_ends -% (all_starts & quote_bounds_incl_carry)) & ~all_ends);
                carry.set(.inside_apostrophes_incl_start_and_carry, (all_ends -% (all_starts & apostrophe_bounds_incl_carry)) & ~all_ends);
                carry.set(.inside_comments, (all_ends -% (all_starts & comment_bounds_incl_carry)) & ~all_ends);
                carry.set(.inside_line_strings, (all_ends -% (all_starts & line_string_bounds_incl_carry)) & ~all_ends);

                printStr(@as([64]u8, @bitCast(chunk))[0..64]);
                // const inside_comment_or_line_string = inside_strings_or_comments & ~disjoint_or(inside_quotes_incl_start_and_carry, inside_apostrophes_incl_start_and_carry);
                // _ = inside_comment_or_line_string;

                printb(all_starts, "[0] all_starts");
                printb(all_ends, "[0] all_ends");

                // TODO: this may or may not be broken by //|\n
                //                                          ^ chunk break

                // Revise `all_ends` such that tokens do not include the newline_or_eof
                all_ends = (all_ends & ~newlines_or_eof) | ((all_ends & newlines_or_eof) >> 1);
                printb(all_ends, "[a] all_ends");

                const no_ends = all_ends == 0;

                // Let us set a dummy end if we ended inside a string/comment/line_string/char_literal
                all_ends = disjoint_or(all_ends, inside_strings_and_comments_including_start & (@as(u64, 1) << 63));
                printb(all_ends, "[b] all_ends");

                // inline for (0..128) |i| {
                //     std.debug.print("{d: >2} '{c}' {} {s}\n", .{
                //         i,
                //         @as(u8, i),
                //         classifier[i],
                //         if (std.meta.intToEnum(Tag, classifier[i])) |e| @tagName(e) else |_| "n/a",
                //     });
                // }

                var classified_chunk = if (CLASSIFIER_STRATEGY == .global)
                    vperm2(classifier, chunk);

                if (CLASSIFIER_STRATEGY == .global)
                    classified_chunk = @select(u8, unmovemask64(inside_strings_or_comments), @as(V, @splat(0)), classified_chunk);

                // We can do Vectorized Classification with just `intersect_byte_halves(upper_nibbles, vpshufb(table, contextless_chunk))`
                // https://validark.github.io/posts/eine-kleine-vectorized-classification/
                var upper_nibbles = if (CLASSIFIER_STRATEGY == .on_demand) blk: {
                    comptime var powers_of_2_up_to_128: [16]u8 = undefined;
                    inline for (&powers_of_2_up_to_128, 0..) |*slot, i| slot.* = if (i < 8) @as(u8, 1) << i else 0xFF;
                    break :blk vpshufb(powers_of_2_up_to_128, chunk >> @splat(4));
                };

                if (CLASSIFIER_STRATEGY == .on_demand)
                    upper_nibbles = @select(u8, unmovemask64(inside_strings_or_comments), @as(V, @splat(0)), upper_nibbles);

                var contextless_chunk = if (CLASSIFIER_STRATEGY == .on_demand or CLASSIFIER_STRATEGY == .none)
                    chunk;

                if (CLASSIFIER_STRATEGY == .on_demand or CLASSIFIER_STRATEGY == .none)
                    contextless_chunk = @select(u8, unmovemask64(inside_strings_or_comments), @as(V, @splat(0)), contextless_chunk);

                if (nonASCII != 0) {
                    @branchHint(.unlikely);

                    inline for (0..@sizeOf(@TypeOf(chunk)) / @sizeOf(Chunk)) |i| {
                        try utf8_checker.validateChunk(std.simd.extract(chunk, @sizeOf(Chunk) * i, @sizeOf(Chunk)));
                    }

                    switch (CLASSIFIER_STRATEGY) {
                        .global => classified_chunk = @select(u8, unmovemask64(nonASCII), @as(V, @splat(0)), classified_chunk),
                        .on_demand => {
                            const nonASCII_bool_vec = unmovemask64(nonASCII);
                            upper_nibbles = @select(u8, nonASCII_bool_vec, @as(V, @splat(0)), upper_nibbles);
                            contextless_chunk = @select(u8, nonASCII_bool_vec, @as(V, @splat(0)), contextless_chunk);
                        },
                        .none => contextless_chunk = @select(u8, unmovemask64(nonASCII), @as(V, @splat(0)), contextless_chunk),
                    }
                }

                // printb(all_starts, "quote or charlit or linestring or comment starts");
                // _ = char_lit_starts;
                // _ = line_string_starts_incl_carry;

                const standalone_symbols: u64 = switch (CLASSIFIER_STRATEGY) {
                    .global => @bitCast(classified_chunk >= @as(V, @splat(last_exclusively_in_single_char_symbol))),
                    .on_demand => blk: {
                        comptime var single_char_ops_map: @Vector(16, u8) = @splat(0);
                        inline for (Operators.single_char_ops) |c| single_char_ops_map[c & 0xF] |= 1 << (c >> 4);
                        break :blk intersect_byte_halves(upper_nibbles, vpshufb(single_char_ops_map, contextless_chunk));
                    },
                    .none => blk: {
                        var standalone_symbols: V = @splat(0);

                        inline for (Operators.single_char_ops) |c| {
                            standalone_symbols |= @select(
                                u8,
                                contextless_chunk == @as(V, @splat(c)),
                                @as(V, @splat(0xFF)),
                                @as(V, @splat(0)),
                            );
                        }

                        break :blk @bitCast(standalone_symbols == @as(V, @splat(0xFF)));
                    },
                };

                const digits: u64 = switch (CLASSIFIER_STRATEGY) {
                    .global => @bitCast(classified_chunk == @as(V, @splat(@intFromEnum(CharClasses.digit)))),
                    // .on_demand => @bitCast(vpshufb(
                    //     [16]u8{ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 0, 0, 0, 0, 0, 0 },
                    //     contextless_chunk,
                    // ) == contextless_chunk),
                    .on_demand, .none => ( // zig fmt: off
                        @as(u64, @bitCast(@as(V, @splat('0')) <= contextless_chunk)) &
                        @as(u64, @bitCast(contextless_chunk <= @as(V, @splat('9'))))
                    ),
                    // zig fmt: on
                };

                const ats: u64 = switch (CLASSIFIER_STRATEGY) {
                    .global => @bitCast(classified_chunk == @as(V, @splat(@intFromEnum(CharClasses.at)))),
                    else => @bitCast(contextless_chunk == @as(V, @splat('@'))),
                };
                defer carry.set(.ats, ats);

                const whitespaces: u64 = switch (CLASSIFIER_STRATEGY) {
                    .global => @bitCast(classified_chunk == @as(V, @splat(@intFromEnum(CharClasses.whitespace)))),
                    .on_demand => @bitCast(vpshufb([16]u8{ ' ', 0, 0, 0, 0, 0, 0, 0, 0, '\t', '\n', 0, 0, '\r', 0, 0 }, contextless_chunk) == contextless_chunk),
                    .none => @as(u64, @bitCast(contextless_chunk == @as(V, @splat(' ')))) |
                        @as(u64, @bitCast(contextless_chunk == @as(V, @splat('\t')))) |
                        carriages |
                        newlines,
                };

                const alpha_underscores: u64 = switch (CLASSIFIER_STRATEGY) {
                    .global => @bitCast(classified_chunk == @as(V, @splat(@intFromEnum(CharClasses.alpha_underscore)))),
                    .on_demand => blk: {
                        comptime var alpha_ops_high_nibble_given_low: @Vector(16, u8) = @splat(0);
                        comptime for (0..128) |c| {
                            switch (c) {
                                'a'...'z', 'A'...'Z', '_' => alpha_ops_high_nibble_given_low[c & 0xF] |= 1 << (c >> 4),
                                else => {},
                            }
                        };
                        break :blk intersect_byte_halves(upper_nibbles, vpshufb(alpha_ops_high_nibble_given_low, contextless_chunk));
                    },
                    .none => (@as(u64, @bitCast(@as(V, @splat('a')) <= contextless_chunk)) & @as(u64, @bitCast(contextless_chunk <= @as(V, @splat('z'))))) |
                        (@as(u64, @bitCast(@as(V, @splat('A')) <= contextless_chunk)) & @as(u64, @bitCast(contextless_chunk <= @as(V, @splat('Z'))))) |
                        @as(u64, @bitCast(contextless_chunk == @as(V, @splat('_')))),
                };

                const alpha_numeric_underscores: u64 = switch (CLASSIFIER_STRATEGY) {
                    .global => // zig fmt: off
                        // The CharClasses are laid out in such a way that this compiles down to a single instruction on AVX-512.
                        @as(u64, @bitCast(classified_chunk == @as(V, @splat(@intFromEnum(CharClasses.alpha_underscore))))) |
                        @as(u64, @bitCast(classified_chunk == @as(V, @splat(@intFromEnum(CharClasses.digit))))),
                    // zig fmt: on
                    .on_demand, .none => alpha_underscores | digits,
                };
                defer carry.set(.alpha_numeric_underscores, alpha_numeric_underscores);

                // quote_starts_incl_carry, char_lit_starts, line_string_starts_incl_carry, comment_starts_incl_carry, classified_chunk, upper_nibbles, contextless_chunk, newlines, carriages, ats, all_starts, all_ends, standalone_symbols, digits, whitespaces, alpha_underscores, alpha_numeric_underscores

                // TODO: make sure that a @ at the end of a chunk can still register as a builtin
                // kinds = @select(u8, unmovemask64((carry.get(.ats) & alpha_numeric_underscores) | ((alpha_numeric_underscores >> 1) & ats)), @as(V, @splat(@intFromEnum(Tag.builtin))), kinds);

                const builtin_starts = ats & (alpha_underscores >> 1);
                all_starts |= builtin_starts;

                const identifier_or_number_or_builtin_ends = alpha_numeric_underscores & ~(alpha_numeric_underscores >> 1);
                // printb(identifier_or_number_or_builtin_ends, "identifier_or_number_or_builtin_ends");
                defer carry.set(.identifier_or_number_or_builtin_ends, identifier_or_number_or_builtin_ends);
                all_ends |= identifier_or_number_or_builtin_ends;
                printb(identifier_or_number_or_builtin_ends, "identifier_or_number_or_builtin_ends!!!!");
                printb(all_ends, "[c] all_ends");

                // Figure out where the @"" are. It's okay to use `unescaped_quotes` because `ats` is not in a string/comment/character literal.
                const string_identifier_starts = ats & (unescaped_quotes >> 1);
                all_starts |= string_identifier_starts;
                // Delete the "" start indicator for @"".
                all_starts ^= string_identifier_starts << 1; // TODO: delete both the ident and this at once?
                // all_starts &= ~(quote_starts_incl_carry & follows_at);

                // const follows_alpha_numeric_underscore = (alpha_numeric_underscores << 1) | carry.get(.alpha_numeric_underscores);

                const follows_at = (ats << 1); // | carry.get(.ats);

                const identifier_or_number_starts = alpha_numeric_underscores & ~(alpha_numeric_underscores << 1) & ~follows_at;
                all_starts |= identifier_or_number_starts;
                // printb(identifier_or_number_starts, "identifier_or_number_starts");

                const identifier_starts = alpha_underscores & ~(alpha_numeric_underscores << 1) & ~follows_at;

                const identifier_ends =
                    identifier_or_number_or_builtin_ends & ~(identifier_or_number_or_builtin_ends -% identifier_starts);
                defer carry.set(.identifier_ends, identifier_ends);

                const number_or_builtin_ends = identifier_or_number_or_builtin_ends ^ identifier_ends;
                // printb(number_or_builtin_ends, "number_or_builtin_ends");
                defer carry.set(.number_or_builtin_ends, number_or_builtin_ends);

                var single_char_ends,
                // These may indicate the first char of a chunk, when a multi-char symbol started in the previous chunk
                const double_char_ends, const triple_char_ends = multi_char_symbol_parser.getMultiCharEndPositions(
                    &carry,
                    if (CLASSIFIER_STRATEGY == .global) classified_chunk else contextless_chunk,
                );

                // It's okay to use `comment_bounds_incl_carry` because `single_char_ends` is guaranteed not to be in a string/comment/char-literal
                single_char_ends &= ~comment_bounds_incl_carry;
                single_char_ends |= standalone_symbols;

                const multi_char_symbol_ends = double_char_ends | triple_char_ends;
                const all_symbol_ends = single_char_ends | multi_char_symbol_ends;

                const triple_char_starts = (triple_char_ends >> 2);
                const multi_char_symbol_starts = (double_char_ends >> 1) | triple_char_starts;
                const all_symbol_starts = single_char_ends | multi_char_symbol_starts;
                // printb(all_symbol_starts, "all_symbol_starts");
                all_starts |= all_symbol_starts;
                // printb(all_symbol_ends, "all_symbol_ends");
                all_ends |= all_symbol_ends;
                printb(all_ends, "[d] all_ends");

                // Maximally, we can have 32 2 character symbols in a 64-byte chunk.
                const multi_char_end_indices = std.simd.extract(bitsToIndices(multi_char_symbol_ends, 0), 0, 32);
                const which_multi_char_symbols_are_triple_chars_compacted = pext(triple_char_ends, multi_char_symbol_ends);

                const multi_char_kinds_compressed: @Vector(32, u8) = std.simd.join(
                    getMultiCharKinds(
                        prev_chunk,
                        chunk,
                        std.simd.extract(multi_char_end_indices, 0, 16),
                        @bitCast(@as(u16, @truncate(which_multi_char_symbols_are_triple_chars_compacted))),
                    ),

                    if (@popCount(multi_char_symbol_starts) < 16)
                        @as(@Vector(16, u8), undefined)
                    else blk: {
                        @branchHint(.unlikely);
                        break :blk getMultiCharKinds(
                            @splat(0),
                            chunk,
                            std.simd.extract(multi_char_end_indices, 16, 16),
                            @bitCast(@as(u16, @truncate(which_multi_char_symbols_are_triple_chars_compacted >> 16))),
                        );
                    },
                );

                const whitespace_starts = whitespaces & ~(whitespaces << 1);
                const whitespace_ends = whitespaces & ~(whitespaces >> 1);

                if (EMIT_WHITESPACE_AND_COMMENTS == .yes) {
                    all_starts |= whitespace_starts;
                    all_ends |= whitespace_ends;
                }

                // const multi_char_symbol_indices = bitsToIndices((double_char_ends >> 1) | double_char_ends, 0);

                const multi_char_has_one_char_in_prev_chunk: u1 = @truncate(double_char_ends | (triple_char_ends >> 1));
                const multi_char_has_two_chars_in_prev_chunk: u1 = @truncate(triple_char_ends);
                const multi_char_starts_before_chunk: u1 = @truncate(multi_char_has_one_char_in_prev_chunk | multi_char_has_two_chars_in_prev_chunk);

                all_starts |= multi_char_starts_before_chunk;

                const number_or_builtin_started_before_chunk: u1 = @truncate(carry.get(.number_or_builtin_ends) & identifier_or_number_starts);
                printb(number_or_builtin_started_before_chunk, "number_or_builtin_started_before_chunk");
                // It's okay to use `unescaped_quotes` because the ats cannot be in a string or comment or character literal.
                const string_identifier_started_before_chunk: u1 = @truncate(unescaped_quotes & carry.get(.ats));
                const builtin_started_before_chunk: u1 = @truncate(alpha_underscores & carry.get(.ats));
                printb(builtin_started_before_chunk, "builtin_started_before_chunk $$$$$$$$$$$");
                const identifier_started_before_chunk: u1 = @truncate(alpha_numeric_underscores & carry.get(.identifier_ends));

                //
                const is_comment_opener_split_between_chunks: u1 = @truncate(all_starts & slashes & carry.get(.slashes));
                const is_line_string_opener_split_between_chunks: u1 = @truncate(all_starts & backslashes & carry.get(.backslashes));

                // We have separate codepaths for handling potential keywords vs long identifiers because potential keywords should have
                // a length that can be stored in one byte whereas for arbitrarily long identifiers this is not the case.
                comptime for (Keywords.unpadded_kws) |kw| assert(kw.len < 15);
                const is_prev_carried_len_short_enough_for_kw: u1 = @intFromBool(prev_carried_len < 15);
                const potential_kw_started_before_chunk: u1 = is_prev_carried_len_short_enough_for_kw & identifier_started_before_chunk;

                // We have two different models of rewriting the previous token.

                // if a builtin or a number or an identifier that is too long
                const first_token_starts_before_chunk: u1 = @truncate(is_first_char_inside_string_or_comment |
                    first_char_ends_quote_or_apostrophe |
                    string_identifier_started_before_chunk |
                    builtin_started_before_chunk |
                    (~is_prev_carried_len_short_enough_for_kw & identifier_started_before_chunk) |
                    number_or_builtin_started_before_chunk |
                    is_comment_opener_split_between_chunks |
                    is_line_string_opener_split_between_chunks);

                const rewritable_token_starts_before_chunk: u1 = potential_kw_started_before_chunk | multi_char_starts_before_chunk;

                // std.debug.print("first_token_starts_before_chunk: {}, ", .{first_token_starts_before_chunk});
                // std.debug.print("potential_kw_started_before_chunk: {}", .{potential_kw_started_before_chunk});
                // std.debug.print("\n", .{});
                // TODO: Optimization idea: try a cmov impl that uses -% 1 as a constant because the compiler probably can't see such a thing.
                // ~(x -% 1) = -x (Arm has a CNEG instruction that might be useful here)
                const carried_end = all_ends & ~(all_ends -% first_token_starts_before_chunk);

                printb(carried_end, "carried_end");
                // printb(first_token_starts_before_chunk, "first_token_starts_before_chunk");
                // printb(carried_end, "carried_end");

                // If we handled any token that started before this chunk, update its length, advancing by two tokens worth if needed.
                // Otherwise, write out two tokens worth of garbage without updating cur_token so we can overwrite it later.
                // std.debug.print("prev_carried_len: {}\n", .{prev_carried_len});
                const carried_len: u32 = if (first_token_starts_before_chunk == 0) 0 else prev_carried_len + @ctz(carried_end) + 1;
                // std.debug.print("carried_len: {}\n", .{carried_len});

                const last_token = &(cur_token.ptr - first_token_starts_before_chunk)[0];
                last_token.len = if (carried_len > std.math.maxInt(u8)) 0 else @intCast(carried_len);
                last_token.kind = if (string_identifier_started_before_chunk == 1)
                    .string_identifier
                else if (builtin_started_before_chunk == 1)
                    .builtin
                else if (is_comment_opener_split_between_chunks == 1)
                    .@"//"
                else if (is_line_string_opener_split_between_chunks == 1)
                    .@"\\\\"
                else
                    last_token.kind;

                cur_token[0..2].* = @bitCast(carried_len);

                printb(all_starts, "[1] all_starts");
                printb(all_ends, "[1] all_ends");
                // _ = before_eofs;
                const invalids = before_eofs & ~((all_ends -% all_starts) | all_ends | whitespaces);
                all_starts |= invalids;

                printb(invalids, "invalids");
                all_ends |= invalids;

                // Remove the fake `start` and the initial `end` if it was a carry-over from a previous chunk
                all_starts &= all_starts -% first_token_starts_before_chunk;
                all_ends &= ~carried_end;
                // std.debug.print("no_ends: {}\n", .{no_ends});
                const adv_amt: usize = if ((!no_ends or cur.len <= 64) and first_token_starts_before_chunk != 0 and carried_len > std.math.maxInt(u8)) 2 else 0;
                // std.debug.print("adv_amt: {}\n", .{adv_amt});
                cur_token = cur_token[adv_amt..];

                cur_token = (cur_token.ptr - rewritable_token_starts_before_chunk)[0 .. cur_token.len + rewritable_token_starts_before_chunk];

                // printb(all_starts, "all_starts");
                // printb(all_ends, "all_ends");~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                // This is a vector that holds the `kind` of each token.
                // It's okay to have tags broadcasted to places that will not be extracted in the end
                var kinds: V = classified_chunk;

                inline for ([_]struct { u64, Tag }{
                    // For each (mask, tag) pair, insert the tag into the starting positions in the mask into `kinds`
                    .{ comment_bounds_incl_carry, .@"//" },
                    // .{ line_string_starts_incl_carry, .@"\\\\" },
                    // .{ quote_starts, .string },
                    // .{ char_lit_starts, .char_literal },

                    .{ string_identifier_starts, .string_identifier },
                    .{ invalids, .invalid },
                    // .{ builtin_starts, .builtin },
                    // .{ alpha_numeric_underscores, .identifier },
                    // .{ digits, .number },
                    // .{ all_symbol_starts, .symbol },
                    // .{ string_identifier_starts, .string_identifier },
                }) |mask_and_tag| {
                    const mask, const tag = mask_and_tag;
                    kinds = @select(u8, unmovemask64(mask), @as(V, @splat(@intFromEnum(tag))), kinds);
                }

                // printb(multi_char_symbol_starts | multi_char_starts_before_chunk, "multi_char_symbol_starts");
                kinds = expand(multi_char_kinds_compressed, kinds, multi_char_symbol_starts | multi_char_starts_before_chunk);

                // inline for (0..32) |i|
                //     std.debug.print("{}: {s}\n", .{ i, if (std.meta.intToEnum(Tag, multi_char_kinds_compressed[i])) |x| @tagName(x) else |_| "n/a" });

                var lens = bitsToIndices(all_ends, 1) -% bitsToIndices(all_starts, 0);
                // printb(multi_char_has_one_char_in_prev_chunk, "multi_char_has_one_char_in_prev_chunk");
                lens += @as(V, [1]u8{multi_char_starts_before_chunk} ++ [1]u8{0} ** 63);
                lens += @as(V, [1]u8{multi_char_has_two_chars_in_prev_chunk} ++ [1]u8{0} ** 63);
                // printb(unescaped_quotes & all_starts, "unescaped_quotes & all_starts");
                // printb(((unescaped_quotes & all_starts) >> 1), "((unescaped_quotes & all_starts) >> 1)");
                // printb(ats, "ats");
                // printb(all_starts, "all_starts");
                // printb(all_starts, "all_starts");
                // all_starts |= string_identifier_starts;

                // The compiler thinks we have to wait until all_starts is fully completed.
                // However, we don't actually have to, because we can simply delete bits from the
                // compression mask we use at the end to undo the splats we do here.
                // inline for ([_]struct { u64, Tag }{
                //     // For each (mask, tag) pair, insert the tag into the starting positions in the mask into `kinds`
                //     .{ (alpha_numeric_underscores >> 1) & ats, .builtin },
                //     .{ unescaped_quotes & all_starts & ~follows_at, .string },
                //     .{ string_identifier_starts, .string_identifier },
                //     .{ unescaped_apostrophes & all_starts, .char_literal },
                //     .{ comment_starts_incl_carry, .@"//" },
                //     .{ line_string_starts_incl_carry & all_starts, .@"\\\\" },
                //     .{ identifier_starts, .identifier },
                //     .{ all_symbol_starts, .symbol },
                // }) |mask_and_tag| {
                //     const mask, const tag = mask_and_tag;
                //     kinds = @select(u8, @as(@Vector(64, bool), @bitCast(mask)), @as(V, @splat(@intFromEnum(tag))), kinds);
                // }

                // kinds = @select(u8, @as(@Vector(64, bool), @bitCast(mask)), @as(V, @splat(@intFromEnum(tag))), kinds);

                // Note: We sometimes use contextless_chunk here because if we used chunk, we'd have to keep that information alive.
                getKeywordExtents(
                    prev_chunk,
                    if (CLASSIFIER_STRATEGY == .global) chunk else contextless_chunk,
                    &lens,
                    &kinds,
                    identifier_starts,
                    identifier_ends,
                    if ((potential_kw_started_before_chunk & is_prev_carried_len_short_enough_for_kw) == 1) @intCast(prev_carried_len) else 0,
                    potential_kw_started_before_chunk,
                );

                // printb(, "(all_ends -% all_starts) | all_ends | whitespaces");

                // TODO: revitalize this codepath
                // mergeWhitespaceAndCommentsWithAdjacentKeywordsAndSymbols(all_starts, all_ends);

                // const whitespace_starts = whitespace & ~(whitespace << 1);
                // (whitespace +% whitespace_starts) & comment_starts_incl_carry;

                printStr(@as([64]u8, @bitCast(if (CLASSIFIER_STRATEGY == .global) chunk else contextless_chunk))[0..64]);
                // printb(all_starts_for_len_calc, "all_starts_for_len_calc");
                // printb(all_ends_for_len_calc, "all_ends_for_len_calc");
                // printb(all_starts, "all_starts");

                if (builtin.mode == .Debug) {
                    var s = all_starts;
                    printb(all_starts, "all_starts");
                    printb(all_ends, "all_ends");
                    var i: u64 = 0;
                    while (s != 0) : ({
                        s &= s -% 1;
                        i += 1;
                    }) {
                        // const e1 = std.meta.intToEnum(CharClasses, kinds[@ctz(s)]);
                        const e2 = std.meta.intToEnum(Tag, kinds[@ctz(s)]);
                        std.debug.print("{} {s} {}\n", .{
                            @ctz(s),
                            // if (e1) |e| @tagName(e) else |_| "n/a",
                            if (e2) |e| @tagName(e) else |_| "n/a",
                            lens[i],
                        });
                    }
                }

                // storeNonTemporal(cur_token.ptr, @as([64]Token, @bitCast(std.simd.interlace(.{ lens, compress(kinds, all_starts) }))));
                cur_token[0..64].* = @as([64]Token, @bitCast(std.simd.interlace(.{ lens, compress(kinds, all_starts) })));
                cur_token = cur_token[@popCount(all_starts)..]; // advance by the number of completed tokens

                const cache_deletion_stat: enum { clflush, prefetch, none } = .none;

                switch (cache_deletion_stat) {
                    .clflush => {
                        const flushable_cache_line = cur_token.ptr - 64;
                        if (@intFromPtr(flushable_cache_line) >= @intFromPtr(tokens.ptr)) {
                            @branchHint(.likely);
                            clflushopt(flushable_cache_line);
                        }
                    },
                    .prefetch => @prefetch(cur_token.ptr - 64, .{ .locality = 0 }),
                    .none => {},
                }

                // TODO: restore optimization maybe?
                // 62 is one less than it otherwise would be to avoid a `+1`.
                prev_carried_len = if (all_starts == 0) prev_carried_len + 64 else 1 + @clz(all_starts);

                // carry.updateViaTopBits(.{
                //     .slashes = slashes,
                //     .backslashes = backslashes,
                //     .non_newlines = non_newlines,
                //     .quote_starts = ~all_ends & (all_ends -% quote_starts),
                //     .char_lit_starts = ~all_ends & (all_ends -% char_lit_starts),
                //     .comment_starts_incl_carry = ~all_ends & (all_ends -% comment_starts_incl_carry),
                //     .line_string_starts_incl_carry = ~all_ends & (all_ends -% line_string_starts_incl_carry),
                //     .inside_strings_and_comments_including_start = inside_strings_or_comments,
                //     .carriages = carriages,
                //     .ats = ats,
                // });

                // carry.print();

                if (0 != before_eofs & @as(u64, switch (CLASSIFIER_STRATEGY) {
                    .global => @bitCast(classified_chunk == @as(V, @splat(@intFromEnum(CharClasses.bad_control)))),
                    .on_demand => blk: {
                        comptime var map: @Vector(16, u8) = @splat(0);
                        comptime for (0..128) |c| {
                            switch (c) {
                                0...'\t' - 1, '\n' + 1...'\r' - 1, '\r' + 1...' ' - 1, 0x7F => map[c & 0xF] |= 1 << (c >> 4),
                                else => {},
                            }
                        };
                        break :blk intersect_byte_halves(upper_nibbles, vpshufb(map, contextless_chunk));
                    },
                    .none => blk: {
                        const codes: u64 = @bitCast(contextless_chunk < @as(V, @splat(@intFromEnum(' '))));
                        const tabs: u64 = @bitCast(contextless_chunk == @as(V, @splat(@intFromEnum('\t'))));
                        const del: u64 = @bitCast(contextless_chunk == @as(V, @splat(@intFromEnum(0x7F))));
                        break :blk (codes ^ (tabs | newlines | carriages)) | del;
                    },
                })) {
                    @branchHint(.unlikely);
                    return error.InvalidControlCharacter;
                }

                if (cur.len <= 64) break;
                cur = cur[64..];
                // std.debug.print("----- NEW CHUNK -----\n", .{});

                // if (comptime builtin.mode == .ReleaseSafe or builtin.mode == .Debug)
                // if (@intFromPtr(cur) >= @intFromPtr(final_cur)) break;
                // if (comptime builtin.mode == .Debug) {
                //     super_duper_pls_delete_me_iter += 1;
                //     if (super_duper_pls_delete_me_iter == 4) break;
                // }
            }

            // if (@intFromPtr(cur) < @intFromPtr(final_cur) or (sourcey.slice().len % 64) != @ctz(eofs)) {
            //     return error.Invalid0ByteInFile;
            // }

            // cur_token = cur_token[if (cur_token[0].len == 0) 3 else 1..];
            cur_token[0] = .{ .len = 1, .kind = .eof };
            cur_token = cur_token[1..];

            if (builtin.mode == .Debug) {
                const end_token = cur_token;
                // cur_token = tokens;

                std.debug.print("-----------------------\n", .{});

                for (tokens[0 .. end_token.ptr - tokens.ptr]) |tok| {
                    std.debug.print("tok.kind: {s}, tok.len: {}\n", .{ @tagName(tok.kind), tok.len });
                }
                std.debug.print("-----------------------\n", .{});

                var iter: TokenInfoIterator = .init(source, tokens[0 .. end_token.ptr - tokens.ptr]);
                var found_invalid = false;
                while (iter.current().kind != .eof) : (iter.advance()) {
                    std.debug.print("{s} {}\n", .{ @tagName(iter.current().kind), iter.current().source.len });
                    if (iter.current().kind == .invalid) found_invalid = true;
                }
                if (found_invalid) return error.InvalidChicken;
                std.debug.print("-----------------------\n", .{});
            }

            // if (@intFromPtr(cur.ptr) < @intFromPtr(end_ptr)) return error.Found0ByteInFile;

            // cur_token[0] = .{ .len = 0, .kind = .eof };
            // cur_token = cur_token[1..];
            // cur_token[0] = .{ .len = 0, .kind = .eof };
            // cur_token = cur_token[1..];
            // cur_token[0] = .{ .len = 0, .kind = .eof };
            // cur_token = cur_token[1..];

            const num_tokens = (@intFromPtr(cur_token.ptr) - @intFromPtr(tokens.ptr)) / @sizeOf(Token);
            // if (builtin.mode == .Debug) {
            //     var cursor: []const u8 = sourcey.slice();

            //     for (tokens[0..num_tokens]) |token| {
            //         const len = token.len;
            //         std.debug.print("kind: {s}, len: {}, str: `", .{ @tagName(token.kind), len });

            //         for (cursor[0..len]) |c| {
            //             switch (c) {
            //                 '\n' => std.debug.print("$", .{}),
            //                 else => std.debug.print("{c}", .{c}),
            //             }
            //         }
            //         std.debug.print("`\n", .{});

            //         cursor = cursor[len..];
            //     }
            // }

            // Overallocate the tokens a little for convenience in the Parser (soon™)
            const new_chunks_data_len = 3 + num_tokens;

            if (gpa.resize(tokens, new_chunks_data_len)) {
                var resized_tokens = tokens;
                resized_tokens.len = new_chunks_data_len;
                return resized_tokens;
            }

            return tokens;
        }

        // We want to merge whitespace and comments together with keywords and symbols.
        // Currently dead code. We currently look right first, then left.
        // The order in which we do this needs to be flipped.

        // whitespace
        // "d   saded    //    \n         f"
        //  011100000111100000001111111111  | whitespace
        //  000000000000010000000000000000  | comment_starts
        //  010000000100000000000000000000  | whitespace_starts

        // .................1..........1.............1.......1.......1..... ⟵ comments_extendable_left
        // ................1..........1.............1....1.......1......... ⟵ left_extended_comment_starts_with_mergables
        // 1......1......1..1.......1..1........1....1.......1.......1..... ⟵ all_starts <>
        // .................1..........1.............1.......1.......1..... ⟵ comments_extendable_left <>
        // 1......1......1.1........1.1.........1...1...................... ⟵ all_starts_for_len_calc
        // .....1......1..1.......1..1........1....1.....................1. ⟵ all_ends_for_len_calc
        // ......1......1..1......1...1.......1.....1....1.......1.......1. ⟵ whitespace_befores
        // ......1......1..1.......1..1........1....1.......1.......1.....1 ⟵ whitespace_ends
        // 1......1......1..1.......1..1........1....1..................... ⟵ all_starts 2
        fn mergeWhitespaceAndCommentsWithAdjacentKeywordsAndSymbols(
            all_starts: u64,
            all_ends: u64,
            whitespaces: u64,
            whitespace_starts: u64,
            whitespace_ends: u64,
            comment_starts: u64,
            all_symbol_starts: u64,
            all_symbol_ends: u64,
            kw_starts: u64,
            kw_ends: u64,
        ) void {
            var all_starts_for_len_calc = all_starts;
            var all_ends_for_len_calc = all_ends;

            // printb(whitespace, "whitespace");
            const whitespace_afters = ~whitespaces & (whitespaces << 1);
            const comment_ends = all_ends & ~(all_ends -% comment_starts);

            // Step 1. Merge whitespace followed by comments into comments.

            // @hello "///'" ++ // bad$ ++$// good$$"me" // m$$  // m$$  // m$
            // .......1......1..1.......1..1........1....1.......1.......1..... ⟵ whitespace_afters
            // .................1..........1.............1.......1.......1..... ⟵ comments_extendable_left
            // ......1......1..1......1...1.......1.....1....1.......1.......1. ⟵ whitespace_starts
            // ................1..........1.............1....1.......1......... ⟵ left_extended_comment_starts_with_mergables
            // Find which comments start right after a group of contiguous whitespace.
            const comments_extendable_left = whitespace_afters & comment_starts;
            const comments_not_extendable_left = ~whitespace_afters & comment_starts;

            // This pext and pdep combination is a common trick when we have two sets of X bits that map to each other, we compress/pext on the later X bits,
            // then broadcast to the corresponding position in the earlier set of X bits.
            // This moves the start of comments further left to include whitespace.
            // We have to use this trick because the carry in an addition only goes one direction.
            // printb(whitespace_afters, "whitespace_afters");
            // printb(whitespace_starts, "whitespace_starts");
            // printb(whitespace_ends, "whitespace_ends");

            // TODO: check boundary conditions
            const left_extended_comment_starts_with_mergables = pdep(pext(comments_extendable_left, whitespace_afters), whitespace_starts);

            // Delete the old comment start positions
            all_starts_for_len_calc ^= comments_extendable_left;
            // Merge in the left-shifted comment start positions
            all_starts_for_len_calc |= left_extended_comment_starts_with_mergables;

            // Step 2. Merge adjacent comments.

            // Find which comments can be merged together, separated only by whitespace (whitespace inside of comments are zeroed)
            //  "" // hello $$$ // hello
            // ...1.........1.................................................. ⟵ left_extended_comment_starts_with_mergables
            // .............1...........1...................................... ⟵ comment_ends
            // .............1.................................................. ⟵ mergable_comments
            const mergable_comments = left_extended_comment_starts_with_mergables & comment_ends;
            all_starts_for_len_calc ^= mergable_comments;
            all_ends_for_len_calc ^= mergable_comments;

            //  "" // hello $$$ // hello
            // ....1............1.............................................. ⟵ comments_extendable_left
            // .............1.................................................. ⟵ mergable_comments
            const deletable_comment_start_positions = comments_extendable_left & ~(comments_extendable_left -% mergable_comments);
            all_starts ^= deletable_comment_start_positions;

            // ...1............................................................ ⟵ left_extended_comment_starts
            const left_extended_comment_starts = (left_extended_comment_starts_with_mergables & ~comment_ends) | comments_not_extendable_left;

            // Step 3. Merge comments followed by whitespace.

            //  "" // hello $$$ // hello$  comptime  var // hello$$$ "/'"  ++
            // .............1...........1........................1............. ⟵ comment_ends
            // ...1.........1...........................1...................... ⟵ left_extended_comment_starts_with_mergables
            // .........................1........................1............. ⟵ comment_ends & ~left_extended_comment_starts_with_mergables
            // 1..1............1..........1.........1...1...........1.....1...1 ⟵ whitespace_ends
            // ...........................1.........................1.......... ⟵ right_extended_comment_ends
            const right_extended_comment_ends = whitespace_ends & ~(whitespace_ends -% (comment_ends & ~left_extended_comment_starts_with_mergables));

            // Always delete comment_ends, because in the initial phase, comments were said to end at a newline.
            // `right_extended_comment_ends` will always have the proper end character for all comments.
            all_ends_for_len_calc &= ~comment_ends;
            all_ends_for_len_calc |= right_extended_comment_ends;

            // Step 4. Merge whitespace/comments with operators and keywords

            const sym_kw_starts = all_symbol_starts | kw_starts;
            const sym_kw_ends = all_symbol_ends | kw_ends;

            // Because comments always end with \n, we kill 2 birds with 1 stone right here.
            const extendable_sym_kw_starts = whitespace_afters & sym_kw_starts;

            const inside_extended_comments = (right_extended_comment_ends << 1) -% left_extended_comment_starts;
            const left_bounds = (whitespace_starts & ~inside_extended_comments) | left_extended_comment_starts;

            all_ends_for_len_calc &= ~(extendable_sym_kw_starts >> 1);

            const left_extended_sym_kw_starts_using_bitreverse = andn(left_bounds, reversedSubtraction(left_bounds, extendable_sym_kw_starts));

            // Because bitReversal is slow on x86, we can extend left by instead extending right, and then shifting the boundary bits left relative to each other.
            // By "relative" shift I mean `pdep(pext(z, x) >> 1, x)`, where in this case `z` is `x & ~(x -% y)`.
            // This almost works without modification, however, we need a catcher bit at the end so that a bit in that position won't be discarded.
            const left_bounds_with_catcher = left_bounds | (@as(u64, 1) << 63);
            const left_extended_sym_kw_starts_using_pdep_n_pext = pdep(pext(andn(left_bounds_with_catcher, left_bounds_with_catcher -% extendable_sym_kw_starts), left_bounds_with_catcher) >> 1, left_bounds_with_catcher);

            const left_extended_sym_kw_starts = if (HAS_FAST_PDEP_AND_PEXT) left_extended_sym_kw_starts_using_pdep_n_pext else left_extended_sym_kw_starts_using_bitreverse;

            if (comptime builtin.mode == .Debug or builtin.mode == .ReleaseSafe) {
                // printb(left_bounds, "left_bounds");
                // printb(left_bounds_with_catcher, "left_bounds_with_catcher");
                // printb(extendable_sym_kw_starts, "extendable_sym_kw_starts");
                // printb(left_bounds_with_catcher -% extendable_sym_kw_starts, "left_bounds_with_catcher -% extendable_sym_kw_starts");
                // printb(left_bounds_with_catcher & ~(left_bounds_with_catcher -% extendable_sym_kw_starts), "left_bounds_with_catcher & ~(left_bounds_with_catcher -% extendable_sym_kw_starts)");
                // printb(left_extended_sym_kw_starts_using_bitreverse, "left_extended_sym_kw_starts_using_bitreverse");
                // printb(left_extended_sym_kw_starts_using_pdep_n_pext, "left_extended_sym_kw_starts_using_pdep_n_pext");
                assert(left_extended_sym_kw_starts_using_bitreverse == left_extended_sym_kw_starts_using_pdep_n_pext);
            }
            // printb(left_bounds, "left_bounds");
            // printb(extendable_sym_kw_starts, "extendable_sym_kw_starts");
            // printb(left_extended_sym_kw_starts, "left_extended_sym_kw_starts");

            // printb(andy(andn(left_bounds, left_bounds -% extendable_sym_kw_starts), left_bounds), "andy(andn(left_bounds, left_bounds -% extendable_sym_kw_starts), left_bounds)");
            // printb(, "pext(andn(left_bounds, left_bounds -% extendable_sym_kw_starts), left_bounds) >> 1");
            all_starts_for_len_calc ^= extendable_sym_kw_starts;
            all_starts_for_len_calc |= left_extended_sym_kw_starts;

            const extendable_sym_kw_ends = (((whitespace_starts | left_extended_comment_starts) & ~left_extended_sym_kw_starts) >> 1) & sym_kw_ends;
            const right_bounds = right_extended_comment_ends | (whitespace_ends & ~inside_extended_comments);
            const right_extended_sym_kw_ends = andn(right_bounds, right_bounds -% extendable_sym_kw_ends);
            const consumed_left_bounds = andn(left_bounds, left_bounds -% extendable_sym_kw_ends);
            const consumed_comments = comment_starts & (~(comment_starts -% extendable_sym_kw_ends) | left_extended_sym_kw_starts);

            // left_extended_sym_kw_starts & comment_starts;
            all_starts ^= consumed_comments;
            all_starts_for_len_calc &= ~consumed_left_bounds;

            all_ends_for_len_calc ^= extendable_sym_kw_ends;
            all_ends_for_len_calc |= right_extended_sym_kw_ends;
        }
    };
}

// const Rp = rpmalloc.RPMalloc(.{});
pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    // var jdz = jdz_allocator.JdzAllocator(.{
    //     .split_large_spans_to_one = true,
    //     .split_large_spans_to_large = true,
    // }).init();
    // defer jdz.deinit();
    // const gpa: Allocator = jdz.allocator();

    // const gpa = std.heap.c_allocator;
    const gpa = std.heap.page_allocator;
    const sources: SourceData = try .readFiles(gpa);
    defer {
        // Leak memory in ReleaseFast because the OS is going to clean it up on program exit.
        // However, in debug mode, we still want to make sure we don't have any leaks that could affect the program while still running.
        if (comptime builtin.mode == .Debug)
            sources.deinit(gpa);
    }

    const bytes = sources.num_bytes;
    const lines = sources.num_lines;

    // try stdout.print("-" ** 72 ++ "\n", .{});
    var num_tokens2: usize = 0;
    const legacy_token_lists: if (RUN_LEGACY_TOKENIZER) []Ast.TokenList.Slice else void = if (RUN_LEGACY_TOKENIZER) try gpa.alloc(Ast.TokenList.Slice, sources.source_list.len);

    const elapsedNanos2: u64 = if (!RUN_LEGACY_TOKENIZER) 0 else blk: {
        const t3 = std.time.nanoTimestamp();
        for (
            sources.source_list.items(.file_contents),
            legacy_token_lists,
        ) |sourcey, *legacy_token_list_slot| {
            assert(std.mem.eql(u8, @TypeOf(sourcey).back_sentinels, "\x00"));
            const source: [:0]const u8 = @ptrCast(sourcey.slice());

            var tokens: Ast.TokenList = .empty;
            defer tokens.deinit(gpa);

            // Empirically, the zig std lib has an 8:1 ratio of source bytes to token count.
            const estimated_token_count = source.len / 8;
            try tokens.ensureTotalCapacity(gpa, estimated_token_count);
            var tokenizer: std.zig.Tokenizer = .init(source);
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
            try stdout.print("\n" ** @intFromBool(RUN_LEGACY_AST or RUN_NEW_AST) ++ "Legacy Tokenizing took             {: >9} ({d:.2} MB/s, {d: >5.2}M loc/s) and used {} memory\n", .{ std.fmt.fmtDuration(elapsedNanos2), throughput, @as(f64, @floatFromInt(lines)) / @as(f64, @floatFromInt(elapsedNanos2)) * 1000, std.fmt.fmtIntSizeDec(num_tokens2 * 5) });
            break :blk elapsedNanos2;
        }
    };

    if (RUN_NEW_TOKENIZER or INFIX_TEST) {
        // const t1 = std.time.nanoTimestamp();
        const Tokenize = Tokenizer(.{});
        const Token = Tokenize.Token;

        // const source_tokens = try gpa.alloc([]Token, sources.source_list.len);
        // const source_tokens2 = if (RUN_COMPRESS_TOKENIZER)
        //     try gpa.alloc([]Token, sources.source_list.len);

        const source_tokens2 = if (RUN_COMPRESS_TOKENIZER)
            try LenAlignedBuffer([]Token, .{ .alignment = 64 * 8, .constant = false }).alloc(gpa, sources.source_list.len);

        defer {
            if (comptime builtin.mode == .Debug) { // Just to make the leak detectors happy
                if (1 != 1) {
                    // for (source_tokens) |source_token| gpa.free(source_token);
                    // gpa.free(source_tokens);
                    if (RUN_COMPRESS_TOKENIZER) {
                        for (source_tokens2.slice()) |source_token| gpa.free(source_token);
                        source_tokens2.deinit(gpa);
                    }
                }
            }
        }

        // inline for (0..100) |_| {
        // for (sources.source_list.items(.file_contents), source_tokens) |source, *source_token_slot| {
        // const tokens = try Tokenizer.tokenize(gpa, source);
        // source_token_slot.* = tokens;

        // var token_iter = TokenInfoIterator.init(source, tokens);

        // while (token_iter.next()) |token| {
        //     std.debug.print("'{s}' ({}) {s}\n", .{ token.source, token.source.len, @tagName(token.kind) });
        // }

        // const b = try Tokenizer.tokenize(gpa, source, 1);

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
        // }
        // }
        const t2 = std.time.nanoTimestamp();
        // const elapsedNanos: u64 = @intCast(t2 - t1);

        if (RUN_COMPRESS_TOKENIZER) {
            for (
                sources.source_list.items(.file_contents),
                sources.source_list.items(.path),
                source_tokens2.slice(),
            ) |source, path, *source_token_slot| {
                source_token_slot.* = Tokenize.tokenize(gpa, source.slice()) catch |e| {
                    std.debug.print("{s}\n", .{path});
                    return e;
                };
                // if (comptime builtin.mode == .Debug)
                //     break;
            }
        }

        const t3 = std.time.nanoTimestamp();
        const elapsedNanos3: u64 = @intCast(t3 - t2);

        var num_tokens: usize = 0;
        for (source_tokens2.allocatedSlice()) |tokens|
            num_tokens +%= tokens.len;

        // Fun fact: bytes per nanosecond is the same ratio as GB/s
        if (RUN_NEW_TOKENIZER and REPORT_SPEED) {
            // const throughput = @as(f64, @floatFromInt(bytes * 1000)) / @as(f64, @floatFromInt(elapsedNanos));
            // try stdout.print("Tokenizing with vectorization took {: >9} ({d:.2} MB/s, {d: >5.2}M loc/s) and used {} memory\n", .{ std.fmt.fmtDuration(elapsedNanos), throughput, @as(f64, @floatFromInt(lines)) / @as(f64, @floatFromInt(elapsedNanos)) * 1000, std.fmt.fmtIntSizeDec(num_tokens * 2) });

            // if (elapsedNanos2 > 0) {
            //     try stdout.print("       That's {d:.2}x faster and {d:.2}x less memory!\n", .{ @as(f64, @floatFromInt(elapsedNanos2)) / @as(f64, @floatFromInt(elapsedNanos)), @as(f64, @floatFromInt(num_tokens2 * 5)) / @as(f64, @floatFromInt(num_tokens * 2)) });
            // }
        }

        if (RUN_COMPRESS_TOKENIZER and REPORT_SPEED) {
            const throughput = @as(f64, @floatFromInt(bytes * 1000)) / @as(f64, @floatFromInt(elapsedNanos3));
            try stdout.print("Tokenizing with compression took   {: >9} ({d:.2} MB/s, {d: >5.2}M loc/s) and used {} memory\n", .{ std.fmt.fmtDuration(elapsedNanos3), throughput, @as(f64, @floatFromInt(lines)) / @as(f64, @floatFromInt(elapsedNanos3)) * 1000, std.fmt.fmtIntSizeDec(num_tokens * 2) });

            if (elapsedNanos2 > 0) {
                try stdout.print("       That's {d:.2}x faster and {d:.2}x less memory than the mainline implementation!\n", .{ @as(f64, @floatFromInt(elapsedNanos2)) / @as(f64, @floatFromInt(elapsedNanos3)), @as(f64, @floatFromInt(num_tokens2 * 5)) / @as(f64, @floatFromInt(num_tokens * 2)) });
            }
            // try stdout.print("       That's {d:.2}x faster than my old implementation!\n", .{@as(f64, @floatFromInt(elapsedNanos)) / @as(f64, @floatFromInt(elapsedNanos3))});
        }

        _ = try std.zig.Ast.parse(gpa, "comptime { a.b.c.*(i()); }", .zig);

        if (INFIX_TEST) {
            // for (sources.source_list.items(.file_contents), source_tokens) |source, tokens| {
            // const parse_tree = try infixToPrefix(gpa, source, tokens);
            // _ = parse_tree;
            // const source_start_pos = if (tokens[0].kind == .whitespace) tokens[0].len else 0;
            // std.debug.print("\n\n\n\n", .{});
            // const token_info_iterator = TokenInfoIterator.init(parse_tree[1..], source[source_start_pos..]);
            // _ = parse(parse_tree[1..], source[source_start_pos..]);
            // _ = parse2(&token_info_iterator);
            // std.debug.print(";", .{});
            // std.debug.print("\nvs{s}\n\n\n\n\n", .{source});
            // }
        }

        if (comptime builtin.mode == .Debug and WRITE_OUT_DATA) {
            var buffer: [1 << 12]u8 = undefined;
            const base_dir_path = "/home/niles/Documents/github/Zig-Parser-Experiment/.token_data2/";
            buffer[0..base_dir_path.len].* = base_dir_path.*;

            for (source_tokens2.slice(), sources.source_list.items(.path), sources.source_list.items(.file_contents)) |tokens, path, file_contents| {
                if (buffer.len - base_dir_path.len <= path.len) return error.BufferTooSmall;
                @memcpy(buffer[base_dir_path.len .. base_dir_path.len + path.len], path[0..path.len]);
                for (buffer[base_dir_path.len .. base_dir_path.len + path.len - 3]) |*c| {
                    if (c.* == '/') c.* = '|';
                }

                buffer[base_dir_path.len + path.len - 3 ..][0..3].* = "txt".*;
                buffer[base_dir_path.len + path.len] = 0;

                std.debug.print("{s}\n", .{buffer[0 .. base_dir_path.len + path.len]});

                const token_file = try std.fs.createFileAbsoluteZ(buffer[0 .. base_dir_path.len + path.len :0], std.fs.File.CreateFlags{ .read = false });
                defer token_file.close();

                // const current_data: []const u8 = @as([*]const u8, @ptrCast(tokens.ptr))[0 .. tokens.len * @sizeOf(Token)];

                var writer = std.io.BufferedWriter(1 << 20, std.fs.File.Writer){ .unbuffered_writer = token_file.writer() };
                var cur_token = tokens[0..];
                var cur: []const u8 = file_contents[0..];

                const int_fmt = "{d}";
                var int_buffer: [std.fmt.count(int_fmt, .{std.math.maxInt(u32)})]u8 = undefined;

                while (cur_token[0].kind != .eof) {
                    const is_large_token = cur_token[0].len == 0;
                    const large_len: u32 = @bitCast(cur_token[1..3].*);
                    const len: u32 = if (is_large_token) large_len else cur_token[0].len;

                    comptime var longest_tag_name = 0;

                    comptime for (std.meta.fieldNames(Tokenize.Tag)) |tag_name| {
                        longest_tag_name = @max(longest_tag_name, tag_name.len + 1);
                    };

                    _ = try writer.write(@tagName(cur_token[0].kind));

                    for (0..longest_tag_name - @tagName(cur_token[0].kind).len) |_| {
                        _ = try writer.write(" ");
                    }

                    const int_data = std.fmt.bufPrint(&int_buffer, int_fmt, .{len}) catch unreachable;

                    for (0..int_buffer.len - int_data.len) |_| {
                        _ = try writer.write(" ");
                    }

                    _ = try writer.write(int_data);
                    _ = try writer.write("    \"");

                    if (writer.end + len > writer.buf.len) {
                        try writer.flush();
                        if (len > writer.buf.len) unreachable;
                    }

                    for (writer.buf[writer.end..][0..len], cur[0..len]) |*slot, c| {
                        slot.* = switch (c) {
                            '\t' => '`',
                            '\r' => '`',
                            '\n' => '`',
                            else => c,
                        };
                    }
                    writer.end += len;

                    _ = try writer.write("\"");
                    _ = try writer.write("\n");

                    cur_token = cur_token[if (is_large_token) 3 else 1..];
                    cur = cur[len..];
                }

                try writer.flush();

                // const old_data = try token_file.readToEndAlloc(gpa, std.math.maxInt(u32));

                // if (old_data.len != old_data_supposed_len or !std.mem.eql(u8, current_data, old_data)) {
                // std.debug.print("Invalid token data!!!!!\n", .{});
                // std.debug.print("{d}\n", .{current_data});
                // std.debug.print("{d}\n", .{old_data});
                // for (current_data, old_data, 0..) |a, b, i| {
                //     if (a != b) {
                //         std.debug.print("{} {} {d} {d}\n", .{ m, i, current_data[i], old_data[i] });
                //         return error.Noodle;
                //     }
                // }
                // }
            }
        }
    }

    const elapsedNanos4: u64 = if (!RUN_LEGACY_AST) 0 else blk: {
        if (!RUN_LEGACY_TOKENIZER) @compileError("Must enable legacy tokenizer to run legacy AST!");
        const legacy_asts = try gpa.alloc(Ast, legacy_token_lists.len);

        const t3 = std.time.nanoTimestamp();
        for (sources.source_list.items(.file_contents), legacy_token_lists, legacy_asts) |source, tokens, *ast_slot| {
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

fn vpshufb(table: anytype, indices: anytype) if (@sizeOf(@TypeOf(indices)) > @sizeOf(@TypeOf(table))) @TypeOf(indices) else @TypeOf(table) {
    if (@inComptime() or SUGGESTED_VEC_SIZE == null) {
        var result: @TypeOf(indices) = undefined;
        for (0..@bitSizeOf(@TypeOf(indices)) / 8) |i| {
            const index = indices[i];
            result[i] = if (index >= 0x80) 0 else table[index % (@bitSizeOf(@TypeOf(table)) / 8)];
        }

        return result;
    }

    if (@sizeOf(@TypeOf(indices)) > @sizeOf(@TypeOf(table))) {
        return vpshufb(std.simd.repeat(@sizeOf(@TypeOf(indices)), table), indices);
    }

    if (@sizeOf(@TypeOf(table)) > SUGGESTED_VEC_SIZE.?) {
        var parts: [@sizeOf(@TypeOf(table)) / SUGGESTED_VEC_SIZE.?]@Vector(SUGGESTED_VEC_SIZE.?, u8) = undefined;
        inline for (&parts, 0..) |*slot, i| {
            slot.* = vpshufb(std.simd.extract(table, i * SUGGESTED_VEC_SIZE.?, SUGGESTED_VEC_SIZE.?), std.simd.extract(indices, i * SUGGESTED_VEC_SIZE.?, SUGGESTED_VEC_SIZE.?));
        }
        return @bitCast(parts);
    }

    const methods = struct {
        extern fn @"llvm.x86.avx512.pshuf.b.512"(@Vector(64, u8), @Vector(64, u8)) @Vector(64, u8);
        extern fn @"llvm.x86.avx2.pshuf.b"(@Vector(32, u8), @Vector(32, u8)) @Vector(32, u8);
        extern fn @"llvm.x86.ssse3.pshuf.b.128"(@Vector(16, u8), @Vector(16, u8)) @Vector(16, u8);
    };

    return switch (@TypeOf(table)) {
        @Vector(64, u8) => if (comptime std.Target.x86.featureSetHas(builtin.cpu.features, .avx512bw)) methods.@"llvm.x86.avx512.pshuf.b.512"(table, indices) else @compileError("CPU target lacks support for vpshufb512"),
        @Vector(32, u8) => if (comptime std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) methods.@"llvm.x86.avx2.pshuf.b"(table, indices) else @compileError("CPU target lacks support for vpshufb256"),
        @Vector(16, u8) => if (comptime std.Target.x86.featureSetHas(builtin.cpu.features, .ssse3)) methods.@"llvm.x86.ssse3.pshuf.b.128"(table, indices) else @compileError("CPU target lacks support for vpshufb128"),
        else => @compileError(std.fmt.comptimePrint("Invalid argument type passed to vpshufb: {}\n", .{@TypeOf(table)})),
    };
}

fn tbl1(table: @Vector(16, u8), indices: anytype) @TypeOf(indices) {
    switch (@TypeOf(indices)) {
        @Vector(16, u8), @Vector(8, u8), @Vector(16, i8), @Vector(8, i8) => {},
        @Vector(8, i16), @Vector(8, u16) => @compileError("[aarch64.neon.tbl1] @Vector(8, u16) is currently not supported for the second operand."),
        else => @compileError("[aarch64.neon.tbl1] Invalid second operand. Should be @Vector(16, u8) or @Vector(8, u8)"),
    }
    return struct {
        extern fn @"llvm.aarch64.neon.tbl1"(@TypeOf(table), @TypeOf(indices)) @TypeOf(indices);
    }.@"llvm.aarch64.neon.tbl1"(table, indices);
}

fn vtbl2(table_part_1: @Vector(8, u8), table_part_2: @Vector(8, u8), indices: @Vector(8, u8)) @Vector(8, u8) {
    comptime assert(builtin.cpu.arch == .arm and std.Target.arm.featureSetHas(builtin.cpu.features, .neon));

    return struct {
        extern fn @"llvm.arm.neon.vtbl2"(@TypeOf(table_part_1), @TypeOf(table_part_2), @TypeOf(indices)) @TypeOf(table_part_1);
    }.@"llvm.arm.neon.vtbl2"(table_part_1, table_part_2, indices);
}

// ---------------------------------------------------------------
//
// The code below this point is licensed under the Apache License.
// Please see the License at the bottom of this file.
//
// ---------------------------------------------------------------

fn shift_in_prev(comptime N: comptime_int, cur: anytype, prev: @TypeOf(cur)) @TypeOf(cur) {
    comptime assert(0 < @as(u4, N) and N < @sizeOf(@TypeOf(cur)));
    if (!@inComptime() and comptime (!HAS_ARM_NEON and builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512bw))) {
        // Workaround for https://github.com/llvm/llvm-project/issues/79799
        const c: @TypeOf(cur) = @bitCast(@shuffle(u64, @as(@Vector(@sizeOf(@TypeOf(cur)) / 8, u64), @bitCast(cur)), @as(@Vector(@sizeOf(@TypeOf(cur)) / 8, u64), @bitCast(prev)), std.simd.mergeShift(~std.simd.iota(i4, 8), std.simd.iota(i4, 8), 6)));
        return asm (
            \\vpalignr %[imm], %[c], %[cur], %[out]
            : [out] "=v" (-> @TypeOf(cur)),
            : [cur] "v" (cur),
              [c] "v" (c),
              [imm] "n" (16 - @as(u8, N)),
        );
    } else {
        const a_shift = @as(comptime_int, N) * 8;
        const b_shift = (@sizeOf(@TypeOf(cur)) - N) * 8;

        return if (USE_SWAR)
            switch (comptime builtin.cpu.arch.endian()) {
                .little => (cur << a_shift) | (prev >> b_shift),
                .big => (cur >> a_shift) | (prev << b_shift),
            }
        else
            std.simd.mergeShift(prev, cur, @sizeOf(@TypeOf(cur)) - N);
    }
}

fn Utf8Checker(comptime options: struct {
    USE_ARM_NEON: bool,
    V: type,
    native_int: type,
}) type {
    return struct {
        const USE_ARM_NEON = options.USE_ARM_NEON;
        const V = options.V;
        const native_int = options.native_int;

        is_invalid_place_to_end: bool,
        prev_input_block: if (USE_ARM_NEON) void else V,
        leftovers: if (USE_ARM_NEON) [3]@Vector(16, u8) else void,

        const empty = @This(){
            .is_invalid_place_to_end = false,
            .prev_input_block = if (USE_ARM_NEON) {} else std.mem.zeroes(V),
            .leftovers = if (USE_ARM_NEON) @bitCast([_]u8{0} ** 48) else {},
        };

        // Default behavior for shuffles across architectures
        // x86_64: If bit 7 is 1, set to 0, otherwise use lower 4 bits for lookup. We can get the arm/risc-v/wasm behavior by adding 0x70 before doing the lookup.
        // ARM: if index is out of range (0-15), set to 0
        // PPC64: use lower 4 bits for lookup
        // MIPS: if bit 6 or bit 7 is 1, set to 0; otherwise use lower 4 bits for lookup (or rather use lower 5 bits for lookup into a table that has 32 elements constructed from 2 input vectors, but if both vectors are the same then it effectively means bits 4,5 are ignored)
        // RISCV: if index is out of range (0-15), set to 0.
        // WASM: if index is out of range (0-15), set to 0.
        fn lookup_chunk(comptime table: [16]u8, indices: anytype) @TypeOf(indices) {
            switch (builtin.cpu.arch) {
                // if high bit is set will result in a 0, otherwise, just looks at lower 4 bits
                .x86_64 => return vpshufb(@as(@TypeOf(indices), @bitCast(table ** (@sizeOf(@TypeOf(indices)) / 16))), indices),
                .aarch64, .aarch64_be => return tbl1(table, indices),
                .arm, .armeb => return switch (@TypeOf(indices)) {
                    @Vector(16, u8) => std.simd.join(vtbl2(table[0..8].*, table[8..][0..8].*, std.simd.extract(indices, 0, 8)), vtbl2(table[0..8].*, table[8..][0..8].*, std.simd.extract(indices, 8, 8))),
                    @Vector(8, u8) => vtbl2(table[0..8].*, table[8..][0..8].*, indices),
                    else => @compileError("Invalid vector size passed to lookup_chunk"),
                },
                else => {
                    var r: @TypeOf(indices) = @splat(0);
                    for (0..@sizeOf(@TypeOf(indices))) |i| r[i] = table[indices[i]];
                    return r;

                    // var r: V = @splat(0);
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
        fn check_special_cases(input: V, prev1: V) V {
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
                inline for (0..@sizeOf(uword)) |i| {
                    const j = i * 8;
                    const prev_ans = if (comptime builtin.mode == .ReleaseSmall)
                        byte_1_low_tbl[@as(u4, @truncate(prev1 >> j))] & byte_1_high_tbl[@as(u4, @truncate(prev1 >> (j + 4)))]
                    else
                        byte_1_tbl[@as(u8, @truncate(prev1 >> j))];
                    result |= @as(V, byte_2_high_tbl[@as(u4, @truncate(input >> (j + 4)))] & prev_ans) << j;
                }
                return result;
            } else {
                return lookup_chunk(byte_1_low_tbl, prev1 & @as(@TypeOf(prev1), @splat(0xF))) &
                    lookup_chunk(byte_1_high_tbl, prev1 >> @splat(4)) &
                    lookup_chunk(byte_2_high_tbl, input >> @splat(4));
            }
        }

        fn must_be_2_3_continuation(prev2: V, prev3: V) V {
            const ones: V = @bitCast([_]u8{0x01} ** @sizeOf(V));
            const msbs: V = @bitCast([_]u8{0x80} ** @sizeOf(V));

            if (USE_SWAR) {
                const is_3rd_byte = prev2 & ((prev2 | msbs) - (0b11100000 - 0x80) * ones);
                const is_4th_byte = prev3 & ((prev3 | msbs) - (0b11110000 - 0x80) * ones);
                return (is_3rd_byte | is_4th_byte) & msbs;
            } else {
                const is_3rd_byte = prev2 -| @as(V, @splat(0b11100000 - 0x80));
                const is_4th_byte = prev3 -| @as(V, @splat(0b11110000 - 0x80));
                return (is_3rd_byte | is_4th_byte) & msbs;
            }
        }

        fn isASCII(input: V) bool {
            // https://github.com/llvm/llvm-project/issues/76812
            return if (USE_SWAR)
                0 == (input & @as(native_int, @bitCast([_]u8{0x80} ** @sizeOf(uword))))
            else if (comptime builtin.cpu.arch == .x86_64 and !std.Target.x86.featureSetHas(builtin.cpu.features, .avx512bw))
                0 == @reduce(.Or, input & @as(V, @splat(0x80)))
            else if (comptime builtin.cpu.arch == .arm or builtin.cpu.arch == .armeb)
                0x80 > @reduce(.Max, input)
            else
                0 == @as(std.meta.Int(.unsigned, @sizeOf(native_int)), @bitCast(input >= @as(@Vector(@sizeOf(native_int), u8), @splat(0x80))));
        }

        fn validateChunksArm(self: *@This(), prev0s: [4]V) !void {
            // Check whether the current bytes are valid UTF-8.
            const true_vec = @as(V, @splat(0xFF));
            const false_vec = @as(V, @splat(0));

            // Rather than operating on 4 chunks separately, we operate on all at once.
            // This is more efficient because if we operate on them separately, we have to generate
            // prev1, prev2, and prev3 for each chunk, which is 3 `ext` instructions per chunk.
            // However, if we do all at once, we only need 3 `ext` instructions to do 4 chunks.

            //                                          prevs: 0 1 2 3
            // -3  1  5  9 13 17 21 25 29 33 37 41 45 49 53 57       |
            // -2  2  6 10 14 18 22 26 30 34 38 42 46 50 54 58     | |
            // -1  3  7 11 15 19 23 27 31 35 39 43 47 51 55 59   | | |
            //  0  4  8 12 16 20 24 28 32 36 40 44 48 52 56 60 | | | |
            //  1  5  9 13 17 21 25 29 33 37 41 45 49 53 57 61 | | |
            //  2  6 10 14 18 22 26 30 34 38 42 46 50 54 58 62 | |
            //  3  7 11 15 19 23 27 31 35 39 43 47 51 55 59 63 |

            //               0   1   2
            // leftovers: { -3, -2, -1 }

            const prev1s: [4]V = ([_]V{shift_in_prev(1, prev0s[3], self.leftovers[2])} ++ prev0s[0..3]).*;
            const prev2s: [4]V = ([_]V{shift_in_prev(1, prev0s[2], self.leftovers[1])} ++ prev1s[0..3]).*;
            const prev3s: [4]V = ([_]V{shift_in_prev(1, prev0s[1], self.leftovers[0])} ++ prev2s[0..3]).*;

            var errs: V = @splat(0);

            inline for (prev0s, prev1s, prev2s, prev3s) |prev0, prev1, prev2, prev3| {
                const sc = check_special_cases(prev0, prev1);
                const must23_80 = must_be_2_3_continuation(prev2, prev3);

                errs |= (sc ^ must23_80) | blk: {
                    const is0x2028or0x2029 =
                        @select(u8, prev2 == @as(V, @splat(0b1110_0010)), true_vec, false_vec) &
                        @select(u8, prev1 == @as(V, @splat(0b1000_0000)), true_vec, false_vec) &
                        (@select(u8, prev0 == @as(V, @splat(0b1010_1000)), true_vec, false_vec) |
                        @select(u8, prev0 == @as(V, @splat(0b1010_1001)), true_vec, false_vec));

                    const is0x85 =
                        @select(u8, prev1 == @as(V, @splat(0b1100_0010)), true_vec, false_vec) &
                        @select(u8, prev0 == @as(V, @splat(0b1000_0101)), true_vec, false_vec);
                    break :blk is0x2028or0x2029 | is0x85;
                };
            }

            if (0 != @reduce(.Max, errs))
                return error.InvalidUtf8;

            // If there are incomplete multibyte characters at the end of the block,
            // write that data into `self.err`.
            // e.g. if there is a 4-byte character, but it's 3 bytes from the end.
            //
            // If the previous chunk's last 3 bytes match this, they're too short (if they ended at EOF):
            // ... 1111____ 111_____ 11______
            self.is_invalid_place_to_end = 1 == (@intFromBool(prev0s[1][15] >= 0b11110000) | @intFromBool(prev0s[2][15] >= 0b11100000) | @intFromBool(prev0s[3][15] >= 0b11000000));
            self.leftovers = .{ prev0s[1], prev0s[2], prev0s[3] };
        }

        fn validateChunk(self: *@This(), input: V) !void {
            // Check whether the current bytes are valid UTF-8.
            // Flip prev1...prev3 so we can easily determine if they are 2+, 3+ or 4+ lead bytes
            // (2, 3, 4-byte leads become large positive numbers instead of small negative numbers)

            const prev_input = self.prev_input_block;
            self.prev_input_block = input;
            const prev1 = shift_in_prev(1, input, prev_input);
            const sc = check_special_cases(input, prev1);
            const prev2 = shift_in_prev(2, input, prev_input);
            const prev3 = shift_in_prev(3, input, prev_input);
            const must23_80 = must_be_2_3_continuation(prev2, prev3);

            const ones: V = @bitCast([_]u8{0x01} ** @sizeOf(V));
            const msbs: V = @bitCast([_]u8{0x80} ** @sizeOf(V));

            const err = (must23_80 ^ sc) | if (USE_SWAR) blk: {
                // Will have a zero byte in `y` if there was a '\u{2028}' or '\u{2029}'
                const x = (input ^ (0b1010_1000 * ones));
                const y = (prev2 ^ (0b1110_0010 * ones)) | (prev1 ^ (0b1000_0000 * ones)) | (x & (x ^ ones));

                // Will have a zero byte in `z` if there was a '\u{85}'
                const z = (prev1 ^ (0b1100_0010 * ones)) | (input ^ (0b1000_0101 * ones));

                // Doesn't necessarily tell us the position, but tells us if there were any zero bytes.
                const has0x2028or0x2029 = (y -% ones) & ~y;
                const has0x85 = (z -% ones) & ~z;
                break :blk (has0x2028or0x2029 | has0x85) & msbs;
            } else blk: {
                const true_vec = @as(V, @splat(0xFF));
                const false_vec = @as(V, @splat(0));

                const is0x2028or0x2029 =
                    @select(u8, prev2 == @as(V, @splat(0b1110_0010)), true_vec, false_vec) &
                    @select(u8, prev1 == @as(V, @splat(0b1000_0000)), true_vec, false_vec) &
                    (@select(u8, input == @as(V, @splat(0b1010_1000)), true_vec, false_vec) |
                    @select(u8, input == @as(V, @splat(0b1010_1001)), true_vec, false_vec));

                const is0x85 =
                    @select(u8, prev1 == @as(V, @splat(0b1100_0010)), true_vec, false_vec) &
                    @select(u8, input == @as(V, @splat(0b1000_0101)), true_vec, false_vec);
                break :blk is0x2028or0x2029 | is0x85;
            };

            if ((0 != if (USE_SWAR)
                err
            else if (comptime builtin.cpu.arch == .arm or builtin.cpu.arch == .armeb)
                @as(native_int, @bitCast(err))
            else
                @reduce(if (comptime builtin.cpu.arch == .x86_64) .Or else .Max, err)))
            {
                return error.InvalidUtf8;
            }

            // If there are incomplete multibyte characters at the end of the block,
            // write that data into `self.err`.
            // e.g. if there is a 4-byte character, but it's 3 bytes from the end.
            //
            // If the previous input's last 3 bytes match this, they're too short (if they ended at EOF):
            // ... 1111____ 111_____ 11______
            self.is_invalid_place_to_end = 0 !=
                if (USE_SWAR)
                msbs & input & ((input | msbs) -% comptime max_value: {
                    var max_array = [1]u8{0} ** @sizeOf(V);
                    max_array[@sizeOf(V) - 3] = 0b11110000 - 0x80;
                    max_array[@sizeOf(V) - 2] = 0b11100000 - 0x80;
                    max_array[@sizeOf(V) - 1] = 0b11000000 - 0x80;
                    break :max_value @as(V, @bitCast(max_array));
                })
            else blk: {
                const max_value = input -| comptime max_value: {
                    var max_array: V = @splat(0xFF);
                    max_array[@sizeOf(V) - 3] = 0b11110000 - 1;
                    max_array[@sizeOf(V) - 2] = 0b11100000 - 1;
                    max_array[@sizeOf(V) - 1] = 0b11000000 - 1;
                    break :max_value max_array;
                };

                // https://github.com/llvm/llvm-project/issues/79779
                if (comptime builtin.cpu.arch == .arm or builtin.cpu.arch == .armeb)
                    break :blk @as(native_int, @bitCast(max_value));
                break :blk @reduce(if (comptime builtin.cpu.arch == .x86_64) .Or else .Max, max_value);
            };
        }

        fn errors(checker: @This()) !void {
            if (checker.is_invalid_place_to_end)
                return error.InvalidUtf8;
        }
    };
}

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

//--------------------------------------------------------------------------------------------
// Peephole optimization helpers:

// Forces the compiler to use the `andn` instruction on some targets.
// Kinda unfortunate but LLVM just doesn't make good decisions regarding op-fusion that much.
// https://github.com/llvm/llvm-project/issues/108840
// https://github.com/llvm/llvm-project/issues/103501 (sorta)
// https://github.com/llvm/llvm-project/issues/85857
// https://github.com/llvm/llvm-project/issues/71389
// https://github.com/llvm/llvm-project/issues/112425
fn andn(src: anytype, mask: @TypeOf(src)) @TypeOf(src) {
    switch (builtin.cpu.arch) {
        .x86_64 => if (std.Target.x86.featureSetHas(builtin.cpu.features, .bmi2)) {
            return asm ("andn %[src], %[mask], %[ret]"
                : [ret] "=r" (-> @TypeOf(src)),
                : [src] "r" (src),
                  [mask] "r" (mask),
            );
        },

        .riscv64 => if (std.Target.riscv.featureSetHas(builtin.cpu.features, .zbb) or std.Target.riscv.featureSetHas(builtin.cpu.features, .zbkb)) {
            return asm ("andn %[ret], %[src], %[mask]"
                : [ret] "=r" (-> @TypeOf(src)),
                : [src] "r" (src),
                  [mask] "r" (mask),
            );
        },

        .aarch64, .aarch64_be => {
            return asm ("bic %[ret], %[src], %[mask]"
                : [ret] "=r" (-> @TypeOf(src)),
                : [src] "r" (src),
                  [mask] "r" (mask),
            );
        },

        .powerpc64 => {
            return asm ("andc %[ret], %[src], %[mask]"
                : [ret] "=r" (-> @TypeOf(src)),
                : [src] "r" (src),
                  [mask] "r" (mask),
            );
        },

        else => {},
    }

    return src & ~mask;
}

fn intersect_byte_halves(a: anytype, b: anytype) std.meta.Int(.unsigned, @typeInfo(@TypeOf(a, b)).vector.len) {
    assert(std.simd.countTrues(@popCount(a) == @as(@TypeOf(a, b), @splat(1))) == @typeInfo(@TypeOf(a, b)).vector.len);

    return @bitCast(if (comptime std.Target.x86.featureSetHas(builtin.cpu.features, .avx512bw))
        @as(@TypeOf(a, b), @splat(0)) != (a & b)
    else
        a == (a & b));
}

// Oftentimes, we want to combine two bitstrings via `|`.
// However, when we know they are disjoint, we can use `+%` instead.
// This enables optimizations on x86 like using `lea` instead of `or`,
// which can fuse a left-shift for one of the operands, and can move to a
// different destination register. Interestingly, the compiler seems afraid of
// doing `kaddq` in `k`-registers, so we should double-check how using this instead
// of an `|` changes the emit.
fn disjoint_or(a: anytype, b: anytype) @TypeOf(a, b) {
    assert((a & b) == 0);
    return a +% b;
}

// Works around https://github.com/llvm/llvm-project/issues/110868
fn unmovemask32(x: u32) @Vector(32, bool) {
    if (comptime std.Target.x86.featureSetHas(builtin.cpu.features, .avx512bw)) {
        return @bitCast(x);
    }

    const bit_positions = comptime std.simd.repeat(32, @as(@Vector(8, u8), @splat(1)) << std.simd.iota(u3, 8));
    const shuffled_x = @shuffle(u8, @as(@Vector(32, u8), @bitCast(@as(@Vector(4, u64), @splat(x)))), undefined, (std.simd.iota(u8, 32) >> @splat(4) << @splat(4)) + (std.simd.iota(u8, 32) >> @splat(3)));
    const T = [2]@Vector(16, u8);
    var bit_positions_and_shuffled_x: T = undefined;

    // Works around https://github.com/llvm/llvm-project/issues/110875
    // Helps sandybridge (avx) and goldmont (sse4_2) targets too
    for (&bit_positions_and_shuffled_x, @as(T, @bitCast(bit_positions)), @as(T, @bitCast(shuffled_x))) |*slot, a, b| {
        slot.* = a & b;
    }

    return bit_positions == @as(@Vector(32, u8), @bitCast(bit_positions_and_shuffled_x));
}

// Works around https://github.com/llvm/llvm-project/issues/110868
fn unmovemask64(x: u64) @Vector(64, bool) {
    if (comptime std.Target.x86.featureSetHas(builtin.cpu.features, .avx512bw)) {
        return @bitCast(x);
    }

    const bit_positions = comptime std.simd.repeat(64, @as(@Vector(8, u8), @splat(1)) << std.simd.iota(u3, 8));
    const shuffled_x = @shuffle(u8, @as(@Vector(64, u8), @bitCast(@as(@Vector(8, u64), @splat(x)))), undefined, (std.simd.iota(u8, 64) >> @splat(4) << @splat(4)) + (std.simd.iota(u8, 64) >> @splat(3)));
    const T = [2]@Vector(32, u8);
    var bit_positions_and_shuffled_x: T = undefined;

    // Works around https://github.com/llvm/llvm-project/issues/110875
    // Helps sandybridge (avx) and goldmont (sse4_2) targets too
    for (&bit_positions_and_shuffled_x, @as(T, @bitCast(bit_positions)), @as(T, @bitCast(shuffled_x))) |*slot, a, b| {
        slot.* = a & b;
    }

    return bit_positions == @as(@Vector(64, u8), @bitCast(bit_positions_and_shuffled_x));
}

/// Equivalent to @bitReverse(@bitReverse(x) -% @bitReverse(y));
// https://github.com/llvm/llvm-project/issues/111046
fn reversedSubtraction(x: u64, y: u64) u64 {
    const Helpers = struct {
        fn bswap(a: u64) u64 {
            var result: u64 = a;
            asm ("bswap %[result]"
                : [result] "+r" (result),
            );
            return result;
        }

        fn vpsubq(a: u64, b: u64) u64 {
            if (@inComptime()) return a -% b;
            return (asm ("vpsubq %[b], %[a], %[ret]"
                : [ret] "=v" (-> @Vector(2, u64)),
                : [a] "v" (a),
                  [b] "v" (b),
            ))[0];
        }

        fn bitReverse1(a: u64) u64 {
            return @byteSwap(@as(u64, @bitCast(@bitReverse(@as(@Vector(8, u8), @bitCast(a))))));
        }

        fn bitReverse2(a: u64) u64 {
            return @bitCast(@bitReverse(@as(@Vector(8, u8), @bitCast(@byteSwap(a)))));
        }

        fn bitReverse3(a: @Vector(2, u64)) [2]u64 {
            return @bitCast(@bitReverse(@as(@Vector(16, u8), @bitCast(@byteSwap(a)))));
        }
    };

    // https://github.com/llvm/llvm-project/issues/112425
    if (comptime std.Target.x86.featureSetHas(builtin.cpu.features, .avx512bw)) {
        const a: u64 = Helpers.bitReverse2(x);
        const b: u64 = Helpers.bitReverse2(y);
        return Helpers.bitReverse1(Helpers.vpsubq(a, b));
    } else {
        const vec = Helpers.bitReverse3(.{ (x), (y) });
        return Helpers.bitReverse1(vec[0] -% vec[1]);
    }
}

//--------------------------------------------------------------------------------------------

fn loadNonTemporal(ptr: anytype) @typeInfo(@TypeOf(ptr)).pointer.child {
    // @prefetch(ptr, .{
    //     .rw = .read,
    //     .locality = 0, // 0 means no temporal locality. That is, the data can be immediately dropped from the cache after it is accessed.
    //     .cache = .data,
    // });
    return struct {
        extern fn @"llvm.x86.avx512.movntdqa"(@TypeOf(ptr)) @typeInfo(@TypeOf(ptr)).pointer.child;
    }.@"llvm.x86.avx512.movntdqa"(ptr);
}

fn storeNonTemporal(ptr: anytype, d: anytype) void {
    if (comptime @sizeOf(@TypeOf(d)) > if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) 64 else 32) {
        const half_size_of = @sizeOf(@TypeOf(d)) / 2;
        const half1, const half2 = @as([2]@Vector(half_size_of, u8), @bitCast(d));
        storeNonTemporal(ptr, half1);
        storeNonTemporal(ptr + half_size_of, half2);
    } else {
        struct {
            extern fn @"llvm.x86.avx.movnt.dq"(@TypeOf(ptr), @TypeOf(d)) void;
        }.@"llvm.x86.avx.movnt.dq"(ptr, d);
    }
}

fn clflushopt(ptr: anytype) void {
    struct {
        extern fn @"llvm.x86.clflushopt"(@TypeOf(ptr)) void;
    }.@"llvm.x86.clflushopt"(ptr);
}

// test "utf8-validator" {
//     var utf8_checker: Utf8Checker(.{
//         .USE_ARM_NEON = true,
//         .V = @Vector(16, u8),
//         .native_int = u128,
//     }) = .empty;
//     var buf: [64]u8 = [1]u8{0} ** 64;
//     var i: u32 = 0;

//     // 0f f1 80 80

//     while (true) {
//         inline for (buf[0..4], @as([4]u8, @bitCast(i))) |*slot, b| {
//             slot.* = b;
//         }

//         const is_valid_cp = std.unicode.utf8ValidateSlice(&@as([4]u8, @bitCast(i)));
//         const is_valid_cp2 = if (utf8_checker.validateChunksArm(std.simd.deinterlace(4, @as(@Vector(64, u8), @bitCast(buf))))) |_| true else |_| false;

//         if (is_valid_cp != is_valid_cp2) {
//             std.debug.print("Stopped at: {x:0>2} {} {}\n", .{ @as([4]u8, @bitCast(i)), is_valid_cp, is_valid_cp2 });
//         }

//         try std.testing.expectEqual(is_valid_cp, is_valid_cp2);

//         i +%= 1;
//         if (i == 0) break;
//     }
// }

const DEFAULT_NOT = true;

fn vpnot(a_: anytype) @TypeOf(a_) {
    if (DEFAULT_NOT) return ~a_;
    const a: @Vector(8, u64) = @bitCast(a_);
    return @bitCast(vpternlog(a, a, a, 51));
}

fn vpternlog(vec_1: @Vector(8, u64), vec_2: @Vector(8, u64), vec_3: @Vector(8, u64), comptime i: i32) @Vector(8, u64) {
    var result = vec_1;
    asm volatile (
        \\ vpternlogq %[imm], %[vec_3], %[vec_2], %[vec_1]
        : [vec_1] "+x" (result),
        : [vec_2] "x" (vec_2),
          [vec_3] "x" (vec_3),
          [imm] "n" (i),
    );
    return result;
}

fn pext(a: anytype, b: anytype) if (@bitSizeOf(@TypeOf(a)) >= @bitSizeOf(@TypeOf(b))) @TypeOf(a) else @TypeOf(b) {
    const T = if (@bitSizeOf(@TypeOf(a)) >= @bitSizeOf(@TypeOf(b))) @TypeOf(a) else @TypeOf(b);

    if (@inComptime() or !HAS_FAST_PDEP_AND_PEXT) {
        @compileError("pls add pext implementation");
        // var src = a;
        // var mask = b;
        // var result: T = 0;

        // while (true) {
        //     // 1. isolate the lowest set bit of mask
        //     const lowest: T = (-%mask & mask);

        //     if (lowest == 0) break;

        //     // 2. populate LSB from src
        //     const LSB: T = @bitCast(@as(std.meta.Int(.signed, @bitSizeOf(T)), @bitCast(src << (@bitSizeOf(T) - 1))) >> (@bitSizeOf(T) - 1));

        //     // 3. copy bit from mask
        //     result |= LSB & lowest;

        //     // 4. clear lowest bit
        //     mask &= ~lowest;

        //     // 5. prepare for next iteration
        //     src >>= 1;
        // }

        // return result;
    }

    if (@bitSizeOf(T) > 64)
        @compileError(std.fmt.comptimePrint("Cannot pext a type larger than 64 bits. Got: {}", .{T}));

    const methods = struct {
        extern fn @"llvm.x86.bmi.pext.32"(u32, u32) u32;
        extern fn @"llvm.x86.bmi.pext.64"(u64, u64) u64;
        extern fn @"llvm.ppc.pextd"(u64, u64) u64;
    };

    return switch (builtin.cpu.arch) {
        .powerpc64, .powerpc64le => methods.@"llvm.ppc.pextd"(a, b),
        .x86, .x86_64 => switch (T) {
            u32 => methods.@"llvm.x86.bmi.pext.32"(a, b),
            else => methods.@"llvm.x86.bmi.pext.64"(a, b),
        },
        else => unreachable,
    };
}

fn pdep(a: anytype, b: anytype) if (@bitSizeOf(@TypeOf(a)) >= @bitSizeOf(@TypeOf(b))) @TypeOf(a) else @TypeOf(b) {
    const T = if (@bitSizeOf(@TypeOf(a)) >= @bitSizeOf(@TypeOf(b))) @TypeOf(a) else @TypeOf(b);

    if (@inComptime() or !HAS_FAST_PDEP_AND_PEXT) {
        var src = a;
        var mask = b;
        var result: T = 0;

        while (true) {
            // 1. isolate the lowest set bit of mask
            const lowest: T = (-%mask & mask);

            if (lowest == 0) break;

            // 2. populate LSB from src
            const LSB: T = @bitCast(@as(std.meta.Int(.signed, @bitSizeOf(T)), @bitCast(src << (@bitSizeOf(T) - 1))) >> (@bitSizeOf(T) - 1));

            // 3. copy bit from mask
            result |= LSB & lowest;

            // 4. clear lowest bit
            mask &= ~lowest;

            // 5. prepare for next iteration
            src >>= 1;
        }

        return result;
    }

    if (@bitSizeOf(T) > 64)
        @compileError(std.fmt.comptimePrint("Cannot pdep a type larger than 64 bits. Got: {}", .{T}));

    const methods = struct {
        extern fn @"llvm.x86.bmi.pdep.32"(u32, u32) u32;
        extern fn @"llvm.x86.bmi.pdep.64"(u64, u64) u64;
        extern fn @"llvm.ppc.pdepd"(u64, u64) u64;
    };

    return switch (builtin.cpu.arch) {
        .powerpc64, .powerpc64le => methods.@"llvm.ppc.pdepd"(a, b),
        .x86, .x86_64 => switch (T) {
            u32 => methods.@"llvm.x86.bmi.pdep.32"(a, b),
            else => methods.@"llvm.x86.bmi.pdep.64"(a, b),
        },
        else => unreachable,
    };
}

fn bzhi(src: u64, mask: u64) u64 {
    if (std.Target.x86.featureSetHas(builtin.cpu.features, .bmi2)) {
        return struct {
            extern fn @"llvm.x86.bmi.bzhi.64"(u64, u64) u64;
        }.@"llvm.x86.bmi.bzhi.64"(src, mask);
    } else {
        const m: u8 = @truncate(mask);
        return src & ~if (m >= 64) @as(u64, 0) else (~@as(u64, 0) << @intCast(m));
    }
}

// Workaround until https://github.com/llvm/llvm-project/issues/79094 is solved.
fn expand8xu8To16xu4AsByteVector(vec: @Vector(8, u8)) @Vector(16, u8) {
    return switch (comptime builtin.cpu.arch.endian()) {
        .little => switch (builtin.cpu.arch) {
            // Doesn't have shifts that operate at byte granularity.
            // To do a shift with byte-granularity, the compiler must insert an `&` operation.
            // Therefore, it's better to do a single `&` after interlacing, and get a 2-for-1.
            // We need to have all these bitCasts because of https://github.com/llvm/llvm-project/issues/89600
            .x86_64 => std.simd.interlace([2]@Vector(8, u8){ vec, @bitCast(@as(@Vector(4, u16), @bitCast(vec)) >> @splat(4)) }) & @as(@Vector(16, u8), @splat(0xF)),

            else => std.simd.interlace(.{ vec & @as(@Vector(8, u8), @splat(0xF)), vec >> @splat(4) }),
        },
        .big => std.simd.interlace(.{ vec >> @splat(4), vec & @as(@Vector(8, u8), @splat(0xF)) }),
    };
}

fn shuffle(table: anytype, indices: anytype) @Vector(@typeInfo(@TypeOf(indices)).vector.len, std.meta.Child(@TypeOf(table))) {
    const T = @Vector(switch (@typeInfo(@TypeOf(table))) {
        .array => |info| info.len,
        .vector => |info| info.len,
        else => @compileError("Invalid type passed to shuffle function."),
    }, std.meta.Child(@TypeOf(table)));
    const child_type = std.meta.Child(@TypeOf(table));
    switch (builtin.cpu.arch) {
        .x86_64 => {
            const UNIQUE_INDICES_AT_ONCE = 16;
            const VEC_LEN = std.simd.suggestVectorLengthForCpu(child_type, builtin.cpu).?;
            const I_LEN = @typeInfo(@TypeOf(indices)).vector.len;

            if (I_LEN < VEC_LEN)
                @compileError("Come implement this please");
            //                return vpshufb(@shuffle(child_type, table, undefined, std.simd.repeat(@sizeOf(@TypeOf(indices)), std.simd.iota(i32, @typeInfo(T).vector.len))), indices);

            // Break the indices into chunks
            var parts: [@divExact(I_LEN, VEC_LEN)]@Vector(VEC_LEN, @typeInfo(T).vector.child) = undefined;
            inline for (&parts, 0..) |*slot, i| {
                const sub_indices = std.simd.extract(indices, i * VEC_LEN, VEC_LEN);
                const Q = @TypeOf(slot.*);
                var result: Q = @splat(0);

                // Break the table into 16-byte chunks
                inline for (0..@typeInfo(T).vector.len / UNIQUE_INDICES_AT_ONCE) |k| {
                    result |= vpshufb(std.simd.repeat(VEC_LEN, std.simd.extract(table, k * UNIQUE_INDICES_AT_ONCE, UNIQUE_INDICES_AT_ONCE)), @as(Q, @splat(0x70)) +| (sub_indices -% @as(Q, @splat(16 * k))));
                }

                slot.* = result;
            }
            return @bitCast(parts);
        },
        else => @compileError(std.fmt.comptimePrint("{s} {s} does not have a shuffle implementation yet!\n", .{ builtin.cpu.arch.genericName(), builtin.cpu.model.name })),
        // .arm, .armeb => {},
        // .aarch64, .aarch64_32, .aarch64_be => {},
    }
}

fn vperm(table: @Vector(64, u8), indices: anytype) @TypeOf(indices) {
    if (@inComptime() or comptime !std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi))
        return shuffle(table, indices & @as(@TypeOf(indices), @splat(0b0011_1111)));

    const padding_len = 64 - @typeInfo(@TypeOf(indices)).vector.len;
    const extended_indices = if (padding_len == 0) indices else std.simd.join(indices, @as(@Vector(padding_len, u8), @splat(undefined)));
    return std.simd.extract(struct {
        extern fn @"llvm.x86.avx512.permvar.qi.512"(@Vector(64, u8), @Vector(64, u8)) @Vector(64, u8);
    }.@"llvm.x86.avx512.permvar.qi.512"(table, extended_indices), 0, @typeInfo(@TypeOf(indices)).vector.len);
}

fn vpermWithFallback(table: @Vector(64, u8), indices: @Vector(64, u8), fallback: @Vector(64, u8), mask: u64) @Vector(64, u8) {
    if (@inComptime() or comptime !std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi))
        return @select(u8, @as(@Vector(64, bool), @bitCast(mask)), vperm(table, indices), fallback);

    return struct {
        extern fn @"llvm.x86.avx512.mask.permvar.qi.512"(@Vector(64, u8), @Vector(64, u8), @Vector(64, u8), u64) @Vector(64, u8);
    }.@"llvm.x86.avx512.mask.permvar.qi.512"(table, indices, fallback, mask);
}

fn padWithUndefineds(T: type, value: anytype) if (@sizeOf(@TypeOf(value)) > @sizeOf(T)) @TypeOf(value) else T {
    const padding_len = @typeInfo(T).vector.len - @typeInfo(@TypeOf(value)).vector.len;
    return if (padding_len <= 0) value else std.simd.join(value, @as(@Vector(padding_len, @typeInfo(T).vector.child), @splat(undefined)));
}

fn vpbroadcast(T: type, a: anytype) T {
    // if ( != u8) @compileError("This workaround is just for byte vectors, home slice.");
    //    if (@inComptime() or comptime (builtin.cpu.arch != .x86_64))
    //        return @splat(@as(@Vector(@sizeOf(T) / @sizeOf(@typeInfo(T).vector.child), @typeInfo(@TypeOf(a)).vector.child), @bitCast(a))[0]);
    //@compileLog(std.simd.extract(a, 0, @sizeOf(@typeInfo(T).vector.child)));
    return @bitCast(std.simd.repeat(@sizeOf(T), asm ("vpbroadcast" ++ switch (@typeInfo(T).vector.child) {
            u8 => "b",
            u16 => "w",
            u32 => "d",
            u64 => "q",
            else => @compileError("[broadcast] Invalid vector type. `child` must be a u8, u16, u32, or u64."),
        } ++ " %[a], %[ret]"
        : [ret] "=v" (-> NATIVE_CHAR_VEC),
        : [a] "v" (padWithUndefineds(@Vector(16, u8), std.simd.extract(a, 0, @sizeOf(@typeInfo(T).vector.child)))),
    )));
}

fn splat(a: anytype) @Vector(64, u8) {
    if (@typeInfo(@TypeOf(a)).vector.child != u8) @compileError("This workaround is just for byte vectors, home slice.");
    if (@inComptime() or comptime builtin.cpu.arch != .x86_64)
        return @splat(a[0]);

    const ret = asm ("vpbroadcastb %[a], %[ret]"
        : [ret] "=v" (-> @Vector(NATIVE_VEC_SIZE, u8)),
        : [a] "v" (padWithUndefineds(@Vector(16, u8), a)),
    );

    return switch (NATIVE_VEC_SIZE) {
        32 => std.simd.join(ret, ret),
        64 => ret,
        else => @compileError(std.fmt.comptimePrint("NATIVE_VEC_SIZE of {} not supported.", .{NATIVE_VEC_SIZE})),
    };
}

fn splat16(a: anytype) @Vector(16, u8) {
    if (@typeInfo(@TypeOf(a)).vector.child != u8) @compileError("This workaround is just for byte vectors, home slice.");
    if (@inComptime() or comptime builtin.cpu.arch != .x86_64)
        return @splat(a[0]);

    return asm ("vpbroadcastb %[a], %[ret]"
        : [ret] "=v" (-> @Vector(16, u8)),
        : [a] "v" (padWithUndefineds(@Vector(16, u8), a)),
    );
}

//    fn vpermFallback(table: @Vector(64, u8), indices: @Vector(64, u8), fallback: @Vector(16, u8), mask: @Vector(16, bool)) @TypeOf(indices) {
//        if (@inComptime() or comptime !std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi))
//            return shuffle(table, indices & @as(@TypeOf(indices), @splat(0b0011_1111)));
//
//        return std.simd.extract(struct {
//            extern fn @"llvm.x86.avx512.mask.permvar.qi.512"(@Vector(64, u8), @Vector(64, u8), @Vector(64, u8), u64) @Vector(64, u8);
//        }.@"llvm.x86.avx512.mask.permvar.qi.512"(table, padWithUndefineds(@Vector(64, u8), indices), padWithUndefineds(@Vector(64, u8), fallback), @as(u16, @bitCast(mask))), 0, @typeInfo(@TypeOf(indices)).vector.len);
//    }

// USE THIS FOR THE BENCHMARK ONLY! THIS WILL BREAK IF THE K-REGISTER DOES NOT MATCH THE HARDCODED ONE!!!
export fn vpermt2b(vec_1: @Vector(64, u8), indices: @Vector(64, u8), vec_2: @Vector(64, u8), mask: u64) @Vector(64, u8) {
    var result = vec_1;
    asm volatile (
        \\ vpermt2b %[vec_2], %[indices], %[vec_1] {k1} {z}
        : [vec_1] "+x" (result),
        : [indices] "x" (indices),
          [vec_2] "x" (vec_2),
          [mask] "k" (mask),
    );
    return result;
}

fn vperm2_zmask(table: @Vector(128, u8), indices: anytype, mask: u64) @TypeOf(indices) {
    // if (@inComptime() or comptime !std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi))
    //     return shuffle(table, indices & @as(@TypeOf(indices), @splat(0b0111_1111)));

    const table_part_1, const table_part_2 = @as([2]@Vector(64, u8), @bitCast(table));
    // USE THIS FOR THE BENCHMARK ONLY! THIS WILL BREAK IF THE K-REGISTER DOES NOT MATCH THE HARDCODED ONE!!!
    // return vpermt2b(table_part_1, indices, table_part_2, mask);
    return std.simd.extract(
        struct {
            extern fn @"llvm.x86.avx512.maskz.vpermt2var.qi.512"(@Vector(64, u8), @Vector(64, u8), @Vector(64, u8), u64) @Vector(64, u8);
        }.@"llvm.x86.avx512.maskz.vpermt2var.qi.512"(
            padWithUndefineds(@Vector(64, u8), indices),
            table_part_1,
            table_part_2,
            mask,
        ),
        0,
        @typeInfo(@TypeOf(indices)).vector.len,
    );
}

fn vperm2(table: @Vector(128, u8), indices: anytype) @TypeOf(indices) {
    if (@inComptime() or comptime !std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi))
        return shuffle(table, indices & @as(@TypeOf(indices), @splat(0b0111_1111)));

    const table_part_1, const table_part_2 = @as([2]@Vector(64, u8), @bitCast(table));
    return std.simd.extract(struct {
        extern fn @"llvm.x86.avx512.vpermi2var.qi.512"(@Vector(64, u8), @Vector(64, u8), @Vector(64, u8)) @Vector(64, u8);
    }.@"llvm.x86.avx512.vpermi2var.qi.512"(table_part_1, padWithUndefineds(@Vector(64, u8), indices), table_part_2), 0, @typeInfo(@TypeOf(indices)).vector.len);
}

fn vperm4(table: @Vector(256, u8), indices: anytype) @TypeOf(indices) {
    if (@inComptime() or comptime !std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi))
        return shuffle(table, indices);

    const table_part_1, const table_part_2, const table_part_3, const table_part_4 = @as([4]@Vector(64, u8), @bitCast(table));
    const padded_indices = padWithUndefineds(@Vector(64, u8), indices);
    const Intrinsic = struct {
        extern fn @"llvm.x86.avx512.vpermi2var.qi.512"(@Vector(64, u8), @Vector(64, u8), @Vector(64, u8)) @Vector(64, u8);
    };

    return std.simd.extract(@select(
        u8,
        padWithUndefineds(@Vector(64, bool), indices < @as(@TypeOf(indices), @splat(128))),
        Intrinsic.@"llvm.x86.avx512.vpermi2var.qi.512"(table_part_1, padded_indices, table_part_2),
        Intrinsic.@"llvm.x86.avx512.vpermi2var.qi.512"(table_part_3, padded_indices, table_part_4),
    ), 0, @typeInfo(@TypeOf(indices)).vector.len);
}

fn vperm2WithFallback(table: @Vector(128, u8), indices: @Vector(64, u8), comptime fallback: @Vector(64, u8)) @Vector(64, u8) {
    if (@inComptime() or comptime !std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi)) {
        if (std.simd.countTrues(fallback == @as(@TypeOf(fallback), @splat(0))) == 64) {
            return shuffle(table, indices);
        } else if (std.simd.countTrues(fallback == @as(@TypeOf(fallback), @splat(0xFF))) == 64) {
            return ~shuffle(~table, indices);
        } else if (std.simd.countTrues(fallback == @as(@TypeOf(fallback), @splat(fallback[0]))) == 64) {
            return shuffle(table -% @as(@TypeOf(table), @splat(fallback[0])), indices) +% fallback;
        } else {
            return @select(u8, indices < @as(@Vector(64, u8), @splat(0x80)), shuffle(table, indices), fallback);
        }
    }

    const table_part_1, const table_part_2 = @as([2]@Vector(64, u8), @bitCast(table));

    return @select(
        u8,
        indices < @as(@Vector(64, u8), @splat(0x80)),
        struct {
            extern fn @"llvm.x86.avx512.vpermi2var.qi.512"(@Vector(64, u8), @Vector(64, u8), @Vector(64, u8)) @Vector(64, u8);
        }.@"llvm.x86.avx512.vpermi2var.qi.512"(table_part_1, indices, table_part_2),
        fallback,
    );
}

// fn vperm2o(table: @Vector(128, u8), indices: anytype) @TypeOf(indices) {
//     if (@inComptime() or comptime !std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi))
//         return ~shuffle(~table, indices);

//     const table_part_1, const table_part_2 = @as([2]@Vector(64, u8), @bitCast(table));

//     return @select(
//         u8,
//         indices < @as(@Vector(64, u8), @splat(0x80)),
//         struct {
//             extern fn @"llvm.x86.avx512.vpermi2var.qi.512"(@Vector(64, u8), @Vector(64, u8), @Vector(64, u8)) @Vector(64, u8);
//         }.@"llvm.x86.avx512.vpermi2var.qi.512"(table_part_1, indices, table_part_2),
//         @as(@Vector(64, u8), @splat(0xFF)),
//     );
// }

// fn vperm2z(table: @Vector(128, u8), indices: @Vector(64, u8)) @Vector(64, u8) {
//     if (@inComptime() or comptime !std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi))
//         return shuffle(table, indices);

//     const table_part_1, const table_part_2 = @as([2]@Vector(64, u8), @bitCast(table));

//     return @select(
//         u8,
//         indices < @as(@Vector(64, u8), @splat(0x80)),
//         struct {
//             extern fn @"llvm.x86.avx512.vpermi2var.qi.512"(@Vector(64, u8), @Vector(64, u8), @Vector(64, u8)) @Vector(64, u8);
//         }.@"llvm.x86.avx512.vpermi2var.qi.512"(table_part_1, indices, table_part_2),
//         @as(@Vector(64, u8), @splat(0)),
//     );
// }

// export fn compressStore(data: @Vector(64, u8), bitstring: u64, dest: [*]u8) void {
//     switch (builtin.cpu.arch) {
//         .x86_64 => {
//             if (@inComptime()) {
//                 var cur_dest = dest[0..];
//                 var cur_bitstr = bitstring;
//                 while (cur_bitstr != 0) {
//                     const slot = @ctz(bitstring);
//                     cur_dest[0] = data[slot];
//                     cur_dest = cur_dest[1..];
//                     cur_bitstr &= cur_bitstr -% 1;
//                 }
//             }
//             if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi2)) {
//                 dest[0..64].* = vpcompress(data, bitstring);
//             }
//         },
//         .arm, .armeb, .aarch64, .aarch64_be => {
//             // FYI: We assume interleaved data on these targets!!
//             comptime var lookups: [256]@Vector(8, u8) = undefined;
//             comptime {
//                 @setEvalBranchQuota(100000);
//                 for (&lookups, 0..) |*slot, i| {
//                     var pos: u8 = 0;
//                     for (0..8) |j| {
//                         const bit: u1 = @truncate(i >> j);
//                         slot[pos] = j / 4 + (j & 3) * 16;
//                         pos += bit;
//                     }

//                     for (pos..8) |j| {
//                         slot[j] = 255;
//                     }
//                 }
//             }

//             const chunks: [4]@Vector(16, u8) = @bitCast(data);

//             const prefix_sum_of_popcounts =
//                 @as(u64, @bitCast(@as(@Vector(8, u8), @popCount(@as(@Vector(8, u8), @bitCast(bitstring)))))) *% 0x0101010101010101;

//             inline for (@as([8]u8, @bitCast(bitstring)), @as([8]u8, @bitCast(prefix_sum_of_popcounts)), 0..) |byte, pos, i| {
//                 dest[pos..][0..8].* = tbl4(chunks[0], chunks[1], chunks[2], chunks[3], lookups[byte] +| @as(@Vector(8, u8), @splat(2 * i)));
//             }
//         },
//     }
// }

fn tbl4(table_part_1: @Vector(16, u8), table_part_2: @Vector(16, u8), table_part_3: @Vector(16, u8), table_part_4: @Vector(16, u8), indices: @Vector(8, u8)) @TypeOf(indices) {
    return struct {
        extern fn @"llvm.aarch64.neon.tbl4"(@TypeOf(table_part_1), @TypeOf(table_part_2), @TypeOf(table_part_3), @TypeOf(table_part_4), @TypeOf(indices)) @TypeOf(indices);
    }.@"llvm.aarch64.neon.tbl4"(table_part_1, table_part_2, table_part_3, table_part_4, indices);
}

// fn run_lengths(bitstr: u32) @Vector(@sizeOf(@TypeOf(bitstr)) * 4, u8) {
//     const iota: u64 = @bitCast(std.simd.iota(u4, 16));
//     var positions_vecs: [2]@Vector(16, u8) = undefined;

//     inline for (&positions_vecs, [2]@TypeOf(ends, starts){ ends, starts }) |*positions_vec, x| {
//         const shift: u6 = @popCount(x & 0xFFFF);
//         assert(shift <= 8);
//         const lower = (pext(iota, pdep(@as(u64, x >> 0), 0x1111111111111111) * 0xF));
//         const upper = (pext(iota, pdep(@as(u64, x >> 16), 0x1111111111111111) * 0xF) << (4 * shift));

//         const upper_16s = vpshufb(@as(@Vector(16, u8), @splat(16)), std.simd.iota(u8, 16) -% @as(@Vector(16, u8), @splat(shift)));

//         // We know the popCount of each 16 bit chunk in `x` is at most 8 by construction.
//         // That means we have to combine a maximum of 16 nibbles, therefore it fits in a u64.
//         // We decompress the 16 nibbles into 16 bytes, then add `upper_16s` to make sure that the upper nibbles start at 16
//         positions_vec.* = expand8xu8To16xu4AsByteVector(@bitCast(lower | upper)) + upper_16s;
//     }

//     return positions_vecs[0] - positions_vecs[1] + @as(@Vector(16, u8), @splat(1));
// }

// From a mask, produces a vector that tells us which indices correspond to set bits in the mask.
// Above the real data is "don't care" data. Not undefined per se, we still want it to be within 0-63,
// because we have a piece of code that does `cur.ptr+indices` that might operate on "don't care" data.
// We could guarantee it to work for ANY byte, by just padding the source file more, but for now we operate
// under the constraints of the current implementation-- that this returns values in the range of 0-63.
fn bitsToIndices(mask: u64, comptime start_index: u7) @Vector(64, u8) {
    if (!@inComptime() and comptime std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi2)) {
        return struct {
            extern fn @"llvm.x86.avx512.mask.compress.b"(@Vector(64, u8), @Vector(64, u8), u64) @Vector(64, u8);
        }.@"llvm.x86.avx512.mask.compress.b"(std.simd.iota(u8, 64) + @as(@Vector(64, u8), @splat(start_index)), @splat(0), mask);
        // Alternatively:
        // }.@"llvm.x86.avx512.mask.compress.b"(std.simd.iota(u8, 64) + @as(@Vector(64, u8), @splat(start_index)), @splat(0), mask);
    } else if (HAS_FAST_PDEP_AND_PEXT) {
        const iota: u64 = @bitCast(std.simd.iota(u4, 16));
        var buffer = [_]u8{0} ** 64;

        inline for (0..4) |i|
            buffer[@popCount(@as(std.meta.Int(.unsigned, i * 16), @truncate(mask)))..][0..16].* =
                expand8xu8To16xu4AsByteVector(@bitCast(pext(iota, pdep(mask >> (i * 16), 0x1111111111111111) * 0xF))) + @as(@Vector(16, u8), @splat(i * 16 + start_index));

        return @bitCast(buffer);
    }
}

fn compress(compressable: @Vector(64, u8), mask: u64) @Vector(64, u8) {
    if (!@inComptime() and comptime std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi2)) {
        return struct {
            extern fn @"llvm.x86.avx512.mask.compress.b"(@Vector(64, u8), @Vector(64, u8), u64) @Vector(64, u8);
        }.@"llvm.x86.avx512.mask.compress.b"(compressable, @splat(0), mask);
    } else if (HAS_FAST_PDEP_AND_PEXT) {
        const iota: u64 = @bitCast(std.simd.iota(u4, 16));
        var buffer = [_]u8{0} ** 64;

        inline for (@as([4]@Vector(16, u8), @bitCast(compressable)), 0..) |compressable_part, i|
            buffer[@popCount(@as(std.meta.Int(.unsigned, i * 16), @truncate(mask)))..][0..16].* =
                vpshufb(compressable_part, expand8xu8To16xu4AsByteVector(@bitCast(pext(iota, pdep(mask >> (i * 16), 0x1111111111111111) * 0xF))) + @as(@Vector(16, u8), @splat(i * 16)));

        return @bitCast(buffer);
    }
}

// Expands a vector of up to max_index bytes into a 64 byte vector based on a mask.
// We could go bigger but we would need more vpshufb's.
fn expand(expandable: anytype, fallback: @Vector(64, u8), pos_mask: u64) @Vector(64, u8) {
    //if (@TypeOf(expandable) == @Vector(1, u8)) {
    //    return @select(u8, unmovemask64(predicate_mask), @as(@Vector(64, u8), @splat(expandable[0])), fallback);
    //} else if (@TypeOf(expandable) == @Vector(2, u8)) {
    //    var ans = @select(u8, unmovemask64(predicate_mask & pos_mask & -%pos_mask), @as(@Vector(64, u8), @splat(expandable[0])), fallback);
    //    ans = @select(u8, unmovemask64(predicate_mask & (predicate_mask -% 1)), @as(@Vector(64, u8), @splat(expandable[1])), fallback);
    //    return ans;
    //}
    // return @select(u8, unmovemask64(predicate_mask), @as(@Vector(64, u8), @bitCast(buffer)), fallback);

    if (@typeInfo(@TypeOf(expandable)).vector.child != u8) @compileError("`expandable` only works on u8's for now.");

    if (!@inComptime() and comptime std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi2)) {
        return struct {
            extern fn @"llvm.x86.avx512.mask.expand.b"(@Vector(64, u8), @Vector(64, u8), u64) @Vector(64, u8);
        }.@"llvm.x86.avx512.mask.expand.b"(padWithUndefineds(@Vector(64, u8), expandable), fallback, pos_mask);
    } else if (HAS_FAST_PDEP_AND_PEXT) {
        var indices_arr: [4]@Vector(16, u8) = undefined;

        inline for (&indices_arr, 0..) |*indices_slot, i| {
            indices_slot.* =
                expand8xu8To16xu4AsByteVector(@bitCast(pdep(pos_mask >> (i * 16), 0x1111111111111111) *% 0x1111111111111110)) + @as(@Vector(16, u8), @splat(@popCount(@as(std.meta.Int(.unsigned, i * 16), @truncate(pos_mask)))));
        }

        const indices: @Vector(64, u8) = @bitCast(indices_arr);

        return shuffle(padWithUndefineds(@Vector(16, u8), expandable), indices);

        //inline for (0..4) |i| {
        //    // shuffle(table: anytype, indices: anytype)
        //    const raw_indices =
        //
        //
        //    if (iters == 1) {
        //        buffer[i * 16 ..][0..16].* = vpshufb(std.simd.extract(expandable_vec, 0, 16), raw_indices);
        //    } else {
        //        var result: @TypeOf(vpshufb(expandable_vec, raw_indices)) = undefined;
        //        inline for (0..@min(i + 1, iters)) |j| {
        //            result |= vpshufb(expandable_vec, raw_indices -% @as(@Vector(16, u8), @splat(j * 0x10)) +| @as(@Vector(16, u8), @splat(0x70)));
        //        }
        //
        //        buffer[i * 16 ..][0..16].* = result;
        //    }
        //}
    } else {
        unreachable;
    }
}

fn pextComptime(src: anytype, comptime mask: @TypeOf(src)) @TypeOf(src) {
    if (mask == 0) return 0;
    const num_one_groups = @popCount(mask & ~(mask << 1));

    if (!@inComptime() and comptime num_one_groups >= 2 and @bitSizeOf(@TypeOf(src)) <= 64 and HAS_FAST_PDEP_AND_PEXT) {
        const methods = struct {
            extern fn @"llvm.x86.bmi.pext.32"(u32, u32) u32;
            extern fn @"llvm.x86.bmi.pext.64"(u64, u64) u64;
            extern fn @"llvm.ppc.pextd"(u64, u64) u64;
        };

        return @intCast(switch (builtin.cpu.arch) {
            .powerpc64, .powerpc64le => methods.@"llvm.ppc.pextd"(src, mask),
            .x86, .x86_64 => if (@bitSizeOf(@TypeOf(src)) <= 32)
                methods.@"llvm.x86.bmi.pext.32"(src, mask)
            else
                methods.@"llvm.x86.bmi.pext.64"(src, mask),
            else => unreachable,
        });
        //return switch (@TypeOf(src)) {
        //u32 => methods.@"llvm.x86.bmi.pext.32"(src, mask),
        //u64 => methods.@"llvm.x86.bmi.pext.64"(src, mask),
        //else => @intCast(pextComptime(@as(if (@bitSizeOf(@TypeOf(src)) <= 32) u32 else u64, src), mask)),
        //};
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

// Debug print functions
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

fn printb(v: anytype, str: []const u8) void {
    if (builtin.mode != .Debug) return;
    comptime var shift = 0;
    inline while (shift != @bitSizeOf(@TypeOf(v))) : (shift += 1) {
        switch (@as(u1, @truncate(v >> shift))) {
            1 => std.debug.print("\x1b[36;1m1\x1b[0m", .{}),
            0 => std.debug.print("\x1b[95m.\x1b[0m", .{}),
        }
    }
    std.debug.print("\x1b[0m ⟵  {s}\n", .{str});
}

fn printStr(str: []const u8) void {
    if (builtin.mode != .Debug) return;
    // std.debug.print("\x1b[31;1m", .{});
    // std.debug.print("\x1b[0m", .{});
    var color: u3 = 1;

    var i: usize = 0;
    defer std.debug.print("\n", .{});

    while (i < str.len) : (i += 1) {
        switch (str[i]) {
            '\n' => {
                std.debug.print("\x1b[30;47;1m{c}\x1b[0m", .{'$'});
                continue;
            },
            0 => {
                std.debug.print("\x1b[30;47;1m{c}\x1b[0m", .{'~'});
                continue;
            },
            '\\', '/' => |c| {
                if (i + 1 < str.len and str[i + 1] == c) {
                    color += @intFromBool(i != 0);
                    if (color == 3) color = 4;
                    if (color == 6) color = 1;
                    // apply new color
                    std.debug.print("\x1b[3{o};1m", .{color});

                    while (true) {
                        std.debug.print("{c}", .{str[i]});
                        i += 1;
                        if (i >= str.len or str[i] == '\n') break;
                    }
                    std.debug.print("\x1b[0m", .{});
                    if (i < str.len) i -= 1;
                    continue;
                }
            },

            '\'', '"' => |c| {
                color += @intFromBool(i != 0);
                if (color == 2) color = 3;
                if (color == 3) color = 4;
                if (color == 6) color = 1;
                // apply new color
                std.debug.print("\x1b[37;4{o};1m", .{color});
                defer std.debug.print("\x1b[0m", .{});

                while (true) {
                    std.debug.print("{c}", .{str[i]});
                    i += 1;
                    if (i >= str.len) return;
                    const escaped = str[i] == '\\';
                    if (escaped)
                        std.debug.print("\\", .{});
                    i += @intFromBool(escaped);
                    if (i >= str.len) return;
                    if ((!escaped and str[i] == c) or str[i] == '\n') break;
                }
                if (str[i] == '\n') {
                    i -= 1;
                    continue;
                }
                std.debug.print("{c}", .{str[i]});
                continue;
            },
            else => {},
        }
        std.debug.print("\x1b[30;47;1m{c}\x1b[0m", .{str[i]});
    }
}

fn printStrForce(str: []const u8) void {
    // std.debug.print("\x1b[31;1m", .{});
    // std.debug.print("\x1b[0m", .{});
    var color: u3 = 1;

    var i: usize = 0;
    defer std.debug.print("\n", .{});

    while (i < str.len) : (i += 1) {
        switch (str[i]) {
            '\n' => {
                std.debug.print("\x1b[30;47;1m{c}\x1b[0m", .{'$'});
                continue;
            },
            0 => {
                std.debug.print("\x1b[30;47;1m{c}\x1b[0m", .{'~'});
                continue;
            },
            '\\', '/' => |c| {
                if (i + 1 < str.len and str[i + 1] == c) {
                    color += @intFromBool(i != 0);
                    if (color == 3) color = 4;
                    if (color == 6) color = 1;
                    // apply new color
                    std.debug.print("\x1b[3{o};1m", .{color});

                    while (true) {
                        std.debug.print("{c}", .{str[i]});
                        i += 1;
                        if (i >= str.len or str[i] == '\n') break;
                    }
                    std.debug.print("\x1b[0m", .{});
                    if (i < str.len) i -= 1;
                    continue;
                }
            },

            '\'', '"' => |c| {
                color += @intFromBool(i != 0);
                if (color == 2) color = 3;
                if (color == 3) color = 4;
                if (color == 6) color = 1;
                // apply new color
                std.debug.print("\x1b[37;4{o};1m", .{color});
                defer std.debug.print("\x1b[0m", .{});

                while (true) {
                    std.debug.print("{c}", .{str[i]});
                    i += 1;
                    if (i >= str.len) return;
                    const escaped = str[i] == '\\';
                    if (escaped)
                        std.debug.print("\\", .{});
                    i += @intFromBool(escaped);
                    if (i >= str.len) return;
                    if ((!escaped and str[i] == c) or str[i] == '\n') break;
                }
                if (str[i] == '\n') {
                    i -= 1;
                    continue;
                }
                std.debug.print("{c}", .{str[i]});
                continue;
            },
            else => {},
        }
        std.debug.print("\x1b[30;47;1m{c}\x1b[0m", .{str[i]});
    }
}

// End Debug print functions

// test "refineMultiCharEndsMasks" {
//     const tokenizer: Tokenizer(.{}) = .{};
//     var multiCharSymbolTokenizer: @TypeOf(tokenizer).MultiCharSymbolTokenizer = .{};
//     var carry: @TypeOf(tokenizer).Carry = .{};
//     _ = multiCharSymbolTokenizer.refineMultiCharEndsMasks(&carry, 0b00001000, 0b00101000);
// }

fn k_op(a: anytype, comptime op: enum { @"+", @"&", @"&~", @"|", @"++", @"~^", @"^" }, b: @TypeOf(a)) @TypeOf(a) {
    const instruction_mneumonic = "k" ++ switch (op) {
        .@"+" => "add",
        .@"&" => "and",
        .@"&~" => "andn",
        .@"|" => "or",
        .@"++" => "unpck",
        .@"~^" => "xnor",
        .@"^" => "xor",
    } ++ switch (@TypeOf(a)) {
        u8 => "b",
        u16 => "w",
        u32 => "d",
        u64 => "q",
        else => @compileError("Invalid type passed to k_op"),
    } ++ if (op != .@"++") "" else switch (@TypeOf(a)) {
        u8 => "w",
        u16 => "d",
        u32 => "q",
        else => @compileError("Invalid type passed to k_op"),
    };

    const valid_k_ops = enum { kaddw, kaddb, kaddq, kaddd, kandw, kandb, kandq, kandd, kandnw, kandnb, kandnq, kandnd, korw, korb, korq, kord, kunpckbw, kunpckwd, kunpckdq, kxnorw, kxnorb, kxnorq, kxnord, kxorw, kxorb, kxorq, kxord };
    const k_op_enum = comptime std.meta.stringToEnum(valid_k_ops, instruction_mneumonic).?;

    const avx_feature_set: std.Target.x86.Feature = switch (k_op_enum) {
        .kaddw => .avx512dq,
        .kaddb => .avx512dq,
        .kaddq => .avx512bw,
        .kaddd => .avx512bw,
        .kandw => .avx512f,
        .kandb => .avx512dq,
        .kandq => .avx512bw,
        .kandd => .avx512bw,
        .kandnw => .avx512f,
        .kandnb => .avx512dq,
        .kandnq => .avx512bw,
        .kandnd => .avx512bw,
        // .knotw => .avx512f,
        // .knotb => .avx512dq,
        // .knotq => .avx512bw,
        // .knotd => .avx512bw,
        .korw => .avx512f,
        .korb => .avx512dq,
        .korq => .avx512bw,
        .kord => .avx512bw,
        // .kortestw => .avx512f,
        // .kortestb => .avx512dq,
        // .kortestq => .avx512bw,
        // .kortestd => .avx512bw,
        // .kshiftlw => .avx512f,
        // .kshiftlb => .avx512dq,
        // .kshiftlq => .avx512bw,
        // .kshiftld => .avx512bw,
        // .kshiftrw => .avx512f,
        // .kshiftrb => .avx512dq,
        // .kshiftrq => .avx512bw,
        // .kshiftrd => .avx512bw,
        // .ktestw => .avx512dq,
        // .ktestb => .avx512dq,
        // .ktestq => .avx512bw,
        // .ktestd => .avx512bw,
        .kunpckbw => .avx512f,
        .kunpckwd => .avx512bw,
        .kunpckdq => .avx512bw,
        .kxnorw => .avx512f,
        .kxnorb => .avx512dq,
        .kxnorq => .avx512bw,
        .kxnord => .avx512bw,
        .kxorw => .avx512f,
        .kxorb => .avx512dq,
        .kxorq => .avx512bw,
        .kxord => .avx512bw,
    };

    if (!comptime std.Target.x86.featureSetHas(builtin.cpu.features, avx_feature_set)) {
        return switch (k_op_enum) { // TODO: audit these to make sure andn is in the right order and kunpckdq is in the right order
            .kaddw, .kaddb, .kaddq, .kaddd => a + b,
            .kandw, .kandb, .kandq, .kandd => a & b,
            .kandnw, .kandnb, .kandnq, .kandnd => a & ~b,
            .korw, .korb, .korq, .kord => a | b,
            .kunpckbw => a << 8 | b,
            .kunpckwd => a << 16 | b,
            .kunpckdq => a << 32 | b,
            .kxnorw, .kxnorb, .kxnorq, .kxnord => ~(a ^ b),
            .kxorw, .kxorb, .kxorq, .kxord => a ^ b,
        };
    }

    return asm (instruction_mneumonic ++ " %[b], %[a], %[ret]"
        : [ret] "=k" (-> @TypeOf(a)),
        : [a] "k" (a),
          [b] "k" (b),
    );
}

fn divCeil(numerator: anytype, denominator: anytype) @TypeOf(numerator / denominator) {
    return (numerator + (denominator - 1)) / denominator;
}
