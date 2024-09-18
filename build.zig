const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // const lib = b.addSharedLibrary(.{
    //     .name = cfg.package_name,
    //     .root_source_file = .{ .path = cfg.stub_path },
    //     .target = target,
    //     .optimize = optimize,
    // });
    // lib.root_module.addImport("exporter", b.createModule(.{
    //     .root_source_file = .{ .path = cfg.exporter_path },
    // }));
    // lib.root_module.addImport("package", b.createModule(.{
    //     .root_source_file = .{ .path = cfg.package_path },
    // }));
    // if (cfg.use_libc) {
    //     lib.linkLibC();
    // }
    // if (cfg.target.cpu_arch == .wasm32) {
    //     lib.use_lld = false;
    //     lib.rdynamic = true;
    // }
    // b.installArtifact(lib);

    const path = "src/main.zig";

    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const num_args = if (b.args) |args| args.len else 0;

    const exe = b.addExecutable(.{
        .name = if (num_args > 0) b.args.?[0] else "exe",
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = b.path(path),
        .target = target,
        .optimize = optimize,
    });

    // exe.addIncludePath(.{ .path = "c-src" }); // Look for C source files
    // exe.linkLibC();

    for ([_][]const u8{"jdz_allocator"}) |dependency_name| {
        const dependency = b.dependency(dependency_name, .{ .target = target, .optimize = optimize });
        exe.root_module.addImport(dependency_name, dependency.module(dependency_name));
    }

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args[1..]);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const unit_tests = b.addTest(.{
        .root_source_file = b.path(path),
        .target = target,
        .optimize = optimize,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
