# Julia 1.13.2 with the trait-world runtime pieces

This branch (`trait-world-1.13`) is Julia v1.13.2 plus the runtime pieces of the trait-bound Julia design that exist so far, numbered as in `../writeup/explorations/julia-support.md`. The design and its notes live in the design repository (`../writeup/`).

The earlier invalidation filter (insertion half in `src/gf.c`, load half in `Compiler/src/reinfer.jl`), the invalidation policy hook (piece 2) and the post-inference hook (piece 4) have been removed: the facts compiler (`design.md` §6, parts A–D) superseded them, and image validation and method insertion are upstream Julia's again. The branch `trait-universe` holds the filter on 1.12 (commit 0bf9125), with which the measurements of sessions 20–21 were made.

## Hooks (branch `trait-world-1.13-hooks`)

1. **Method-definition hooks** (`src/gf.c`, `src/staticdata.c`, `src/staticdata_utils.c`):
   - `jl_set_method_definition_hook(f)`: `f(m::Method, after::Bool)` is called before a method enters the method table (`after = false`; throwing refuses it, the method never becomes visible and no world is allocated) and once it is visible (`after = true`).
   - `jl_set_image_methods_hook(f)`: `f(ms::Vector{Any}, pkg::String)` is called with a package image's new methods before `jl_activate_methods`; it returns the methods to leave inactive (they never become visible), or `nothing`. An exception is reported and ignored.
   - Both run in the latest world, outside the world-counter lock, not for definitions a hook makes itself, and not while the system image is built. `nothing` removes a hook. The callables are rooted by the caller.

3. **Field-offset intrinsics** (`src/intrinsics.h`, `src/intrinsics.cpp`, `src/runtime_intrinsics.c`, `Compiler/src/tfuncs.jl`):
   - `Core.Intrinsics.getfield_at_offset(obj, T, off)` loads the value of type `T` at byte offset `off` of a boxed (mutable or abstractly typed) object, with the alias metadata of a mutable field (`jtbaa_mutab`) and, for a reference, a null check.
   - `setfield_at_offset(obj, T, off, x)` stores it, with the write barrier for a reference.
   - Inline fields holding references, and non-constant `T`, go to the runtime functions `jl_getfield_at_offset` and `jl_setfield_at_offset`.
   - TraitWorld uses them instead of `llvmcall` with hand-written TBAA. A loop that reads a field through the object while storing into another array vectorizes: 0.121 ns per element against 0.120 plain.

5. **Native code for upstream and foreign CodeInstances in package images** (`src/staticdata_utils.c`, `Compiler/src/precompile.jl`):
   - `jl_set_image_keep_compiled(1)` (TraitWorld calls it when a package using it is precompiled): a CodeInstance inferred during the precompilation for an upstream method, which no call of the package reaches statically (a dynamic call, `invokelatest`, an entry's worker at upstream types), is kept in the image if it was compiled, with its machine code. Before, it was dropped.
   - CodeInstances of another owner (an `AbstractInterpreter`'s own cache, `cache_owner`) get machine code in the image, compiled from their cached IR (the interpreter must keep it: `may_discard_trees(interp) = false`). Before, only their inferred code was saved, and a session compiled them again.
   - Measured (`$TMPDIR/p5`, a package whose workload calls an upstream function dynamically and through the upstream package's own interpreter): the upstream function's first call 0.33 ms instead of 21.5 ms; the interpreter's code 0.39 ms instead of 11.1 s (the compiler's specializations for the interpreter are kept too).

## Build

Built in session 23 (2026-10-16) at `julia-1.13/`, a worktree of `julia/` on this branch: `julia-1.13/julia`. `tw-build.sh` runs `make -j4` niced under a memory watchdog (it kills the build below 4 GB available). The build downloads from cache.julialang.org, github.com, api.github.com, codeload.github.com, objects.githubusercontent.com, release-assets.githubusercontent.com, raw.githubusercontent.com and curl.haxx.se/curl.se.

A cold build of this checkout needs the network for its dependencies (`make` downloads LLVM and other binaries) and takes 40–60 minutes; run it alone on the machine.

    make -j4                       # cold build
    make -C src -j4                # the C library alone, after a change in src/
    make -j4 julia-sysimg-release  # the system image, after a change in base/ or Compiler/

On 1.12 the clone was moved after building, which left the standard-library symlinks under `usr/share/julia/stdlib` and `base/JuliaSyntax` dangling; a build at this path avoids that.
