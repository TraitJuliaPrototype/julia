# Julia 1.13.2 with the trait-world invalidation filter

This branch (`trait-world-1.13`) is Julia v1.13.2 plus the runtime pieces of the trait-bound Julia design that exist so far. The design and its notes live in the design repository (`../writeup/notes/`); the runtime plan is `../writeup/explorations/runtime.md`.

| Commit | What |
|---|---|
| insertion half (`src/gf.c`) | method insertion does not invalidate trait-world code through methods outside its view (owned by neither its function's nor its argument types' owners) |
| load half (`Compiler/src/reinfer.jl`) | image validation keeps trait-world code whose edges fail only through outside, owned methods; a pirate still invalidates. The Base-callee rule of the 1.12 version is untested here |

Trait-world code is recognized by the module markers `__trait_module__` and `__trait_world_machinery__` (TraitWorld sets the module list at `__init__`). The load half is on unless `JULIA_TW_FILTER=0`; `JULIA_TW_PROTECT=<package names>` (measurements only) treats those packages as trait-world code.

The earlier branch `trait-universe` holds the same patches on 1.12 (commit 0bf9125), with which the measurements of sessions 20–21 were made.

## Hooks (branch `trait-world-1.13-hooks`)

1. **Method-definition hooks** (`src/gf.c`, `src/staticdata.c`, `src/staticdata_utils.c`):
   - `jl_set_method_definition_hook(f)`: `f(m::Method, after::Bool)` is called before a method enters the method table (`after = false`; throwing refuses it, the method never becomes visible and no world is allocated) and once it is visible (`after = true`).
   - `jl_set_image_methods_hook(f)`: `f(ms::Vector{Any}, pkg::String)` is called with a package image's new methods before `jl_activate_methods`; it returns the methods to leave inactive (they never become visible), or `nothing`. An exception is reported and ignored.
   - Both run in the latest world, outside the world-counter lock, not for definitions a hook makes itself, and not while the system image is built. `nothing` removes a hook. The callables are rooted by the caller.

2. **Invalidation policy hook** (`src/gf.c`, `src/staticdata*.c`, `Compiler/src/reinfer.jl`):
   - `jl_set_invalidation_policy(f)`: `f(m::Method)::Union{Module,Nothing}` is called per new method, before the world-counter lock (and for an image's methods before activation): a package whose types every call the method applies to must mention, so that it cannot change protected code that does not mention that package's types, `nothing` when it invalidates as usual. Inside the lock the test per CodeInstance is structural: its module is registered (`jl_set_trait_world_filter`) and its signature mentions no type of that package. The load half calls the same callable through `Core.invokelatest` (`ReinferUtils._tw_policy`), because validation runs in the compiler's early world.
   - The hard-coded policy (trigger owners, Julia's library names) is gone: TraitWorld's `_invalidation_policy` decides (rule 4).
   - **A fix to the filter itself:** a skipped CodeInstance keeps its backedge. Before, the backedge was removed although the code was kept, so a later pirate could not find and invalidate it (stale code).
   - The callee rule at insertion needs nothing new: a protected caller is never invalidated through a callee's invalidation, and keeps calling the callee's compiled code.
   - **Until TraitWorld registers its policy** (its own image, and images loaded with it, are validated before its `__init__` runs), the load half applies a built-in default with the same rule (`_tw_default_policy`): the package outside Julia and the trait world whose type the signature requires (an argument of its type, `Type{T}` of one, a Julia type parametrized by one, a `Vararg` of one).
   - **A fix for TraitWorld's own image:** the trait-world markers (`__trait_module__`, `__trait_world_machinery__`) are looked up in the latest world. Validation runs in the compiler's world, where the bindings of the image being loaded are not yet defined, so TraitWorld's own code was never protected at load. With CSV, DataFrames and CairoMakie loaded first, 212 of its edges failed and its checks recompiled at the first solve (0.56–0.8 s).

3. **Field-offset intrinsics** (`src/intrinsics.h`, `src/intrinsics.cpp`, `src/runtime_intrinsics.c`, `Compiler/src/tfuncs.jl`):
   - `Core.Intrinsics.getfield_at_offset(obj, T, off)` loads the value of type `T` at byte offset `off` of a boxed (mutable or abstractly typed) object, with the alias metadata of a mutable field (`jtbaa_mutab`) and, for a reference, a null check.
   - `setfield_at_offset(obj, T, off, x)` stores it, with the write barrier for a reference.
   - Inline fields holding references, and non-constant `T`, go to the runtime functions `jl_getfield_at_offset` and `jl_setfield_at_offset`.
   - TraitWorld uses them instead of `llvmcall` with hand-written TBAA. A loop that reads a field through the object while storing into another array vectorizes: 0.121 ns per element against 0.120 plain.

4. **Post-inference hook** (`Compiler/src/typeinfer.jl`):
   - `Base.Compiler._tw_infer_hook[] = f`: after `typeinf_ext_toplevel` has inferred and queued a method instance of a trait-world module, `f(ci::CodeInstance)` is called in the latest world; an exception is ignored.
   - TraitWorld audits the new code's edges there (`_on_inferred`), where it is compiled. Its per-call guards compare one stamp, which only the events that can change an audit bump: a pirate of a sealed package defined or loaded (piece 1), a seal, a mode change, a finding at inference. Before, every world change made each guard re-audit its signature. A typed call: 31.4 ns against 29.9 ns plain (33.3 ns with the world-counter guard).
   - Measured with TraitDifferentialEquations and CSV, DataFrames and CairoMakie loaded after it (session note 23, row 7): no ODECore, ordinary-layer or machinery code invalidated, first solve 96 ms (0.7–1.05 s before). Loaded before it: first solve still 0.7 s, from Base callees of TraitWorld's own image that fail at validation (the default policy above does not yet take the intersection of a `Union`'s required packages).
   - The image hook's checks cost 0.38 s over the 63,000 methods of those loads (4.8 s before owners were memoized per image).

5. **Native code for upstream and foreign CodeInstances in package images** (`src/staticdata_utils.c`, `Compiler/src/precompile.jl`):
   - `jl_set_image_keep_compiled(1)` (TraitWorld calls it when a package using it is precompiled): a CodeInstance inferred during the precompilation for an upstream method, which no call of the package reaches statically (a dynamic call, `invokelatest`, an entry's worker at upstream types), is kept in the image if it was compiled, with its machine code. Before, it was dropped.
   - CodeInstances of another owner (an `AbstractInterpreter`'s own cache, `cache_owner`) get machine code in the image, compiled from their cached IR (the interpreter must keep it: `may_discard_trees(interp) = false`). Before, only their inferred code was saved, and a session compiled them again.
   - Measured (`$TMPDIR/p5`, a package whose workload calls an upstream function dynamically and through the upstream package's own interpreter): the upstream function's first call 0.33 ms instead of 21.5 ms; the interpreter's code 0.39 ms instead of 11.1 s (the compiler's specializations for the interpreter are kept too).

## Build

Built in session 23 (2026-10-16) at `julia-1.13/`, a worktree of `julia/` on this branch: `julia-1.13/julia`. `tw-build.sh` runs `make -j4` niced under a memory watchdog (it kills the build below 4 GB available). The build downloads from cache.julialang.org, github.com, api.github.com, codeload.github.com, objects.githubusercontent.com, release-assets.githubusercontent.com, raw.githubusercontent.com and curl.haxx.se/curl.se.

The load half had to change for the build (it had been tested only by injection): image validation runs in the compiler's world, fixed early in the bootstrap, before Base's `ENV`, `split` and `pathof` exist. Its globals are `RefValue`s, the environment is read with `getenv`, and the standard libraries are a literal list. TraitWorld registers trait modules with the insertion half when they start and protects its own machinery (its registries hold no edges).

A cold build of this checkout needs the network for its dependencies (`make` downloads LLVM and other binaries) and takes 40–60 minutes; run it alone on the machine.

    make -j4                       # cold build
    make -C src -j4                # the C library alone, after a change in src/
    make -j4 julia-sysimg-release  # the system image, after a change in base/ or Compiler/

On 1.12 the clone was moved after building, which left the standard-library symlinks under `usr/share/julia/stdlib` and `base/JuliaSyntax` dangling; a build at this path avoids that.

Without a rebuild, the load half can be injected into a stock Julia 1.13.2 (`../experiments/runtime-injection/`); the insertion half needs this build.
