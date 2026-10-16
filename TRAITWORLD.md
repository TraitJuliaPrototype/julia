# Julia 1.13.2 with the trait-world invalidation filter

This branch (`trait-world-1.13`) is Julia v1.13.2 plus the runtime pieces of the trait-bound Julia design that exist so far. The design and its notes live in the design repository (`../writeup/notes/`); the runtime plan is `../writeup/explorations/runtime.md`.

| Commit | What |
|---|---|
| insertion half (`src/gf.c`) | method insertion does not invalidate trait-world code through methods outside its view (owned by neither its function's nor its argument types' owners) |
| load half (`Compiler/src/reinfer.jl`) | image validation keeps trait-world code whose edges fail only through outside, owned methods; a pirate still invalidates. The Base-callee rule of the 1.12 version is untested here |

Trait-world code is recognized by the module markers `__trait_module__` and `__trait_world_machinery__` (TraitWorld sets the module list at `__init__`). The load half is on unless `JULIA_TW_FILTER=0`; `JULIA_TW_PROTECT=<package names>` (measurements only) treats those packages as trait-world code.

The earlier branch `trait-universe` holds the same patches on 1.12 (commit 0bf9125), with which the measurements of sessions 20–21 were made.

## Build

A cold build of this checkout needs the network for its dependencies (`make` downloads LLVM and other binaries) and takes 40–60 minutes; run it alone on the machine.

    make -j4                       # cold build
    make -C src -j4                # the C library alone, after a change in src/
    make -j4 julia-sysimg-release  # the system image, after a change in base/ or Compiler/

On 1.12 the clone was moved after building, which left the standard-library symlinks under `usr/share/julia/stdlib` and `base/JuliaSyntax` dangling; a build at this path avoids that.

Without a rebuild, the load half can be injected into a stock Julia 1.13.2 (`../experiments/runtime-injection/`); the insertion half needs this build.
