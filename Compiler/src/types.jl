# This file is a part of Julia. License is MIT: https://julialang.org/license
#

const WorkThunk = Any
# #@eval struct WorkThunk
#    thunk::Core.OpaqueClosure{Tuple{Vector{Tasks}}, Bool}
#    WorkThunk(work) = new($(Expr(:opaque_closure, :(Tuple{Vector{Tasks}}), :Bool, :Bool, :((tasks) -> work(tasks))))) # @opaque Vector{Tasks}->Bool (tasks)->work(tasks)
# end
# (p::WorkThunk)() = p.thunk()

"""
    AbstractInterpreter

An abstract base class that allows multiple dispatch to determine the method of
executing Julia code. The native Julia-LLVM pipeline is enabled by using the
`NativeInterpreter` concrete instantiation of this abstract class, others can be
swapped in as long as they follow the `AbstractInterpreter` API.

If `interp::NewInterpreter` is an `AbstractInterpreter`, it is expected to provide at least
the following methods to satisfy the `AbstractInterpreter` API requirement:
- `InferenceParams(interp::NewInterpreter)` - return an `InferenceParams` instance
- `OptimizationParams(interp::NewInterpreter)` - return an `OptimizationParams` instance
- `get_inference_world(interp::NewInterpreter)` - return the world age for this interpreter
- `get_inference_cache(interp::NewInterpreter)` - return the local inference cache
- `cache_owner(interp::NewInterpreter)` - return the owner of any new cache entries

If `CodeInstance`s compiled using `interp::NewInterpreter` are meant to be executed with `invoke`,
a method `codegen_cache(interp::NewInterpreter) -> IdDict{CodeInstance, CodeInfo}` must be defined,
and inference must be triggered via `typeinf_ext_toplevel` with source mode `SOURCE_MODE_ABI`.
"""
abstract type AbstractInterpreter end

abstract type AbstractLattice end

struct InvalidIRError <: Exception end

struct ArgInfo
    fargs::Union{Nothing,Vector{Any}}
    argtypes::Vector{Any}
end

struct StmtInfo
    """
    If `used` is false, we know that the return value is statically unused and
    need thus not be computed.
    """
    used::Bool
    saw_latestworld::Bool
end

struct SpecInfo
    nargs::Int
    isva::Bool
    propagate_inbounds::Bool
    method_for_inference_limit_heuristics::Union{Nothing,Method}
end
SpecInfo(src::CodeInfo) = SpecInfo(
    Int(src.nargs), src.isva,
    src.propagate_inbounds,
    src.method_for_inference_limit_heuristics::Union{Nothing,Method})

"""
    v::VarState

A special wrapper that represents a local variable of a method being analyzed.
This does not participate in the native type system nor the inference lattice, and it thus
should be always unwrapped to `v.typ` when performing any type or lattice operations on it.
`v.undef` represents undefined-ness of this static parameter. If `true`, it means that the
variable _may_ be undefined at runtime, otherwise it is guaranteed to be defined.
If `v.typ === Bottom` it means that the variable is strictly undefined.
"""
struct VarState
    typ
    undef::Bool
    VarState(@nospecialize(typ), undef::Bool) = new(typ, undef)
end

struct AnalysisResults
    result
    next::AnalysisResults
    AnalysisResults(@nospecialize(result), next::AnalysisResults) = new(result, next)
    AnalysisResults(@nospecialize(result)) = new(result)
    # NullAnalysisResults() = new(nothing)
    # global const NULL_ANALYSIS_RESULTS = NullAnalysisResults()
end
const NULL_ANALYSIS_RESULTS = AnalysisResults(nothing)

"""
    result::InferenceResult

A type that represents the result of running type inference on a chunk of code.
There are two constructor available:
- `InferenceResult(mi::MethodInstance, [𝕃::AbstractLattice])` for regular inference,
  without extended lattice information included in `result.argtypes`.
- `InferenceResult(mi::MethodInstance, argtypes::Vector{Any}, overridden_by_const::BitVector)`
  for constant inference, with extended lattice information included in `result.argtypes`.
"""
mutable struct InferenceResult
    #=== constant fields ===#
    const linfo::MethodInstance
    const argtypes::Vector{Any}
    const overridden_by_const::Union{Nothing,BitVector}

    #=== mutable fields ===#
    result                            # extended lattice element if inferred, nothing otherwise
    exc_result                        # like `result`, but for the thrown value
    src                               # ::Union{CodeInfo, IRCode, OptimizationState} if inferred copy is available, nothing otherwise
    valid_worlds::WorldRange          # if inference and optimization is finished
    ipo_effects::Effects              # if inference is finished
    effects::Effects                  # if optimization is finished
    analysis_results::AnalysisResults # AnalysisResults with e.g. result::ArgEscapeCache if optimized, otherwise NULL_ANALYSIS_RESULTS
    is_src_volatile::Bool             # `src` has been cached globally as the compressed format already, allowing `src` to be used destructively
    tombstone::Bool

    #=== uninitialized fields ===#
    ci::CodeInstance                  # CodeInstance if this result may be added to the cache
    ci_as_edge::CodeInstance          # CodeInstance as the edge representing locally cached result
    function InferenceResult(mi::MethodInstance, argtypes::Vector{Any}, overridden_by_const::Union{Nothing,BitVector})
        result = exc_result = src = nothing
        valid_worlds = WorldRange()
        ipo_effects = effects = Effects()
        analysis_results = NULL_ANALYSIS_RESULTS
        return new(mi, argtypes, overridden_by_const, result, exc_result, src,
            valid_worlds, ipo_effects, effects, analysis_results, #=is_src_volatile=#false, false)
    end
end
function InferenceResult(mi::MethodInstance, 𝕃::AbstractLattice=fallback_lattice)
    argtypes = matching_cache_argtypes(𝕃, mi)
    return InferenceResult(mi, argtypes, #=overridden_by_const=#nothing)
end

function stack_analysis_result!(inf_result::InferenceResult, @nospecialize(result))
    return inf_result.analysis_results = AnalysisResults(result, inf_result.analysis_results)
end

function traverse_analysis_results(callback, (;analysis_results)::Union{InferenceResult,CodeInstance})
    analysis_results isa AnalysisResults || return nothing
    while isdefined(analysis_results, :next)
        if (result = callback(analysis_results.result)) !== nothing
            return result
        end
        analysis_results = analysis_results.next
    end
    return nothing
end

"""
    inf_params::InferenceParams

Parameters that control abstract interpretation-based type inference operation.

---
- `inf_params.max_methods::Int = 3`\\
  Type inference gives up analysis on a call when there are more than `max_methods` matching
  methods. This trades off between compiler latency and generated code performance.
  Typically, considering many methods means spending _lots_ of time obtaining poor type
  information, so this option should be kept low. [`Base.Experimental.@max_methods`](@ref)
  can have a more fine-grained control on this configuration with per-module or per-method
  annotation basis.
---
- `inf_params.max_union_splitting::Int = 4`\\
  Specifies the maximum number of union-tuples to swap or expand before computing the set of
  matching methods or conditional types.
---
- `inf_params.max_apply_union_enum::Int = 8`\\
  Specifies the maximum number of union-tuples to swap or expand when inferring a call to
  `Core._apply_iterate`.
---
- `inf_params.max_tuple_splat::Int = 32`\\
  When attempting to infer a call to `Core._apply_iterate`, abort the analysis if the tuple
  contains more than this many elements.
---
- `inf_params.tuple_complexity_limit_depth::Int = 3`\\
  Specifies the maximum depth of large tuple type that can appear as specialized method
  signature when inferring a recursive call graph.
---
- `inf_params.ipo_constant_propagation::Bool = true`\\
  If `false`, disables analysis with extended lattice information, i.e. disables any of
  the concrete evaluation, semi-concrete interpretation and constant propagation entirely.
  [`Base.@constprop :none`](@ref Base.@constprop) can have a more fine-grained control on
  this configuration with per-method annotation basis.
---
- `inf_params.aggressive_constant_propagation::Bool = false`\\
  If `true`, forces constant propagation on any methods when any extended lattice
  information available. [`Base.@constprop :aggressive`](@ref Base.@constprop) can have a
  more fine-grained control on this configuration with per-method annotation basis.
---
- `inf_params.assume_bindings_static::Bool = false`\\
  If `true`, assumes that no new bindings will be added, i.e. a non-existing binding at
  inference time can be assumed to always not exist at runtime (and thus e.g. any access to
  it will `throw`). Defaults to `false` since this assumption does not hold in Julia's
  semantics for native code execution.
---
- `inf_params.force_enable_inference::Bool = false`\\
  If `true`, inference will be performed on functions regardless of whether it was disabled
  at the module level via `Base.Experimental.@compiler_options`.
---
"""
struct InferenceParams
    max_methods::Int
    max_union_splitting::Int
    max_apply_union_enum::Int
    max_tuple_splat::Int
    tuple_complexity_limit_depth::Int
    ipo_constant_propagation::Bool
    aggressive_constant_propagation::Bool
    assume_bindings_static::Bool
    ignore_recursion_hardlimit::Bool
    force_enable_inference::Bool

    function InferenceParams(
        max_methods::Int,
        max_union_splitting::Int,
        max_apply_union_enum::Int,
        max_tuple_splat::Int,
        tuple_complexity_limit_depth::Int,
        ipo_constant_propagation::Bool,
        aggressive_constant_propagation::Bool,
        assume_bindings_static::Bool,
        ignore_recursion_hardlimit::Bool,
        force_enable_inference::Bool,
    )
        return new(
            max_methods,
            max_union_splitting,
            max_apply_union_enum,
            max_tuple_splat,
            tuple_complexity_limit_depth,
            ipo_constant_propagation,
            aggressive_constant_propagation,
            assume_bindings_static,
            ignore_recursion_hardlimit,
            force_enable_inference,
        )
    end
end
function InferenceParams(
    params::InferenceParams = InferenceParams( # default constructor
        #=max_methods::Int=# BuildSettings.MAX_METHODS,
        #=max_union_splitting::Int=# 4,
        #=max_apply_union_enum::Int=# 8,
        #=max_tuple_splat::Int=# 32,
        #=tuple_complexity_limit_depth::Int=# 3,
        #=ipo_constant_propagation::Bool=# true,
        #=aggressive_constant_propagation::Bool=# false,
        #=assume_bindings_static::Bool=# false,
        #=ignore_recursion_hardlimit::Bool=# false,
        #=force_enable_inference::Bool=# false
    );
    max_methods::Int = params.max_methods,
    max_union_splitting::Int = params.max_union_splitting,
    max_apply_union_enum::Int = params.max_apply_union_enum,
    max_tuple_splat::Int = params.max_tuple_splat,
    tuple_complexity_limit_depth::Int = params.tuple_complexity_limit_depth,
    ipo_constant_propagation::Bool = params.ipo_constant_propagation,
    aggressive_constant_propagation::Bool = params.aggressive_constant_propagation,
    assume_bindings_static::Bool = params.assume_bindings_static,
    ignore_recursion_hardlimit::Bool = params.ignore_recursion_hardlimit,
    force_enable_inference::Bool = params.force_enable_inference,
)
    return InferenceParams(
        max_methods,
        max_union_splitting,
        max_apply_union_enum,
        max_tuple_splat,
        tuple_complexity_limit_depth,
        ipo_constant_propagation,
        aggressive_constant_propagation,
        assume_bindings_static,
        ignore_recursion_hardlimit,
        force_enable_inference,
    )
end

"""
    opt_params::OptimizationParams

Parameters that control optimizer operation.

---
- `opt_params.inlining::Bool = inlining_enabled()`\\
  Controls whether or not inlining is enabled.
---
- `opt_params.inline_cost_threshold::Int = 100`\\
  Specifies the number of CPU cycles beyond which it's not worth inlining.
---
- `opt_params.inline_nonleaf_penalty::Int = 1000`\\
  Specifies the penalty cost for a dynamic dispatch.
---
- `opt_params.inline_tupleret_bonus::Int = 250`\\
  Specifies the extra inlining willingness for a method specialization with non-concrete
  tuple return types (in hopes of splitting it up). `opt_params.inline_tupleret_bonus` will
  be added to `opt_params.inline_cost_threshold` when making inlining decision.
---
- `opt_params.max_tuple_splat::Int = 32`\\
  When attempting to inline `Core._apply_iterate`, abort the optimization if the tuple
  contains more than this many elements.
---
- `opt_params.compilesig_invokes::Bool = true`\\
  If `true`, gives the inliner license to change which `MethodInstance` to invoke when
  generating `:invoke` expression based on the [`@nospecialize`](@ref) annotation,
  in order to avoid over-specialization.
---
- `opt_params.assume_fatal_throw::Bool = false`\\
  If `true`, gives the optimizer license to assume that any `throw` is fatal and thus the
  state after a `throw` is not externally observable. In particular, this gives the
  optimizer license to move side effects (that are proven not observed within a particular
  code path) across a throwing call. Defaults to `false`.
---
- `opt_params.preserve_local_sources::Bool = false`\\
  If `true`, the inliner is restricted from modifying locally-cached sources that are
  retained in `CallInfo` objects and always makes their copies before inlining them into
  caller context. Defaults to `false`.
---
"""
struct OptimizationParams
    inlining::Bool
    inline_cost_threshold::Int
    inline_nonleaf_penalty::Int
    inline_tupleret_bonus::Int
    max_tuple_splat::Int
    compilesig_invokes::Bool
    assume_fatal_throw::Bool
    preserve_local_sources::Bool

    function OptimizationParams(
        inlining::Bool,
        inline_cost_threshold::Int,
        inline_nonleaf_penalty::Int,
        inline_tupleret_bonus::Int,
        max_tuple_splat::Int,
        compilesig_invokes::Bool,
        assume_fatal_throw::Bool,
        preserve_local_sources::Bool)
        return new(
            inlining,
            inline_cost_threshold,
            inline_nonleaf_penalty,
            inline_tupleret_bonus,
            max_tuple_splat,
            compilesig_invokes,
            assume_fatal_throw,
            preserve_local_sources)
    end
end
function OptimizationParams(
    params::OptimizationParams = OptimizationParams(
        #=inlining::Bool=# inlining_enabled(),
        #=inline_cost_threshold::Int=# 100,
        #=inline_nonleaf_penalty::Int=# 1000,
        #=inline_tupleret_bonus::Int=# 250,
        #=max_tuple_splat::Int=# 32,
        #=compilesig_invokes::Bool=# true,
        #=assume_fatal_throw::Bool=# false,
        #=preserve_local_sources::Bool=# false);
    inlining::Bool = params.inlining,
    inline_cost_threshold::Int = params.inline_cost_threshold,
    inline_nonleaf_penalty::Int = params.inline_nonleaf_penalty,
    inline_tupleret_bonus::Int = params.inline_tupleret_bonus,
    max_tuple_splat::Int = params.max_tuple_splat,
    compilesig_invokes::Bool = params.compilesig_invokes,
    assume_fatal_throw::Bool = params.assume_fatal_throw,
    preserve_local_sources::Bool = params.preserve_local_sources)
    return OptimizationParams(
        inlining,
        inline_cost_threshold,
        inline_nonleaf_penalty,
        inline_tupleret_bonus,
        max_tuple_splat,
        compilesig_invokes,
        assume_fatal_throw,
        preserve_local_sources)
end

"""
    NativeInterpreter <: AbstractInterpreter

This represents Julia's native type inference algorithm and the Julia-LLVM codegen backend.
"""
struct NativeInterpreter <: AbstractInterpreter
    # The world age we're working inside of
    world::UInt

    # method table to lookup for during inference on this world age
    method_table::CachedMethodTable{InternalMethodTable}

    # Cache of inference results for this particular interpreter
    inf_cache::Vector{InferenceResult}
    codegen::IdDict{CodeInstance,CodeInfo}

    # Parameters for inference and optimization
    inf_params::InferenceParams
    opt_params::OptimizationParams

    # whether calls are inferred only at facts (compiler §1.1)
    facts::Bool
    # how many inferences of the other policy this one is nested in (compiler §2.4)
    crossings::Int
    # `return_type`'s run-time answers: every method inferred plainly, under an owner of its own (§1.4)
    plainrt::Bool
    # the closed build's check in a session (compiler §7.6, §2.7): the owner of its code, or `nothing`.
    # It infers by a closed build's rules, every method plainly, into a cache of its own
    closed::Any
    # the check's method table (compiler §7.6): the session's with the build's replacements of Base's
    # methods laid over it, or `nothing` (a check without replacements, and every other interpreter)
    overlay_table::Union{Nothing,CachedMethodTable{OverlayMethodTable}}
end

"""
The owner of the code of one closed build's check in a session (compiler §2.7, §7.6). `overlay` holds
the methods the build replaces in Base, as an overlay method table, or `nothing`.
"""
mutable struct ClosedCheck
    overlay::Union{Nothing,MethodTable}
end
ClosedCheck() = ClosedCheck(nothing)

function NativeInterpreter(world::UInt = get_world_counter();
                           inf_params::InferenceParams = InferenceParams(),
                           opt_params::OptimizationParams = OptimizationParams(),
                           facts::Bool = false, crossings::Int = 0, plainrt::Bool = false,
                           @nospecialize(closed = nothing))
    curr_max_world = get_world_counter()
    # Sometimes the caller is lazy and passes typemax(UInt).
    # we cap it to the current world age for correctness
    if world == typemax(UInt)
        world = curr_max_world
    end
    # If they didn't pass typemax(UInt) but passed something more subtly
    # incorrect, fail out loudly.
    @assert world <= curr_max_world
    method_table = CachedMethodTable(InternalMethodTable(world))
    inf_cache = Vector{InferenceResult}() # Initially empty cache
    codegen = IdDict{CodeInstance,CodeInfo}()
    closed === nothing || (plainrt = true)
    overlay_table = (closed isa ClosedCheck && closed.overlay !== nothing) ?
        CachedMethodTable(OverlayMethodTable(world, closed.overlay::MethodTable)) : nothing
    return NativeInterpreter(world, method_table, inf_cache, codegen, inf_params, opt_params, facts && !plainrt, crossings, plainrt, closed, overlay_table)
end

# Quickly and easily satisfy the AbstractInterpreter API contract
InferenceParams(interp::NativeInterpreter) = interp.inf_params
OptimizationParams(interp::NativeInterpreter) = interp.opt_params
get_inference_world(interp::NativeInterpreter) = interp.world
get_inference_cache(interp::NativeInterpreter) = interp.inf_cache
# code of either policy is cached under Julia's own owner, with the policy in its flags (compiler §2.3);
# `return_type`'s run-time answers under one of their own (§1.4)
cache_owner(interp::NativeInterpreter) = interp.closed !== nothing ? interp.closed : interp.plainrt ? :plain_rt : nothing

# the policy of the code an interpreter makes and looks up under Julia's own owner (compiler §2.3): 0
# plain, 1 facts, -1 the policy the runtime reaches a method with
code_policy(::AbstractInterpreter) = Cint(-1)
code_policy(interp::NativeInterpreter) = interp.plainrt ? Cint(-1) : Cint(interp.facts)
"A CodeInstance records the policy it is compiled in, before it is published (compiler §2.3)."
function set_code_policy!(ci::CodeInstance, interp::AbstractInterpreter)
    p = code_policy(interp)
    p >= 0 && ccall(:jl_ci_set_policy, Cvoid, (Any, Cint), ci, p)
    return ci
end

engine_reserve(interp::AbstractInterpreter, mi::MethodInstance) = engine_reserve(mi, cache_owner(interp), code_policy(interp))
engine_reserve(mi::MethodInstance, @nospecialize owner) = ccall(:jl_engine_reserve, Any, (Any, Any), mi, owner)::CodeInstance
engine_reserve(mi::MethodInstance, @nospecialize(owner), policy::Cint) =
    ccall(:jl_engine_reserve_policy, Any, (Any, Any, Cint), mi, owner, policy)::CodeInstance
# engine_fulfill(::AbstractInterpreter, ci::CodeInstance, src::CodeInfo) = ccall(:jl_engine_fulfill, Cvoid, (Any, Any), ci, src) # currently the same as engine_reject, so just use that one
engine_reject(::AbstractInterpreter, ci::CodeInstance) = ccall(:jl_engine_fulfill, Cvoid, (Any, Ptr{Cvoid}), ci, C_NULL)

function already_inferred_quick_test end
function lock_mi_inference end
function unlock_mi_inference end

"""
    add_remark!(::AbstractInterpreter, sv::InferenceState, remark)

Emit an analysis remark during inference for the current line (i.e. `sv.currpc`).
These annotations are ignored by default, but can be used by external tooling to annotate
inference results.
"""
function add_remark! end

may_optimize(::AbstractInterpreter) = true
may_compress(::AbstractInterpreter) = true
may_discard_trees(::AbstractInterpreter) = true
may_discard_trees(::NativeInterpreter) =
    ccall(:jl_get_type_infer_preserve_ir, Int8, ()) == 0
precompile_keep_ir(::AbstractInterpreter) = false
precompile_keep_ir(::NativeInterpreter) =
    ccall(:jl_get_precompile_keep_ir, Int8, ()) != 0

"""
    method_table(interp::AbstractInterpreter)::MethodTableView

Returns a method table this `interp` uses for method lookup.
External `AbstractInterpreter` can optionally return `OverlayMethodTable` here
to incorporate customized dispatches for the overridden methods.
"""
method_table(interp::AbstractInterpreter) = InternalMethodTable(get_inference_world(interp))
function method_table(interp::NativeInterpreter)
    ot = interp.overlay_table
    return ot === nothing ? interp.method_table : ot
end

"""
    codegen_cache(interp::AbstractInterpreter) -> Union{Nothing, IdDict{CodeInstance, CodeInfo}}

Optionally return a cache associating a `CodeInfo` to a `CodeInstance` that should be added to the JIT
for future execution via `invoke(f, ::CodeInstance, args...)`. This cache is used during `typeinf_ext_toplevel`,
and may be safely discarded between calls to this function.

By default, a value of `nothing` is returned indicating that `CodeInstance`s should not be added to the JIT.
Attempting to execute them via `invoke` will result in an error.
"""
codegen_cache(::AbstractInterpreter) = nothing
codegen_cache(interp::NativeInterpreter) = interp.codegen

"""
By default `AbstractInterpreter` implements the following inference bail out logic:
- `bail_out_toplevel_call(::AbstractInterpreter, sig, ::InferenceState)`: bail out from
   inter-procedural inference when inferring top-level and non-concrete call site `callsig`
- `bail_out_call(::AbstractInterpreter, rt, ::InferenceState)`: bail out from
  inter-procedural  inference when return type `rt` grows up to `Any`
- `bail_out_apply(::AbstractInterpreter, rt, ::InferenceState)`: bail out from
  `_apply_iterate` inference when return type `rt` grows up to `Any`

It also bails out from local statement/frame inference when any lattice element gets down to `Bottom`,
but `AbstractInterpreter` doesn't provide a specific interface for configuring it.
"""
function bail_out_toplevel_call end, function bail_out_call end, function bail_out_apply end

"""
    infer_compilation_signature(::AbstractInterpreter)::Bool

For some call sites (for example calls to varargs methods), the signature to be compiled
and executed at run time can differ from the argument types known at the call site.
This flag controls whether we should always infer the compilation signature in addition
to the call site signature.
"""
infer_compilation_signature(::AbstractInterpreter) = false
infer_compilation_signature(::NativeInterpreter) = true

typeinf_lattice(::AbstractInterpreter) = InferenceLattice(BaseInferenceLattice.instance)
ipo_lattice(::AbstractInterpreter) = InferenceLattice(IPOResultLattice.instance)
optimizer_lattice(::AbstractInterpreter) = SimpleInferenceLattice.instance

function code_cache(interp::AbstractInterpreter)
  cache = InternalCodeCache(cache_owner(interp), code_policy(interp))
  worlds = WorldRange(get_inference_world(interp))
  return WorldView(cache, worlds)
end

"""
    DefinitionCheckInterpreter(world, hook, state)

The definition-time check (compiler §6.7): Julia's inference of a method at its own signature, with an
ephemeral cache, so that every function the method reaches is inferred again, and at each known call
`hook(interp, f, argtypes, sv)`, the library's, which records a use outside a set's contract in `state`.
The interpreter is the compiler's own: a library that runs the check defines no method of the
compiler's functions, so loading it invalidates none of the compiler's code (compiler C-40).
"""
struct DefinitionCheckCache
    dict::IdDict{MethodInstance,CodeInstance}
end
struct DefinitionCheckInterpreter <: AbstractInterpreter
    world::UInt
    inf_params::InferenceParams
    opt_params::OptimizationParams
    inf_cache::Vector{InferenceResult}
    code_cache::DefinitionCheckCache
    hook::Any       # hook(interp, f, argtypes::Vector{Any}, sv) -> nothing, called in the latest world
    state::Any      # the library's record of this check
end
DefinitionCheckInterpreter(world::UInt, @nospecialize(hook), @nospecialize(state)) =
    DefinitionCheckInterpreter(world, InferenceParams(), OptimizationParams(), InferenceResult[],
                               DefinitionCheckCache(IdDict{MethodInstance,CodeInstance}()), hook, state)
InferenceParams(interp::DefinitionCheckInterpreter) = interp.inf_params
OptimizationParams(interp::DefinitionCheckInterpreter) = interp.opt_params
get_inference_world(interp::DefinitionCheckInterpreter) = interp.world
get_inference_cache(interp::DefinitionCheckInterpreter) = interp.inf_cache
cache_owner(::DefinitionCheckInterpreter) = :DefinitionCheck
code_cache(interp::DefinitionCheckInterpreter) = WorldView(interp.code_cache, WorldRange(interp.world))
get(wvc::WorldView{DefinitionCheckCache}, mi::MethodInstance, default) = get(wvc.cache.dict, mi, default)
getindex(wvc::WorldView{DefinitionCheckCache}, mi::MethodInstance) = getindex(wvc.cache.dict, mi)
haskey(wvc::WorldView{DefinitionCheckCache}, mi::MethodInstance) = haskey(wvc.cache.dict, mi)
setindex!(wvc::WorldView{DefinitionCheckCache}, ci::CodeInstance, mi::MethodInstance) = (setindex!(wvc.cache.dict, ci, mi); wvc)

get_escape_cache(interp::AbstractInterpreter) = GetNativeEscapeCache(interp)

abstract type CallInfo end

@nospecialize

function add_edges!(edges::Vector{Any}, info::CallInfo)
    if info === NoCallInfo()
        return nothing # just a minor optimization to avoid dynamic dispatch
    end
    add_edges_impl(edges, info)
    nothing
end
nsplit(info::CallInfo) = nsplit_impl(info)::Union{Nothing,Int}
getsplit(info::CallInfo, idx::Int) = getsplit_impl(info, idx)::MethodLookupResult
getresult(info::CallInfo, idx::Int) = getresult_impl(info, idx)#=::Union{Nothing,ConstResult}=#

add_edges_impl(::Vector{Any}, ::CallInfo) = error("""
    All `CallInfo` is required to implement `add_edges_impl(::Vector{Any}, ::CallInfo)`""")
nsplit_impl(::CallInfo) = nothing
getsplit_impl(::CallInfo, ::Int) = error("""
    A `info::CallInfo` that implements `nsplit_impl(info::CallInfo)::Int` must implement `getsplit_impl(info::CallInfo, idx::Int)::MethodLookupResult`
    in order to correctly opt in to inlining""")
getresult_impl(::CallInfo, ::Int) = nothing

@specialize
