# The facts mode of the native interpreter (writeup/plans/2026-10-17-policy-in-compiler-plan.md,
# part (A)): a call is inferred, with its edges, only at a fact signature; any other call is compiled
# as a dynamic call, typed `Any`, recording nothing.
#
# The mode is a field of `NativeInterpreter`, so that one process can hold code of both kinds (part
# (B) marks trait-world code). Its default is the process switch: an image built with
# JULIA_COMPILE_BY_FACTS=1 (Base and the standard libraries compiled by facts) keeps it on; the
# variable, read once per process by the runtime, overrides it either way. `all` is a test bed: use
# a depot of its own, since package images record the system image but not the switch.
# Two things stay plain: the compiler's own code (it runs in its own world, as plain Julia's
# compiler does), and `return_type` at run time (its answer is plain inference's, so that values
# are Julia's; `return_type_interp`).

"Set while a system image is built with the mode on, and kept in it: the image's default."
const FACTS_IMAGE = RefValue{Bool}(false)
"The process switch: on for an image built with the mode, or as JULIA_COMPILE_BY_FACTS says."
function facts_default()
    v = ccall(:jl_compile_by_facts, Cint, ())     # 1 on, 0 off, 2 unset (read once per process)
    v == Cint(2) && return FACTS_IMAGE[]
    v == Cint(1) && !FACTS_IMAGE[] && (FACTS_IMAGE[] = true)
    return v == Cint(1)
end
facts_mode(interp::AbstractInterpreter) = interp isa NativeInterpreter && interp.facts

# --- (B): code compiled by facts beside plain code ------------------------------------------------
# A module compiled by facts (`@compiler_options facts=true`, as `@trait_module` declares) has its
# methods compiled by facts under their own cache owner, whoever calls them; everything a facts frame
# calls is compiled by facts under that owner too. Plain code calling a facts method names the
# facts CodeInstance. At run time, code of that owner dispatches to code of that owner (part (C),
# `jl_apply_generic_facts` in the runtime).

"The cache owner of code compiled by facts beside plain code."
const FACTS_OWNER = :facts
"""
Whether `m` is compiled by facts: a method of a module compiled by facts, unless it is a boundary
(`Expr(:meta, :facts, 0)` in its body: compiled in its caller's mode, as a trait entry is).
"""
is_facts_method(@nospecialize(m)) = m isa Method && ccall(:jl_method_is_facts, Cint, (Any,), m) != 0
"A facts interpreter beside `interp`: its world, parameters and codegen cache, the facts owner."
facts_interp(interp::NativeInterpreter) =
    NativeInterpreter(interp.world, interp.method_table, InferenceResult[], interp.codegen,
                      interp.inf_params, interp.opt_params, true, FACTS_OWNER)
"""
The interpreter that infers code of cache owner `owner` beside `interp` (for compile queues, whose
items may be CodeInstances of either owner).
"""
function interp_for_owner(interp::AbstractInterpreter, @nospecialize(owner))
    cache_owner(interp) === owner && return interp
    interp isa NativeInterpreter || return interp
    owner === FACTS_OWNER && return facts_interp(interp)
    owner === nothing || return interp
    return NativeInterpreter(interp.world, interp.method_table, InferenceResult[], interp.codegen,
                             interp.inf_params, interp.opt_params, facts_default(), nothing)
end
"The owner whose code dispatch runs for `mi`: the facts owner for a method of a facts module."
default_owner(mi::MethodInstance) = is_facts_method(mi.def) ? FACTS_OWNER : nothing
"Whether plain inference in `interp` must leave `method` to an inference by facts."
leaves_to_facts(interp::AbstractInterpreter, @nospecialize(method)) =
    interp isa NativeInterpreter && interp.owner !== FACTS_OWNER && is_facts_method(method)

"""
A call from plain code to a facts method: the callee is inferred by facts, in an inference of its own
(its own stack, as a request from the runtime would be), and the call names that CodeInstance, with
the edge to it. Plain code never infers a facts method itself.
"""
function facts_edge(interp::NativeInterpreter, method::Method, mi::MethodInstance, caller::AbsIntState,
                    edgecycle::Bool, edgelimited::Bool)
    ci = typeinf_ext(facts_interp(interp), mi, SOURCE_MODE_NOT_REQUIRED)
    ci isa CodeInstance && return return_cached_result(interp, method, ci, caller, edgecycle, edgelimited)
    return Future(MethodCallResult(interp, caller, method, Any, Any, Effects(), nothing, edgecycle, edgelimited))
end

"Whether `mi` is the compiler's own code (a method of `Compiler` or a module inside it)."
function is_compiler_code(mi::MethodInstance)
    m = mi.def
    mod = m isa Method ? m.module : m::Module
    while true
        mod === Compiler && return true
        p = parentmodule(mod)
        p === mod && return false
        mod = p
    end
end

"The most concrete signatures one fact may stand for: Julia's own bound on union splitting."
const MAX_FACT_SPLITS = 4

"""
Whether `T` (a call's argument types) is a fact signature: a dispatch tuple; a tuple whose
tuple-typed arguments have unions of concrete element types (it admits exactly the values of its
splits, as top-level union splitting); or a closure's call at kinds (only its owner adds methods).
"""
is_fact_sig(@nospecialize(T)) = T isa DataType && (isdispatchtuple(T) || fact_splits_ok(T) || vararg_fact(T) ||
    (T <: Tuple && length(T.parameters) >= 1 && is_closure_type(T.parameters[1]) && kinds_ok(T.parameters)))

"""
A tuple type every value of which is a dispatch tuple with the same owners in it: concrete elements
(not kinds), tuple elements of the same kind, and a `Vararg{E}` tail of such an `E`. A call
signature of this form is a fact, and so is an argument of this type: only the stakeholders of its
members or a pirate can change dispatch at any of them (the user, session 29: "Anything that
satisfies 'can only be changed by stakeholders or pirates' is fine"). Julia's core test of issue
#10981 splats 2·10^6 arguments; a function of varargs passes them on as `Tuple{Vararg{E}}`.
"""
function vararg_fact(@nospecialize(T))
    (T isa DataType && T <: Tuple) || return false
    isdispatchtuple(T) && return false                 # a dispatch tuple is a fact already
    ps = T.parameters
    for i in 1:length(ps)
        p = ps[i]
        if isvarargtype(p)
            i == length(ps) || return false
            concrete_member(unwrapva(p)) || return false
        else
            concrete_member(p) || return false
        end
    end
    return true
end
"Whether every value of type `p` has the same concrete type's owners: a concrete type (not a kind), or a tuple type of such members."
concrete_member(@nospecialize(p)) = (isdispatchelem(p) && !iskindtype(p)) || vararg_fact(p)

function fact_splits_ok(@nospecialize(T))
    (T isa DataType && T <: Tuple) || return false
    ps = T.parameters
    (isempty(ps) || isvarargtype(ps[end])) && return false
    n = 1
    for p in ps
        k = isdispatchelem(p) ? 1 : tuple_splits(p)
        k == 0 && return false
        n *= k
        n > MAX_FACT_SPLITS && return false
    end
    return true
end
"The number of concrete tuple types `P` admits when its elements are concrete or unions of concrete types; 0 if not such a tuple."
function tuple_splits(@nospecialize(P))
    (P isa DataType && P <: Tuple) || return 0
    ps = P.parameters
    (isempty(ps) || isvarargtype(ps[end])) && return 0
    n = 1
    for q in ps
        if isdispatchelem(q)
            k = 1
        elseif q isa Union
            us = uniontypes(q)
            for u in us
                isdispatchelem(u) || return 0
            end
            k = length(us)
        else
            k = tuple_splits(q)
        end
        k == 0 && return 0
        n *= k
        n > MAX_FACT_SPLITS && return 0
    end
    return n
end

"A closure's type: its instance's name is generated (`#…`), unlike a named function's (`sum`)."
function is_closure_type(@nospecialize(F))
    (F isa DataType && F <: Function) || return false
    s = F.name.singletonname
    p = ccall(:jl_symbol_name, Ptr{UInt8}, (Any,), s)
    return Core.Intrinsics.pointerref(p, 1, 1) === UInt8(0x23)     # '#'
end
function kinds_ok(ps::SimpleVector)
    for p in ps
        (p isa Type && (isdispatchelem(p) || p === DataType || p === UnionAll || p === Union)) || return false
    end
    return true
end

const FACTS_ABSTRACT_SITE = FailedMethodMatch("facts mode: argument types not concrete, dispatched at run time")

"`find_method_matches` under the facts mode: every split must be a fact signature."
function facts_find_method_matches(interp::AbstractInterpreter, argtypes::Vector{Any}, @nospecialize(atype),
                                   max_union_splitting::Int, max_methods::Int)
    𝕃 = typeinf_lattice(interp)
    if is_union_split_eligible(𝕃, argtypes, max_union_splitting)
        for arg_n in switchtupleunion(𝕃, argtypes)
            sig_n = argtypes_to_type(arg_n)
            sig_n === Union{} && continue
            is_fact_sig(sig_n) || return FACTS_ABSTRACT_SITE
        end
        return find_union_split_method_matches(interp, argtypes, max_methods)
    end
    is_fact_sig(atype) || return FACTS_ABSTRACT_SITE
    return find_simple_method_matches(interp, atype, max_methods)
end

# --- `return_type` under the facts mode, step 2 (as TraitWorld's `_native_return_type`) -------------
# `return_type` asks what *plain* inference gives. When the facts mode's own answer cannot be wider
# than plain inference's (a constant, a concrete type, Union{}), it is folded; otherwise plain
# inference answers at the signature, in its own cache (no code is generated from it), and its
# answer is folded only when it rests on facts at concrete signatures, with an edge to that
# inference result: a load changes the folded answer exactly when it changes plain Julia's.

"Plain inference for `return_type` under the facts mode: its own cache owner, never used for code."
struct PlainRTInterp <: AbstractInterpreter
    world::UInt
    inf_params::InferenceParams
    opt_params::OptimizationParams
    inf_cache::Vector{InferenceResult}
end
PlainRTInterp(world::UInt) = PlainRTInterp(world, InferenceParams(), OptimizationParams(), InferenceResult[])
InferenceParams(i::PlainRTInterp) = i.inf_params
OptimizationParams(i::PlainRTInterp) = i.opt_params
get_inference_world(i::PlainRTInterp) = i.world
get_inference_cache(i::PlainRTInterp) = i.inf_cache
cache_owner(::PlainRTInterp) = :facts_plain_return_type

struct PlainRTInfo <: CallInfo
    sigs::Vector{Any}
    cis::Vector{CodeInstance}
end
function add_edges_impl(edges::Vector{Any}, info::PlainRTInfo)
    for k in 1:length(info.sigs)
        push!(edges, info.sigs[k]); push!(edges, info.cis[k])
    end
    nothing
end

"""
The interpreter for `return_type` at run time: under the facts mode, plain inference in its own cache
owner, so that the answer is plain Julia's and no plain code enters the cache code is compiled from.
"""
return_type_interp(world::UInt) = facts_default() ? PlainRTInterp(world) : NativeInterpreter(world)

"The concrete signatures `sig` admits (top-level unions and unions in tuple arguments split, at most `MAX_FACT_SPLITS`), or `nothing`."
function concrete_splits(@nospecialize(sig))
    (sig isa DataType && sig <: Tuple) || return nothing
    isdispatchtuple(sig) && return Any[sig]
    ps = sig.parameters
    (isempty(ps) || isvarargtype(ps[end])) && return nothing
    alts = Vector{Any}[]
    n = 1
    for p in ps
        a = isdispatchelem(p) ? Any[p] : p isa Union ? Any[uniontypes(p)...] : tuple_alts(p)
        a === nothing && return nothing
        for u in a
            (isdispatchelem(u) || (u isa DataType && isdispatchtuple(u))) || return nothing
        end
        n *= length(a)
        n > MAX_FACT_SPLITS && return nothing
        push!(alts, a)
    end
    out = Any[Any[]]
    for a in alts
        out = cartesian_push(out, a)
    end
    return Any[Tuple{c...} for c in out]
end
"Every partial combination in `out` extended by every alternative in `a`."
function cartesian_push(out::Vector{Any}, a::Vector{Any})
    new = Any[]
    for c in out, u in a
        push!(new, Any[(c::Vector{Any})..., u])
    end
    return new
end
function tuple_alts(@nospecialize(P))
    tuple_splits(P) == 0 && return nothing
    out = Any[Any[]]
    for q in P.parameters
        a = isdispatchelem(q) ? Any[q] : q isa Union ? Any[uniontypes(q)...] : tuple_alts(q)
        a === nothing && return nothing
        out = cartesian_push(out, a)
    end
    return Any[Tuple{c...} for c in out]
end

"Whether `ci`'s answer rests on facts at concrete signatures only, through at most `limit` CodeInstances."
function rests_on_facts(ci::CodeInstance, limit::Int = 2000)
    seen = IdSet{CodeInstance}()
    todo = CodeInstance[ci]
    while !isempty(todo)
        c = pop!(todo)
        c in seen && continue
        push!(seen, c)
        length(seen) > limit && return false
        es = c.edges
        i = 1
        while i <= length(es)
            e = es[i]
            if e isa Int || e isa Core.MethodTable || e isa MethodInstance
                return false
            elseif e isa Core.Binding
                i += 1
            elseif e isa CodeInstance
                is_fact_sig(get_ci_mi(e).specTypes) || return false
                push!(todo, e)
                i += 1
            elseif e isa Type && i < length(es)
                callee = es[i + 1]
                callee isa CodeInstance || return false
                m = get_ci_mi(callee).def
                (is_fact_sig(e) || (m isa Method && e == m.sig)) || return false
                push!(todo, callee)
                i += 2
            else
                return false
            end
        end
    end
    return true
end

"Plain inference's `return_type` answer at `sig`, merged over its concrete splits as plain `return_type` merges its matches, folded with an edge per split when every answer rests on facts; else `nothing`."
function facts_plain_return_type(interp::AbstractInterpreter, @nospecialize(sig), sv::AbsIntState)
    splits = concrete_splits(sig)
    splits === nothing && return nothing
    world = get_inference_world(interp)
    rt = Union{}
    sigs = Any[]; cis = CodeInstance[]
    valid = WorldRange()
    for s in splits
        lo, hi = RefValue{UInt}(typemin(UInt)), RefValue{UInt}(typemax(UInt))
        ms = _methods_by_ftype(s, nothing, 1, world, false, lo, hi, Ptr{Int32}(C_NULL))
        (ms isa Vector && length(ms) == 1) || return nothing
        match = ms[1]::MethodMatch
        match.fully_covers || return nothing
        mi = specialize_method(match)
        nci = typeinf_ext_toplevel(PlainRTInterp(world), mi, SOURCE_MODE_NOT_REQUIRED)
        nci isa CodeInstance || return nothing
        rests_on_facts(nci) || return nothing
        valid = intersect(valid, WorldRange(max(lo[], nci.min_world), min(hi[], nci.max_world)))
        # decoded as the compiler decodes it: `rettype_const` may hold a lattice element
        # (an `InterConditional` for a `Bool` predicate), not a constant
        r = cached_return_type(nci)
        rt_s = r isa Const ? Core.Typeof(r.val) : widenconst(r)
        rt = tmerge(fallback_lattice, rt, rt_s)
        push!(sigs, s); push!(cis, nci)
    end
    update_valid_age!(sv, valid)
    return CallMeta(Const(rt), Union{}, Effects(EFFECTS_TOTAL; nortcall = false), MethodResultPure(PlainRTInfo(sigs, cis)))
end

# --- (D): a call on a dynamic value, split at the call site ---------------------------------------
# `dynamic_call(K, f, x, args...)` is `f(x, args...)` where `x` is a value of an open set of types
# (a trait's dynamic value, `K` the trait). Inference splits it over the candidate types registered
# for `K` (at most `MAX_FACT_SPLITS`): `f` at their union, each split a fact with its edge, plus a
# fallback for any other type; the inliner emits `isa` branches with direct calls and the dynamic
# call in the last branch, as for union splitting. The candidates are read without an edge: a type
# registered later reaches the fallback, which is correct, and invalidates nothing.

"`f(x, args...)::RT` for a value `x` of the open set of types registered for `K` (split at the call site)."
dynamic_call(@nospecialize(K), ::Type{RT}, f, x, args...) where {RT} = f(x, args...)::RT

mutable struct DynamicCandidates
    @atomic table::Any          # an IdDict{Any,Vector{Any}}, replaced on each registration, never mutated
end
const DYNAMIC_CANDIDATES = DynamicCandidates(nothing)

"Register type `C` as a candidate of `K` (TraitWorld does, for each trait-object type a dynamic value of `K` holds)."
function register_dynamic_candidate!(@nospecialize(K), @nospecialize(C))
    while true
        old = @atomic DYNAMIC_CANDIDATES.table
        if old !== nothing
            v = get(old::IdDict{Any,Vector{Any}}, K, nothing)
            if v !== nothing
                for c in v::Vector{Any}
                    c === C && return nothing
                end
            end
        end
        new = old === nothing ? IdDict{Any,Vector{Any}}() : copy(old::IdDict{Any,Vector{Any}})
        v = Any[get(new, K, Any[])::Vector{Any}..., C]
        new[K] = v
        (@atomicreplace DYNAMIC_CANDIDATES.table old => new).success && return nothing
    end
end
"The candidate types of `K` now: the first `MAX_FACT_SPLITS` registered."
function dynamic_candidates(@nospecialize(K))
    t = @atomic DYNAMIC_CANDIDATES.table
    t === nothing && return Any[]
    v = get(t::IdDict{Any,Vector{Any}}, K, nothing)
    v === nothing && return Any[]
    v = v::Vector{Any}
    return length(v) <= MAX_FACT_SPLITS ? v : v[1:MAX_FACT_SPLITS]
end

"""
The call info of a `dynamic_call`: the call of `f` (at the candidates' union when `x`'s type is open);
the inliner always rewrites the statement to that call. `open`: a value of another type may come,
so the split keeps the dynamic call as its last branch.
"""
struct DynamicSplitInfo <: CallInfo
    info::CallInfo
    open::Bool
    rt::Any        # the declared result: the fallback's result is asserted to it
end
add_edges_impl(edges::Vector{Any}, info::DynamicSplitInfo) = add_edges!(edges, info.info)
nsplit_impl(info::DynamicSplitInfo) = nsplit(info.info)
getsplit_impl(info::DynamicSplitInfo, idx::Int) = getsplit(info.info, idx)
getresult_impl(info::DynamicSplitInfo, idx::Int) = getresult(info.info, idx)

"Inference of `dynamic_call(K, f, x, args...)`: `f` at the union of `K`'s candidates (a narrower known type of `x` kept), and a fallback."
function abstract_dynamic_call(interp::AbstractInterpreter, argtypes::Vector{Any}, si::StmtInfo, sv::AbsIntState)
    K = argtypes[2]
    RTa = argtypes[3]
    RT = RTa isa Const && RTa.val isa Type ? RTa.val : (isType(widenconst(RTa)) ? widenconst(RTa).parameters[1] : Any)
    RT isa Type || (RT = Any)
    newargs = Any[argtypes[4:end]...]
    cands = K isa Const ? dynamic_candidates(K.val) : Any[]
    X = widenconst(argtypes[5])
    narrowed = false
    if !isempty(cands) && !isdispatchelem(X)
        U = Union{}
        for C in cands
            U = Union{U, C}
        end
        U = typeintersect(X, U)
        if U !== Union{}
            newargs[2] = U
            narrowed = true
        end
    end
    call = abstract_call(interp, ArgInfo(nothing, newargs), si, sv)::Future
    return Future{CallMeta}(call, interp, sv) do call, _, _
        # the statement's own type stays what the call gives (the inliner rewrites it to the call of
        # `f`, which asserts nothing); only a split's join is typed `RT`, its fallback asserted
        narrowed || return CallMeta(call.rt, call.exct, call.effects, DynamicSplitInfo(call.info, false, RT), call.refinements)
        # a value of another type reaches the fallback: its result and effects are unknown
        return CallMeta(Any, Any, merge_effects(call.effects, Effects()), DynamicSplitInfo(call.info, true, RT))
    end
end
