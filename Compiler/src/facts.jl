# This file is a part of Julia. License is MIT: https://julialang.org/license

# The facts inference: in code compiled by facts, a call is inferred, with an edge, only at a fact
# signature, a signature at which only the function's or the argument types' owners can change
# dispatch. Every other call is a run-time site, dispatched when it runs, and clears the `nortsite`
# effect, which is carried to every caller. Such code is then invalidated only by a definition
# its owners (or a pirate) make.

"Whether the interpreter infers calls only at facts."
facts_mode(::AbstractInterpreter) = false
facts_mode(interp::NativeInterpreter) = interp.facts

"The effects of a call compiled without an edge: unknown, and under the facts mode a run-time site."
unedged_effects(interp::AbstractInterpreter) = facts_mode(interp) ? EFFECTS_RTSITE : Effects()

"The most concrete signatures a fact's union splits into."
const MAX_FACT_SPLITS = 4
"The interpreter's default bound on union splitting, for the checks that have no interpreter at hand."
const FACT_UNION_SPLITTING = InferenceParams().max_union_splitting

"""
Whether `T` (a call's argument types) is a fact signature: a dispatch tuple; a tuple whose
tuple-typed arguments have unions of concrete element types that split by Julia's cost; concrete
members followed by a `Vararg` of a concrete member; or a closure's call at kinds.
"""
is_fact_sig(@nospecialize(T), max_union_splitting::Int = FACT_UNION_SPLITTING) =
    T isa DataType && (isdispatchtuple(T) || fact_splits_ok(T, max_union_splitting) || vararg_fact(T) ||
    (T <: Tuple && length(T.parameters) >= 1 && is_closure_type(T.parameters[1]) && kinds_ok(T.parameters)))

"Concrete members followed by a `Vararg{E}` tail of a concrete member `E`, every element a concrete member."
function vararg_fact(@nospecialize(T))
    T isa DataType || return false
    T <: Tuple || return false
    isdispatchtuple(T) && return false
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
"A concrete type that is not a kind, or a tuple type of such members."
concrete_member(@nospecialize(p)) = (isdispatchelem(p) && !iskindtype(p)) || vararg_fact(p)

"Whether `T`'s tuple-typed arguments have unions of concrete element types within Julia's union-splitting cost."
function fact_splits_ok(@nospecialize(T), max_union_splitting::Int)
    T isa DataType || return false
    T <: Tuple || return false
    ps = T.parameters
    (isempty(ps) || isvarargtype(ps[end])) && return false
    lens = Int[]
    for p in ps
        isdispatchelem(p) && continue
        tuple_union_lengths!(lens, p) || return false
    end
    return union_split_cost(lens) <= max_union_splitting
end
function tuple_union_lengths!(lens::Vector{Int}, @nospecialize(P))
    P isa DataType || return false
    P <: Tuple || return false
    ps = P.parameters
    (isempty(ps) || isvarargtype(ps[end])) && return false
    for q in ps
        if isdispatchelem(q)
            continue
        elseif q isa Union
            us = uniontypes(q)
            for u in us
                isdispatchelem(u) || return false
            end
            push!(lens, length(us))
        else
            tuple_union_lengths!(lens, q) || return false
        end
    end
    return true
end
"Julia's union-splitting cost (`unionsplitcost`) of unions of these lengths."
function union_split_cost(lens::Vector{Int})
    nu, max = 1, 2
    for nti in lens
        if nti > max
            max, nti = nti, max
        end
        nu, ovf = Core.Intrinsics.checked_smul_int(nu, nti)
        ovf && return typemax(Int)
    end
    return nu
end

"A closure's type: its instance's name is generated (`#…`), unlike a named function's."
function is_closure_type(@nospecialize(F))
    F isa DataType || return false
    F <: Function || return false
    s = F.name.singletonname
    p = ccall(:jl_symbol_name, Ptr{UInt8}, (Any,), s)
    return Core.Intrinsics.pointerref(p, 1, 1) === UInt8(0x23)     # '#'
end
function kinds_ok(ps::SimpleVector)
    for p in ps
        p isa Type || return false
        ((isdispatchelem(p) || p === DataType || p === UnionAll || p === Union)) || return false
    end
    return true
end

const FACTS_ABSTRACT_SITE = FailedMethodMatch("facts mode: argument types not a fact signature, dispatched at run time")

"`find_method_matches` under the facts mode: every split must be a fact signature."
function facts_find_method_matches(interp::AbstractInterpreter, argtypes::Vector{Any}, @nospecialize(atype),
                                   max_union_splitting::Int, max_methods::Int)
    𝕃 = typeinf_lattice(interp)
    if is_union_split_eligible(𝕃, argtypes, max_union_splitting)
        for arg_n in switchtupleunion(𝕃, argtypes)
            sig_n = argtypes_to_type(arg_n)
            sig_n === Union{} && continue
            is_fact_sig(sig_n, max_union_splitting) || return FACTS_ABSTRACT_SITE
        end
        return find_union_split_method_matches(interp, argtypes, max_methods)
    end
    is_fact_sig(atype, max_union_splitting) || return FACTS_ABSTRACT_SITE
    return find_simple_method_matches(interp, atype, max_methods)
end

# The policy per method: plain, facts, or the policy of the code that reaches it (a boundary).
const POLICY_PLAIN = Cint(0)
const POLICY_FACTS = Cint(1)
const POLICY_CALLER = Cint(2)
const REACH_PLAIN = Cint(0)
const REACH_FACTS = Cint(1)
const REACH_RUNTIME = Cint(2)
method_policy(m::Method) = ccall(:jl_method_policy, Cint, (Any,), m)
method_policy_facts(m::Method) = method_policy(m) === POLICY_FACTS
"Whether the code of `mi` is compiled by facts where the runtime reaches it (a compile request)."
mi_policy_facts(mi::MethodInstance) = ccall(:jl_mi_policy_at, Cint, (Any, Cint), mi, REACH_RUNTIME) != 0
"Whether a callee of policy `p` is inferred under the other policy than the caller's `facts`."
other_policy(p::Cint, facts::Bool) = p !== POLICY_CALLER && (p === POLICY_FACTS) !== facts

"Whether `mi`'s code is compiled by facts where code of policy `facts` reaches it."
mi_policy_facts_at(mi::MethodInstance, facts::Bool) =
    ccall(:jl_mi_policy_at, Cint, (Any, Cint), mi, facts ? REACH_FACTS : REACH_PLAIN) != 0
"The policy a CodeInstance of Julia's own owner was compiled in (`JL_CI_FLAGS_FACTS`)."
ci_policy_facts(ci::CodeInstance) = (getfield(ci, :flags, :monotonic)::UInt8 & 0x20) !== 0x00

"""
The native interpreter of the other policy, in the same world and with the same parameters; one
crossing further where it is nested in an inference (`nested`). It shares the codegen cache, so a
queue that compiles what it infers finds the source of code of either policy.
"""
function policy_interp(interp::NativeInterpreter, facts::Bool; nested::Bool = true)
    world = get_inference_world(interp)
    return NativeInterpreter(world, CachedMethodTable(InternalMethodTable(world)), InferenceResult[],
                             interp.codegen, InferenceParams(interp), OptimizationParams(interp),
                             facts, interp.crossings + Int(nested))
end

"Whether `interp` infers `mi`'s top frame under the other policy than its method's (reflection)."
infers_other_policy(::AbstractInterpreter, ::MethodInstance) = false
infers_other_policy(interp::NativeInterpreter, mi::MethodInstance) =
    mi.def isa Method && mi_policy_facts_at(mi, interp.facts) !== interp.facts

"""
The interpreter that infers code of policy `facts` at the top level: `interp` where it has that
policy, else the other policy's, sharing its codegen cache.
"""
interp_for_policy(interp::AbstractInterpreter, ::Bool) = interp
interp_for_policy(interp::NativeInterpreter, facts::Bool) =
    interp.facts === facts ? interp : policy_interp(interp, facts; nested = false)

# How many inferences of the other policy may nest: past it a call is answered as unknown, so that
# two policies' inferences of a cycle do not recurse into each other.
const MAX_POLICY_CROSSINGS = 4

"""
A callee whose policy differs from the caller's is inferred under its own policy, in an inference
of its own: its cached code of that policy, or a new inference; the caller's result
rests on the callee's CodeInstance.
"""
function typeinf_other_policy(interp::NativeInterpreter, method::Method, mi::MethodInstance,
                              caller::AbsIntState, edgecycle::Bool, edgelimited::Bool)
    other = policy_interp(interp, !interp.facts)
    codeinst = get(code_cache(other), mi, nothing)
    if !(codeinst isa CodeInstance)
        interp.crossings >= MAX_POLICY_CROSSINGS &&
            return Future(MethodCallResult(interp, caller, method, Any, Any, unedged_effects(interp), nothing, edgecycle, edgelimited))
        codeinst = typeinf_ext(other, mi, SOURCE_MODE_NOT_REQUIRED)
    end
    codeinst isa CodeInstance ||
        return Future(MethodCallResult(interp, caller, method, Any, Any, unedged_effects(interp), nothing, edgecycle, edgelimited))
    return return_cached_result(interp, method, codeinst, caller, edgecycle, edgelimited)
end
