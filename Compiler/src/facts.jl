# This file is a part of Julia. License is MIT: https://julialang.org/license

# The facts inference (compiler §1): in code compiled by facts, a call is inferred, with an edge,
# only at a fact signature. Every other call is a run-time site, dispatched when it runs, and clears
# the `nortsite` effect, which is carried to every caller.

"Whether the interpreter infers calls only at facts (compiler §1.1)."
facts_mode(::AbstractInterpreter) = false
facts_mode(interp::NativeInterpreter) = interp.facts

"The effects of a call compiled without an edge: unknown, and under the facts mode a run-time site."
unedged_effects(interp::AbstractInterpreter) = facts_mode(interp) ? EFFECTS_RTSITE : Effects()

"The most concrete signatures a fact's union splits into, and a split's candidates (§1.2, §7.2)."
const MAX_FACT_SPLITS = 4
"The interpreter's default bound on union splitting, for the checks that have no interpreter at hand."
const FACT_UNION_SPLITTING = InferenceParams().max_union_splitting

"""
Whether `T` (a call's argument types) is a fact signature (compiler §1.2): a dispatch tuple; a
tuple whose tuple-typed arguments have unions of concrete element types that split by Julia's cost;
concrete members followed by a `Vararg` of a concrete member; or a closure's call at kinds.
"""
is_fact_sig(@nospecialize(T), max_union_splitting::Int = FACT_UNION_SPLITTING) =
    T isa DataType && (isdispatchtuple(T) || fact_splits_ok(T, max_union_splitting) || vararg_fact(T) ||
    (T <: Tuple && length(T.parameters) >= 1 && is_closure_type(T.parameters[1]) && kinds_ok(T.parameters)))

"Concrete members followed by a `Vararg{E}` tail of a concrete member `E`, every element a concrete member."
function vararg_fact(@nospecialize(T))
    (T isa DataType && T <: Tuple) || return false
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
    (T isa DataType && T <: Tuple) || return false
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
    (P isa DataType && P <: Tuple) || return false
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

const FACTS_ABSTRACT_SITE = FailedMethodMatch("facts mode: argument types not a fact signature, dispatched at run time")

"`find_method_matches` under the facts mode: every split must be a fact signature (compiler §1.2)."
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

# The policy per method (compiler §2.3): a method compiled by facts by itself; a boundary and a
# neutral method plain (C-44, C-45).
method_policy_facts(m::Method) = ccall(:jl_method_is_facts, Cint, (Any,), m) != 0
mi_policy_facts(mi::MethodInstance) = (def = mi.def; def isa Method && method_policy_facts(def))

"The native interpreter of the other policy, in the same world and with the same parameters, one crossing further."
policy_interp(interp::NativeInterpreter, facts::Bool) =
    NativeInterpreter(get_inference_world(interp); inf_params = InferenceParams(interp),
                      opt_params = OptimizationParams(interp), facts, crossings = interp.crossings + 1)

# How many inferences of the other policy may nest: past it a call is answered as unknown, so that
# two policies' inferences of a cycle do not recurse into each other.
const MAX_POLICY_CROSSINGS = 4

"""
A callee whose policy differs from the caller's is inferred under its own policy, in an inference
of its own (compiler §2.4): its cached code of that policy, or a new inference; the caller's result
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
