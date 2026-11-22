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
Whether `T` (a call's argument types) is a fact signature: a call of an owner-only function at any
signature; a dispatch tuple; a tuple whose tuple-typed arguments have unions of concrete element
types that split by Julia's cost; concrete members followed by a `Vararg` of a concrete member; or a
closure's call at kinds.
"""
is_fact_sig(@nospecialize(T), max_union_splitting::Int = FACT_UNION_SPLITTING) =
    T isa DataType && (isdispatchtuple(T) || owner_only_call(T) || fact_splits_ok(T, max_union_splitting) || vararg_fact(T) ||
    (T <: Tuple && length(T.parameters) >= 1 && is_closure_type(T.parameters[1]) && kinds_ok(T.parameters)))

"""
Whether `F` (a function's type) is owner-only: only its owner may define its methods, so a call of it
is a fact at any signature (a method another module defines for it is a pirate's). A function its
owner declared so (`jl_set_owner_only`), or a function of a module whose `owner_only` option is on
(its submodules included) that the module neither exports nor declares `public`, a closure of such a
module among them, but for a trait's operation, open to the set's members (`jl_set_open_operation`).
Read where the call is inferred.
"""
function is_owner_only(@nospecialize(F))
    F isa DataType || return false
    F <: Function || return false
    tn = F.name
    m = tn.module
    name = tn.singletonname
    ccall(:jl_binding_owner_only_p, Cint, (Any, Any), m, name) != 0 && return true
    # a trait's operation is open to its members, exported or not
    ccall(:jl_binding_open_operation_p, Cint, (Any, Any), m, name) != 0 && return false
    (ccall(:jl_get_module_trait_options, Cint, (Any,), m) & Cint(64)) != 0 || return false
    return ccall(:jl_module_public_p, Cint, (Any, Any), m, name) == 0
end

"A call of an owner-only function: the function's type first, or after a keyword call's `NamedTuple`."
function owner_only_call(@nospecialize(T))
    T <: Tuple || return false
    ps = T.parameters
    n = length(ps)
    n >= 1 || return false
    f = ps[1]
    isvarargtype(f) && return false
    f === typeof(Core.kwcall) && return n >= 3 && !isvarargtype(ps[3]) && is_owner_only(ps[3])
    return is_owner_only(f)
end

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
                             facts, interp.crossings + Int(nested), false)
end

"Whether `interp` infers `mi`'s top frame under the other policy than its method's (facts )."
infers_other_policy(::AbstractInterpreter, ::MethodInstance) = false
infers_other_policy(interp::NativeInterpreter, mi::MethodInstance) =
    !interp.plainrt && mi.def isa Method && mi_policy_facts_at(mi, interp.facts) !== interp.facts

"""
The interpreter that infers code of policy `facts` at the top level: `interp` where it has that
policy, else the other policy's, sharing its codegen cache. `return_type`'s run-time interpreter
infers every method plainly, under an owner of its own.
"""
interp_for_policy(interp::AbstractInterpreter, ::Bool) = interp
interp_for_policy(interp::NativeInterpreter, facts::Bool) =
    interp.plainrt || interp.facts === facts ? interp : policy_interp(interp, facts; nested = false)

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

"""
Whether a record is deep G2: every dispatch edge is at a fact signature, and every callee among its
edges is a CodeInstance that carries the G2 bit. Such code is invalidated only by a definition of
the owners of the functions and types its facts name, at any depth. A callee finishes before its
callers and its bit is final then, so the edges' own bits decide the class, with no walk. `own` is the
CodeInstance itself, or the members of its inference cycle (their CodeInstances and MethodInstances),
which are judged as one: an edge to a member counts, whether it names the member's CodeInstance or its
MethodInstance. Another CodeInstance of the same MethodInstance is judged by its bit. Any other callee
named by its MethodInstance alone is outside the class: its backedge carries whatever code it gets.
A membership's method, which a folded positive's edge names by its instance (an instance with no
code), needs nothing more: only the membership's end breaks it. An edge that names a Method alone
names no code. An invoke at a declared signature is not gated; a binding edge is not a dispatch.
"""
function edges_satisfy_g2(edges::SimpleVector, own::Union{CodeInstance,IdSet{Any}})
    for e in edges
        if e isa CodeInstance
            is_own(own, e) && continue
            # its G2 bit (`JL_CI_FLAGS_G2`)
            (getfield(e, :flags, :monotonic)::UInt8 & 0x10) == 0x00 && return false
        elseif e isa MethodInstance
            # a folded positive membership names its membership's method by its instance, which has no
            # code: a membership edge, which the class allows
            is_own(own, e) || is_membership_mi(e) || return false
        end
    end
    for (invokesig, item) in ForwardToBackedgeIterator(edges)
        if item isa Core.Binding
            continue
        elseif item isa MethodTable
            # a call's dispatch at a fact signature, or a negative's absence edge: only a membership
            # of the pair, by an owner of the set or the type or by a pirate, breaks it
            (is_fact_sig(invokesig) || is_implements_sig(invokesig)) || return false
        elseif item isa MethodInstance && is_membership_mi(item)
            continue
        elseif item isa MethodInstance
            # an `invoke` edge is skipped; an ordinary call is checked at its own signature: a call
            # that reached an instance wider than its match carries that signature (`CallSigBackedge`),
            # by which a later method invalidates it
            (invokesig === nothing || invokesig isa CallSigBackedge) || continue
            is_fact_sig(invokesig isa CallSigBackedge ? invokesig.sig : item.specTypes) || return false
        end
    end
    return true
end

"""
An edge that names a callee by its MethodInstance alone names the callee's CodeInstance where one exists
when the caller finishes: inference had no CodeInstance for the callee where it reached the
call (a recursion through it, a limited inference), and the callee was finished before the caller. The
CodeInstance is the one of the policy the caller's call reaches, valid in the caller's worlds. The backedge
is the same, on the instance: a definition that changes dispatch at the callee ends the caller as before;
of the callee's code, only the CodeInstance named ends it. The caller's own instance, the members of its
inference cycle (not yet cached), a membership's instance and an interpreter of no policy are left as they
are.
"""
function name_callee_code(interp::AbstractInterpreter, caller::MethodInstance, edges::SimpleVector, worlds::WorldRange)
    interp isa NativeInterpreter && !interp.plainrt || return edges
    out = nothing
    for i in 1:length(edges)
        e = edges[i]
        e isa MethodInstance || continue
        (e === caller || !(e.def isa Method) || is_membership_mi(e)) && continue
        cache = WorldView(InternalCodeCache(cache_owner(interp), Cint(mi_policy_facts_at(e, interp.facts))), worlds)
        ci = get(cache, e, nothing)
        ci isa CodeInstance || continue
        out === nothing && (out = Any[edges...])
        out[i] = ci
    end
    return out === nothing ? edges : Core.svec(out...)
end

"Whether `mi` is an instance of a membership's method, which a folded positive's edge names."
is_membership_mi(mi::MethodInstance) = (m = mi.def; m isa Method && is_implements_sig(m.sig))

"Whether `sig` is a signature of `Core.implements`, whose methods are the memberships."
function is_implements_sig(@nospecialize(sig))
    b = unwrap_unionall(sig)
    return b isa DataType && length(b.parameters) == 3 && b.parameters[1] === typeof(Core.implements)
end

is_own(own::CodeInstance, e::CodeInstance) = own === e
is_own(own::CodeInstance, e::MethodInstance) = get_ci_mi(own) === e
is_own(own::IdSet{Any}, @nospecialize(e)) = e in own

"Set the G2 bit of `ci` where its record is deep G2."
set_g2!(ci::CodeInstance, edges::SimpleVector) =
    (edges_satisfy_g2(edges, ci) && ccall(:jl_ci_set_g2, Cvoid, (Any,), ci); nothing)

# Fully typed mode: a method that asks for it is compiled by facts; where its code at a signature holds
# a run-time site, itself or in a callee, the optimizer puts a call of `jl_not_fully_typed` at the code's
# start, which reaches the hook a tool sets: it warns at the first run there, or raises at every run.
# Fully typed code gets no check.

"The method's fully typed mode: 0 off, 1 warn, 2 enforce (its own, else its module's)."
method_fully_typed(m::Method) = ccall(:jl_method_fully_typed, Cint, (Any,), m)

"""
The fault of `mi`'s code under fully typed mode: 0 none, 1 a run-time site (in it or a callee). The
result's type is no fault: a type is a fault where it is used. Only for code compiled by facts of a
method that asks for the mode.
"""
function fully_typed_fault(interp::AbstractInterpreter, mi::MethodInstance, effects::Effects, @nospecialize(rt))
    facts_mode(interp) || return 0
    def = mi.def
    def isa Method || return 0
    method_fully_typed(def) == 0 && return 0
    is_nortsite(effects) || return 1
    return 0
end

"The check at the start of code that fails fully typed mode: `jl_not_fully_typed(svec(mi, level, warned, kind, rt))`."
function insert_fully_typed_check!(ir::IRCode, interp::AbstractInterpreter, mi::MethodInstance, result::InferenceResult)
    kind = fully_typed_fault(interp, mi, result.ipo_effects, result.result)
    kind == 0 && return ir
    site = svec(mi, Int(method_fully_typed(mi.def::Method)), false, Int(kind), widenconst(ignorelimited(result.result)))
    stmt = Expr(:foreigncall, Expr(:tuple, QuoteNode(:jl_not_fully_typed)), Nothing, svec(Any), 0,
                QuoteNode(:ccall), QuoteNode(site))
    insert_node!(ir, 1, NewInstruction(stmt, Nothing, NoCallInfo(), nothing, IR_FLAG_NULL))
    return compact!(ir)
end

# The code a run-time site reaches whatever the values: while an image is built, a
# run-time site in facts code whose types match one method that specializes on none of the site's
# abstract arguments reaches that method's one instance; the image holds it, as plain inference
# compiles a callee at abstract types. The site still dispatches when it runs and records no edge.

"The instances the run-time sites of the facts code `src` reach whatever the values."
function rtsite_targets(interp::AbstractInterpreter, src::CodeInfo, sptypes::Vector{VarState})
    out = MethodInstance[]
    for stmt in src.code
        isexpr(stmt, :(=)) && (stmt = stmt.args[2])
        isexpr(stmt, :call) || continue
        mi = rtsite_target(interp, stmt.args, src, sptypes)
        mi === nothing || push!(out, mi)
    end
    return out
end

function rtsite_target(interp::AbstractInterpreter, args::Vector{Any}, src::CodeInfo, sptypes::Vector{VarState})
    argtypes = Any[]
    for a in args
        applicable(argextype, a, src, sptypes) || return nothing
        # an argument that is no value of the IR (an expression nested in the call): no target
        a isa Expr && !(a.head === :static_parameter || a.head === :boundscheck || a.head === :copyast) && return nothing
        t = widenconst(argextype(a, src, sptypes))
        t === Bottom && return nothing
        push!(argtypes, t)
    end
    isempty(argtypes) && return nothing
    ft = argtypes[1]
    (isdispatchelem(ft) && !(ft <: Builtin)) || return nothing
    atype = argtypes_to_type(argtypes)
    (atype isa DataType && !isdispatchtuple(atype)) || return nothing
    matches = findall(atype, method_table(interp); limit = 1)
    matches === nothing && return nothing
    length(matches.matches) == 1 || return nothing
    m = (only(matches.matches)::MethodMatch).method
    nospec = reinterpret(UInt32, m.nospecialize)
    decl = unwrap_unionall(m.sig)
    decl isa DataType || return nothing
    nd = length(decl.parameters)
    ps = atype.parameters
    for i in 2:length(ps)
        p = ps[i]
        isvarargtype(p) && return nothing
        isdispatchelem(p) && continue
        # an abstract argument: not specialized on (`@nospecialize`), at a declared type the runtime
        # compiles at whatever the value (gf.c, `jl_compilation_sig`)
        iarg = m.isva ? min(i - 1, m.nargs - 1) : i - 1
        (1 <= iarg <= 32 && nd >= 1) || return nothing
        bit = UInt32(1) << (iarg - 1)
        nospec & bit != 0 || return nothing
        d = unwrapva(decl.parameters[min(i, nd)])
        (has_free_typevars(d) || iskindtype(d)) && return nothing
    end
    return compileable_specialization_for_call(interp, atype)
end
