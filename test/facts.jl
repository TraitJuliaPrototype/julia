# This file is a part of Julia. License is MIT: https://julialang.org/license

# The facts policy: a module compiled by facts infers a call, with its edge, only at a fact signature,
# and dispatches every other call when it runs.

using Test
const CC = Base.Compiler

# the return type and the effects of `f` at `T` under the facts inference or plain inference, with an
# interpreter of the current world
facts_rt(f, T; facts::Bool = true) = Base.code_typed(f, T; interp = CC.NativeInterpreter(; facts))[1][2]
facts_effects(f, T) = Base.infer_effects(f, T; interp = CC.NativeInterpreter(; facts = true))

module OwnerOnly
Base.Experimental.@compiler_options owner_only=true
export pub_exported
public pub_declared
helper(x::Int) = 1
helper(x::Float64) = 2.0
pub_exported(x::Int) = 1
pub_exported(x::Float64) = 2.0
pub_declared(x::Int) = 1
pub_declared(x::Float64) = 2.0
call_helper(x::Real) = helper(x)
call_exported(x::Real) = pub_exported(x)
call_declared(x::Real) = pub_declared(x)
call_kw(x::Real) = kwhelper(x; k = 1)
kwhelper(x::Int; k) = k
kwhelper(x::Float64; k) = 2.0
module Sub
inner(x::Int) = 1
inner(x::Float64) = 2.0
call_inner(x::Real) = inner(x)
end
end

module NotOwnerOnly
other(x::Int) = 1
other(x::Float64) = 2.0
call_other(x::Real) = other(x)
declared(x::Int) = 1
declared(x::Float64) = 2.0
call_declared(x::Real) = declared(x)
end

@testset "owner-only functions" begin
    @test ccall(:jl_get_module_options, Cint, (Any,), OwnerOnly) == 64
    @test ccall(:jl_get_module_options, Cint, (Any,), OwnerOnly.Sub) == 64   # inherited at creation
    @test ccall(:jl_get_module_options, Cint, (Any,), NotOwnerOnly) == 0
    @test CC.is_owner_only(typeof(OwnerOnly.helper))
    @test CC.is_owner_only(typeof(OwnerOnly.Sub.inner))
    @test !CC.is_owner_only(typeof(OwnerOnly.pub_exported))
    @test !CC.is_owner_only(typeof(OwnerOnly.pub_declared))
    @test !CC.is_owner_only(typeof(NotOwnerOnly.other))

    # owner-only by the module's option: a fact at `Real`, inferred; plain inference the same
    @test facts_rt(OwnerOnly.call_helper, (Real,)) === Union{Int, Float64}
    @test facts_rt(OwnerOnly.call_helper, (Real,); facts = false) === Union{Int, Float64}
    @test CC.is_nortsite(facts_effects(OwnerOnly.call_helper, (Real,)))
    @test facts_rt(OwnerOnly.Sub.call_inner, (Real,)) === Union{Int, Float64}
    @test facts_rt(OwnerOnly.call_kw, (Real,)) === Union{Int, Float64}
    # exported, or declared public: not owner-only, a run-time site at `Real`
    @test facts_rt(OwnerOnly.call_exported, (Real,)) === Any
    @test facts_rt(OwnerOnly.call_declared, (Real,)) === Any
    @test !CC.is_nortsite(facts_effects(OwnerOnly.call_exported, (Real,)))

    # a module without the option: a run-time site; a function its owner declares owner-only (before
    # any caller is inferred), a fact
    @test facts_rt(NotOwnerOnly.call_other, (Real,)) === Any
    @test !CC.is_owner_only(typeof(NotOwnerOnly.declared))
    ccall(:jl_set_owner_only, Cvoid, (Any, Any), NotOwnerOnly, :declared)
    @test ccall(:jl_binding_owner_only_p, Cint, (Any, Any), NotOwnerOnly, :declared) == 1
    @test CC.is_owner_only(typeof(NotOwnerOnly.declared))
    @test facts_rt(NotOwnerOnly.call_declared, (Real,)) === Union{Int, Float64}
    @test facts_rt(NotOwnerOnly.call_other, (Real,)) === Any

    # an unknown option bit is refused
    @test_throws ErrorException ccall(:jl_set_module_option, Cvoid, (Any, Cint, Cint), NotOwnerOnly, 3, 1)
end

# the CodeInstances of the native owner in `f`'s instance at `T`
function native_cis(f, T)
    mi = Base.method_instance(f, T)
    out = Core.CodeInstance[]
    ci = mi.cache
    while true
        ci.owner === nothing && push!(out, ci)
        isdefined(ci, :next) || break
        ci = ci.next
    end
    return out
end
const CI_FLAG_G2 = 0x10      # deep G2: its dispatch edges, and every callee's at any depth, at fact signatures
const CI_FLAG_FACTS = 0x20   # compiled by facts

module DeepEndings
Base.Experimental.@compiler_options facts=true
g(x::Int) = 1
f(x::Int) = g(x) + 1
h(x::Int) = g(x) + 2
end
deep_plain_g(x::Int) = 1
deep_plain_f(x::Int) = deep_plain_g(x) + 1

@testset "what a definition ends of the deep class" begin
    take() = (r = ccall(:jl_take_deep_endings, Any, ()); r === nothing ? Any[] : r)
    @test DeepEndings.f(1) == 2 && DeepEndings.h(1) == 3 && deep_plain_f(1) == 2
    ci_f = only(native_cis(DeepEndings.f, (Int,)))
    @test ci_f.flags & (CI_FLAG_G2 | CI_FLAG_FACTS) == CI_FLAG_G2 | CI_FLAG_FACTS
    @test ccall(:jl_set_deep_endings, Cint, (Cint,), 1) == 0
    try
        # a definition that replaces g(::Int) ends f's and h's code: kind :definition, the cause the
        # new method
        @eval DeepEndings g(x::Int) = 10
        list = take()
        @test length(list) >= 6 && length(list) % 3 == 0
        newg = which(DeepEndings.g, (Int,))
        @test all(==(:definition), list[1:3:end])
        @test all(c -> c === newg, list[2:3:end])
        @test ci_f in list[3:3:end]
        @test all(ci -> ci.flags & (CI_FLAG_G2 | CI_FLAG_FACTS) == CI_FLAG_G2 | CI_FLAG_FACTS, list[3:3:end])
        @test DeepEndings.f(1) == 11 && DeepEndings.h(1) == 12

        # plain code that a definition ends is not kept
        @eval deep_plain_g(x::Int) = 5
        @test isempty(take())
        @test deep_plain_f(1) == 6

        # a deletion: kind :deletion, the cause the method deleted
        @eval DeepEndings g(x::Integer) = 20
        take()
        @test DeepEndings.f(1) == 11
        Base.delete_method(which(DeepEndings.g, (Int,)))
        list = take()
        @test !isempty(list)
        @test all(==(:deletion), list[1:3:end])
        @test all(c -> c === newg, list[2:3:end])
        @test DeepEndings.f(1) == DeepEndings.g(1) + 1   # the method the deleted one replaced, or g(::Integer)
    finally
        @test ccall(:jl_set_deep_endings, Cint, (Cint,), 0) == 1
    end
    # switched off: nothing is kept
    @eval DeepEndings g(x::Int) = 30
    @test isempty(take())
    @test DeepEndings.f(1) == 31
end

module FullyTypedEnforce
Base.Experimental.@compiler_options fully_typed=enforce
typed(x::Int) = x + 1
untyped(v::Vector{Any}) = v[1] + 1
end
module FullyTypedWarn
Base.Experimental.@compiler_options fully_typed=warn
untyped(v::Vector{Any}) = v[1] + 1
end
module FullyTypedOff
untyped(v::Vector{Any}) = v[1] + 1
end
@eval FullyTypedOff bymeta(v::Vector{Any}) = $(Expr(:block, Expr(:meta, :fully_typed_enforce), :(v[1] + 1)))

@testset "fully typed mode" begin
    policy(f, T) = (Base.method_instance(f, T).cache.flags & CI_FLAG_FACTS) != 0 ? :facts : :plain
    E, W, O = FullyTypedEnforce, FullyTypedWarn, FullyTypedOff
    @test ccall(:jl_get_module_options, Cint, (Any,), E) == 32
    @test ccall(:jl_get_module_options, Cint, (Any,), W) == 16
    @test ccall(:jl_method_fully_typed, Cint, (Any,), which(E.typed, (Int,))) == 2
    @test ccall(:jl_method_fully_typed, Cint, (Any,), which(W.untyped, (Vector{Any},))) == 1
    @test ccall(:jl_method_fully_typed, Cint, (Any,), which(O.untyped, (Vector{Any},))) == 0
    @test ccall(:jl_method_fully_typed, Cint, (Any,), which(O.bymeta, (Vector{Any},))) == 2
    @test ccall(:jl_method_policy, Cint, (Any,), which(E.typed, (Int,))) == 1   # facts

    # no hook: fully typed code runs; code with a run-time site raises under enforce, runs under warn
    @test E.typed(1) == 2
    @test policy(E.typed, (Int,)) === :facts
    @test_throws "not fully typed" E.untyped(Any[1])
    @test W.untyped(Any[1]) == 2
    @test O.untyped(Any[1]) == 2
    @test policy(O.untyped, (Vector{Any},)) === :plain
    @test_throws "not fully typed" O.bymeta(Any[1])

    # a hook gets the site (the instance, the level, warned, the kind): under warn once per site, under
    # enforce at every run
    seen = Any[]
    hook(site) = (push!(seen, site); site[2] == 2 && error("hook: not fully typed"); nothing)
    ccall(:jl_set_fully_typed_hook, Cvoid, (Any,), hook)
    try
        @eval FullyTypedWarn untyped2(v::Vector{Any}) = v[1] * 2
        @test W.untyped2(Any[3]) == 6
        @test W.untyped2(Any[4]) == 8
        @test length(seen) == 1
        @test seen[1][1] === Base.method_instance(W.untyped2, (Vector{Any},))
        @test seen[1][2] == 1 && seen[1][4] == 1
        @test_throws "hook: not fully typed" E.untyped(Any[1])
        @test_throws "hook: not fully typed" E.untyped(Any[1])
        @test length(seen) == 3
        @test E.typed(5) == 6
        @test length(seen) == 3
    finally
        ccall(:jl_set_fully_typed_hook, Cvoid, (Any,), nothing)
    end
end

equiv_f(x::Int) = x + 12

@testset "a compile request takes the code of a compiled equivalent entry" begin
    @test equiv_f(1) == 13
    mi = Base.method_instance(equiv_f, (Int,))
    world = Base.get_world_counter()
    compiled = only(filter(ci -> ci.invoke != C_NULL, native_cis(equiv_f, (Int,))))
    entry() = Core.CodeInstance(mi, nothing, compiled.rettype, compiled.exctype, nothing, compiled.inferred,
        Int32(0), compiled.min_world, compiled.max_world, compiled.ipo_purity_bits, nothing,
        compiled.debuginfo, compiled.edges)
    # an equivalent entry without code at the head of the instance's cache, the compiled one after it
    # (as where an entry further down the chain was compiled after the first was inserted)
    uncompiled = entry()
    @atomic uncompiled.next = mi.cache
    @atomic mi.cache = uncompiled
    @test mi.cache === uncompiled
    request = entry()
    # the first entry by its ABI is the one without code; the request takes the compiled one
    @test ccall(:jl_get_ci_equiv, Any, (Any, UInt), request, world) === uncompiled
    @test CC.equivalent_cached_ci(request, world) === compiled
    # no equivalent entry: the request itself
    other = Core.CodeInstance(mi, nothing, Any, compiled.exctype, nothing, compiled.inferred,
        Int32(0), compiled.min_world, compiled.max_world, compiled.ipo_purity_bits, nothing,
        compiled.debuginfo, compiled.edges)
    @test CC.equivalent_cached_ci(other, world) === other
    @test equiv_f(1) == 13
end
