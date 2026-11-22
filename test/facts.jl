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
