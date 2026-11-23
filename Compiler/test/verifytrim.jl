# This file is a part of Julia. License is MIT: https://julialang.org/license

include("setup_Compiler.jl")

# revise: Core.include(Compiler.TrimVerifier, joinpath(@__DIR__, "../src/verifytrim.jl"))

using Test
using .Compiler: typeinf_ext_toplevel, TrimVerifier, TRIM_SAFE, TRIM_UNSAFE
using .TrimVerifier: get_verify_typeinf_trim, verify_print_error, CallMissing, CCallableMissing

function sprint(f, args...)
    return Base.sprint((io, f, args...) -> f(IOContext{IO}(io), args...), f, args...)
end

let infos = Any[]
    errors, parents = get_verify_typeinf_trim(infos)
    @test isempty(errors)
    @test isempty(parents)
end

finalizer(@nospecialize(f), @nospecialize(o)) = Core.finalizer(f, o)

let infos = typeinf_ext_toplevel(Any[Core.svec(Nothing, Tuple{typeof(finalizer), typeof(identity), Any})], [Base.get_world_counter()], TRIM_UNSAFE, false)
    errors, parents = get_verify_typeinf_trim(infos)
    @test !isempty(errors) # unresolvable finalizer

    # the only error should be a CallMissing error for the Core.finalizer builtin
    (warn, desc) = only(errors)
    @test !warn
    @test desc isa CallMissing
    @test occursin("finalizer", desc.desc)
    repr = sprint(verify_print_error, desc, parents, warn)
    # New format uses multiline for unstable types
    @test occursin(r"^unresolved finalizer registered from statement (Core\.)?finalizer\("s, repr)
    @test occursin(r"f::Any"s, repr)
    @test occursin(r"o::Any"s, repr)
    @test occursin(r"::Nothing\n\nStacktrace:"s, repr)
    @test occursin(r"\[1\] finalizer\(f::Any, o::Any\)"s, repr)
end

# test that basic `cfunction` generation is allowed, when the dispatch target can be resolved
make_cfunction() = @cfunction(+, Float64, (Int64,Int64))
let infos = typeinf_ext_toplevel(Any[Core.svec(Ptr{Cvoid}, Tuple{typeof(make_cfunction)})], [Base.get_world_counter()], TRIM_UNSAFE, false)
    errors, parents = get_verify_typeinf_trim(infos)
    @test isempty(errors)
end

# use TRIM_UNSAFE to bypass verifier inside typeinf_ext_toplevel
make_cfunction_bad(@nospecialize(f::Any)) = @cfunction($f, Float64, (Int64,Int64))::Base.CFunction
let infos = typeinf_ext_toplevel(Any[Core.svec(Base.CFunction, Tuple{typeof(make_cfunction_bad), Any})], [Base.get_world_counter()], TRIM_UNSAFE, false)
    errors, parents = get_verify_typeinf_trim(infos)
    @test !isempty(errors) # missing cfunction

    (is_warning, desc) = only(errors)
    @test !is_warning
    @test desc isa CallMissing
    @test occursin("cfunction", desc.desc)
    repr = sprint(verify_print_error, desc, parents, is_warning)
    @test occursin(r"^unresolved cfunction from statement"s, repr)
    @test occursin(r"::Base.CFunction\n\nStacktrace:"s, repr)
    @test occursin(r"\[1\] make_cfunction_bad\(f::Any\)"s, repr)
    resize!(infos, 1)
    @test infos[1] isa Core.SimpleVector && infos[1][1] isa Type && infos[1][2] isa Type
    errors, parents = get_verify_typeinf_trim(infos)
    (warn, desc) = only(errors)
    @test !warn
    @test desc isa CCallableMissing
    @test desc.rt == Base.CFunction
    @test desc.sig == Tuple{typeof(make_cfunction_bad), Any}
    @test occursin("unresolved ccallable", desc.desc)
    repr = sprint(verify_print_error, desc, parents, warn)
    @test repr == "unresolved ccallable for Tuple{$(typeof(make_cfunction_bad)), Any} => Base.CFunction\n\n"
end

let infos = typeinf_ext_toplevel(Any[Core.svec(Base.SecretBuffer, Tuple{Type{Base.SecretBuffer}})], [Base.get_world_counter()], TRIM_UNSAFE, false)
    @test length(infos) > 4
    errors, parents = get_verify_typeinf_trim(infos)
    @test isempty(errors)

    resize!(infos, 1)
    @test infos[1] isa Core.SimpleVector && infos[1][1] isa Type && infos[1][2] isa Type
    errors, parents = get_verify_typeinf_trim(infos)
    (warn, desc) = only(errors)
    @test !warn
    @test desc isa CCallableMissing
    @test desc.rt == Base.SecretBuffer
    @test desc.sig == Tuple{Type{Base.SecretBuffer}}
    @test occursin("unresolved ccallable", desc.desc)
    repr = sprint(verify_print_error, desc, parents, warn)
    @test repr == "unresolved ccallable for Tuple{Type{Base.SecretBuffer}} => Base.SecretBuffer\n\n"
end

let infos = typeinf_ext_toplevel(Any[Core.svec(Float64, Tuple{typeof(+), Int32, Int64})], [Base.get_world_counter()], TRIM_UNSAFE, false)
    errors, parents = get_verify_typeinf_trim(infos)
    (warn, desc) = only(errors)
    @test !warn
    @test desc isa CCallableMissing
    @test desc.rt == Int64
    @test desc.sig == Tuple{typeof(+), Int32, Int64}
    @test occursin("ccallable declared return type", desc.desc)
    repr = sprint(verify_print_error, desc, parents, warn)
    @test repr == "ccallable declared return type does not match inference for Tuple{typeof(+), Int32, Int64} => Int64\n\n"
end

let infos = typeinf_ext_toplevel(Any[Core.svec(Int64, Tuple{typeof(ifelse), Bool, Int64, UInt64})], [Base.get_world_counter()], TRIM_UNSAFE, false)
    errors, parents = get_verify_typeinf_trim(infos)
    (warn, desc) = only(errors)
    @test warn  # this is a warning since Union{Int64, UInt64} <: Int64 is false but not an error
    @test desc isa CCallableMissing
    @test occursin("ccallable declared return type", desc.desc)
    repr = sprint(verify_print_error, desc, parents, warn)
    @test repr == "ccallable declared return type does not match inference for Tuple{typeof(ifelse), Bool, Int64, UInt64} => Union{Int64, UInt64}\n\n"
end

let infos = typeinf_ext_toplevel(Any[Core.svec(Union{Int64,UInt64}, Tuple{typeof(ifelse), Bool, Int64, UInt64})], [Base.get_world_counter()], TRIM_SAFE, false)
    errors, parents = get_verify_typeinf_trim(infos)
    @test isempty(errors)
    infos = typeinf_ext_toplevel(Any[Core.svec(Real, Tuple{typeof(ifelse), Bool, Int64, UInt64})], [Base.get_world_counter()], TRIM_SAFE, false)
    errors, parents = get_verify_typeinf_trim(infos)
    @test isempty(errors)
end


mi = Base.method_instance(sum, (Vector{Union{Int64,Float64, Float32,UInt32}},))
let infos = typeinf_ext_toplevel(Any[mi], [Base.get_world_counter()], TRIM_UNSAFE, false)
    errors, parents = get_verify_typeinf_trim(infos)
    @test !isempty(errors)
end

# an invoke whose code the build did not compile is resolved where the build compiles the instance's code of
# the policy the call through `jl_invoke` asks
@noinline vt_f(x::Int) = x + 1
vt_g(x::Int) = 2 * vt_f(x)
@test vt_g(1) == 4
let w = Base.get_world_counter(), interp = Compiler.NativeInterpreter(w)
    mi_g = Compiler.specialize_method(only(Base._methods_by_ftype(Tuple{typeof(vt_g), Int}, 1, w)))
    ci_g = Compiler.typeinf_ext(interp, mi_g, Compiler.SOURCE_MODE_GET_SOURCE)
    src_g = interp.codegen[ci_g]
    ci_f = only(st for st in src_g.code if Meta.isexpr(st, :invoke)).args[1]::Core.CodeInstance
    mi_f = ci_f.def
    src_f = something(get(interp.codegen, ci_f, nothing),
                      interp.codegen[Compiler.typeinf_ext(interp, mi_f, Compiler.SOURCE_MODE_GET_SOURCE)])
    function other(facts::Bool)          # another code instance of `vt_f`, of the policy given
        c = Core.CodeInstance(mi_f, nothing, ci_f.rettype, ci_f.exctype, nothing, nothing, Int32(0), ci_f.min_world,
                              ci_f.max_world, UInt32(0), nothing, nothing, Core.svec())
        fl = getfield(c, :flags, :monotonic)
        setfield!(c, :flags, facts ? (fl | 0x20) : (fl & ~0x20), :monotonic)
        return c
    end
    refused(list) = [e for (warn, e) in first(get_verify_typeinf_trim(list, nothing, w)) if !warn && e isa CallMissing]
    asked = Compiler.mi_policy_facts_at(mi_f, Compiler.ci_policy_facts(ci_f))
    @test isempty(refused(Any[ci_g, src_g, ci_f, src_f]))
    @test isempty(refused(Any[ci_g, src_g, other(asked), src_f]))
    r = refused(Any[ci_g, src_g, other(!asked), src_f])
    @test length(r) == 1 && occursin("policy", r[1].desc)
    @test length(refused(Any[ci_g, src_g])) == 1
end
