# This file is a part of Julia. License is MIT: https://julialang.org/license

# make sure that typeinf is executed before turning on typeinf_ext
# this ensures that typeinf_ext doesn't recurse before it can add the item to the workq
# especially try to make sure any recursive and leaf functions have concrete signatures,
# since we won't be able to specialize & infer them at runtime

function activate_codegen!()
    ccall(:jl_set_typeinf_func, Cvoid, (Any,), typeinf_ext_toplevel)
    # Register the new unified compile and emit function
    ccall(:jl_set_compile_and_emit_func, Cvoid, (Any,), compile_and_emit_native)
    Core.eval(Compiler, quote
        let typeinf_world_age = Base.tls_world_age()
            @eval Core.OptimizedGenerics.CompilerPlugins.typeinf(::Nothing, mi::MethodInstance, source_mode::UInt8) =
                Base.invoke_in_world($(Expr(:$, :typeinf_world_age)), typeinf_ext_toplevel, mi, Base.tls_world_age(), source_mode, Compiler.TRIM_NO)
        end
    end)
end

global bootstrapping_compiler::Bool = false
function bootstrap!()
    global bootstrapping_compiler = true
    let time() = ccall(:jl_clock_now, Float64, ())
        println("Compiling the compiler. This may take several minutes ...")

        ssa_inlining_pass!_tt = Tuple{typeof(ssa_inlining_pass!), IRCode, InliningState{NativeInterpreter}, Bool}
        optimize_tt = Tuple{typeof(optimize), NativeInterpreter, OptimizationState{NativeInterpreter}, InferenceResult}
        typeinf_ext_tt = Tuple{typeof(typeinf_ext), NativeInterpreter, MethodInstance, UInt8}
        typeinf_tt = Tuple{typeof(typeinf), NativeInterpreter, InferenceState}
        typeinf_edge_tt = Tuple{typeof(typeinf_edge), NativeInterpreter, Method, Any, SimpleVector, InferenceState, Bool, Bool}
        # the entry of `return_type` at run time (`collect` of a generator, `map` over an empty
        # collection): without it, its first call in every session compiled it (80-90 ms)
        return_type_tt = Tuple{typeof(_return_type), NativeInterpreter, DataType}
        # `return_type`'s own interpreter (facts.jl): the run-time entry in both settings, the
        # inference behind a fold of `return_type` in code compiled by facts, and what inference
        # reaches by dynamic dispatch for it. Without them, the first run-time `return_type` of a
        # session compiled the compiler for it (about 5 s)
        plain_rt_tt = Tuple{typeof(_return_type), PlainRTInterp, DataType}
        plain_rt_ext_tt = Tuple{typeof(typeinf_ext_toplevel), PlainRTInterp, MethodInstance, UInt8}
        plain_rt_dyn = Any[
            Tuple{typeof(optimize), PlainRTInterp, OptimizationState{PlainRTInterp}, InferenceResult},
            Tuple{typeof(ssa_inlining_pass!), IRCode, InliningState{PlainRTInterp}, Bool},
            Tuple{typeof(typeinf), PlainRTInterp, InferenceState},
            Tuple{typeof(typeinf_edge), PlainRTInterp, Method, Any, SimpleVector, InferenceState, Bool, Bool},
            Tuple{typeof(typeinf_local), PlainRTInterp, InferenceState, CurrentState},
            Tuple{typeof(doworkloop), PlainRTInterp, InferenceState},
            Tuple{typeof(finishinfer!), InferenceState, PlainRTInterp, Int, IdDict{MethodInstance, CodeInstance}},
            Tuple{typeof(finish!), PlainRTInterp, InferenceState, UInt64, UInt64},
            Tuple{typeof(finish_nocycle), PlainRTInterp, InferenceState, UInt64},
            Tuple{typeof(is_already_cached), PlainRTInterp, InferenceResult, CodeInstance}]
        fs = Any[
            # we first create caches for the optimizer, because they contain many loop constructions
            # and they're better to not run in interpreter even during bootstrapping
            compact!, ssa_inlining_pass!_tt, optimize_tt,
            # then we create caches for inference entries
            typeinf_ext_tt, typeinf_tt, typeinf_edge_tt, return_type_tt, plain_rt_tt, plain_rt_ext_tt,
            plain_rt_dyn...,
        ]
        # tfuncs can't be inferred from the inference entries above, so here we infer them manually
        for x in T_FFUNC_VAL
            push!(fs, x[3])
        end
        for i = 1:length(T_IFUNC)
            if isassigned(T_IFUNC, i)
                x = T_IFUNC[i]
                push!(fs, x[3])
            else
                println(stderr, "WARNING: tfunc missing for ", reinterpret(IntrinsicFunction, Int32(i)))
            end
        end
        starttime = time()
        world = get_world_counter()
        for f in fs
            if isa(f, DataType) && f.name === typename(Tuple)
                tt = f
            else
                tt = Tuple{typeof(f), Vararg{Any}}
            end
            matches = _methods_by_ftype(tt, 10, world)::Vector
            if isempty(matches)
                println(stderr, "WARNING: no matching method found for `", tt, "`")
            else
                for m in matches
                    # remove any TypeVars from the intersection
                    m = m::MethodMatch
                    params = Any[m.spec_types.parameters...]
                    for i = 1:length(params)
                        params[i] = unwraptv(params[i])
                    end
                    mi = specialize_method(m.method, Tuple{params...}, m.sparams)
                    #isa_compileable_sig(mi) || println(stderr, "WARNING: inferring `", mi, "` which isn't expected to be called.")
                    typeinf_ext_toplevel(mi, world, isa_compileable_sig(mi) ? SOURCE_MODE_ABI : SOURCE_MODE_NOT_REQUIRED, TRIM_NO)
                end
            end
        end
        endtime = time()
        println("Base.Compiler ──── ", sub_float(endtime,starttime), " seconds")
    end
    activate_codegen!()
    global bootstrapping_compiler = false
    nothing
end

function activate!(; reflection=true, codegen=false)
    if reflection
        Base.REFLECTION_COMPILER[] = Compiler
    end
    if codegen
        bootstrap!()
    end
end
