# This file is a part of Julia. License is MIT: https://julialang.org/license

# The targets of the compiler's split tables (compiler §7.3; Compiler/src/splittable.jl, the
# runtime's `jl_split_table_ptr` in src/gf.c): a value of a type outside a split's candidates is
# called through a `:caller_world` pointer to the call's target compiled for its type, made here the
# first time the type is met at the call's signature.

"""
The target of a split table's pointer: `f(x::T, args...)::R`, with `f` and `x` passed boxed and the
other arguments as `Compiler.split_ccall_type` says; asserted to their types at the site (`A`), so
the call inside is static.
"""
struct _SplitTarget{F,T,R,A} end
for n in 0:Compiler.MAX_SPLIT_TABLE_ARGS
    local as = Any[Symbol(:a, k) for k in 1:n]
    local typed = Any[:($(as[k])::fieldtype(A, $k)) for k in 1:n]
    @eval (::_SplitTarget{F,T,R,A})(@nospecialize(f), @nospecialize(x), $(as...)) where {F,T,R,A} = (f::F)(x::T, $(typed...))::R
end

"The `:caller_world` pointer to the target for type `T` of a split table's call (`f` of type `F`, arguments `A` at the site, result `R`)."
@generated function _split_pointer(::Type{F}, ::Type{T}, ::Type{R}, ::Type{A}) where {F,T,R,A}
    target = _SplitTarget{F,T,R,A}()
    cats = Any[Compiler.split_ccall_type(a) for a in A.parameters]
    return Expr(:cfunction, Ptr{Cvoid}, QuoteNode(target), Compiler.split_ccall_rt(R),
                Expr(:call, GlobalRef(Core, :svec), Any, Any, cats...), QuoteNode(:caller_world))
end

"""
A split table's miss (`jl_split_table_ptr`): the pointer for type `T`, made and added to the table.
The pointer is made outside the update; the table's state is replaced, never mutated.
"""
function _split_table_miss!(tbl::Compiler.SplitTable, @nospecialize(T))
    # a session without the compiler makes no pointer: the call dispatches when it runs (compiler §7.3)
    ccall(:jl_get_LLVM_VERSION, UInt32, ()) == 0 && return C_NULL
    fp = _split_pointer(tbl.f, T, tbl.rt, tbl.args)::Ptr{Cvoid}
    s = ccall(:jl_split_session, UInt64, ())
    while true
        old = @atomic :acquire tbl.state
        new = Compiler.split_state_add(old, s, T, fp)
        (@atomicreplace :acquire_release :monotonic tbl.state old => new).success && return fp
    end
end
