# This file is a part of Julia. License is MIT: https://julialang.org/license

# The target of a split's pointers (compiler §7.2, §7.3; traits §7.9). A pointer dispatches to the target in its caller's world, so they are defined before
# any code that can split: a generated function's generator runs in the world it was defined in, and
# a caller compiled later serves that world. The table's miss (base/splittable.jl) comes later; it
# runs in the latest world.

const _SPLIT_TARGET_ARGS = (:a1, :a2, :a3, :a4, :a5, :a6)   # without `Symbol(:a, k)`: no printing yet

"""
The target of a split table's pointer: `f(x::T, args...)`, its result asserted to the declared
result `R` (traits §7.9), with `f` and `x` passed boxed and the other arguments as
`Compiler.split_ccall_type` says; asserted to their types at the site (`A`), so the call inside is
static.
"""
struct _SplitTarget{F,T,R,A} end
for n in 0:Compiler.MAX_SPLIT_TABLE_ARGS
    local as = Any[_SPLIT_TARGET_ARGS[k] for k in 1:n]
    local typed = Any[:($(as[k])::fieldtype(A, $k)) for k in 1:n]
    @eval (::_SplitTarget{F,T,R,A})(@nospecialize(f), @nospecialize(x), $(as...)) where {F,T,R,A} =
        ((f::F)(x::T, $(typed...)))::R
end

