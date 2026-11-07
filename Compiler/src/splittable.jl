# This file is a part of Julia. License is MIT: https://julialang.org/license

# The table of a call's targets (compiler §7.3): a value of a type outside an operation's split
# reaches the split's last branch, where the inliner emits a lookup of the value's type in the
# table of the call's signature (`jl_split_table_ptr`, src/gf.c) and a typed call through the
# pointer it returns: a `:caller_world` pointer to the call's target compiled for that type, made
# the first time the type is met (`Base._split_table_miss!`). The pointer runs in the caller's
# world, as the dynamic call does, so the table needs no invalidation. A table saved in a package
# image belongs to another session and counts as empty. Not in a trimmed program.

"The most arguments after the split value a split table's call takes (the targets' fixed arities, `Base._SplitTarget`)."
const MAX_SPLIT_TABLE_ARGS = 6

"""
A split table's state: pairs (type address, pointer) by open addressing in `kv` (0 is empty), the
session that filled it, and the types (rooted). Replaced on each addition, never mutated.
"""
mutable struct SplitState
    const kv::Memory{UInt}
    const session::UInt64
    const keys::Memory{Any}
end
"""
The table of one call signature at a split's last branch: the called function's type `f`, the other
arguments' types at the site `args` (a tuple type), the declared result type `rt`. The pointer passes
and returns values as `split_ccall_type` says. The runtime reads its first field (`jl_split_table_ptr`).
"""
mutable struct SplitTable
    @atomic state::SplitState
    const f::Any
    const args::Any
    const rt::Any
end
function SplitTable(@nospecialize(f), @nospecialize(args), @nospecialize(rt))
    kv = Memory{UInt}(undef, 2)
    kv[1] = kv[2] = UInt(0)
    return SplitTable(SplitState(kv, UInt64(0), Memory{Any}(undef, 0)), f, args, rt)
end

mutable struct SplitTables
    @atomic table::Any          # an IdDict{Any,SplitTable}, replaced on each addition, never mutated
end
const SPLIT_TABLES = SplitTables(nothing)

"The table of a call signature, made the first time it is asked for: one per signature and session."
function split_table(@nospecialize(f), @nospecialize(args), @nospecialize(rt))
    key = (f, args, rt)
    while true
        old = @atomic SPLIT_TABLES.table
        if old !== nothing
            t = get(old::IdDict{Any,SplitTable}, key, nothing)
            t === nothing || return t::SplitTable
        end
        new = old === nothing ? IdDict{Any,SplitTable}() : copy(old::IdDict{Any,SplitTable})
        t = SplitTable(f, args, rt)
        new[key] = t
        (@atomicreplace SPLIT_TABLES.table old => new).success && return t
    end
end

"The state `old` with type `T`'s pointer `fp` added, in session `s` (a state of another session is dropped)."
function split_state_add(old::SplitState, s::UInt64, @nospecialize(T), fp::Ptr{Cvoid})
    a = reinterpret(UInt, pointer_from_objref(T))
    same = old.session === s
    oldkeys = same ? old.keys : Memory{Any}(undef, 0)
    n = 1
    for k in oldkeys
        k === T || (n += 1)
    end
    cap = 4
    while cap < 2n
        cap *= 2
    end
    kv = Memory{UInt}(undef, 2cap)
    for i in 1:2cap
        kv[i] = UInt(0)
    end
    keys = Memory{Any}(undef, n)
    j = 0
    if same
        okv = old.kv
        for i in 0:(length(okv) ÷ 2 - 1)
            k = okv[2i + 1]
            (k == UInt(0) || k == a) && continue
            split_kv_insert!(kv, k, okv[2i + 2])
        end
        for k in oldkeys
            k === T && continue
            j += 1
            keys[j] = k
        end
    end
    split_kv_insert!(kv, a, reinterpret(UInt, fp))
    keys[j + 1] = T
    return SplitState(kv, s, keys)
end
function split_kv_insert!(kv::Memory{UInt}, k::UInt, v::UInt)
    mask = length(kv) ÷ 2 - 1
    i = Int((k >> 4) & UInt(mask))
    while kv[2i + 1] != UInt(0)
        i = (i + 1) & mask
    end
    kv[2i + 1] = k
    kv[2i + 2] = v
    return nothing
end

"The type a split table's key records for an argument at the site: its concrete type, or `Any`."
split_table_type(@nospecialize(T)) = (isconcretetype(T) && !iskindtype(T)) ? T : Any
"""
How a split table's pointer passes a value of type `T` (the C ABI of `ccall` and `cfunction`): by
value where `T` is an isbits type of nonzero size, boxed (`Any`) otherwise, so that a mutable
object keeps its identity and a struct with references is never copied field by field.
"""
split_ccall_type(@nospecialize(T)) = (T isa DataType && isbitstype(T) && sizeof(T) > 0) ? T : Any
"How a split table's pointer returns a result of type `R`: as `split_ccall_type`, and `Nothing` as no value."
split_ccall_rt(@nospecialize(R)) = R === Nothing ? Nothing : split_ccall_type(R)
