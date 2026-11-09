# This file is a part of Julia. License is MIT: https://julialang.org/license

# `IO`'s contract (traits §9b; compiler §7.1, §7.5). `IO` is Julia's closed abstract type, and a
# method typed `io::IO` is compiled per stream type. The contract's operations carry their declared
# results. So at a value typed `IO` (a field, an element, an argument declared `@nospecialize`), an
# operation's call has its declared result and branches over the candidates.

# The operations, each Base's method at `::IO` with its declared result. A write's count is an
# `Int`, as `IOBuffer` and `IOStream` return it. A call through `IO` converts another stream's count
# (Core's streams and `Filesystem.File` return a `UInt`), as `f(…)::Int` would (traits §7.9).
let ops = Any[]
    for (f, sig, rt) in Any[
            (write,        Tuple{IO, UInt8},              Int),
            (unsafe_write, Tuple{IO, Ptr{UInt8}, UInt},   Int),
            (read,         Tuple{IO, Type{UInt8}},        UInt8),
            (unsafe_read,  Tuple{IO, Ptr{UInt8}, UInt},   Any),
            (flush,        Tuple{IO},                     Any),
            (lock,         Tuple{IO},                     Any),
            (unlock,       Tuple{IO},                     Any),
            (get,          Tuple{IO, Any, Any},           Any),
            (haskey,       Tuple{IO, Any},                Bool)]
        m = which(f, sig)
        m.sig === Tuple{typeof(f), sig.parameters...} || error("IO's operation $f: Base's method at ::IO is $m")
        ccall(:jl_method_set_declared_result, Cvoid, (Any, Any, Cint), m, rt, 1)
        push!(ops, m)
    end
    ccall(:jl_declare_contract, Cvoid, (Any, Any), IO, Core.svec(ops...))
end

# The candidates of the split (compiler §7.2), in this order: at a value typed `IO`, an operation
# branches over these with direct calls, and goes through the table past them (§7.3).
for T in (IOBuffer, IOStream, TTY, PipeEndpoint)
    ccall(:jl_register_candidate, Cvoid, (Any, Any), IO, T)
end
