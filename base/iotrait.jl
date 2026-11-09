# This file is a part of Julia. License is MIT: https://julialang.org/license

# `IO`'s contract (traits §9b; compiler §7.1, §7.5). `IO` is Julia's closed abstract type, and a
# method typed `io::IO` is compiled per stream type. At a value typed `IO` (a field, an element, an
# argument declared `@nospecialize`), an operation's call branches over the candidates.

# The operations, each Base's method at `::IO`. A closed type's members are Julia's streams, which no
# check reaches, so no result can be promised for them: each operation's result is `Any`, Julia's own
# answer at such a receiver, and the split neither converts nor asserts (open issue 171).
let ops = Any[]
    for (f, sig) in Any[
            (write,        Tuple{IO, UInt8}),
            (unsafe_write, Tuple{IO, Ptr{UInt8}, UInt}),
            (read,         Tuple{IO, Type{UInt8}}),
            (unsafe_read,  Tuple{IO, Ptr{UInt8}, UInt}),
            (flush,        Tuple{IO}),
            (lock,         Tuple{IO}),
            (unlock,       Tuple{IO}),
            (get,          Tuple{IO, Any, Any}),
            (haskey,       Tuple{IO, Any})]
        m = which(f, sig)
        m.sig === Tuple{typeof(f), sig.parameters...} || error("IO's operation $f: Base's method at ::IO is $m")
        ccall(:jl_method_set_declared_result, Cvoid, (Any, Any, Cint), m, Any, 1)
        push!(ops, m)
    end
    # the split's candidates (compiler §7.2), in this order: at a value typed `IO`, an operation
    # branches over these with direct calls, and goes through the table past them (§7.3)
    ccall(:jl_declare_contract, Cvoid, (Any, Any, Any), IO, Core.svec(ops...),
          Core.svec(IOBuffer, IOStream, TTY, PipeEndpoint))
end
