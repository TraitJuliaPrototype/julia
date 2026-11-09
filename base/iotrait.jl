# This file is a part of Julia. License is MIT: https://julialang.org/license

# `IO`, Base's trait of streams (traits §9b; compiler §7.5). `Core.IO` is a set from Core's build
# (boot.jl): a method typed `io::IO` is compiled once, whatever the stream, Julia's own included;
# `IO` is closed after load.

# a method that can only raise at a type that joins `IO` later is no shadow (compiler §5.6)
macro _serves_no_member()
    return Expr(:meta, :serves_no_member)
end

# The operations' defaults where Base has none: each raises the `MethodError` the call would.
eof(io::IO) = (@_serves_no_member; throw(MethodError(eof, (io,))))
isopen(io::IO) = (@_serves_no_member; throw(MethodError(isopen, (io,))))
close(io::IO) = (@_serves_no_member; throw(MethodError(close, (io,))))

# The candidates of the split (compiler §7.2, §7.5), registered before any other member: in code
# compiled once, an operation on the stream branches over these with direct calls.
for T in (IOBuffer, IOStream, TTY, PipeEndpoint)
    ccall(:jl_register_candidate, Cvoid, (Any, Any), IO, T)
end

# The operations (traits §9b), each Base's method at `::IO` with its declared result: inference
# reads it at a receiver typed `IO`, and the split asserts it on its branches and its fallback.
for (f, sig, rt) in Any[
        (write,        Tuple{IO, UInt8},              Integer),
        (unsafe_write, Tuple{IO, Ptr{UInt8}, UInt},   Integer),
        (read,         Tuple{IO, Type{UInt8}},        UInt8),
        (unsafe_read,  Tuple{IO, Ptr{UInt8}, UInt},   Any),
        (eof,          Tuple{IO},                     Bool),
        (isopen,       Tuple{IO},                     Bool),
        (close,        Tuple{IO},                     Any),
        (flush,        Tuple{IO},                     Any),
        (lock,         Tuple{IO},                     Any),
        (unlock,       Tuple{IO},                     Any),
        (get,          Tuple{IO, Any, Any},           Any),
        (haskey,       Tuple{IO, Any},                Bool)]
    m = which(f, sig)
    m.sig === Tuple{typeof(f), sig.parameters...} || error("IO's operation $f: Base's method at ::IO is $m")
    ccall(:jl_method_set_declared_result, Cvoid, (Any, Any, Cint), m, rt, 1)
end

# `Dynamic{IO}(io)` is an `IO` (traits §9b): a member by Base's entry, forwarding by its object, so
# plain code typed `io::IO` takes it and an `IOContext`'s properties reach `show` through it.
let Z = TypeVar(:Z, IO)
    ccall(:jl_declare_membership, Any, (Any, Any, Any, Any, Any),
          UnionAll(Z, Core.Dynamic{Z}), IO, Base, :object, :field)
end
@Core.latestworld
_dynamic_io(d::Core.Dynamic) = getfield(d, :object)::IO
write(d::Core.Dynamic{<:IO}, x::UInt8) = write(_dynamic_io(d), x)
unsafe_write(d::Core.Dynamic{<:IO}, p::Ptr{UInt8}, n::UInt) = unsafe_write(_dynamic_io(d), p, n)
read(d::Core.Dynamic{<:IO}, ::Type{UInt8}) = read(_dynamic_io(d), UInt8)
unsafe_read(d::Core.Dynamic{<:IO}, p::Ptr{UInt8}, n::UInt) = unsafe_read(_dynamic_io(d), p, n)
eof(d::Core.Dynamic{<:IO}) = eof(_dynamic_io(d))
isopen(d::Core.Dynamic{<:IO}) = isopen(_dynamic_io(d))
close(d::Core.Dynamic{<:IO}) = close(_dynamic_io(d))
flush(d::Core.Dynamic{<:IO}) = flush(_dynamic_io(d))
lock(d::Core.Dynamic{<:IO}) = lock(_dynamic_io(d))
unlock(d::Core.Dynamic{<:IO}) = unlock(_dynamic_io(d))
get(d::Core.Dynamic{<:IO}, key, default) = get(_dynamic_io(d), key, default)
haskey(d::Core.Dynamic{<:IO}, key) = haskey(_dynamic_io(d), key)
