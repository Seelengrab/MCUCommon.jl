module MCUCommon

export Register, RegisterBits, Pin, volatile_load, volatile_store!,
    keep, delay_ms, delay_us, delay, CPU_FREQUENCY_HZ, datapin,
    dataregister, @pindefs

@enum Access Unknown=0 Read Write ReadWrite ReadWriteOnce

"""
    Register{Reg, T <: Base.BitInteger}

Represents a register named `Reg` (a `Symbol`), holding data in the size of `T`.

!!! warn "Internal fields"
    Internally, this is represented as a `Ptr{T}`, pointing to the memory mapped register at the given address.
    This is an implementation detail and may change at any time.
"""
struct Register{Reg, T <: Base.BitInteger}
    ptr::Ptr{T}
    Register{Reg, T}(x::Ptr) where {Reg, T} = new{Reg, T}(convert(Ptr{T}, x))
    Register{Reg, T}(x::Base.BitInteger) where {Reg, T} = new{Reg, T}(Ptr{T}(x % UInt)) # Ptr only takes Union{Int, UInt, Ptr}...
end

"""
   RegisterBits{Reg, T}

Represents a bitmask and/or registerstate to be set/used. `Reg` and `T` designate the corresponding `Register{Reg, T}` these bits are for.

!!! note "Error paths"
    This type relies on constant propagation to eliminate error paths.

!!! warn "Internal fields"
    The bits stored in this object are considered internal. Accessing them directly to construct equivalently set bits for a different
    register is not supported.
"""
struct RegisterBits{Reg, T}
    bits::T
end

"""
    Pin{Reg, bit}

Represents the bit at position `bit` in the register `Reg`. `bit` is 1-based.
`Reg` is the actual `Register` object, not its type.

!!! note "Error paths"
    This type relies on constant propagation to eliminate error paths, as well as for compile-time elimination of the entire object.
    Instabilities in either type parameter will lead to runtime error paths being emitted.
"""
struct Pin{RT, Reg, Bit, Mask}
    # RT   : The actual type of `Reg`
    # Reg  : An actual register object, to allow setting just a single bit in isolation without keeping the register around in the source code
    # Bit  : The i-th bit in `Reg`, 1-based
    # Mask : The bitmask used on `Reg` to access this Pin. Constructed from `Bit`.
    function Pin{Reg, Bit}() where {Reg, Bit}
        Reg isa Register || throw(ArgumentError("`$Reg` is not a `Register` object!"))
        T = typeof(Reg)
        1 <= Bit <= 8*sizeof(eltype(T)) || throw(ArgumentError("Register `$Reg` only has `$(8*sizeof(eltype(T)))` bits, while bit `$Bit` was requested."))
        mask = eltype(Reg)(1 << (Bit - 1))
        new{T, Reg, Bit, mask}()
    end
end

struct Field{RT, Reg, Mode, Offset, Width, Mask}
    # RT     : The actual type of `Reg`
    # Reg    : An actual register object, to allow setting just a single bit in isolation without keeping the register around in the source code
    # Offset : The offset of the field in the register
    # Width  : The width of the field in the register
    function Field{Reg, Mode, Offset, Width}() where {Reg, Mode, Offset, Width}
        Reg isa Register || throw(ArgumentError("`$Reg is not a `Register` object!"))
        Offset isa Integer || throw(ArgumentError("`$Offset` is not an integer!"))
        Width isa Integer || throw(ArgumentError("`$Width` is not an integer!"))
        Mode isa Access || throw(ArgumentError("`$Mode` is not an access mode!"))
        T = typeof(Reg)
        1 <= (Offset + Width) <= 8*sizeof(eltype(T)) || throw(ArgumentError("Register `$Reg` only has `$(8*sizeof(eltype(T)))` bits, while Offset+Width = $Offset+$Width = $(Offset+Width) bits were requested."))
        mask = ~(~zero(eltype(Reg)) << Width) << Offset
        new{T, Reg, Mode, Offset, Width, mask}()
    end
end

Base.eltype(::Type{R}) where {Reg, T, R <: Register{Reg, T}} = T
Base.zero(::Register{R, T}) where {R, T} = RegisterBits{R, T}(zero(T))
Base.one(::Register{R, T}) where {R, T} = RegisterBits{R, T}(one(T))

Base.getindex(r::Register) = volatile_load(r)
Base.setindex!(r::Register{Reg, T}, data::T)                    where {Reg, T}          = volatile_store!(r, data)
Base.setindex!(r::Register{Reg, T},   rb::RegisterBits{Reg, T}) where {Reg, T}          = volatile_store!(r, rb.bits)
Base.setindex!(r::RT,                   ::Pin{RT, Reg, b, m})   where {RT, Reg, b, m}   = volatile_store!(r, m)

######
# logical operations with Pins & Registers
######

Base.:(|)(rba::RegisterBits{Reg, T}, rbb::RegisterBits{Reg, T}) where {Reg, T} = RegisterBits{Reg, T}(rba.bits | rbb.bits)
Base.:(&)(rba::RegisterBits{Reg, T}, rbb::RegisterBits{Reg, T}) where {Reg, T} = RegisterBits{Reg, T}(rba.bits & rbb.bits)
Base.:(⊻)(rba::RegisterBits{Reg, T}, rbb::RegisterBits{Reg, T}) where {Reg, T} = RegisterBits{Reg, T}(rba.bits ⊻ rbb.bits)
Base.:(|)(rbb::RegisterBits{Reg, T}, v::T)                      where {Reg, T} = RegisterBits{Reg, T}(v | rbb.bits)
Base.:(&)(rbb::RegisterBits{Reg, T}, v::T)                      where {Reg, T} = RegisterBits{Reg, T}(v & rbb.bits)
Base.:(⊻)(rbb::RegisterBits{Reg, T}, v::T)                      where {Reg, T} = RegisterBits{Reg, T}(v ⊻ rbb.bits)
Base.:(|)(v::T,                      rbb::RegisterBits{Reg, T}) where {Reg, T} = RegisterBits{Reg, T}(v | rbb.bits)
Base.:(&)(v::T,                      rbb::RegisterBits{Reg, T}) where {Reg, T} = RegisterBits{Reg, T}(v & rbb.bits)
Base.:(⊻)(v::T,                      rbb::RegisterBits{Reg, T}) where {Reg, T} = RegisterBits{Reg, T}(v ⊻ rbb.bits)
Base.:(~)(rb::RegisterBits{Reg, T})                             where {Reg, T} = RegisterBits{Reg, T}(~rb.bits)

# RT is the type of the register this Pin is for
# Reg is the register itself
# bx is the Bit of the pin (one based)
# mx is the bitmask to set/read that pin
Base.:(|)(::Pin{RT, Reg, ba, ma},  ::Pin{RT, Reg, bb, mb}) where {R, T, RT <: Register{R, T}, Reg, ba, bb, ma, mb} = RegisterBits{R, T}(ma | mb)
Base.:(&)(::Pin{RT, Reg, ba, ma},  ::Pin{RT, Reg, bb, mb}) where {R, T, RT <: Register{R, T}, Reg, ba, bb, ma, mb} = RegisterBits{R, T}(ma & mb)
Base.:(|)(::Pin{RT, Reg, bb, mb}, v::T)                    where {R, T, RT <: Register{R, T}, Reg, bb, mb}         = RegisterBits{R, T}(v | mb)
Base.:(&)(::Pin{RT, Reg, bb, mb}, v::T)                    where {R, T, RT <: Register{R, T}, Reg, bb, mb}         = RegisterBits{R, T}(v & mb)
Base.:(|)(v::T,                    ::Pin{RT, Reg, bb, mb}) where {R, T, RT <: Register{R, T}, Reg, bb, mb}         = RegisterBits{R, T}(v | mb)
Base.:(&)(v::T,                    ::Pin{RT, Reg, bb, mb}) where {R, T, RT <: Register{R, T}, Reg, bb, mb}         = RegisterBits{R, T}(v & mb)
Base.:(~)(::Pin{RT, Reg, ba, ma})                          where {R, T, RT <: Register{R, T}, Reg, ba, ma}         = RegisterBits{R, T}(~ma)

Base.:(|)(p::Pin,                 rb::RegisterBits)                                                  = rb | p
Base.:(&)(p::Pin,                 rb::RegisterBits)                                                  = rb & p
Base.:(|)(rb::RegisterBits{R, T},   ::Pin{RT, Reg, b, m}) where {R, T, RT<:Register{R,T}, Reg, b, m} = RegisterBits{R, T}(rb.bits | m)
Base.:(&)(rb::RegisterBits{R, T},   ::Pin{RT, Reg, b, m}) where {R, T, RT<:Register{R,T}, Reg, b, m} = RegisterBits{R, T}(rb.bits & m)

Base.getindex(_::Pin{RT, Reg, b, m}) where {RT, Reg, b, m} = (volatile_load(Reg) & m) != zero(m)
function Base.setindex!(_::Pin{Register{R, T}, Reg, b, m}, val::Bool) where {R, T, Reg, b, m}
    cur = volatile_load(Reg)

    res = ifelse(val, cur | m, cur & ~m)

    return volatile_store!(Reg, res)
end

# gracefullly stolen from VectorizationBase.jl
const LLVM_TYPES = IdDict{Type{<:Union{Bool,Base.HWReal,Float16}},String}(
  Float16 => "half",
  Float32 => "float",
  Float64 => "double",
  # Bit => "i1", # I don't think we have these here?
  Bool => "i8",
  Int8 => "i8",
  UInt8 => "i8",
  Int16 => "i16",
  UInt16 => "i16",
  Int32 => "i32",
  UInt32 => "i32",
  Int64 => "i64",
  UInt64 => "i64"
  # Int128 => "i128", # let's not worry about these
  # UInt128 => "i128",
  # UInt256 => "i256",
  # UInt512 => "i512",
  # UInt1024 => "i1024",
)

"""
    volatile_store!(r::Register{T}, v::T) -> Nothing
    volatile_store!(p::Ptr{T}, v::T) -> Nothing

Stores a value `T` in the register/pointer `r`/`p`. Unlike `unsafe_store!`, is not elided by LLVM.
"""
function volatile_store! end

"""
  volatile_load(r::Register{T}) -> T
  volatile_load(p::Ptr{T}) -> T

Loads a value of type `T` from the register/ptr `r`/`p`. Unlike `unsafe_load`, is not elided by LLVM.
"""
function volatile_load end

"""
    dataregister(::Register)

Gives the data direction register for the given register.
"""
function dataregister end

"""
   datapin(::Pin)

Gives the data direction pin for the given pin.
"""
datapin(::Pin{PT, Port, Bit}) where {PT, Port, Bit} = Pin{dataregister(Port), Bit}()

"""
    keep(x)

Forces LLVM to keep a value alive, by pretending to clobber its memory. The memory is never actually accessed,
which makes this a no-op in the final assembly.
"""
function keep end

Base.@assume_effects :nothrow :terminates_globally volatile_store!(x::Register{Reg, T}, v::T) where {Reg, T} = volatile_store!(x.ptr, v)
Base.@assume_effects :nothrow :terminates_globally volatile_load(x::Register) = volatile_load(x.ptr)

for T in keys(LLVM_TYPES)
    ptrs = LLVM_TYPES[T]
    store = """
          %ptr = inttoptr i64 %0 to $ptrs*
          store volatile $ptrs %1, $ptrs* %ptr, align 1
          ret void
          """
    vs = :(function volatile_store!(x::Ptr{$T}, v::$T)
        @inline
        return Base.llvmcall(
            $store,
            Cvoid,
            Tuple{Ptr{$T},$T},
            x,
            v
        )
    end)
    load = """
           %ptr = inttoptr i64 %0 to $ptrs*
           %val = load volatile $ptrs, $ptrs* %ptr, align 1
           ret $ptrs %val
           """
    ld = :(function volatile_load(x::Ptr{$T})
        @inline
        return Base.llvmcall(
            $load,
            $T,
            Tuple{Ptr{$T}},
            x
        )
    end)
    @eval $vs
    @eval $ld

    str = """
          call void asm sideeffect "", "X,~{memory}"($ptrs %0)
          ret void
          """
    k = :(function keep(x::$T)
        @inline
        return Base.llvmcall(
            $str,
            Cvoid,
            Tuple{$T},
            x
    )
    end)
    @eval $k
end

macro pindefs(name::Symbol, pins::Expr)
    pindefs(name, pins)
end

macro pindefs(reg::Symbol, name::Symbol, N::Int)
    expr = Expr(:block)
    expr.args = [ Symbol(name, i) for i in 0:N-1 ]
    pindefs(reg, expr)
end

function pindefs(name::Symbol, pins::Expr)
    pins.head !== :block && throw(ArgumentError("Given expression is not a block!"))
    res = Expr(:block)
    bit = 1
    nesc = esc(name)
    for e in pins.args
        e isa LineNumberNode && continue
        if e isa Symbol
            e_esc = esc(e)
            e != :_ && push!(res.args, :(const $e_esc = Pin{$nesc, $bit}()))
            bit += 1
        elseif (e isa Expr && e.head === :call
                && length(e.args) == 3
                && e.args[1] === Symbol(":")
                && e.args[2] === :_
                && e.args[3] isa Int)
            # intentional padding/reserved
            bit += e.args[3]
        else
            throw(ArgumentError("Invalid subexpression: `$e`"))
        end
    end
    res
end

end # module MCUCommon
