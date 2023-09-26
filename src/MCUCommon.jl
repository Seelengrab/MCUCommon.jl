module MCUCommon

using FieldFlags

export Register, RegisterBits, Pin, volatile_load, volatile_store!,
    keep, delay_ms, delay_us, delay, CPU_FREQUENCY_HZ, datapin,
    dataregister, @pindefs

@enum Access Unknown=0 Read Write ReadWrite ReadWriteOnce
const WriteableAccess = (Write, ReadWrite, ReadWriteOnce)
const ReadableAccess = (Read, ReadWrite, ReadWriteOnce)

"""
    Register{Reg, T}

Represents a register named `Reg` (a `Symbol`), holding data of type `T`, which is expected to be a bitfield from FieldFlags.jl.

!!! warn "Internal fields"
    Internally, this is represented as a `Ptr{T}`, pointing to the memory mapped register at the given address.
    This is an implementation detail and may change at any time.
"""
struct Register{Reg, T}
    ptr::Ptr{T}
    Register{Reg, T}(x::Ptr) where {Reg, T} = new{Reg, T}(convert(Ptr{T}, x))
    Register{Reg, T}(x::Base.BitInteger) where {Reg, T} = new{Reg, T}(Ptr{T}(x % UInt)) # Ptr only takes Union{Int, UInt, Ptr}...
end

"""
    Field{Reg, Mode, Width, Name}

Represents a field of a register.

 * `RT`: The type of the register.
 * `Reg`: The actual register object.
 * `Mode`: The access mode of this field.
 * `Width`: The width, in bits, of this field.
 * `Name`: The name of the field.
"""
struct Field{RT, Reg, Mode, Width, Name}
    # RT     : The actual type of `Reg`
    # Reg    : An actual register object, to allow setting just a single field in isolation without keeping the register around in the source code
    # Offset : The offset of the field in the register
    # Width  : The width of the field in the register
    function Field{Reg, Mode, Width, Name}() where {Reg, Mode, Name, Width}
        Reg isa Register || throw(ArgumentError("`$Reg is not a `Register` object!"))
        Mode isa Access || throw(ArgumentError("`$Mode` is not an access mode!"))
        Name isa Symbol || throw(ArgumentError("`$Name` is not a `Symbol`!"))
        Width isa Integer || throw(ArgumentError("`$Width` is not an integer!"))
        T = typeof(Reg)
        new{T, Reg, Mode, Width, Name}()
    end
end

const Pin{RT, Reg, Mode, Name} = Field{RT, Reg, Mode, 1, Name}

Base.eltype(::Type{R}) where {Reg, T, R <: Register{Reg, T}} = T

function Base.getindex(r::Register)
    volatile_load(r)
end
function Base.getindex(_::Field{RT, Reg, Mode, Width, Name}) where {RName, RT <: Register{RName}, Reg, Mode, Width, Name} 
    Mode ∉ ReadableAccess && throw(ArgumentError("Field `$Name` of register `$RName` is not readable!"))
    getproperty(volatile_load(Reg), Name)
end

function Base.setindex!(r::RT, data::T) where {RT <: Register, RR, Mode, T <: Field{RT, RR, Mode}} 
    Mode ∉ WriteableAccess && throw(ArgumentError("Field `$Name` of register `$RName` is not writeable!"))
    volatile_store!(r, data)
end
function Base.setindex!(_::P, val::Bool) where {RName, RT <: Register{RName}, RR, Name, Mode, P <: Pin{RT, RR, Mode, Name}}
    Mode ∉ WriteableAccess && throw(ArgumentError("Field `$Name` of register `$RName` is not writeable!"))
    cur = volatile_load(RR)

    setproperty!(cur, Name, val)

    return volatile_store!(RR, cur)
end

# gracefullly stolen from VectorizationBase.jl

"""
    volatile_store!(r::Register{T}, v::T) -> Nothing

Stores a value `T` in the register `r`. This call is not elided by LLVM.
"""
function volatile_store! end

"""
  volatile_load(r::Register{T}) -> T
  volatile_load(p::Ptr{T}) -> T

Loads a value of type `T` from the register `r`. This call is not elided by LLVM.
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

Base.@assume_effects :nothrow :terminates_globally @generated function volatile_store!(x::Register{Reg,T}, v::T) where {Reg,T}
    ptrs = 8*sizeof(T)
    inner_type = fieldtype(T, 1)
    store = """
          %ptr = inttoptr i64 %0 to i$ptrs*
          store volatile i$ptrs %1, i$ptrs* %ptr, align 1
          ret void
          """
    :(return Base.llvmcall(
        $store,
        Cvoid,
        Tuple{Ptr{$T},$inner_type},
        x.ptr,
        getfield(v, :fields)
    ))
end

Base.@assume_effects :nothrow :terminates_globally @generated function volatile_load(x::Register{Reg,T}) where {Reg,T}
    ptrs = 8*sizeof(T)
    inner_type = fieldtype(T, 1)
    load = """
           %ptr = inttoptr i64 %0 to i$ptrs*
           %val = load volatile i$ptrs, i$ptrs* %ptr, align 1
           ret i$ptrs %val
           """
   :(return $T(Base.llvmcall(
        $load,
        $inner_type,
        Tuple{Ptr{$T}},
        x.ptr
    )))
end

@generated function keep(x::T) where T
    ptrs = 8*sizeof(T)
    str = """
          call void asm sideeffect "", "X,~{memory}"(i$ptrs %0)
          ret void
          """
    :(return Base.llvmcall(
        $str,
        Cvoid,
        Tuple{$T},
        x
    ))
end

macro regdef(block::Expr)
    block.head === :struct || throw(ArgumentError("Not a valid register definition!"))
    filter!(x -> !(x isa LineNumberNode), block.args[3].args) # just to make definitions easier
    !(block.args[2] isa Expr) && throw(ArgumentError("Not a valid register definition - missing address for Register!"))
    regname = block.args[2].args[1]
    regaddr = block.args[2].args[2]
    regfieldname = Symbol(regname, :Fields)

    bitfieldexpr = deepcopy(block)
    bitfieldexpr.args[2] = regfieldname
    bitfieldfields = bitfieldexpr.args[3].args
    map(eachindex(bitfieldfields)) do i
        field = bitfieldfields[i].args[3]
        if field isa Expr
            # drop the access mode, if it exists
            bitfieldfields[i].args[3] = field.args[1]
        elseif !(field isa Integer)
            throw(ArgumentError("Not a valid register definition!"))
        end
    end
    # let FieldFlags.jl handle the struct creation
    bitfield = FieldFlags.bitfield(bitfieldexpr)

    # now onto the individual register/pin definitions
    regdef = :(const $(esc(regname)) = Register{$(QuoteNode(regname)), $(esc(regfieldname))}($regaddr))

    fields = Expr(:block)
    for field in block.args[3].args
        fieldname = field.args[2]
        # support explicit placeholders
        fieldname === :_ && continue
        mode_size = field.args[3]
        if mode_size isa Integer
            fieldsize = mode_size
            fieldmode = :ReadWrite
        elseif mode_size isa Expr
            fieldsize = mode_size.args[1]
            fieldmode = mode_size.args[2]
        else
            throw(ArgumentError("Malformed register definition!"))
        end
        escname = esc(fieldname)
        fielddef = :(const $escname = Field{$(esc(regname)),$fieldmode,$fieldsize,$(QuoteNode(fieldname))}())
        push!(fields.args, fielddef)
    end

    return :(
        $bitfield;
        $regdef;
        $fields
    )
end

end # module MCUCommon
