using Flux

struct ConstantLayer
    w
end
ConstantLayer(N::Integer) = ConstantLayer(zeros(N))
(m::ConstantLayer)(x) = m.w
Flux.@functor ConstantLayer

struct A2TNetwork
  base::Chain
  attn::Chain
  solutions::Array{Function} # function takes in state and outputs vector of action values
end

function (m::A2TNetwork)(input)
    b = m.base(input) #output is (Na, b)
    w = m.attn(input) #output is (Nt+1, b)
    B, Nt = size(input, 2), size(w,1) - 1
    qs = [hcat([s(input[:,i]) for s in m.solutions]...) for i=1:B] #output is (Na, Nt)
    Flux.stack(qs .* Flux.unstack(w[1:Nt, :], 2), 2) .+ w[Nt+1:Nt+1, :] .* b
end

Flux.@functor A2TNetwork

Flux.trainable(m::A2TNetwork) = (m.base, m.attn)

function Base.iterate(m::A2TNetwork, i=1)
    i > length(m.base.layers) + length(m.attn.layers) && return nothing
    if i <= length(m.base.layers)
        return (m.base[i], i+1)
    elseif i <= length(m.base.layers) + length(m.attn.layers)
        return (m.attn[i - length(m.base.layers)], i+1)
    end
end

function Base.deepcopy(m::A2TNetwork)
  A2TNetwork(deepcopy(m.base), deepcopy(m.attn), deepcopy(m.solutions))
end

