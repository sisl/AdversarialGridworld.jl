const ThreeAgentPos = SVector{6,Int}

const syma_ego = [[:up,], [:down,], [:left,], [:right,]]
const syma_adv = [[:up, :up], [:down, :up], [:left, :up], [:right, :up],
                [:up, :down], [:down, :down], [:left, :down], [:right, :down],
                [:up, :left], [:down, :left], [:left, :left], [:right, :left],
                [:up, :right], [:down, :right], [:left, :right], [:right, :right]]
const aind_ego = Dict(syma_ego[i] => i for i=1:length(syma_ego))
const aind_adv = Dict(syma_adv[i] => i for i=1:length(syma_adv))
const S2 = ThreeAgentPos
const A2 = Array{Symbol}

# Gridworld with adversary
@with_kw mutable struct DoubleAdversarialGridworldMDP <: MDP{S2, A2}
    size::Tuple{Int, Int} = (10,10)
    rewards::Dict{GWPos, Float64} = Dict()
    walls::Vector{GWPos} = []
    tprob::Float64 = 0.7
    discount::Float64 = 0.95
    agent_gets_action = :ego # :ego or :adversary
    ego_policy = (s, rng::AbstractRNG = Random.GLOBAL_RNG) -> rand(rng, syma_ego)
    adversary_policy = (s, rng::AbstractRNG = Random.GLOBAL_RNG) -> rand(rng, syma_adv)
    failure_penalty = 1.
end

valid_pos(mdp::DoubleAdversarialGridworldMDP, pos::GWPos) = !(pos in mdp.walls || any((pos .> mdp.size) .| (pos .< GWPos(1,1))))

function POMDPs.states(mdp::DoubleAdversarialGridworldMDP)
    lengths = (mdp.size[1], mdp.size[2], mdp.size[1], mdp.size[2], mdp.size[1], mdp.size[2])
    ss = S2[]
    for ijk in CartesianIndices(lengths)
        s = S2(ijk.I...)
        valid_pos(mdp, ego_pos(s)) && valid_pos(mdp, adversary1_pos(s)) && valid_pos(mdp, adversary2_pos(s))  &&  push!(ss, s)
    end
    push!(ss, S2(-1,-1,-1,-1,-1,-1))
    ss
end

function random_valid_pos(mdp::DoubleAdversarialGridworldMDP, rng::AbstractRNG = Random.GLOBAL_RNG, exclude = [], max_trials = 1000)
    trial = 0
    while trial < max_trials
        pos = GWPos(rand(rng, 1:mdp.size[1]), rand(rng, 1:mdp.size[2]))
        if valid_pos(mdp, pos) && !(haskey(mdp.rewards, pos) || pos in exclude)
            return pos
        end
        trial += 1
    end
 end

function POMDPs.initialstate(mdp::DoubleAdversarialGridworldMDP, rng::AbstractRNG = Random.GLOBAL_RNG)
    ego = random_valid_pos(mdp, rng)
    adversary1 = random_valid_pos(mdp, rng, [ego])
    adversary2 = random_valid_pos(mdp, rng, [ego, adversary1])
    Deterministic(S2(ego..., adversary1..., adversary2...))
end

POMDPs.actions(mdp::DoubleAdversarialGridworldMDP) = mdp.agent_gets_action == :ego ? syma_ego : syma_adv
POMDPs.actionindex(mdp::DoubleAdversarialGridworldMDP, a::A2) = mdp.agent_gets_action == :ego ? aind_ego[a] : aind_adv[a]

ego_pos(s::S2) = GWPos(s[1], s[2])
adversary1_pos(s::S2) = GWPos(s[3], s[4])
adversary2_pos(s::S2) = GWPos(s[5], s[6])
agents_overlap(s::S2) = (ego_pos(s) == adversary1_pos(s)) && (ego_pos(s) == adversary2_pos(s))


POMDPs.isterminal(mdp::DoubleAdversarialGridworldMDP, s::S2) = any(s .< 0)
POMDPs.discount(mdp::DoubleAdversarialGridworldMDP) = mdp.discount

# Returns a sample next state and reward
function POMDPs.gen(mdp::DoubleAdversarialGridworldMDP, s::S2, a::A2, rng::AbstractRNG = Random.GLOBAL_RNG)
    if haskey(mdp.rewards, ego_pos(s)) || agents_overlap(s) || isterminal(mdp, s)
        return (sp = S2(-1,-1,-1,-1,-1,-1), r = reward(mdp, s))
    else
        # If this MDP controls the agent then use the adversary policy for the adversary
        # Do the opposite if the adversary is being controlled by the MDP action
        if mdp.agent_gets_action == :ego
            rdir = (rand(rng) < mdp.tprob) ? dir[a[1]] : dir[rand(rng, syma[a[1] .!= syma])]
            new_ego = ego_pos(s) + rdir
            a_adv = mdp.adversary_policy(s, rng)
            new_adv1 = adversary1_pos(s) + dir[a_adv[1]]
            new_adv2 = adversary2_pos(s) + dir[a_adv[2]]
        else
            rdir1 = (rand(rng) < mdp.tprob) ? dir[a[1]] : dir[rand(rng, syma[a[1] .!= syma])]
            rdir2 = (rand(rng) < mdp.tprob) ? dir[a[2]] : dir[rand(rng, syma[a[2] .!= syma])]
            new_ego = ego_pos(s) + dir[mdp.ego_policy(s, rng)[1]]
            new_adv1 = adversary1_pos(s) + rdir1
            new_adv2 = adversary2_pos(s) + rdir2
        end

        # Make sure the moves are in bound and not hitting a wall
        new_ego = valid_pos(mdp, new_ego) ? new_ego : ego_pos(s)
        new_adv1 = valid_pos(mdp, new_adv1) ? new_adv1 : adversary1_pos(s)
        new_adv2 = valid_pos(mdp, new_adv2) ? new_adv2 : adversary2_pos(s)

        return (sp = S2(new_ego..., new_adv1..., new_adv2...), r = reward(mdp, s))
    end
end


# Returns the reward for the provided state
function POMDPs.reward(mdp::DoubleAdversarialGridworldMDP, s::S2)
    isterminal(mdp, s) && return 0
    r = (get(mdp.rewards, ego_pos(s), 0.0) - mdp.failure_penalty*agents_overlap(s)) /  mdp.failure_penalty
    mdp.agent_gets_action == :ego ? r : -r
end

function tocolor(mdp::DoubleAdversarialGridworldMDP, r::Float64)
    maxr = maximum(values(mdp.rewards))
    minr = -maxr
    frac = (r-minr)/(maxr-minr)
    return get(ColorSchemes.redgreensplit, frac)
end

# Renders the mdp
function POMDPModelTools.render(mdp::DoubleAdversarialGridworldMDP, s::S2)
    nx, ny = mdp.size
    cells = []
    for x in 1:nx, y in 1:ny
        pos = GWPos(x,y)
        reward_index = findfirst([pos] .== keys(mdp.rewards))
        wall_index = findfirst([pos] .== mdp.walls)
        ctx = context((x-1)/nx, (ny-y)/ny, 1/nx, 1/ny)
        color = "white"
        if !isnothing(reward_index)
            color = tocolor(mdp, get(mdp.rewards, pos, 0))
        elseif !isnothing(wall_index)
            color = "black"
        end
        cell = compose(ctx, Compose.rectangle(), fill(color))
        push!(cells, cell)
    end
    grid = compose(context(), Compose.stroke("gray"), cells...)
    outline = compose(context(), Compose.rectangle())

    ego = nothing
    x,y = ego_pos(s)
    if all([x,y] .> 0)
        agent_ctx = context((x-1)/nx, (ny-y)/ny, 1/nx, 1/ny)
        ego = compose(agent_ctx, Compose.circle(0.5, 0.5, 0.4), Compose.stroke("black"), fill("blue"))
    end

    adversary1 = nothing
    x,y = adversary1_pos(s)
    if all([x,y] .> 0)
        agent_ctx = context((x-1)/nx, (ny-y)/ny, 1/nx, 1/ny)
        adversary1 = compose(agent_ctx, Compose.circle(0.5, 0.5, 0.4), Compose.stroke("black"), fill("orange"))
    end

    adversary2 = nothing
    x,y = adversary2_pos(s)
    if all([x,y] .> 0)
        agent_ctx = context((x-1)/nx, (ny-y)/ny, 1/nx, 1/ny)
        adversary2 = compose(agent_ctx, Compose.circle(0.5, 0.5, 0.4), Compose.stroke("black"), fill("red"))
    end
    agents_comp = compose(context(), ego, adversary1, adversary2)

    sz = min(w, h)
    return compose(context((w-sz)/2, (h-sz)/2, sz, sz), agents_comp, grid, outline)
end

