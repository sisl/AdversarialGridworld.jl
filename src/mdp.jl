const GWPos = SVector{2,Int}
const TwoAgentPos = SVector{4,Int}
const dir = Dict(:up=>GWPos(0,1), :down=>GWPos(0,-1), :left=>GWPos(-1,0), :right=>GWPos(1,0), :stay=>GWPos(0,0),
                 :upleft=>GWPos(-1,1), :upright=>GWPos(1,1), :downright=>GWPos(1,-1), :downleft=>GWPos(-1,-1))
const aind = Dict(:up=>1, :down=>2, :left=>3, :right=>4, :stay=>5, :upleft=>6, :upright=>7, :downright=>8, :downleft=>9)
const syma = [:up, :down, :left, :right, :stay, :upleft, :upright, :downleft, :downright]
const S = TwoAgentPos
const A = Symbol

# Gridworld with adversary
@with_kw mutable struct AdversarialGridworldMDP <:MDP{S, A}
    size::Tuple{Int, Int} = (10,10)
    rewards::Dict{GWPos, Float64} = Dict()
    walls::Vector{GWPos} = []
    tprob::Float64 = 0.7
    discount::Float64 = 0.95
    agent_gets_action = :ego # :ego or :adversary
    ego_policy = (s, rng::AbstractRNG = Random.GLOBAL_RNG) -> rand(rng, syma)
    adversary_policy = (s, rng::AbstractRNG = Random.GLOBAL_RNG) -> rand(rng, syma)
    failure_penalty = 5
end

valid_pos(mdp, pos::GWPos) = !(pos in mdp.walls || any((pos .> mdp.size) .| (pos .< GWPos(1,1))))

function random_valid_pos(mdp::AdversarialGridworldMDP, rng::AbstractRNG = Random.GLOBAL_RNG, exclude = [], max_trials = 1000)
    trial = 0
    while trial < max_trials
        pos = GWPos(rand(rng, 1:mdp.size[1]), rand(rng, 1:mdp.size[2]))
        if valid_pos(mdp, pos) && !(haskey(mdp.rewards, pos) || pos in exclude)
            return pos
        end
        trial += 1
    end
 end

function POMDPs.initialstate(mdp::AdversarialGridworldMDP, rng::AbstractRNG = Random.GLOBAL_RNG)
    ego = random_valid_pos(mdp, rng)
    adversary = random_valid_pos(mdp, rng, [ego])
    S(ego..., adversary...)
end

POMDPs.actions(mdp::AdversarialGridworldMDP) = syma
POMDPs.actionindex(mdp::AdversarialGridworldMDP, a::A) = aind[a]

ego_pos(s::S) = GWPos(s[1], s[2])
adversary_pos(s::S) = GWPos(s[3], s[4])
agents_overlap(s::S) = ego_pos(s) == adversary_pos(s)


POMDPs.isterminal(mdp::AdversarialGridworldMDP, s::S) = any(s .< 0)
POMDPs.discount(mdp::AdversarialGridworldMDP) = mdp.discount

# Returns a sample next state and reward
function POMDPs.gen(mdp::AdversarialGridworldMDP, s::S, a::A, rng::AbstractRNG = Random.GLOBAL_RNG)
    if haskey(mdp.rewards, ego_pos(s)) || agents_overlap(s) || isterminal(mdp, s)
        return (sp = S(-1,-1,-1,-1), r = reward(mdp, s))
    else
        # Compute the direction based on the provided action
        rdir = (rand(rng) < mdp.tprob) ? dir[a] : dir[rand(rng, syma[a .!= syma])]

        # If this MDP controls the agent then use the adversary policy for the adversary
        # Do the opposite if the adversary is being controlled by the MDP action
        if mdp.agent_gets_action == :ego
            new_ego = ego_pos(s) + rdir
            new_adv = adversary_pos(s) + dir[mdp.adversary_policy(s, rng)]
        else
            new_ego = ego_pos(s) + dir[mdp.ego_policy(s, rng)]
            new_adv = adversary_pos(s) + rdir
        end

        # Make sure the moves are in bound and not hitting a wall
        new_ego = valid_pos(mdp, new_ego) ? new_ego : ego_pos(s)
        new_adv = valid_pos(mdp, new_adv) ? new_adv : adversary_pos(s)

        return (sp = S(new_ego..., new_adv...), r = reward(mdp, s))
    end
end


# Returns the reward for the provided state
function POMDPs.reward(mdp::AdversarialGridworldMDP, s::S)
    isterminal(mdp, s) && return 0
    r = (get(mdp.rewards, ego_pos(s), 0.0) - mdp.failure_penalty*agents_overlap(s)) /  mdp.failure_penalty
    mdp.agent_gets_action == :ego ? r : -r
end

function tocolor(mdp::AdversarialGridworldMDP, r::Float64)
    maxr = maximum(values(mdp.rewards))
    minr = -maxr
    frac = (r-minr)/(maxr-minr)
    return get(ColorSchemes.redgreensplit, frac)
end

# Renders the mdp
function POMDPModelTools.render(mdp::AdversarialGridworldMDP, s::S)
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

    if all(s .> 0)
        x,y = ego_pos(s)
        agent_ctx = context((x-1)/nx, (ny-y)/ny, 1/nx, 1/ny)
        ego = compose(agent_ctx, Compose.circle(0.5, 0.5, 0.4), Compose.stroke("black"), fill("blue"))

        x,y = adversary_pos(s)
        agent_ctx = context((x-1)/nx, (ny-y)/ny, 1/nx, 1/ny)
        adversary = compose(agent_ctx, Compose.circle(0.5, 0.5, 0.4), Compose.stroke("black"), fill("orange"))

        agents_comp = compose(context(), ego, adversary)

        sz = min(w, h)
        return compose(context((w-sz)/2, (h-sz)/2, sz, sz), agents_comp, grid, outline)
    else
        sz = min(w, h)
        return compose(context((w-sz)/2, (h-sz)/2, sz, sz), grid, outline)
    end
end

