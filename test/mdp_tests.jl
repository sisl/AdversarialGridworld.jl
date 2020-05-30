using AdversarialGridworld
using Test
using Random
using StaticArrays
using POMDPs

@test GWPos == SVector{2,Int}
@test TwoAgentPos == SVector{4,Int}
@test dir == Dict(:up=>GWPos(0,1), :down=>GWPos(0,-1), :left=>GWPos(-1,0), :right=>GWPos(1,0), :stay=>GWPos(0,0),
                 :upleft=>GWPos(-1,1), :upright=>GWPos(1,1), :downright=>GWPos(1,-1), :downleft=>GWPos(-1,-1))
@test aind == Dict(:up=>1, :down=>2, :left=>3, :right=>4, :stay=>5, :upleft=>6, :upright=>7, :downright=>8, :downleft=>9)
@test syma == [:up, :down, :left, :right, :stay, :upleft, :upright, :downleft, :downright]

# Test construction
mdp = AdversarialGridworldMDP(rewards = Dict(GWPos(1,1) => 1, GWPos(10,10) => 2, GWPos(1,10) => -1, GWPos(10,1) => -2), walls = [GWPos(5,5), GWPos(5,6), GWPos(5,7)])
amdp = AdversarialGridworldMDP(agent_gets_action = :adversary, rewards = Dict(GWPos(1,1) => 1, GWPos(10,10) => 2, GWPos(1,10) => -1, GWPos(10,1) => -2), walls = [GWPos(5,5), GWPos(5,6), GWPos(5,7)])
@test mdp.size == (10,10)
@test mdp.rewards[GWPos(10,10)] == 2
@test GWPos(5,5) in mdp.walls
@test mdp.tprob == 0.7
@test mdp.agent_gets_action == :ego
@test amdp.agent_gets_action == :adversary
@test mdp.adversary_policy(initialstate(mdp)) isa Symbol
@test mdp.ego_policy(initialstate(mdp)) isa Symbol
@test mdp.failure_penalty == 5

# Test valid_pos
@test !valid_pos(mdp, GWPos(11,10))
@test !valid_pos(mdp, GWPos(0,10))
@test !valid_pos(mdp, GWPos(5,5))
@test valid_pos(mdp, GWPos(9,9))

# Test random position generation and initial state
exclude = [GWPos(4,4)]
for i=1:100
    pos = random_valid_pos(mdp,  Random.GLOBAL_RNG, exclude)
    @test valid_pos(mdp, pos)
    @test !(pos in exclude)
    @test !(haskey(mdp.rewards, pos))
end

for i=1:100
    s = initialstate(mdp)
    @test ego_pos(s) != adversary_pos(s)
end

# Test action space and actions
@test actions(mdp) == syma
@test actionindex(mdp, :up) == 1
@test actionindex(mdp, :down) == 2
@test actionindex(mdp, :left) == 3
@test actionindex(mdp, :right) == 4

# Test agent position functions
S = TwoAgentPos
@test ego_pos(S(1,2,3,4)) == GWPos(1,2)
@test adversary_pos(S(1,2,3,4)) == GWPos(3,4)
@test !agents_overlap(S(1,2,3,4))
@test agents_overlap(S(1,2,1,2))

# Test discount
@test discount(mdp) == 0.95

# Test isterminal
isterminal(mdp, S(1,1,3,4))
@test !isterminal(mdp, S(1,1,3,4))
@test isterminal(mdp, S(-1,-1,-1,-1))

# TODO: Test gen
# Stay in place
gen(mdp, S(1,1,3,4), :up)

# Move
@test reward(mdp, S(-1,-1,-1,-1)) == 0
@test reward(mdp, S(2,2,2,2)) == -1.0
@test reward(mdp, S(1,1,2,2)) == 1 / mdp.failure_penalty
@test reward(mdp, S(1,1,1,1)) == (1. -mdp.failure_penalty) / mdp.failure_penalty
@test reward(mdp, S(1,2,2,1)) == 0

@test reward(amdp, S(2,2,2,2)) == 1.
@test reward(amdp, S(1,1,2,2)) == -0.2
@test reward(amdp, S(1,1,1,1)) == -(1. -mdp.failure_penalty) / mdp.failure_penalty
@test reward(amdp, S(1,2,2,1)) == 0


# render(mdp, initialstate(mdp))