module AdversarialGridworld
    using POMDPs
    using POMDPModelTools
    using StaticArrays
    using Parameters
    using Base.Cartesian
    using Random
    using Compose
    using ColorSchemes
    using LocalApproximationValueIteration
    using GridInterpolations
    using LocalFunctionApproximation
    using Serialization
    using POMDPSimulators
    using Statistics
    using Cairo, Fontconfig

    export GWPos, TwoAgentPos, aind, dir, syma, AdversarialGridworldMDP, DoubleAdversarialGridworldMDP, valid_pos,
            ThreeAgentPos, aind_ego, aind_adv, syma_ego, syma_adv,
            random_valid_pos, ego_pos, adversary_pos, agents_overlap
    include("mdp.jl")
    include("mdp_2adv.jl")

    export load_items, convert_policies_to_tables, solve_for_policies,
            generate_task_set, generate_task, solve_for_policy, value_to_table,
            action_values, policy_to_table, average_performance
    include("task_generator.jl")

end