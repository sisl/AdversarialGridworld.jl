using TensorBoardLogger, ValueHistories
using Plots

train_bl_


hist = convert(MVHistory, logger)

function load_history(folder)
    logger = TBLogger(folder, tb_append)
    hist = MVHistory()
    try
    TensorBoardLogger.map_summaries(logger) do tag, iter, val
        push!(hist, Symbol(tag), iter, val)
    end
    catch end
    hist
end

## Plot Training history on seen task
train_baseline_hist = load_history("log/train_baseline/")
train_A2T_hist = load_history("log/train_A2T/")
train_A2t_const_hist = load_history("log/train_A2T_ConstantW_highlr/")
train_lb, train_lb_std, train_ub, train_ub_std = deserialize("train_bounds.jls")

plot([1,2e6], [train_lb, train_lb], ribbon=(2.98*train_lb_std), label = "Random Policy",
    title = "Training History on Seen Task",
    xlabel="Samples", ylabel="reward",
    xlims=(1e2, 2e6),
    xscale = :log,
    legend = :bottomleft)
plot!([1,2e6], [train_ub, train_ub], ribbon=(2.98*train_ub_std), label = "Optimal Policy")
plot!(get(train_baseline_hist[:eval_reward])..., label = "DQN Baseline")
plot!(get(train_A2T_hist[:eval_reward])..., label = "A2T", color = :black)
plot!(get(train_A2t_const_hist[:eval_reward])..., alpha = 0.75, color = :turquoise4, label = "A2T Constant Weights")
savefig("train_history_seen.pdf")


## Plot Training history on unseen task =
test_baseline_hist = load_history("log/test_baseline/")
test_A2T_hist = load_history("log/test_A2T/")
test_A2t_const_hist = load_history("log/test_A2T_ConstantW_highlr/")
test_lb, test_lb_std, test_ub, test_ub_std = deserialize("test_bounds.jls")

plot([1,2e6], [test_lb, test_lb], ribbon=(2.98*test_lb_std), label = "Random Policy",
    title = "Training History on Unseen Task",
    xlabel="Samples", ylabel="reward",
    xlims=(1e1, 2e6),
    xscale = :log,
    legend = :bottomleft)
plot!([1,2e6], [test_ub, test_ub], ribbon=(2.98*test_ub_std), label = "Optimal Policy")
plot!(get(test_baseline_hist[:eval_reward])..., label = "DQN Baseline")
plot!(get(test_A2T_hist[:eval_reward])..., label = "A2T", color = :black)
plot!(get(test_A2t_const_hist[:eval_reward])..., alpha = 0.75, color = :turquoise4, label = "A2T Constant Weights")
savefig("train_history_unseen.pdf")

