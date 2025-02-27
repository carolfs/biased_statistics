using HypothesisTests
using JuMP
using NLopt
using Random
using Statistics
using Dates

include("hybrid.jl")

function hybridfit(simdata)
    loglik = -Inf
    α1val, α2val, λval, β1val, β2val, wval, pval = 0., 0., 0., 0., 0., 0., 0.
    for _ = 1:100
        model = Model(NLopt.Optimizer)
        set_optimizer_attribute(model, "algorithm", :LD_SLSQP)
        @variable(model, 0 <= α1 <= 1, start=rand())
        @variable(model, 0 <= α2 <= 1, start=rand())
        @variable(model, 0 <= λ <= 1, start=rand())
        @variable(model, 0 <= β1 <= 20, start=rand()*20.)
        @variable(model, 0 <= β2 <= 20, start=rand()*20.)
        @variable(model, 0 <= w <= 1, start=rand())
        @variable(model, -5 <= p <= 5, start=(-5. + rand()*10.))
        llfunc(α1, α2, λ, β1, β2, w, p) = hybridloglik(simdata, α1, α2, λ, β1, β2, w, p)
        register(model, :llfunc, 7, llfunc; autodiff=true)
        @NLobjective(model, Max, llfunc(α1, α2, λ, β1, β2, w, p))
        JuMP.optimize!(model)
        if objective_value(model) > loglik
            α1val, α2val, λval, β1val, β2val, wval, pval = value(α1), value(α2), value(λ), value(β1), value(β2), value(w), value(p)
            loglik = objective_value(model)
        end
    end
    loglik, α1val, α2val, λval, β1val, β2val, wval, pval
end

const numagentspergroup = 100

function getmbweight(trials)
    while true
        try
            return hybridfit(trials)[7]
        catch
            println("ERROR: ", err.msg)
            println(stacktrace())
        end
    end
end

function runttestnumtrials(repl)
    starttime = now()
    println("Starting replication ", repl, " at ", Dates.format(starttime, "HH:MM"), "...")
    w1 = [getmbweight(simhybridagent(paramslownoise, 120)) for _ = 1:numagentspergroup]
    w2 = [getmbweight(simhybridagent(paramslownoise, 200)) for _ = 1:numagentspergroup]
    res = UnequalVarianceTTest(w1, w2)
    println("Finished replication ", repl, " after ", (now() - starttime), "!")
    "$repl,$numagentspergroup,$numagentspergroup,$(mean(w1)),$(mean(w2)),$(res.t),$(pvalue(res))"
end

function runttestnoiselevels(repl)
    starttime = now()
    println("Starting replication ", repl, " at ", Dates.format(starttime, "HH:MM"), "...")
    w1 = [getmbweight(simhybridagent(paramslownoise, 200)) for _ = 1:numagentspergroup]
    w2 = [getmbweight(simhybridagent(paramshighnoise, 200)) for _ = 1:numagentspergroup]
    res = UnequalVarianceTTest(w1, w2)
    println("Finished replication ", repl, " after ", (now() - starttime), "!")
    "$repl,$numagentspergroup,$numagentspergroup,$(mean(w1)),$(mean(w2)),$(res.t),$(pvalue(res))"
end

const numrepls = 1_000
const numbatches = 10

function runttests(numbatch)
    @assert 1 <= numbatch <= (numrepls ÷ numbatches)
    # Run t-tests for different numbers of trials
    output_flnm = "ttest_tst_numtrials_results_$numbatch.csv"
    dorepls = (numrepls ÷ numbatches)
    if !isfile(output_flnm)
        open(output_flnm, "w") do outf
            println(outf, "replication,n1,n2,avgmbw1,avgmbw2,tvalue,pvalue")
            for repl = ((numbatch - 1)*dorepls + 1):numbatch*dorepls
                result = runttestnumtrials(repl)
                println(outf, result)
            end
        end
    end
    # Run t-tests for different noise levels
    output_flnm = "ttest_tst_noiselevels_results_$numbatch.csv"
    if !isfile(output_flnm)
        open(output_flnm, "w") do outf
            println(outf, "replication,n1,n2,avgmbw1,avgmbw2,tvalue,pvalue")
            for repl = ((numbatch - 1)*dorepls + 1):numbatch*dorepls
                result = runttestnumtrials(repl)
                println(outf, result)
            end
        end
    end
end
