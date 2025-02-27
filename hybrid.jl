# Simulate the two-stage task from Daw et al. (2011)

using Random
using StatsFuns

struct HybridParams
    α1::Float64
    α2::Float64
    λ::Float64
    β1::Float64
    β2::Float64
    w::Float64
    p::Float64
end

struct Trial
    choice1::Int
    state2::Int
    choice2::Int
    reward::Int
end

const minrwrdprob = 0.25
const maxrwrdprob = 0.75
const rwrdprobdiffusion = 0.025
const commonprob = 0.7

function getrandomrwrd(state2, choice2, rwrdprobs)
    # Get a reward (0 or 1) with this probability.
    return Int(rand() < rwrdprobs[2*(state2-1)+choice2])
end

function diffuserwrdprobs(prob)
    # Diffuse reward probability and reflect it on boundaries.
    nextval = prob + (randn() * rwrdprobdiffusion) % 1.0
    if nextval > maxrwrdprob
        nextval = 2 * maxrwrdprob - nextval
    end
    if nextval < minrwrdprob
        nextval = 2 * minrwrdprob - nextval
    end
    if nextval > maxrwrdprob
        nextval = 2 * maxrwrdprob - nextval
    end
    return nextval
end

function getrandomrwrdprobs()
    # Create random reward probabilities within the allowed interval.
    return rand(4) .* (maxrwrdprob - minrwrdprob) .+ minrwrdprob
end

function getrandomstate2(choice)
    # Return the final state given a first-stage choice (1 or 2).
    if rand() < commonprob
        return choice
    end
    return 3 - choice
end

function simhybridagent(params::HybridParams, numtrials::Int)
    # Simulation of the simple hybrid model.
    q = zeros(2)
    v = zeros(2, 2)
    rwrdprobs = getrandomrwrdprobs()
    choice1 = 1
    trials = Vector{Trial}(undef, numtrials)
    for t in eachindex(trials)
        r = params.w * 0.4 * (maximum(v[2, :]) - maximum(v[1, :])) + (1.0 - params.w) * (q[2] - q[1])
        if t > 1
            if choice1 == 2
                r += params.p
            else
                r -= params.p
            end
        end
        p1 = logistic(params.β1 * r)
        choice1 = Int(rand() < p1) + 1
        state2 = getrandomstate2(choice1)
        choice2 = Int(rand() < logistic(params.β2 * (v[state2, 2] - v[state2, 1]))) + 1
        reward = getrandomrwrd(state2, choice2, rwrdprobs)
        q[choice1] = (1.0 - params.α1) * q[choice1] + params.α1 * ((1.0 - params.λ) * v[state2, choice2] + params.λ * reward)
        v[state2, choice2] = (1.0 - params.α2) * v[state2, choice2] + params.α2 * reward
        rwrdprobs = diffuserwrdprobs.(rwrdprobs)
        trials[t] = Trial(choice1, state2, choice2, reward)
    end
    trials
end

log_inv_logit(x) = -log1pexp(-x)
log1m_inv_logit(x) = -x - log1pexp(-x)

function hybridloglik(trials, α1::T, α2::T, λ::T, β1::T, β2::T, w::T, p::T) where T
    ll = 0.0
    q = zeros(T, 2)
    v = zeros(T, 2, 2)
    prevchoice1 = 1
    for (t, trial) in enumerate(trials)
        r = w * 0.4 * (maximum(v[2, :]) - maximum(v[1, :])) + (1.0 - w) * (q[2] - q[1])
        if t > 1
            if prevchoice1 == 2
                r += p
            else
                r -= p
            end
        end
        x1 = β1 * r
        if trial.choice1 == 2
            ll += log_inv_logit(x1)
        else
            ll += log1m_inv_logit(x1)
        end
        x2 = β2 * (v[trial.state2, 2] - v[trial.state2, 1])
        if trial.choice2 == 2
            ll += log_inv_logit(x2)
        else
            ll += log1m_inv_logit(x2)
        end
        q[trial.choice1] = (1.0 - α1) * q[trial.choice1] + α1 * ((1.0 - λ) * v[trial.state2, trial.choice2] + λ * trial.reward)
        v[trial.state2, trial.choice2] = (1.0 - α2) * v[trial.state2, trial.choice2] + α2 * trial.reward
        prevchoice1 = trial.choice1
    end
    ll
end

function testsimhybrid()
    open("testhybridagent.csv", "w") do outf
        println(outf, "trial,choice1,state2,choice2,reward")
        trials = simhybridagent(HybridParams(0.4, 0.6, 0.9, 2.5, 3.5, 0.5, 0.1), 50_000)
        for (t, trial) in enumerate(trials)
            println(outf, t, ",", trial.choice1, ",", trial.state2, ",", trial.choice2, ",", trial.reward)
        end
    end
end

const numtrials = 200
const paramshighnoise = HybridParams(0.5, 0.4, 0.6, 2.0, 2.0, 0.7, 0.1)
const paramslownoise = HybridParams(0.5, 0.4, 0.6, 5.0, 5.0, 0.7, 0.1)

function runweightnoisesim()
    open("weightnoisesim.csv", "w") do outf
        println(outf, "replication,participant,noise,trial,choice1,state2,choice2,reward")
        repl = 1
        println("Simulating replication ", repl, "...")
        part_num = 1
        for (params, noisecondition) in ((paramshighnoise, "high"), (paramslownoise, "low"))
            for _ = 1:200
                trials = simhybridagent(params, numtrials)
                for (t, trial) in enumerate(trials)
                    println(
                        outf, repl, ",", part_num, ",", noisecondition, ",", t, ",", trial.choice1, ",",
                        trial.state2, ",", trial.choice2, ",", trial.reward)
                end
                part_num += 1
            end
        end
    end
end
