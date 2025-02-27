functions {
    real hybrid(int T, array[] int choice1, array[] int state2, array[] int choice2,
        array[] int reward, real alpha1, real alpha2, real lmbd, real beta1, 
        real beta2, real w, real p) {

        real log_lik = 0.;
        array[2] real q = {0., 0.};
        array[2, 2] real v = {{0., 0.}, {0., 0.}};

        for (t in 1:T) {
            real x1;
            real x2;
            x1 = // Model-based value
                w*0.4*(max(v[2]) - max(v[1])) +
                // Model-free value
                (1 - w)*(q[2] - q[1]);
            // Perseveration
            if (t > 1) {
                if (choice1[t - 1] == 2)
                    x1 += p;
                else
                    x1 -= p;
            }
            // Exploration
            x1 *= beta1;
            // First stage choice
            if (choice1[t] == 2)
                log_lik += log_inv_logit(x1);
            else
                log_lik += log1m_inv_logit(x1);

            // Second stage choice
            x2 = beta2*(v[state2[t], 2] - v[state2[t], 1]);
            if (choice2[t] == 2)
                log_lik += log_inv_logit(x2);
            else
                log_lik += log1m_inv_logit(x2);

            // Learning
            q[choice1[t]] += alpha1*(v[state2[t], choice2[t]] - q[choice1[t]]) +
                alpha1*lmbd*(reward[t] - v[state2[t], choice2[t]]);
            v[state2[t], choice2[t]] += alpha2*(reward[t] - v[state2[t], choice2[t]]);
        }
        return log_lik;
    }
}
data {
    // Total number of agents
    int<lower=1> N;
    // Condition for each agent
    array[N] int<lower=1, upper=2> condition;
    // Maximum number of trials
    int<lower=1> T;
    // Number of trials per condition
    array[2] int num_trials;
    array[N, T] int<lower=1, upper=2> choice1; // First stage actions
    array[N, T] int<lower=1, upper=2> choice2; // Second stage actions
    array[N, T] int<lower=1, upper=2> state2; // Second stage states
    array[N, T] int<lower=0, upper=1> reward; // Rewards
}
parameters {
    array[N] real<lower=0, upper=1> w;
    array[2] real<lower=0, upper=1> mu;
    vector<lower=0>[2] kappa;
}
model {
    kappa ~ lognormal(6, 0.5);
    w ~ beta_proportion(mu[condition], kappa[condition]);
    for(i in 1:N) {
        target += hybrid(num_trials[condition[i]], choice1[i], state2[i], choice2[i], reward[i], 0.5, 0.4, 0.6, 5.,
            5., w[i], 0.1);
    }
}
