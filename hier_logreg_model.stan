data {
    int M; // number of participants
    int N; // maximum number of trials per participant
    int K; // number of group predictors
    array[M, N] int<lower=0, upper=1> y; // stay for each trial
    array[M] matrix[N, K] x; // group predictors for each trial
}
parameters {
    matrix[K, M] z;
    // Distribution of coefficients
    vector[K] mu;
    cholesky_factor_corr[K] L_Omega;
    vector<lower=0>[K] tau;
}
model {
    matrix[K, K] m = diag_pre_multiply(tau, L_Omega);
    to_vector(z) ~ std_normal();
    for (p in 1:M) {
        vector[K] coefs = m * z[,p] + mu;
        y[p] ~ bernoulli_logit(x[p] * coefs);
    }
    // Priors
    L_Omega ~ lkj_corr_cholesky(2);
    mu ~ normal(0., 10.);
    tau ~ normal(0., 10.);
}
