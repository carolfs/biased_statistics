functions {
    // Assumed transitions
    // Station transitions [(1, 2), (3, 4), (5, 6)]
    // Spaceship transitions [1, 2, 3, 1, 2, 3]
    real mbstationval(int station, array[] real qaliens) {
        if (station == 1) {
            return fmax(qaliens[1], qaliens[2]);
        }
        else if (station == 2) {
            return fmax(qaliens[3], qaliens[1]);
        }
        else {
            return fmax(qaliens[2], qaliens[3]);
        }
    }
    real hybridstationvals(int station, real whtop, array[] real qaliens, array[] real qstations) {
        return whtop * mbstationval(station, qaliens) + (1 - whtop) * qstations[station];
    }
    real mbspaceshipval(int spaceship, array[] real qaliens) {
        return qaliens[(spaceship - 1) % 3 + 1];
    }
    real hybridspaceshipvals(int spaceship, real whmid, array[] real qaliens, array[] real qspaceships) {
        return whmid * mbspaceshipval(spaceship, qaliens) + (1 - whmid) * qspaceships[spaceship];
    }
    real hybrid_loglik(int num_trials, array[] int higheff, array[] int station0, array[] int station1, array[] int choice0, array[] int choice1, array[] int state2,
        array[] real reward, real alpha, real bhtop, real bhmid, real blow, real lambda, real whtop, real whmid, real wlow) {
        array[3] real qaliens = rep_array(0., 3);
        array[6] real qspaceships = rep_array(0., 6);
        array[3] real qstations = rep_array(0., 3);
        real ll = 0.0;
        for (t in 1:num_trials) {
            real dalien = reward[t] - qaliens[state2[t]];
            real dspaceship = qaliens[state2[t]] - qspaceships[choice1[t]];
            if (higheff[t]) {
                array[2] int spaceships = {2*choice0[t] - 1, 2*choice0[t]};
                real dstation = qspaceships[choice1[t]] - qstations[choice0[t]];
                if (choice0[t] == station0[t]) {
                    ll += log_inv_logit(bhtop*(
                        hybridstationvals(station0[t], whtop, qaliens, qstations) - 
                        hybridstationvals(station1[t], whtop, qaliens, qstations)
                    ));
                }
                else {
                    ll += log_inv_logit(bhtop*(
                        hybridstationvals(station1[t], whtop, qaliens, qstations) - 
                        hybridstationvals(station0[t], whtop, qaliens, qstations)
                    ));
                }
                if (choice1[t] == spaceships[1]) {
                    ll += log_inv_logit(bhmid*(
                        hybridspaceshipvals(spaceships[1], whmid, qaliens, qspaceships) -
                        hybridspaceshipvals(spaceships[2], whmid, qaliens, qspaceships)
                    ));
                }
                else {
                    ll += log_inv_logit(bhmid*(
                        hybridspaceshipvals(spaceships[2], whmid, qaliens, qspaceships) -
                        hybridspaceshipvals(spaceships[1], whmid, qaliens, qspaceships)
                    ));
                }
                qstations[choice0[t]] += alpha * dstation + lambda * alpha * dspaceship + lambda^2 * alpha * dalien;
            }
            else {
                array[3] int spaceships = (choice1[t] % 2 == 0) ? {2, 4, 6} : {1, 3, 5};
                array[3] real spaceshipvals;
                for (i in 1:3) {
                    spaceshipvals[i] = blow*hybridspaceshipvals(spaceships[i], wlow, qaliens, qspaceships);
                }
                ll += spaceshipvals[(choice1[t] - 1) %/% 2 + 1] - log_sum_exp(spaceshipvals);
            }
            qspaceships[choice1[t]] += alpha * dspaceship + lambda * alpha * dalien;
            qaliens[state2[t]] += alpha * dalien;
        }
        return ll;
    }
}
data {
    int num_subjects;
    array[num_subjects] int num_trials;
    array[num_subjects, 200] int higheff;
    array[num_subjects, 200] int station0;
    array[num_subjects, 200] int station1;
    array[num_subjects, 200] int choice0;
    array[num_subjects, 200] int choice1;
    array[num_subjects, 200] int state2;
    array[num_subjects, 200] real reward;
}
parameters {
    matrix[8, num_subjects] z;
    cholesky_factor_corr[8] L_Omega;
    vector<lower=0,upper=pi()/2>[8] tau_unif; 
    // Group
    real<lower=0, upper=1> alpha_mu;
    real<lower=0> bhtop_mu;
    real<lower=0> bhmid_mu;
    real<lower=0> blow_mu;
    real<lower=0, upper=1> lambda_mu;
    real<lower=0, upper=1> whtop_mu;
    real<lower=0, upper=1> whmid_mu;
    real<lower=0, upper=1> wlow_mu;
}
transformed parameters {
    matrix[8, num_subjects] subjparams;
    matrix[8, 8] Omega;
    {
        vector[8] tau = tan(tau_unif); // Cauchy(0, 1) prior
        matrix[8, num_subjects] m = diag_pre_multiply(tau, L_Omega) * z;
        vector[8] mu;
        Omega = L_Omega * L_Omega';
        mu[1] = logit(alpha_mu);
        mu[2] = log(bhtop_mu);
        mu[3] = log(bhmid_mu);
        mu[4] = log(blow_mu);
        mu[5] = logit(lambda_mu);
        mu[6] = logit(whtop_mu);
        mu[7] = logit(whmid_mu);
        mu[8] = logit(wlow_mu);
        for (s in 1:num_subjects) {
            subjparams[,s] = mu + m[,s];
        }
    }
}
model {
    array[num_subjects] real alpha;
    array[num_subjects] real bhtop;
    array[num_subjects] real bhmid;
    array[num_subjects] real blow;
    array[num_subjects] real lambda;
    array[num_subjects] real whtop;
    array[num_subjects] real whmid;
    array[num_subjects] real wlow;
    for (i in 1:num_subjects) {
        alpha[i] = inv_logit(subjparams[1][i]);
        bhtop[i] = exp(subjparams[2][i]);
        bhmid[i] = exp(subjparams[3][i]);
        blow[i] = exp(subjparams[4][i]);
        lambda[i] = inv_logit(subjparams[5][i]);
        whtop[i] = inv_logit(subjparams[6][i]);
        whmid[i] = inv_logit(subjparams[7][i]);
        wlow[i] = inv_logit(subjparams[8][i]);
    }
    to_vector(z) ~ std_normal();
    L_Omega ~ lkj_corr_cholesky(2);
    bhtop_mu ~ gamma(4.82, 1./0.88);
    bhmid_mu ~ gamma(4.82, 1./0.88);
    blow_mu ~ gamma(4.82, 1./0.88);
    for (i in 1:num_subjects) {
        target += hybrid_loglik(num_trials[i], higheff[i], station0[i], station1[i], choice0[i], choice1[i], state2[i],
            reward[i], alpha[i], bhtop[i], bhmid[i], blow[i], lambda[i], whtop[i], whmid[i], wlow[i]);
    }
}
