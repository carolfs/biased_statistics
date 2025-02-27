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
    real hybridstationval(int station, real whtop, array[] real qaliens, array[] real qstations) {
        return whtop * mbstationval(station, qaliens) + (1 - whtop) * qstations[station];
    }
    real mbspaceshipval(int spaceship, array[] real qaliens) {
        return qaliens[(spaceship - 1) % 3 + 1];
    }
    real hybridspaceshipval(int spaceship, real whmid, array[] real qaliens, array[] real qspaceships) {
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
                        hybridstationval(station0[t], whtop, qaliens, qstations) - 
                        hybridstationval(station1[t], whtop, qaliens, qstations)
                    ));
                }
                else {
                    ll += log_inv_logit(bhtop*(
                        hybridstationval(station1[t], whtop, qaliens, qstations) - 
                        hybridstationval(station0[t], whtop, qaliens, qstations)
                    ));
                }
                if (choice1[t] == spaceships[1]) {
                    ll += log_inv_logit(bhmid*(
                        hybridspaceshipval(spaceships[1], whmid, qaliens, qspaceships) -
                        hybridspaceshipval(spaceships[2], whmid, qaliens, qspaceships)
                    ));
                }
                else {
                    ll += log_inv_logit(bhmid*(
                        hybridspaceshipval(spaceships[2], whmid, qaliens, qspaceships) -
                        hybridspaceshipval(spaceships[1], whmid, qaliens, qspaceships)
                    ));
                }
                qstations[choice0[t]] += alpha * dstation + lambda * alpha * dspaceship + lambda^2 * alpha * dalien;
            }
            else {
                array[3] int spaceships = (choice1[t] % 2 == 0) ? {2, 4, 6} : {1, 3, 5};
                real spaceshipvals[3];
                for (i in 1:3) {
                    spaceshipvals[i] = blow*hybridspaceshipval(spaceships[i], wlow, qaliens, qspaceships);
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
    int num_trials;
    array[num_trials] int higheff;
    array[num_trials] int station0;
    array[num_trials] int station1;
    array[num_trials] int choice0;
    array[num_trials] int choice1;
    array[num_trials] int state2;
    array[num_trials] real reward;
}
parameters {
    real<lower=0, upper=20> blow;
    real<lower=0, upper=20> bhtop;
    real<lower=0, upper=20> bhmid;
    real<lower=0, upper=1> alpha;
    real<lower=0, upper=1> lambda;
    real<lower=0, upper=1> wlow;
    real<lower=0, upper=1> whtop;
    real<lower=0, upper=1> whmid;
}
transformed parameters {
    real loglik = hybrid_loglik(num_trials, higheff, station0, station1, choice0, choice1, state2,
        reward, alpha, bhtop, bhmid, blow, lambda, whtop, whmid, wlow);
}
model {
    bhtop ~ gamma(4.82, 1./0.88);
    bhmid ~ gamma(4.82, 1./0.88);
    blow ~ gamma(4.82, 1./0.88);
    target += loglik;
}
