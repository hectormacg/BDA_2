
data {
    int<lower=0> N; // number of observations
    int<lower=0> K; // number of predictors
    matrix[N, K] X;  // matrix design
    vector[N] area_log;
    int y[N]; // observed counts
}
parameters {
    vector[K] beta;
    real reciprocal_phi;
}
transformed parameters {
    vector[N] eta;
    real phi;
    eta = area_log + X * beta;
    phi = 1. / reciprocal_phi;
}

model {
    reciprocal_phi ~ cauchy(0., 5);
    beta[1] ~ normal(0,1.5);
    beta[2:K] ~ normal(0,1);
    y ~ neg_binomial_2_log(eta, phi);
}

generated quantities{
    vector[N] yrep;  //replicates
    vector[N] mu;
    vector[N] log_lik;
    mu = exp(eta);

    // Generate replicated data using vectorized _rng form
    for (n in 1:N) {
         log_lik[n] = neg_binomial_2_log_lpmf(y[n] | eta[n], phi);   
         yrep[n] = neg_binomial_2_rng(mu[n], phi);
    }
}
