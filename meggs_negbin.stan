
data {
    int<lower=0> N; // number of observations
    int<lower=0> K; // number of predictors
    matrix[N, K] X;  // matrix design
    vector[N] area_log;
    int y[N]; // observed counts
    real m0;
    real<lower=0> sigma0;
    real<lower=0> sigma1;

}
parameters {
    vector[K] beta;
     real<lower=0> phi;
}
transformed parameters {
    vector[N] eta;
    eta = area_log + X * beta;
}

model {
    phi ~ gamma(2, 0.01);
    beta[1] ~ normal(m0, sigma0);
    beta[2:K] ~ normal(0, sigma1);
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
