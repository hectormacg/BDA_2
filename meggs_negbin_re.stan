
data {
    int<lower=0> N; // number of observations
    int<lower=0> K; // number of predictors
    int<lower=0> M; // number of farms
    matrix[N, K] X;  // matrix design
    vector[N] area_log;
    int j_farm[N];
    int y[N]; // observed counts
}
parameters {
    vector[K] beta;
    real reciprocal_phi;
    real mu_theta;
    real<lower=0> tau_theta;
    vector[M] theta;
}
transformed parameters {
    vector[N] eta;
    real phi;
    real<lower=0> sigma_theta;
    phi = 1. / reciprocal_phi;
    sigma_theta = 1/sqrt(tau_theta);
     //eta = area_log + X * beta + theta[j_farm];
    for (n in 1:N){
        eta[n] =  area_log[n] + X[n] * beta + theta[j_farm[n]];
    }
}

model {
    reciprocal_phi ~ cauchy(0., 5);
    beta[1] ~ normal(0,1.5);
    beta[2:K] ~ normal(0,1);
    mu_theta ~ normal(0, 1);
    tau_theta ~ gamma(0.1, 0.1);
    theta ~ normal(mu_theta, sigma_theta);
    for (n in 1:N){
         
         y[n] ~ neg_binomial_2_log(eta[n], phi);
    }
}

generated quantities{
    //vector[N] yrep;  //replicates
    vector[N] mu;
    vector[N] log_lik;
    mu = exp(eta);

    // Generate replicated data using vectorized _rng form
    for (n in 1:N) {
         log_lik[n] = neg_binomial_2_log_lpmf(y[n] | eta[n], phi);   
         //yrep[n] = neg_binomial_2_rng(mu[n], phi);
    }
}
