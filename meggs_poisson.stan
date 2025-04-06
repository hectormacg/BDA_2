
data {
    int<lower=0> N; // number of observations
    int<lower=0> K; // number of predictors
    matrix[N, K] X;  // matrix design
    vector[N] area_log;
    int y[N]; // observed counts
}
parameters {
    vector[K] beta;
}

transformed parameters {
    real alpha;
    alpha = 0.;
}


model {
    beta[1] ~ normal(0,1.5);
    beta[2:K] ~ normal(0,1);
    y ~ poisson_log(area_log + X * beta);
}

generated quantities{
    //vector[N] yrep;  //replicates
    vector[N] log_lambda;
    vector[N] lambda; 
    real beta_3_4_sum;
    vector[N] log_lik;
    beta_3_4_sum=exp(beta[3] + beta[4]);

    // Calculate linear predictor (log rate parameter) efficiently
    log_lambda =  area_log + X * beta;
    lambda=exp(log_lambda);

    // Generate replicated data using vectorized _rng form
    for (n in 1:N) {
         log_lik[n] = poisson_lpmf(y[n] | lambda[n]);
         
    }
}
