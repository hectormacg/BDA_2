
data {
  int<lower=1> N;
  int<lower=0> num_eggs[N];
  vector[N] log_area;
  int<lower=0,upper=1> sprayed[N];
  int<lower=0,upper=1> lead[N];
  int<lower=1> J;                // number of farms
  int<lower=1,upper=J> farm_id[N];  // farm index for each observation
}

parameters {
  real beta_sprayed;
  real beta_lead;
  real beta_interaction;

  real mu_theta;               // hyper-mean for random effect
  real<lower=0> sigma_theta;   // hyper-sd for random effect
  vector[J] theta_raw;         // standardized random effects

  real<lower=0> phi;           // overdispersion
}

transformed parameters {
  vector[J] theta = mu_theta + sigma_theta * theta_raw;
}

model {
  // Priors
  beta_sprayed ~ normal(0, 2);
  beta_lead ~ normal(0, 2);
  beta_interaction ~ normal(0, 2);
  mu_theta ~ normal(0, 1);
  sigma_theta ~ exponential(1);
  theta_raw ~ normal(0, 1);
  phi ~ gamma(2, 0.1);

  // Likelihood
  for (n in 1:N) {
    num_eggs[n] ~ neg_binomial_2_log(
      log_area[n] +
      theta[farm_id[n]] +
      beta_sprayed * sprayed[n] +
      beta_lead * lead[n] +
      beta_interaction * sprayed[n] * lead[n],
      phi
    );
  }
}

generated quantities {
  vector[N] log_lik;
  int num_eggs_rep[N];  // posterior predictive samples

  for (n in 1:N) {
    real log_mu = log_area[n] +
      theta[farm_id[n]] +
      beta_sprayed * sprayed[n] +
      beta_lead * lead[n] +
      beta_interaction * sprayed[n] * lead[n];

    log_lik[n] = neg_binomial_2_log_lpmf(num_eggs[n] | log_mu, phi);
    num_eggs_rep[n] = neg_binomial_2_log_rng(log_mu, phi);  // simulated count
  }
}


