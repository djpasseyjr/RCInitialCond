import pickle as pkl

SOFTROBO_PRIOR = {
    'delta': 0.3736117214,
    'gamma': 18.66636932,
    'mean_degree': 1.7242465519999999,
    'ridge_alpha': 1.268554237,
    'sigma': 0.3125062064,
    'spect_rad': 0.8922393143999999,
}

THOMAS_PRIOR = {
    'gamma': 12.6,
    'mean_degree': 2.2,
    'ridge_alpha': 0.0005,
    'sigma': 1.5,
    'spect_rad': 12.0
}

LORENZ_PRIOR = {
    'gamma': 5.632587,
    'mean_degree': 0.21,
    'ridge_alpha': 2e-07,
    'sigma': 0.078,
    'spect_rad': 14.6
}

ROSSLER_PRIOR = {
    'gamma': 19.1,
    'mean_degree': 2.0,
    'ridge_alpha': 6e-07,
    'sigma': 0.063,
    'spect_rad': 8.472
}

with open("thomas_prior.pkl", "wb") as file:
    pkl.dump(THOMAS_PRIOR, file)

with open("lorenz_prior.pkl", "wb") as file:
    pkl.dump(LORENZ_PRIOR, file)

with open("rossler_prior.pkl", "wb") as file:
    pkl.dump(ROSSLER_PRIOR, file)

with open("softrobo_prior.pkl", "wb") as file:
    pkl.dump(SOFTROBO_PRIOR, file)
