
import torch

def integrate_evolution(eps_total, eps_plastic, eps_elastic, R, p, R0, h, E):
    """
    Itegration of evolution laws.
    """

    for t in range(0, len(eps_total)):
        # Loop on the time step

        if t > 0:

            delta_epsilon = eps_total[t] - eps_total[t - 1]
            sigma = E * eps_total[t]

            f = sigma - (R[t] + R0)
            # print(f'{i = }, {f = } ')

            if f < 0:
                eps_elastic[t] = eps_elastic[t - 1] + delta_epsilon
            else:
                deps_plastic = E * delta_epsilon / (E + h)
                eps_plastic[t] = eps_plastic[t - 1] + deps_plastic
                eps_elastic[t] = eps_total[t] - eps_plastic[t]

                p[t] = eps_plastic[t]

    R = h * p
    return eps_plastic, eps_elastic, R, p

def compute_loss(eps_total_U, eps_total_true_noisy, E, eps_elastic_U, eps_elastic_V, sigma_V, h_guess, p_U, R_V):
    """
    Compute each component of the mCRE
    """

    dd_loss = torch.sum((eps_total_U - eps_total_true_noisy) ** 2).item()

    cre_elastic = torch.sum(1 / 2 * E * eps_elastic_U ** 2 - 1 / 2 * E * eps_elastic_V ** 2 +
                            sigma_V * (eps_elastic_V - eps_elastic_U)).item()

    cre_plastic = torch.sum(1 / 2 * h_guess * p_U ** 2 + 1 / (2 * h_guess) * R_V ** 2 - R_V * p_U).item()

    return dd_loss, cre_elastic, cre_plastic


def step_1_mcre_latin(E, R0, h, F_imposed, eps_total_true, alpha):
    '''
    Compute the first step of the mCRE with a Latin inspired scheme.
    '''

    eps_plastic_U = torch.zeros_like(F_imposed)
    p_U, R_U = torch.zeros_like(F_imposed), torch.zeros_like(F_imposed)
    eps_plastic_V = torch.zeros_like(F_imposed)
    p_V, R_V = torch.zeros_like(F_imposed), torch.zeros_like(F_imposed)

    # Iteration 0 (elastic initialisation)
    eps_elastic_U = (F_imposed + alpha * eps_total_true)/(E+alpha)
    eps_elastic_V = F_imposed / E

    eps_total_U = eps_plastic_U + eps_elastic_U
    eps_total_V = eps_plastic_V + eps_elastic_V

    relative_update = 1
    while relative_update > 1e-5:

        # Local step (integration of evolution laws)
        eps_plastic_U, eps_elastic_U, R_U, p_U = integrate_evolution(eps_total_U, eps_plastic_U, eps_elastic_U, R_U,
                                                                     p_U, R0, h, E)
        eps_plastic_V, eps_elastic_V, R_V, p_V = integrate_evolution(eps_total_V, eps_plastic_V, eps_elastic_V, R_V,
                                                                     p_V, R0, h, E)

        # Global step
        d_eps_elastic_U = (E * eps_elastic_U - F_imposed + alpha * (eps_total_U - eps_total_true)) / (E + alpha)
        d_eps_elastic_V = (E * eps_elastic_V - F_imposed) / E

        relative_update = max(torch.max(torch.abs(d_eps_elastic_U[1:] / eps_elastic_U[1:])), torch.max(torch.abs(d_eps_elastic_V[1:] / eps_elastic_V[1:])))
        eps_elastic_U -= d_eps_elastic_U
        eps_elastic_V -= d_eps_elastic_V

        eps_total_U = eps_plastic_U + eps_elastic_U
        eps_total_V = eps_plastic_V + eps_elastic_V
        sigma_U = E * eps_elastic_U
        sigma_V = E * eps_elastic_V

    return eps_total_U, eps_elastic_U, eps_plastic_U, sigma_U, R_U, p_U, eps_total_V, eps_elastic_V, eps_plastic_V, sigma_V, R_V, p_V


def update_h(R_V, p_U, h_guess, lr):
    '''
    Update the value of h with a gradient descent step
    '''

    d_cre_p_dh = 0
    for i in range(0, len(R_V)):
        d_cre_p_dh += 1 / 2 * p_U[i] ** 2 - R_V[i] ** 2 / (2 * h_guess ** 2)

    h_guess = h_guess - lr * d_cre_p_dh

    return h_guess