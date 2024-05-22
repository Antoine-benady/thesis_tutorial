
import torch
import copy


def compute_loss(eps_total_true_noisy, E, eps_elastic_U, eps_elastic_V, sigma_V):
    """
    Compute each component of the mCRE
    """

    dd_loss = torch.sum((eps_elastic_U - eps_total_true_noisy) ** 2).item()

    cre_elastic = torch.sum(1 / 2 * E * eps_elastic_U ** 2 - 1 / 2 * E * eps_elastic_V ** 2 +
                            sigma_V * (eps_elastic_V - eps_elastic_U)).item()


    return dd_loss, cre_elastic


def step_1_mcre(E, F_imposed, eps_total_true, alpha):
    '''
    Compute the first step of the mCRE with a Latin inspired scheme.
    '''

    # Iteration 0 (elastic initialisation)
    eps_elastic_U = (F_imposed + alpha * eps_total_true)/(E+alpha)
    eps_elastic_V = F_imposed / E

    sigma_U = E * eps_elastic_U
    sigma_V = E * eps_elastic_V

    return eps_elastic_U, sigma_U, eps_elastic_V, sigma_V


def update_E(eps_elastic_U, eps_elastic_V, E_guess, lr):
    '''
    Update the value of h with a gradient descent step
    '''

    d_cre_dE = 0
    for j in range(0, len(eps_elastic_V)):
        d_cre_dE += 1 / 2 * (eps_elastic_U[j] ** 2 - eps_elastic_V[j] ** 2)

    E_guess = E_guess - lr * d_cre_dE

    return E_guess


def compute_grad_mcre(eps_elastic_U, eps_elastic_V):
    '''
    Update the value of h with a gradient descent step
    '''

    d_cre_dE = 1 / 2 * (eps_elastic_U ** 2 - eps_elastic_V ** 2)

    return d_cre_dE

