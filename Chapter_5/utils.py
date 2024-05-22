
import torch
import copy
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
    while relative_update > 1e-3:

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

    dict_sol_U = {'eps_total':eps_total_U,
                  'eps_elastic':eps_elastic_U,
                  'sigma':sigma_U,
                  'R':R_U,
                  'p':p_U
                  }

    dict_sol_V = {'eps_total': eps_total_V,
                  'eps_elastic': eps_elastic_V,
                  'sigma': sigma_V,
                  'R': R_V,
                  'p': p_V
                  }

    return dict_sol_U, dict_sol_V



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


def update_h(R_V, p_U, h_guess, lr):
    '''
    Update the value of h with a gradient descent step
    '''

    d_cre_p_dh = 0
    for i in range(0, len(R_V)):
            d_cre_p_dh += 1 / 2 * p_U[i] ** 2 - R_V[i] ** 2 / (2 * h_guess ** 2)

    h_guess = h_guess - lr * d_cre_p_dh

    return h_guess

def compute_loss_evolution(dict_sol_U, dict_sol_V, eps_total_true_noisy, E, h_guess):
    """
    Compute each component of the mCRE
    """

    dd_loss = torch.sum((dict_sol_U['eps_total'] - eps_total_true_noisy) ** 2).item()

    cre_elastic = torch.sum(1 / 2 * E * dict_sol_U['eps_elastic'] ** 2 - 1 / 2 * E * dict_sol_V['eps_elastic'] ** 2 +
                            dict_sol_V['sigma'] * (dict_sol_V['eps_elastic'] - dict_sol_U['eps_elastic'])).item()

    cre_plastic = torch.sum(1 / 2 * h_guess * dict_sol_U['p'] ** 2 + 1 / (2 * h_guess) * dict_sol_V['R'] ** 2 - dict_sol_V['R'] * dict_sol_U['p']).item()

    return dd_loss, cre_elastic, cre_plastic


def step_1_mcre_MDKF_evol(sol_t_minus_1, alpha, F_imposed_t, eps_total_true_t, E, R0, h):

    sol_t = copy.deepcopy(sol_t_minus_1)
    # Iteration 0 (initialisation)
    sol_t['U']['eps_elastic'] = (F_imposed_t + alpha * eps_total_true_t)/(E+alpha)
    sol_t['V']['eps_elastic'] = F_imposed_t / E

    sol_t['U']['eps_total'] = 0 + sol_t['U']['eps_elastic']
    sol_t['V']['eps_total'] = 0 + sol_t['V']['eps_elastic']


    relative_update = 1
    while relative_update > 1e-4:
        # Local step
        sol_t['U'] = integrate_evolution_for_MDKF(copy.deepcopy(sol_t['U']), sol_t_minus_1['U'], R0, h, E)
        sol_t['V'] = integrate_evolution_for_MDKF(copy.deepcopy(sol_t['V']), sol_t_minus_1['V'], R0, h, E)

        # Global step
        d_eps_elastic_U = (E * sol_t['U']['eps_elastic'] - F_imposed_t + alpha * (sol_t['U']['eps_total'] - eps_total_true_t)) / (E + alpha)
        d_eps_elastic_V = (E * sol_t['V']['eps_elastic'] - F_imposed_t) / E

        relative_update = max(abs(d_eps_elastic_U/sol_t['U']['eps_elastic']),abs(d_eps_elastic_V/sol_t['V']['eps_elastic']))
        sol_t['U']['eps_elastic'] -= d_eps_elastic_U
        sol_t['V']['eps_elastic'] -= d_eps_elastic_V

        sol_t['U']['eps_total'] = sol_t['U']['eps_plastic'] + sol_t['U']['eps_elastic']
        sol_t['V']['eps_total'] = sol_t['V']['eps_plastic'] + sol_t['V']['eps_elastic']
        sol_t['U']['sigma'] = E * sol_t['U']['eps_elastic']
        sol_t['V']['sigma'] = E * sol_t['V']['eps_elastic']

    return sol_t


def integrate_evolution_for_MDKF(sol_t, sol_t_minus_1, R0, h, E):

    delta_epsilon = sol_t['eps_total'] - sol_t_minus_1['eps_total']
    sigma = E * sol_t['eps_total']

    f = sigma - (sol_t['R'] + R0)

    # print(f'{i = }, {f = } ')

    if f < 0:
        sol_t['eps_elastic'] = sol_t_minus_1['eps_elastic'] + delta_epsilon
    else:

        deps_plastic = E * delta_epsilon / (E + h)

        sol_t['eps_plastic'] = sol_t_minus_1['eps_plastic'] + deps_plastic
        sol_t['eps_elastic'] = sol_t['eps_total'] - sol_t['eps_plastic']

        sol_t['p'] = sol_t['eps_plastic']

    sol_t['R'] = h * sol_t['p']
    return sol_t


def compute_grad_mcre_evolution_for_MDKF(p_U, p_V):
    '''
    Update the value of h with a gradient descent step
    '''

    d_cre_dh = 1 / 2 * (p_U ** 2 - p_V ** 2)

    return d_cre_dh

