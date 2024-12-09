# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:40:29 2023

@author: allis
"""

import matplotlib.pyplot as plt
import numpy as np


def growth(lambda_1, C_tot, lambda_2):
    # Logistic proliferation
    return lambda_1 * (1 - lambda_2 * C_tot)


def natural_release(rho, C):
    return rho * C


def RT_release(psi, C):
    return max(0, psi * C)


def A_natural_out(sigma, A):
    return -1 * sigma * A


def immune_death_T(iota, C, T):
    return -1 * iota * T * C


def T_natural_out(eta, T):
    return - eta * T


def tumor_volume(C, T, vol_C, vol_T):
    return C * vol_C + T * vol_T


def tum_kinetic(phi, tau_1, tau_2, t):
    if t <= tau_1:
        a = 0
    elif t > tau_2:
        a = 1
    else:
        a = (t - tau_1) / (tau_2 - tau_1)
    return -1 * a * phi


def immune_death_dePillis(C, T, p, q, s, p1, p_1, mi, vol_flag, time_flag, t, t_treat, delta_t, j=None):
    # if j!= None and j>650:
    #         print(j)
    #         print("Ta immune1", Ta_lym[:,j])
    m = 0
    # if j!= None and j>650:
    #         print("Ta immune1", Ta_lym[:,j])
    if vol_flag == 0 or time_flag == 0:
        pass
    else:
        if abs(t - t_treat) < delta_t / 2:
            p_1 = p_1 - mi * p_1 * delta_t + p1
            m = 1
        else:
            p_1 = p_1 - mi * p_1 * delta_t
    # if j!= None and j>650:
    #        print("Ta immune2", Ta_lym[:,j])
    if C == 0:
        f = 0
    else:
        # print((s + (T / C) ** q))
        #print('p', p_1)
        f = p * (1 + p_1) * (T / C) ** q / (s + (T / C) ** q)
        #print('f', f)
        if np.isnan(f):
            f = 0  # p * (1 + p_1)
        # if j <= 550:
        #     print('inside function', f)
    return f, m, p_1


def markov_TCP_analysis(im_death, prol, C, delta_t):
    cell_num = int(np.rint(C))
    # print("cell coubt", cell_num)
    f = prol * delta_t  # Birth probability
    g = im_death * delta_t  # Dead probability

    e = f + g

    # normalises the probabilities if the sum is more than 1
    if e > 1:
        f = f/e
        g = g/e
    # generates an array choosing whether the cells multiplie, die or stay constant
    # nested min max for probability of staying constant makes sure probability stays between 0 and 1
    nothingProbability = min(max(0, 1 - f - g), 1)

    # print("birth", type(f))
    # print("death", type(g))
    # print("nothing", type(nothingProbability))
    if isinstance(f, np.ndarray):
        f = f[0]
    if isinstance(g, np.ndarray):
        g = g[0]
    if isinstance(nothingProbability, np.ndarray):
        nothingProbability = nothingProbability[0]
    probabilities = np.array([f, g, nothingProbability], dtype=float).flatten()
    #print("probability", probabilities)
    # print(probabilities.ndim)
    cell_array = np.random.choice(np.array([2, 0, 1]).flatten(), size=(1, cell_num), replace=True, p=probabilities)
    # print("cell aray", cell_array)
# Create a list to store the randomly selected values
#     cell_array = []

    C = np.sum(np.array(cell_array))
    return C

def tumour_radius(C_tot, Ta_tum, V_C, V_T, u=1):
    volume = tumor_volume(C_tot, u*Ta_tum, V_C, V_T)
    return (3*volume/(4*np.pi))**(1/3)

def A_activate_T(a, b, K, h, c4, c_4, ni, t_treat, t, delta_t, T, A, vol_flag, time_flag, Ta, Tb, j=None, multiplier = 1):
    # working as expected
    m = 0
    newTa = Ta
    newTb = Tb
    if vol_flag == 0 or time_flag == 0:
        pass
    else:
        if abs(t - t_treat) < delta_t / 2:
            # c4 is anti-CTLA4 concentration for each injection
            # c_4 is anti CTLA4 concentration as function of time
            # increment c_4 by c4 if treatment occurs at timestep
            # print('treatment')
            #print('time', t)
            c_4 = c_4 - ni * c_4 * delta_t + c4
            m = 1
        else:
            c_4 = max(0, c_4 - ni * c_4 * delta_t)
        # if c_4 > 2.0:
            #print('current c4', c_4, 'at time', t)

    if c4 == 0:
        c_4 = 0
    T_ac = a * T * A            # active
    T_in = b / (1 + multiplier*c_4) * T * A  # inactive
# check if T or A become negative
    # T(t+1) < 0, K is initial count of T
    T0_flag = T + delta_t * (- 1*T_ac - T_in + h) < 0
    A0_flag = A + delta_t * (- 1*T_ac - T_in) < 0  # A(t+1) < 0

    if T0_flag or A0_flag:
        # if any of them are negative
        if T0_flag:
            #print('T0 flag')
            delta_t_1 = -1*T / (-1 * T_ac - T_in + h)  # T = 0
            T_1 = 0
            A_1 = max(0, A + delta_t_1 * (-1 * T_ac - T_in))

            newTa = Ta + delta_t_1 * T_ac
            newTb = Tb + delta_t_1 * T_in
        elif A0_flag:
            #print('A neg')
            delta_t_1 = -1*A / (- 1*T_ac - 1 * T_in)  # A = 0
            A_1 = 0
            #print('A no treatment', A + delta_t * (-1* T_ac - b*T*A))
            #print('A treated', A + delta_t * (-1* T_ac - b/(1+c_4)*T*A))
            T_1 = max(0, T + delta_t_1 * (-1 * T_ac - T_in + h))
            newTa = Ta + delta_t_1 * T_ac
            newTb = Tb + delta_t_1 * T_in
        else:
            #print('else')
            delta_t_2 = -1*A / (- 1*T_ac - T_in)  # A = 0
            delta_t_3 = -1*T / (-1 * T_ac - T_in + h)  # T = 0
            delta_t_1 = min(delta_t_2, delta_t_3)
            A_1 = 0
            T_1 = 0
            newTa = Ta + delta_t_1 * T_ac
            newTb = Tb + delta_t_1 * T_in
            delta_t_1 = delta_t_3

        delta_t_2 = delta_t - delta_t_1
        T = min(K, T_1 + delta_t_2 * h)
        A = A_1
    else:
        #print('fine')
        T = min(K, T + delta_t * (-1 * T_ac - T_in + h))
        #print('K', K)
        #print(T + delta_t * (-1 * T_ac - T_in + h))
        A = max(0, A + delta_t * (-1 * T_ac - T_in))

        newTa = Ta + delta_t * T_ac
        newTb = Tb + delta_t * T_in
        # print('Tb treat', newTb)
        # print('Tb no treat', Tb + delta_t*b*A*T)
        # print('Tb diff', newTb - (Tb + delta_t*b*A*T))
    return T, A, newTa, newTb, m, c_4


def cropArray(array, j):
    return array[:, 0:j]

def hypoxia(C_H, r, d_max, u, Ta_tum, vol_C, vol_T):
    hypoxia_step = ((4*np.pi*(r - d_max)**3 - 3*u*Ta_tum*vol_T)/(3*vol_C)) - C_H
    return hypoxia_step


def radioimmuno_response_model(param, delta_t, free, t_f1, t_f2, D, t_rad, t_treat_c4, t_treat_p1, LQL, activate_vd, use_Markov):
    # Extract all the parameters
    C_0 = param[0]
    lambda_1_N = param[1]
    lambda_1_H = param[2]
    alpha_C = param[3]
    beta_C = param[4]
    OER = param[5]
    phi = param[6]
    tau_dead_1 = param[7]
    tau_dead_2 = param[8]
    vol_C = param[9]
    A_0 = param[10]
    rho = param[11]
    psi = param[12]
    sigma = param[13]
    tau_1 = param[14]
    Ta_tum_0 = param[15]
    alpha_T = param[16]
    beta_T = param[17]
    tau_2 = param[18]
    eta = param[19]
    T_lym_0 = param[20]
    h = param[21]
    iota = param[22]
    vol_T = param[23]
    c4 = param[24]
    r = param[25]
    ni = param[26]
    a = param[27]
    b = (r - 1) * a
    p = param[28]
    q = param[29]
    s = param[30]
    d_max = param[31]
    u = param[32]
    recovery = param[33]
    lambda_2_N = param[34]
    lambda_2_H = param[35]
    beta_2 = param[36]
    p1 = param[37]
    mi = param[38]
    multiplier = param[39]
    c4_list = []
    p1_list = []
    #print('c4', c4)
    # Create discrete time array
    time = np.arange(0, t_f1 + t_f2 + 1 + delta_t, delta_t)
    m = len(time)

    # Select LQL or modified LQ
    if LQL == 1 and D[0] > 0:
        beta_C = min(beta_C, 2 * beta_C * (beta_2 *
                     D[0] - 1 + np.exp(-1*beta_2 * D[0])) / beta_2**2)
    else:
        beta_C = beta_C * (1 + beta_2 * np.sqrt(float(D[0])))

    # Activate vascular death if activate_vd is 1 and first dose > 15Gy
    if activate_vd == 1 and D[0] > 15:
        vascular_death = 0
    else:
        vascular_death = 1

    # Initialise variables
    C_N = np.zeros((1, m))     #normoxic tumour cells 
    C_H = np.zeros((1, m)) #hypoxic tumour cells
    C = np.zeros((1, m)) # Tumor cells (tumor))
    #C_no_treat = np.zeros((1,m))
    A = np.zeros((1, m))       # Antigens (activation zone))
    #A_no_treat = np.zeros((1,m))
    Ta_tum = np.zeros((1, m))  # Activated T-cells (tumor))
    #Ta_tum_no_treat = np.zeros((1, m))
    # T-cell available to be activated (activation zone))
    T_lym = np.zeros((1, m))
    #T_lym_no_treat = np.zeros((1,m))
    Ta_lym = np.zeros((1, m))  # Activated T-cells (activation zone))
    #Ta_lym_no_treat = np.zeros((1,m))
    Tb_lym = np.zeros((1, m))  # Inactivated T-cells (activation zone))
    #Tb_lym_no_treat = np.zeros((1,m))
    vol = np.zeros((1, m))     # Tumor volume

    # Delay index
    del_1 = max(0, round(tau_1/delta_t) - 1)
    del_2 = max(0, round(tau_2/delta_t) - 1)

    d = len(D)
    C_dead_N = np.zeros((1, d)) # Damaged normoxic tumor cells at each RT time
    C_dead_H = np.zeros((1, d)) # Damaged hypoxic tumor cells at each RT time  
   # C_dead = np.zeros((1, d))  # Damaged tumor cells at each RT time
    # Alive damaged tumor cells evolution for each RT dose
    M = np.zeros((d, m))
    M_N = np.zeros((d, m))
    M_H = np.zeros((d, m))
    # Total alive damaged tumor cells at each time step
    C_dam = np.zeros((1, m))
    C_dam_N = np.zeros((1, m))
    C_dam_H = np.zeros((1, m))
    C_tot_N =  np.zeros((1, m)) #total normoxic tumour cells
    C_tot_H =  np.zeros((1, m)) #total hypoxic tumour cells
    C_tot = np.zeros((1, m))            # Total alive tumor cells
    radius = np.zeros((1,m))
    radius_H = np.zeros((1,m))
    #C_tot_no_treat = C
    # Surviving fraction with LQ model parameters
    SF_N = np.zeros((1, d))    # Normoxic tumor cells surviving fraction
    SF_H = np.zeros((1, d))    # Hypoxic tumor cells surviving fraction
    SF_T = np.zeros((1, d))    # Lymphocytes surviving fraction

    # Variables initial value
    C[0] = C_0
    C_N[0] = C_0
    C_H[0] = 0
    # C_no_treat[0] = C_0
    A[0] = A_0
    # A_no_treat[0] = A_0
    Ta_tum[0] = Ta_tum_0
    # Ta_tum_no_treat[0] = Ta_tum_0
    T_lym[0] = T_lym_0
    # T_lym_no_treat[0] = T_lym_0
    C_tot[0] = C_0
    C_tot_N[0] = C_0
    C_tot_H[0] = 0
    # C_tot_no_treat[0] = C_0
    # Free behavior in time or volume
    free_flag = free[0]   # 1 for free behavior, 0 otherwise
    free_op = free[1]     # 1 for time, 0 for volume

    t_eq = -1
    vol_flag = 1          # 1 if initial volume was achieved, 0 otherwise
    time_flag = 1         # 1 if initial time was achieved, 0 otherwise

    # if free_flag == 1:
    # if free_op == 0:
    #vol_in = free[2]
    #vol_flag = 0
    # else:
    #t_in = free[2]
    #time_flag = 0
    # else:
    #m = t_f2/delta_t + 1

    p_1 = 0
    c_4 = 0
    tf_id = max(1, round(t_f2 / delta_t))
    k = 0                 # Radiation vector index
    ind_c4 = 0            # c4 treatment vector index
    ind_p1 = 0            # p1 treatment vector index
    # print(m)
    # print(del_1)
    # print(del_2)
# initialise all the arrays to have the initial conditions
    V_0 = tumor_volume(C_0, Ta_tum_0, vol_C, vol_T)
    r_0 = tumour_radius(C_0, Ta_tum_0, vol_C, vol_T)
    for i in range(max(del_1, del_2) + 1):
        C[:, i] = C_0
        C_N[:, i] = C_0
        #C_H[:, i] = 0.01*C_0
        C_H[:, i] = 0
        A[:, i] = A_0
        Ta_tum[:, i] = Ta_tum_0
        T_lym[:, i] = T_lym_0
        #Ta_lym[:,i] = 0
        Tb_lym[:, i] = 0
        C_tot[:, i] = C_0
        # C_no_treat[:, i] = C_0
        # A_no_treat[:, i] = A_0
        # Ta_tum_no_treat[:, i] = Ta_tum_0
        # T_lym_no_treat[:, i] = T_lym_0
        # Tb_lym_no_treat[:, i] = 0
        # C_tot_no_treat[:, i] = C_0
        vol[:, i] = V_0
        radius[:,i] = r_0
        c4_list.append(0)
        p1_list.append(0)

    # Algorithm
    j = i
    im_death_H = Ta_lym.copy()
    im_death_N = Ta_lym.copy()
    if isinstance(p1, np.ndarray):
        pass
    #im_death_no_treat = Ta_lym
    #print("max possible j", m-1)
    # print(C.shape)
    while j <= m-1:
        # if j>=1820:
        #     print(j)
        #     print("C as per start of main function", C[:,j])
        #     print("C total", C_tot[:,j])
        #     print("Tb_lym", Tb_lym[:,j])
        # growth rate of C due to natural tumor growth
        prol_N = growth(lambda_1_N, C_tot[:, j], lambda_2_N)[0]
        prol_H = growth(lambda_1_H, C_tot[:, j], lambda_2_H)[0]
        #print(prol)
        p_11 = p_1
        ind_p11 = ind_p1
        storeTalym = (Ta_lym[:, j][0],)
        #storeTalym_no_treat = (Ta_lym_no_treat[:, j][0],)
        # if j>650:
        #print("C", C[:,j])
        #print("Store Ta3", storeTalym)
        #print("Tb_lym", Tb_lym[:,j])
        # p1_flag = 0
        # if vol_flag == 0 or time_flag == 0:
        #     pass
        # else:
        #     if abs(time[j+1] - t_treat_p1[ind_p1]) < delta_t / 2:
        #         p_1 = p_1 - mi * p_1 * delta_t + p1
        #         p1_flag = 1
        #     else:
        #         p_1 = p_1 - mi * p_1 * delta_t
        # im_death[:,j] = p * (1 + p_1) * (Ta_tum[:,j] / C_tot[:,j]) ** q / (s + (Ta_tum[:,j] / C_tot[:,j]) ** q)
        # if np.isnan(im_death[:,j]):
        #   #stop division by 0 - if C tot is 0, C is 0 and so there is no change because of immune cell death
        #     im_death[:,j] = 0
        # print(t_treat_p1)
        # print(C_tot.shape)
        # print(Ta_tum.shape)
        # print(im_death.shape)
        # print(time)
        # print(t_treat_p1)
        # if D[0] != 0 and abs(time[j+1] - t_rad[k]) <= delta_t/2:
        #     print('start', C[:,j])
        if C_tot_H[:,j] == 0:
            if isinstance(p1, np.ndarray):
                [im_death_N[:, j], p1_flag, p_1] = immune_death_dePillis(C_tot_N[:, j], Ta_tum[:, j], p, q, s, p1[ind_p1], p_1, mi, vol_flag, time_flag, time[j+1], t_treat_p1[ind_p1], delta_t, j)
            else:
                [im_death_N[:, j], p1_flag, p_1] = immune_death_dePillis(C_tot_N[:, j], Ta_tum[:, j], p, q, s, p1, p_1, mi, vol_flag, time_flag, time[j+1], t_treat_p1[ind_p1], delta_t, j)
            immune_N = (im_death_N[:, j][0],)
        else:
            if isinstance(p1, np.ndarray):
                [im_death_N[:, j], p1_flag, p_1] = immune_death_dePillis(C_tot_N[:, j], (1-u)*Ta_tum[:, j], p, q, s, p1[ind_p1], p_1, mi, vol_flag, time_flag, time[j+1], t_treat_p1[ind_p1], delta_t, j)

            else:
                [im_death_N[:, j], p1_flag, p_1] = immune_death_dePillis(C_tot_N[:, j], (1 - u) * Ta_tum[:, j], p, q, s,
                                                                     p1, p_1, mi, vol_flag, time_flag,
                                                                     time[j + 1], t_treat_p1[ind_p1], delta_t, j)
            immune_N = (im_death_N[:, j][0],)
        if C_tot_H[:,j] != 0:
            if isinstance(p1, np.ndarray):
                immune_H_res = immune_death_dePillis(C_tot_H[:, j], u*Ta_tum[:, j], p, q, s, p1[ind_p1], p_1, mi, vol_flag, time_flag, time[j+1], t_treat_p1[ind_p1], delta_t, j)
            else:
                immune_H_res = immune_death_dePillis(C_tot_H[:, j], u*Ta_tum[:, j], p, q, s, p1, p_1, mi, vol_flag, time_flag, time[j+1], t_treat_p1[ind_p1], delta_t, j)
            #print(immune_H)
            im_death_H[:, j] = immune_H_res[0]
        else:
            #[im_death_H[:, j], p1_flag, p_1] = immune_death_dePillis(C_tot[:, j], u*Ta_tum[:, j], p, q, s, p1, p_1, mi, vol_flag, time_flag, time[j+1], t_treat_p1[ind_p1], delta_t, j) 
            im_death_H[:, j] = 0
        immune_H = (im_death_H[:, j][0],)
        # if j <=550:
        #     print('next call')
        #[im_death_no_treat[:,j], x,y] = immune_death_dePillis(C_tot_no_treat[:, j], Ta_tum_no_treat[:, j], p, q, s, p1, p_1, mi, vol_flag, time_flag, time[j+1], t_treat_p1[ind_p1], delta_t, j)
        
        #immune_no_treat = (im_death_no_treat[:, j][0],)
        # if j <= 550:
        #     print('immune cell death', immune)
        #     print('immune cell death no treat', immune_no_treat)
        #     print('diff', immune[0] - immune_no_treat[0])
        #     print('treat', p * (1 + p_1) * (Ta_tum[:,j] / C_tot[:,j]) ** q / (s + (Ta_tum[:,j] / C_tot[:,j]) ** q))
        #     print('no treat', p *  (1 + p_1) * (Ta_tum_no_treat[:,j] / C_tot_no_treat[:,j]) ** q / (s + (Ta_tum_no_treat[:,j] / C_tot_no_treat[:,j]) ** q))
        Ta_lym[:, j] = storeTalym[0]
        #Ta_lym_no_treat[:,j] = storeTalym_no_treat[0]
        # if j>650:
        #print("Tb_lym", Tb_lym[:,j])
        #print("store Ta", storeTalym)
        if p1_flag == 1:
                #checks if the anti-PD-1 doses are given as an array or not. If not an array, it will be just a single number (every fraction has equal dose)
            # if treatment was administered, increase the t treat p1 index by 1 (up to max of len(t_treat_p1) - 1 so no errors occur)
            ind_p1 = min(ind_p1 + 1, len(t_treat_p1) - 1)
        # Markov
        # if C[:, j] <= 1000 and use_Markov:
        #     C[:, j+1] = markov_TCP_analysis(im_death[:, j]
        #                                     [0], prol, C[:, j][0], delta_t)
        if C_N[:, j] == 0:
            newC_N = (0,)
            #newC_no_treat = (0,)
        elif C_N[:, j] <= 1500 and use_Markov:
            #print('n', C_N[:, j][0])
            newC_N = (markov_TCP_analysis(immune_N[0], prol_N, C_N[:, j][0], delta_t),)
        else:
            newC_N = (max(0, (C_N[:, j] + delta_t * (prol_N - immune_N[0]) * C_N[:, j])[0]),)
            #print('before', newC[0])
            #newC_no_treat = (max(0, C_no_treat[:, j] + delta_t * (prol - immune[0]) * C_no_treat[:, j]),)
        if C_H[:, j] == 0:
            newC_H = (0,)
            #newC_no_treat = (0,)
        elif C_H[:, j] <= 1500 and use_Markov:
            #print('h', C_H[:, j][0])
            newC_H = (markov_TCP_analysis(immune_H[0], prol_H, C_H[:, j][0], delta_t),)
        else:
            newC_H = (max(0, (C_H[:, j] + delta_t * (prol_H - immune_H[0]) * C_H[:, j])[0]),)
        if isinstance(c4, np.ndarray):
            T_lym[:, j + 1], A[:, j + 1], Ta_lym[:, j + 1], Tb_lym[:, j + 1], c4_flag, c_4 = A_activate_T(
                a, b, T_lym_0, h, c4[ind_c4], c_4, ni, t_treat_c4[ind_c4], time[j + 1], delta_t, T_lym[:, j], A[:, j],
                vol_flag, time_flag, Ta_lym[:, j], Tb_lym[:, j], j, multiplier)

        else:
            T_lym[:, j+1], A[:, j+1], Ta_lym[:, j+1], Tb_lym[:, j+1], c4_flag, c_4 = A_activate_T(
            a, b, T_lym_0, h, c4, c_4, ni, t_treat_c4[ind_c4], time[j+1], delta_t, T_lym[:, j], A[:, j], vol_flag, time_flag, Ta_lym[:, j], Tb_lym[:, j], j, multiplier)

        c4_list.append(c_4)
        #print(time[j])
        #print(c4_list[-1])
        p1_list.append(p_1)
        #T_lym_no_treat[:, j+1], A_no_treat[:, j+1], Ta_lym_no_treat[:, j+1], Tb_lym_no_treat[:, j+1], *_ = A_activate_T(
            #a, b, T_lym_0, h, 0, 0, ni, t_treat_c4[ind_c4 - 1], time[j+1], delta_t, T_lym_no_treat[:, j], A_no_treat[:, j], vol_flag, time_flag, Ta_lym_no_treat[:, j], Tb_lym_no_treat[:, j], j)

        if c4_flag == 1:
            ind_c4 = min(ind_c4 + 1, len(t_treat_c4) - 1)
        # get the rate at which antigen is released by tumor cells, delayed due to delay between antigen release and t cell activation
        nat_rel = natural_release(rho, C_tot[:, (j+1) - del_1])
        #nat_rel_no_treat = natural_release(rho, C_tot_no_treat[:, (j+1) - del_1])
        # calc how many cells died in the timestep, delayed due to delay between antigen release and t cell activation
        dead_step = M[:, j-del_1] - M[:, j+1-del_1]
        dead_step[dead_step < 0] = 0  # clear negative differences to be 0

        # sum up for total of all cells that died in timestep due to all RT doses
        dead_step = np.sum(dead_step)

        RT_rel = RT_release(psi, dead_step)
        # exponential decay of antigen
        A_nat_out = A_natural_out(sigma, A[:, j])
        #A_nat_out_no_treat = A_natural_out(sigma, A_no_treat[:, j])
        # getting next value of A by using small change formula
        A[:, j+1] = A[:, j+1] + delta_t * (nat_rel + A_nat_out + RT_rel*phi)
        #A_no_treat[:, j+1] = A_no_treat[:, j+1] + delta_t * (nat_rel_no_treat + A_nat_out_no_treat) + RT_rel
        #print('A treatment', A[:, j+1])
        # if  A[:, j] + delta_t * (- 1*a*A[:, j] - b/(1+c_4)*T_lym[:, j-1]*A[:, j]) < 0:
        #     print('negative treat')
        # if A[:, j] + delta_t * (- 1*a*A[:, j] - b*T_lym[:, j-1]*A[:, j]) < 0:
        #     print('negative no treat')
        #     print('A no treat', delta_t*(nat_rel + A_natural_out(sigma, A[:, j])) + RT_rel)
        # else:
        #     print('A no treatment', A[:, j] + delta_t * (- 1*a*A[:, j] - b * T_lym[:, j]*A[:, j] + nat_rel + A_natural_out(sigma, A[:, j])) + RT_rel)
        # T cell
        # interaction between tumor cell and Ta cells
        T_out = immune_death_T(iota, C[:, j] + C_dam[:, j], Ta_tum[:, j])
        #T_out_no_treat = immune_death_T(iota, C_no_treat[:, j] + C_dam[:, j], Ta_tum_no_treat[:, j])
        
        #print('T lym diff', T_lym[:,j+1] - T_lym_no_treat[:,j+1])
        #print('A diff', A[:,j+1] - A_no_treat[:,j+1]) #as expected, no ctla4 decreases A
        # if j >= 1981 and j <=1983:
        # print(j)
        #print("T out", T_out)
        # exponential natural elimination of Ta
        T_nat_out = T_natural_out(eta, Ta_tum[:, j])
        #T_nat_out_no_treat = T_natural_out(eta, Ta_tum_no_treat[:, j])
        
        #Ta_tum[:,j+1] = Ta_tum[:,j] + vascular_death * Ta_lym[:,(j+1) - del_2] + delta_t * (T_out + T_nat_out )
        Ta_tum[:, j+1] = Ta_tum[:, j] + vascular_death * delta_t * a * A[:, j - del_2]*T_lym[:, j - del_2] + delta_t * (T_out + T_nat_out)
        #Ta_tum_no_treat[:, j+1] = Ta_tum_no_treat[:, j] + vascular_death * delta_t * a * A_no_treat[:, j - del_2]*T_lym_no_treat[:, j - del_2] + delta_t * (T_out_no_treat + T_nat_out_no_treat)
        #print('Ta tum diff', Ta_tum[:,j+1] - Ta_tum_no_treat[:,j+1])
        # if j <= 550:
        #     print(time[j+1])
        #     #print(del_2*delta_t)
        #     print('t out diff', T_out - T_out_no_treat)
        #     print('t nat out diff', T_nat_out - T_nat_out_no_treat)
        #     print('Ta tum increase', delta_t * a * A[:, j - del_2]*T_lym[:, j - del_2])
        #     print('Ta tum decrease', delta_t * (T_out + T_nat_out))
        #     print('Ta tum step', vascular_death * delta_t * a * A[:, j - del_2]*T_lym[:, j - del_2] + delta_t * (T_out + T_nat_out))
        #     print('Ta tum increase no treat', delta_t * a * A_no_treat[:, j - del_2]*T_lym_no_treat[:, j - del_2])
        #     print('Ta tum decrease no treat', delta_t * (T_out_no_treat + T_nat_out_no_treat))
        #     print('Ta tum step no treat', vascular_death * delta_t * a * A_no_treat[:, j - del_2]*T_lym_no_treat[:, j - del_2] + delta_t * (T_out_no_treat + T_nat_out_no_treat))
        #     print('A diff', A[:,j+1] - A_no_treat[:,j+1])
        #     print('Ta tum diff', Ta_tum[:,j+1] - Ta_tum_no_treat[:,j+1])
        #     print()
        # if j>650:
        #     print("Tb_lym", Tb_lym[:,j])
        #print("C", C[:,j])
        #print("Ta as per middle of main function", Ta_lym[:,j])
        #print("Ta as per middle of main function", Ta_lym[:,j+1])
        if (time[j+1] > t_eq and activate_vd == 1 and D[0] >= 15):
            vascular_death = min(1, recovery * (time[j+1 - t_eq]))

        # if vol_flag == 1 and time_flag == 1 and D[0] != 0 and abs(time[j+1] - t_rad[k]) <= delta_t/2:
        C_dam_N_new = (0,)
        C_dam_H_new = (0,)    
        if D[0] != 0 and abs(time[j+1] - t_rad[k]) <= delta_t/2:
            # print("RT")
            # print(j)
            # print(time[j])
            # calculaate survival fractions of cancer cells and T cells
            SF_N[:, k] = np.exp(-1 * alpha_C * D[k] - beta_C * D[k] ** 2)
            SF_H[:, k] = np.exp((-1 * alpha_C * OER* D[k] - beta_C * D[k] ** 2)/(OER**2))
            #print('SF N', SF_N[:,k])
            #print('SF H', SF_H[:,k])
            #print('C', C[:,j])
            SF_T[:, k] = np.exp(-1 * alpha_T * D[k] - beta_T * D[k] ** 2)
            # updates cancer cell count by killing off (1-SFC)*C of the cancer cells
           # print('before', newC[0])
            C_remain_N = newC_N[0]*SF_N[:,k][0]
            C_dead_N[:, k] = newC_N[0] - C_remain_N
            C_remain_H = newC_H[0]*SF_H[:,k][0]
            C_dead_H[:, k] = newC_H[0] - C_remain_H
           #print(C[:,j])
            # print(SF_C[:,k])
            # print("before RT kill", C[:, j+1])
            # C[:,j+1] = C[:,j+1] - C_dead[:,k]
            
            newC_N = (C_remain_N,)
            newC_H = (C_remain_H,)
            #print('after treat', newC[0])
            #print('treat', newC[0])
            # print(C[:, j])
            # print(C[0][500:-1])
            # print(Ta_tum[:,j])
            # print("before RT kill", Ta_tum[:, j+1])
            Ta_tum[:, j+1] = Ta_tum[:, j+1] - (1 - SF_T[:, k]) * Ta_tum[:, j+1]
            # print(Ta_tum[:, j+1])
            # print(Ta_tum[0][500:-1])
            for ii in range(d):
              # C_kin is the -omega function (natural clearing), im_death_d is immune cell death
                C_kin_N = tum_kinetic(phi, tau_dead_1, tau_dead_2, time[j+1] - t_rad[ii])
                if isinstance(p1, np.ndarray):

                    im_death_dN = immune_death_dePillis(C_tot_N[:, j], (1-u)*Ta_tum[:, j], p, q, s, p1[ind_p11], p_11, mi, vol_flag, time_flag, time[j+1], t_treat_p1[ind_p11], delta_t)[0]
                else:
                    im_death_dN = immune_death_dePillis(C_tot_N[:, j], (1 - u) * Ta_tum[:, j], p, q, s, p1, p_11, mi,
                                          vol_flag, time_flag, time[j + 1], t_treat_p1[ind_p11], delta_t)[0]
                M_N[ii, j+1] = max(0, M_N[ii, j] + delta_t *(C_kin_N - im_death_dN) * M_N[ii, j])
                C_kin_H = tum_kinetic(phi, tau_dead_1, tau_dead_2, time[j+1] - t_rad[ii])
                if C_tot_H[:, j+1] == 0:
                    im_death_dH = 0
                else:
                    if isinstance(p1, np.ndarray):
                        im_death_dH = immune_death_dePillis(C_tot_H[:, j], u*Ta_tum[:, j], p, q, s, p1[ind_p11], p_11, mi, vol_flag, time_flag, time[j+1], t_treat_p1[ind_p11], delta_t)[0]
                    else:
                        im_death_dH = immune_death_dePillis(C_tot_H[:, j], u * Ta_tum[:, j], p, q, s, p1, p_11, mi, vol_flag,
                                              time_flag, time[j + 1], t_treat_p1[ind_p11], delta_t)[0]
                M_H[ii, j+1] = max(0, M_H[ii, j] + delta_t *(C_kin_H - im_death_dH) * M_H[ii, j])               

            M_N[k, j+1] = M_N[k, j+1] + C_dead_N[:, k]
            M_H[k, j+1] = M_H[k, j+1] + C_dead_H[:, k]
        # The sum of the columns of M is the total damaged tumor cells that
        # are going to die in each time step
            C_dam_N_new = (np.sum(M_N[:, j+1]),)
            C_dam_H_new = (np.sum(M_H[:, j+1]),)
            #print(C_dam_new)
            k = min(k + 1, len(t_rad) - 1)
            
        # elif vol_flag == 1 and time_flag == 1 and D[0] != 0:
        elif D[0] != 0:
            for ii in range(d):
              # C_kin is the -omega function (natural clearing), im_death_d is immune cell death
                C_kin_N = tum_kinetic(phi, tau_dead_1, tau_dead_2, time[j+1] - t_rad[ii])
                if isinstance(p1, np.ndarray):
                    im_death_dN = immune_death_dePillis(C_tot_N[:, j], (1 - u) * Ta_tum[:, j], p, q, s, p1[ind_p11], p_11, mi, vol_flag,
                                      time_flag, time[j + 1], t_treat_p1[ind_p11], delta_t)[0]
                else:
                    im_death_dN = immune_death_dePillis(C_tot_N[:, j], (1 - u) * Ta_tum[:, j], p, q, s, p1, p_11, mi,
                                                    vol_flag, time_flag, time[j + 1], t_treat_p1[ind_p11], delta_t)[0]
                
                M_N[ii, j+1] = max(0, M_N[ii, j] + delta_t *(C_kin_N - im_death_dN) * M_N[ii, j])
                C_kin_H = tum_kinetic(phi, tau_dead_1, tau_dead_2, time[j+1] - t_rad[ii])
                if C_tot_H[:, j+1] == 0:
                    im_death_dH = 0
                else:
                    if isinstance(p1, np.ndarray):
                        im_death_dH = immune_death_dePillis(C_tot_H[:, j], u * Ta_tum[:, j], p, q, s, p1[ind_p11], p_11, mi, vol_flag,
                                              time_flag, time[j + 1], t_treat_p1[ind_p11], delta_t)[0]
                    else:
                        im_death_dH = immune_death_dePillis(C_tot_H[:, j], u * Ta_tum[:, j], p, q, s, p1, p_11, mi, vol_flag,
                                              time_flag, time[j + 1], t_treat_p1[ind_p11], delta_t)[0]
                M_H[ii, j+1] = max(0, M_H[ii, j] + delta_t *(C_kin_H - im_death_dH) * M_H[ii, j])
            # The sum of the columns of M is the total damaged tumor cells that
        # are going to die in each time step
            C_dam_N_new = (np.sum(M_N[:, j+1]),)
            C_dam_H_new = (np.sum(M_H[:, j+1]),)
            #print(C_dam_new)
        # print(j)
        # print(Ta_tum[:,j+1])
        # get rid of negative values
        
        
        # if C_no_treat[:, j+1] < 0:
        #     C_no_treat[:, j+1] = 0
        # if A_no_treat[:, j+1] < 0:
        #     A_no_treat[:, j+1] = 0
        # if Ta_tum_no_treat[:, j+1] < 0:
        #     Ta_tum_no_treat[:, j+1] = 0
        # update total count of cancer cells (damaged and healthy)

        # if j>650:
        #     print("Tb_lym", Tb_lym[:,j])
        #print("Ta as per later middle of main function", Ta_lym[:,j])
        #print("Ta as per later middle of main function", Ta_lym[:,j+1])
        if vol_flag != 1 and vol[j+1] >= vol_in:
            t_eq = time[j+1]
            t_rad = t_rad + t_eq
            t_treat_p1 = t_treat_p1 + t_eq
            t_treat_c4 = t_treat_c4 + t_eq
            vol_flag = 1
        elif time_flag != 1 and time[j+1] >= t_in:
            m = j + 1 + tf_id
            t_rad = np.array(t_rad) + t_in
            t_treat_p1 = np.array(t_treat_p1) + t_in
            t_treat_c4 = np.array(t_treat_c4) + t_in
            time_flag = 1
        
        # C_no_treat[:, j+1] = newC_no_treat[0]
        if newC_N[0] < 0.5:
            newC_N = (0,)
        if newC_H[0] < 0.5:
            newC_H = (0,)        
        if C_dam_N_new[0] < 0.5:
            C_dam_N_new = (0,)
        if C_dam_H_new[0] < 0.5:
            C_dam_H_new = (0,)
        if A[:, j+1] < 0:
            A[:, j+1] = 0
        if Ta_tum[:, j+1] < 0:
            Ta_tum[:, j+1] = 0
        
        
        
        #print('before', newC[0])
        C_dam_N[:,j+1] = C_dam_N_new[0]
        C_dam_H[:,j+1] = C_dam_H_new[0]
        #C_tot_new = (newC[0] + C_dam_new[0],)
        
        #print(C_tot_new[0])
        # if D[0] != 0 and abs(time[j+1] - t_rad[k]) <= delta_t/2:
        #     print('C var', C[:,j+1])
       
        # if D[0] != 0 and abs(time[j+1] - t_rad[k]) <= delta_t/2:
        #     print('C var', C[:,j+1])
            # C_tot_no_treat[:, j+1] = newC_no_treat[0] + C_dam[:, j+1]
            
            # calculate tumour volume at the time step by V = C*VC + Ta*VT
        
        
        #print('c4', c_4)
        #print('A treatment', A[:, j+1])
        #print('A no treatment', A_no_treat[:, j+1])
        
        
        
        
        C_N[:, j+1] = newC_N[0]
        C_H[:, j+1] = newC_H[0]
        C[:, j+1] = newC_N[0] + newC_H[0]
        # if D[0] != 0 and abs(time[j+1] - t_rad[k]) <= delta_t/2:
        #     print('after assign', newC[0])
        #     print('C after assign', C[:,j+1])
        #print('before', C[:,j+1])
       
        C_tot[:,j+1] = C[:,j+1] + C_dam_N_new[0] + C_dam_H_new[0]
        C_tot_N[:,j+1] = C_dam_N_new[0] + newC_N[0]
        C_tot_H[:,j+1] = C_dam_H_new[0] + newC_H[0]
        #C[:, j+1] = newC[0]
        # if D[0] != 0 and abs(time[j+1] - t_rad[k]) <= delta_t/2:
        #     print('after C tot', newC[0])
        #     print('C after C tot', C[:,j+1])
        #print('after', C[:,j+1])
        vol[:, j+1] = tumor_volume(C_tot[:,j+1], Ta_tum[:, j+1], vol_C, vol_T)
        radius[:, j+1] = tumour_radius(C_tot[:,j+1], Ta_tum[:, j+1], vol_C, vol_T)
        radius_H[:, j+1] = tumour_radius(C_tot_H[:,j+1], Ta_tum[:, j+1], vol_C, vol_T, u)
        if radius[:,j+1] - radius_H[:,j+1] > d_max and C_tot_N[:,j+1] != 0.0:
            hypoxia_step = hypoxia(C_tot_H[:,j+1], radius[:,j+1], d_max, u, Ta_tum[:,j+1], vol_C, vol_T)
            #print('tumour cells become hypoxic', time[j+1])
            #print(radius[:,j+1] - radius_H[:,j+1] )
            #print(radius[:,j] - radius_H[:,j] )
            #print(C_tot_N[:,j+1]*vol_C + (1-u)*Ta_tum[:,j+1]*vol_T)
            #print(C_tot_N[:,j]*vol_C + (1-u)*Ta_tum[:,j]*vol_T)
            # print(newC_N[0])
            # print((hypoxia_step*C_N[:,j+1])/C_tot_N[:,j+1])            
            C_N[:, j+1] = max(0, newC_N[0] - (hypoxia_step*newC_N[0])/C_tot_N[:,j+1])
            # print(C_N[:, j+1])
            # print()
            C_H[:, j+1] = newC_H[0] + (hypoxia_step*newC_N[0])/C_tot_N[:,j+1]
            C_dam_N[:, j+1] = max(0, C_dam_N_new[0] - (hypoxia_step*C_dam_N_new[0])/C_tot_N[:,j+1])
            C_tot_N[:,j+1] = C_N[:, j+1] + C_dam_N[:, j+1]
            C_dam_H[:, j+1] = C_dam_H_new[0] + (hypoxia_step*C_dam_N_new[0])/C_tot_N[:,j+1]
            C_tot_H[:,j+1] = C_H[:, j+1] + C_dam_H[:, j+1]
        elif radius[:,j+1] - radius_H[:,j+1] < d_max and C_tot_H[:,j+1] != 0:
            #print('tumour cells reoxygenate', time[j+1])
            hypoxia_step = hypoxia(C_tot_H[:,j+1], radius[:,j+1], d_max, u, Ta_tum[:,j+1], vol_C, vol_T)
            #print(hypoxia_step)
            C_N[:, j+1] = newC_N[0] - (hypoxia_step*newC_H[0])/C_tot_H[:,j+1]
            C_H[:, j+1] = max(0, newC_H[0] + (hypoxia_step*newC_H[0])/C_tot_H[:,j+1])
            C_dam_N[:, j+1] =C_dam_N_new[0] - (hypoxia_step*C_dam_H_new[0])/C_tot_H[:,j+1]
            C_tot_N[:,j+1] = C_N[:, j+1] + C_dam_N[:, j+1]
            C_dam_H[:, j+1] = max(0, C_dam_H_new[0] + (hypoxia_step*C_dam_H_new[0])/C_tot_H[:,j+1])
            C_tot_H[:,j+1] = C_H[:, j+1] + C_dam_H[:, j+1]
        # elif C_tot_N[:,j+1] == 0:
        #     C_N[:, j+1] = 0
        #     C_dam_N[:, j+1] = 0

        radius_H[:, j+1] = tumour_radius(C_tot_H[:,j+1], Ta_tum[:, j+1], vol_C, vol_T, u)
        
        j = j + 1
        #print('A treatment', A[:, j])
        #print('A no treatment', A_no_treat[:, j])
        #print()
        if time[j-1] > t_f1 and vol_flag == 0:

            time = time[0:j]

            vol = cropArray(vol, j)
            C_tot = cropArray(C_tot, j)
            C = cropArray(C, j)
            C_dam = cropArray(C_dam, j)
            C_tot_N = cropArray(C_tot_N,j)
            C_tot_H = cropArray(C_tot_H,j)
            C_N = cropArray(C_N,j)
            C_H = cropArray(C_H,j)
            C_dam_N = cropArray(C_dam_N,j)
            C_dam_H = cropArray(C_dam_H,j)
            A = cropArray(A, j)
            Ta_tum = cropArray(Ta_tum, j)
            T_lym = cropArray(T_lym, j)
            Ta_lym = cropArray(Ta_lym, j)
            Tb_lym = cropArray(Tb_lym, j)
            radius = cropArray(radius, j)
            radius_H = cropArray(radius_H, j)
            p1_list = p1_list[0:j]
            # print(len(time))
            # print(len(p1_list))
            #print('last time', time[-1])
            # return vol, t_eq, time, C_tot, C, C_dam, A, Ta_tum, T_lym, Ta_lym, Tb_lym
        if time[j-1] > t_eq + t_f2 and vol_flag == 1:
            # print(len(time))
            # print(len(p1_list))
            time = time[0:j]
            vol = cropArray(vol, j)
            C_tot = cropArray(C_tot, j)
            C = cropArray(C, j)
            C_dam = cropArray(C_dam, j)
            C_tot_N = cropArray(C_tot_N,j)
            C_tot_H = cropArray(C_tot_H,j)
            C_N = cropArray(C_N,j)
            C_H = cropArray(C_H,j)
            C_dam_N = cropArray(C_dam_N,j)
            C_dam_H = cropArray(C_dam_H,j)
            A = cropArray(A, j)
            #A_no_treat = cropArray(A_no_treat, j)
            Ta_tum = cropArray(Ta_tum, j)
            T_lym = cropArray(T_lym, j)
            Ta_lym = cropArray(Ta_lym, j)
            Tb_lym = cropArray(Tb_lym, j)
            c4_list = c4_list[0:j]
            p1_list = p1_list[0:j]
            radius = cropArray(radius, j)
            radius_H = cropArray(radius_H, j)
            # print(len(time))
            # print(len(p1_list))
            #print('A treatment', A[:,j-1])
            #print('b', b)
            #print('b treat', b/(1+c_4))
            # if A[:,j-2] + delta_t * (- 1*a*A[:,j-2] - b*T_lym[:,j-2]*A[:,j-2]) < 0:
            #print('A no treat', delta_t*( nat_rel + A_natural_out(sigma, A[:,j-2])) + RT_rel)
            # else:
            #print('A no treatment', A[:,j-2]+ delta_t * (- 1*a*A[:,j-2] - b*T_lym[:,j-2]*A[:,j-2] + nat_rel + A_natural_out(sigma, A[:,j-2])) + RT_rel)
            #print('last time', time[-1])
            return vol, radius, radius_H, t_eq, time, C_tot, C, C_dam, C_tot_N, C_tot_H, C_N, C_H, C_dam_N, C_dam_H, A, Ta_tum, T_lym, Ta_lym, Tb_lym, c4_list, p1_list


# param = [500000.0, 0.4043764660304215, 0.3348825012978519, 0.0444792478133069, 0.051958556519556, 1.0, 1.5, 1.0000000000000004e-06, 0.0, 1.82061785504427e-21, 1.6370031265953986, 0.0492366376540959, 0.6416711700693031, 0.1617903030935988, 0.0416924514099231, 0.00416924514099231, 2.0,
#           0.0674559557659555, 299838.7440652358, 0.198909083172271, 9.211522519585746e-09, 8.284618352937945e-07, 4.722366482869665e-51, 5.0, 0.00001, 2e-5, 0.1379556739056123, 0.4073542114448485, 0.0481351408570356, 0.0099999999999999, 1.1404642118810832e-106, 0.0053955684115056, 0.2, 0.1]

# # param[25] = 10**-8
# # param[-2] = 0.2
#param = [500000.0, 0.3235011728243372, 0.0768587131869118, 0.0480366957418198, 0.4692307692307693, 1.0, 1.5, 1.000000000140727e-06, 0.0, 1.82061785504427e-21, 7.466844120599544e-13, 0.0408028409088006, 0.5133369360554425, 0.129432242474879, 0.0333539611279384, 0.0033353961127938, 2.0, 0.0674559557659557, 299838.1752100708, 0.397818166344542, 1.8423045039171492e-08, 6.627694682350356e-07, 0.0, 5.0, 2.147618075197616e+30, 3.049754060177746e+18, 0.1655468086867347, 0.3258833691558788, 0.0385081126856285, 0.0079999999999999, 2.2809284237621664e-106, 0.0101259381920267, 0.0, 0.0718136483092672, 2.245831895609047e+80]
#param = [500000.0, 0.4666148832758964, 0.1921467829672796, 0.0480366957418198, 0.4692307692307693, 1.0, 1.5, 1.0000000001123534e-06, 0.0, 1.82061785504427e-21, 1.866711030149886e-12, 0.074612219239879, 1.2833423401386062, 0.129432242474879, 0.0494100656486492, 0.0033353961127938, 2.0, 0.0674559557659551, 299837.34525097534, 0.1671682639657163, 7.369218015668599e-09, 1.656923670587589e-06, 0.2, 5.0, 2.147618075197616e+30, 3.049754060177746e+18, 0.1655468086867347, 0.3258833691558788, 0.0385081126856285, 0.0079999999999999, 9.123713695048666e-107, 0.0101259381920224, 0.4, 0.0718136483092672, 2.245831895609047e+80]


# # param[24] = 0.0005  # ctla4 decay rate
# # param[25] = 10**-5
# # param[-1] = 10**20
# import pandas as pd
# params = pd.read_csv('hypoxia_parameters.csv').values.tolist()
# #print(params)
# param = params[0]
#print(param[31])
#plt.plot(time, p_1)
#plt.show()
# param[37] = 0.8
# t_treat_p1 = np.array([13])
# vol_new, radius, radius_H, _, time, C_tot, C_new, C_dam, C_tot_N, C_tot_H, C_N, C_H, C_dam_N, C_dam_H, A, Ta_tum, T_lym, Ta_lym, Tb_lym, c_4, p_1 = radioimmuno_response_model(param, 0.05, free, t_f1, t_f2, D, t_rad, t_treat_c4, t_treat_p1, LQL, activate_vd, use_Markov)
#plt.plot(time, C[0] - C_new[0], label = 'pd')
#print(np.min(C[0]))
#print(np.min(C_new[0]))
# plt.plot(time, C[0])
# plt.plot(time, C_new[0], label='delay')
# plt.legend()
#plt.plot(time, max_depth, color='red')
#print(radius[0])
#print(C_H[0])
# for i in range(len(C_N[0])):
#     if C_N[0][i] == 0.0:
#         print(time[i])
#         break
# for i in range(len(radius[0])):
#     if i > 0 and (radius[0][i] >= param[31] and radius[0][i-1] <= param[31]):
#         print(time[i])
#         break
#plt.plot(time, radius_H[0])
#print(radius - radius_H)
#print(C_N[0])
#print(C_tot_N/C_tot)
#plt.plot(time, C_H[0], color='red')
#print(radius[0] - radius_H[0])
# # #print(Ta_tum)
# # c4=0
#param[24] = 0.3
# #param[-2] = 0
# #param[25] = 10**-1
# #param[-1] = 10**40
# D = [30]
#param[32] = 0.1
#param[31] = 0.9 #no hypoxia
#vol_new, radius_new, radius_H_new, _, time, C_tot_new, C_new, C_dam, C_tot_N, C_tot_H, C_N_new, C_H_new, C_dam_N, C_dam_H, A_new, Ta_tum_new, T_lym_new, Ta_lym_new, Tb_lym_new, c_4_new = radioimmuno_response_model(param, 0.05, free, t_f1, t_f2, D, t_rad, t_treat_c4, t_treat_p1, LQL, activate_vd, use_Markov)
#plt.plot(time, C_H_new[0], color='blue')
#plt.plot(time, vol_new[0] - vol[0], color='red')
#rint(C)
# plt.plot(time, C[0])
# def getTreatmentTime(times, C):
#   for i in range(len(C[0])):
#     if float(C[0][i]) == float(0):
#       print(times[i])
      
# getTreatmentTime(time, C)
# print(C[0])
#print(Ta_tum_new)
#print(Ta_tum - Ta_tum_new)
#print(c_4)
#print(c_4_new)
# plt.plot(time, vol[0], '--', color ='red', label ="volumes")
# plt.title('Volume vs Time for PD 1 and CTLA4 Treatment')
# plt.legend()
#plt.plot(time, vol[0], '--', color ='red', label ="treat")
#plt.plot(time, vol_new[0], '--', color ='blue', label ="control")
# plt.title('ctla4 concentration for PD 1 and CTLA4 Treatment')
#plt.legend()
# plt.plot(time, A[0], '--', color ='red', label ="treat")
# plt.plot(time, A_control[0], '--', color ='blue', label ="control")
# plt.legend()
# print(vol[0] - vol_new[0])
