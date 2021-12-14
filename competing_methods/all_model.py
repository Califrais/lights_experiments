import os
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Versions/4.0/Resources"
from rpy2 import robjects
from rpy2.robjects import pandas2ri, numpy2ri
from sklearn.preprocessing import LabelEncoder
import numpy as np
from lights.inference import prox_QNMCEM
from prettytable import PrettyTable
from lifelines.utils import concordance_index as c_index_score
from time import time
from lights.simulation import SimuJointLongitudinalSurvival
import pandas as pd
from scipy.stats import beta
def load_data(simu):
    if simu:
        n_long_features = 5
        n_time_indep_features = 10
        n_samples = 200
        simu = SimuJointLongitudinalSurvival(seed=123,
                                             n_long_features=n_long_features,
                                             n_samples=n_samples,
                                             n_time_indep_features=n_time_indep_features,
                                             sparsity=0.5)
        X, Y, T, delta, _ = simu.simulate()
        id = np.arange(n_samples)
        time_dep_feat = ['long_feature_%s' % (l + 1)
                         for l in range(n_long_features)]
        time_indep_feat = ['X_%s' % (l + 1)
                           for l in range(n_time_indep_features)]

        data_lights = pd.DataFrame(data=np.column_stack((id, T, delta, X, Y)),
                            columns=["id", "T_survival", "delta"] +
                                    time_indep_feat + time_dep_feat)
        df1 = pd.DataFrame(data=np.column_stack((id, T, delta, X)),
                           columns=["id", "T_survival", "delta"] +
                                   time_indep_feat)
        # generate t_max
        a = 2
        b = 5
        r = beta.rvs(a, b, size=n_samples)
        t_max = T * (1 - r)
        for i in range(n_samples):
            Y_i_ = []
            for l in range(n_long_features):
                Y_il = Y.loc[i][l]
                times_il = Y_il.index.values
                t_long_chosen = (times_il <= t_max[i])
                if not np.any(t_long_chosen):
                    t_max[i] = times_il[0]
                    t_long_chosen = (times_il <= t_max[i])
                times_il = times_il[t_long_chosen]
                y_il = Y_il.values.flatten()[t_long_chosen].tolist()
                n_il = len(times_il)
                tmp = data_lights.loc[i, time_dep_feat[l]]
                if tmp[tmp.index.values <= t_max[i]].empty:
                    data_lights[time_dep_feat[l]][i] = tmp[tmp.index.values == tmp.index.values[0]]
                    t_max[i] = tmp.index.values[0]
                else:
                    data_lights[time_dep_feat[l]][i] = tmp[tmp.index.values <= t_max[i]]
                Y_i_.append(y_il)

            Y_i = np.column_stack(
                (np.array([id[i]] * n_il), times_il, np.array([t_max[i]] * n_il), np.array(Y_i_).T))
            if i == 0:
                Y_ = Y_i
            else:
                Y_ = np.row_stack((Y_, Y_i))
        data_lights["T_max"] = t_max
        df2 = pd.DataFrame(data=Y_, columns=["id", "T_long", "T_max"] + time_dep_feat)
        data = pd.merge(df2, df1, on="id")

    else:
        # load PBC Seq
        robjects.r.source(os.getcwd() + "/competing_methods/load_PBC_Seq.R")
        time_dep_feat = ["serBilir", "SGOT", "albumin"]
        time_indep_feat = ["age", "drug", "sex"]
        data_R = robjects.r["load"]()
        # TODO: encoder and normalize
        data = pd.DataFrame(data_R).T
        data.columns = data_R.colnames
        data["drug"] = LabelEncoder().fit_transform(data["drug"])
        data["sex"] = LabelEncoder().fit_transform(data["sex"])
        id_list = np.unique(data["id"])
        n_samples = len(id_list)
        n_long_features = len(time_dep_feat)
        data_lights = data.drop_duplicates(subset=["id"])

        # generate t_max
        a = 2
        b = 5
        r = beta.rvs(a, b, size=n_samples)
        T = data_lights["T_survival"].values
        t_max = T * (1 - r)

        Y = []
        t_max_R = []
        for i in range(n_samples):
            tmp = data[(data["id"] == id_list[i]) & (data["T_long"] < t_max[i])]
            if tmp.empty:
                t_max[i] = data[(data["id"] == id_list[i])]["T_long"][0]
                n_i = 1
            else:
                n_i = tmp.shape[0]
            data = data[(data["id"] != id_list[i]) |
                    ((data["id"] == id_list[i]) & (data["T_long"] < t_max[i]))]
            y_i = []
            for l in range(n_long_features):
                Y_il = data[["T_long", time_dep_feat[l]]][
                    (data["id"] == id_list[i]) & (data['T_long'] < t_max[i])]
                # TODO: Add value of 1/365 (the first day of survey instead of 0)
                y_i += [pd.Series(Y_il[time_dep_feat[l]].values,
                                  index=Y_il["T_long"].values + 1 / 365)]
            Y.append(y_i)
            t_max_R += [t_max[i]] * n_i
        data_lights[time_dep_feat] = Y
        data_lights["T_max"] = t_max
        data["T_max"] = np.array(t_max_R).flatten()
    return (data, data_lights, time_dep_feat, time_indep_feat)

def extract_lights_feat(data, time_indep_feat, time_dep_feat):
    X = np.float_(data[time_indep_feat].values)
    Y = data[time_dep_feat]
    T = np.float_(data[["T_survival"]].values.flatten())
    delta = np.int_(data[["delta"]].values.flatten())

    return (X, Y, T, delta)

def extract_R_feat(data):
    data_id = data.drop_duplicates(subset=["id"])
    T = data_id[["T_survival"]].values.flatten()
    delta = data_id[["delta"]].values.flatten()
    with robjects.conversion.localconverter(robjects.default_converter +
                                            pandas2ri.converter +
                                            numpy2ri.converter):
        data_R = robjects.conversion.py2rpy(data)
        T_R = robjects.conversion.py2rpy(T)
        delta_R = robjects.conversion.py2rpy(delta)

    return (data_R, T_R, delta_R)

def all_model(n_runs = 1, simu=True):

    seed = 0
    test_size = .2
    data, data_lights, time_dep_feat, time_indep_feat = load_data(simu)
    t = PrettyTable(['Algos', 'C_index', 'time'])
    for i in range(n_runs):
        seed += 1
        id_list = data_lights["id"]
        nb_test_sample = int(test_size * len(id_list))
        np.random.seed(seed)
        id_test = np.random.choice(id_list, size=nb_test_sample, replace=False)
        data_lights_train = data_lights[data_lights.id.isin(id_test)]
        data_lights_test = data_lights[data_lights.id.isin(id_test)]
        X_lights_train, Y_lights_train, T_train, delta_train = \
            extract_lights_feat(data_lights_train, time_indep_feat, time_dep_feat)
        X_lights_test, Y_lights_test, T_test, delta_test = \
            extract_lights_feat(data_lights_test, time_indep_feat, time_dep_feat)

        data_train = data[data.id.isin(id_test)]
        data_test = data[data.id.isin(id_test)]
        data_R_train, T_R_train, delta_R_train = extract_R_feat(data_train)
        data_R_test, T_R_test, delta_R_test = extract_R_feat(data_test)

        # The penalized Cox model.
        robjects.r.source(os.getcwd() + "/competing_methods/CoxNet.R")
        X_R_train = robjects.r["Cox_get_long_feat"](data_R_train, time_dep_feat,
                                                  time_indep_feat)
        X_R_test = robjects.r["Cox_get_long_feat"](data_R_test, time_dep_feat,
                                                 time_indep_feat)
        best_lambda = robjects.r["Cox_cross_val"](X_R_train, T_R_train, delta_R_train)
        start = time()
        trained_CoxPH = robjects.r["Cox_fit"](X_R_train, T_R_train,
                                              delta_R_train, best_lambda)
        Cox_pred = robjects.r["Cox_score"](trained_CoxPH, X_R_test)
        Cox_marker = np.array(Cox_pred[:])
        Cox_c_index = c_index_score(T_test, Cox_marker, delta_test)
        Cox_c_index = max(Cox_c_index, 1 - Cox_c_index)
        t.add_row(["Cox", "%g" % Cox_c_index, "%.3f" % (time() - start)])

        # Multivariate joint latent class model.
        start = time()
        robjects.r.source(os.getcwd() + "/competing_methods/MJLCMM.R")
        trained_long_model, trained_mjlcmm = robjects.r["MJLCMM_fit"](data_R_train,
                                             robjects.StrVector(time_dep_feat),
                                             robjects.StrVector(time_indep_feat),
                                             alpha=2)
        MJLCMM_pred = robjects.r["MJLCMM_score"](trained_long_model,
                                                 trained_mjlcmm,
                                                 time_indep_feat, data_R_test)
        MJLCMM_marker = np.array(MJLCMM_pred.rx2('pprob')[2])
        MJLCMM_c_index = c_index_score(T_test, MJLCMM_marker, delta_test)
        MJLCMM_c_index = max(MJLCMM_c_index, 1 - MJLCMM_c_index)
        t.add_row(["MJLCMM", "%g" % MJLCMM_c_index, "%.3f" % (time() - start)])

        # Multivariate shared random effect model.
        start = time()
        robjects.r.source(os.getcwd() + "/competing_methods/JMBayes.R")
        trained_JMBayes = robjects.r["fit"](data_R_train,
                                            robjects.StrVector(time_dep_feat),
                                            robjects.StrVector(time_indep_feat))
        JMBayes_marker = np.array(robjects.r["score"](trained_JMBayes, data_R_test))
        JMBayes_c_index = c_index_score(T_test, JMBayes_marker, delta_test)
        JMBayes_c_index = max(JMBayes_c_index, 1 - JMBayes_c_index)
        t.add_row(["JMBayes", "%g" % JMBayes_c_index, "%.3f" % (time() - start)])

        # lights
        start = time()
        fixed_effect_time_order = 1
        learner = prox_QNMCEM(fixed_effect_time_order=fixed_effect_time_order,
                              max_iter=5, initialize=True, print_every=1,
                              compute_obj=True, simu=False,
                              asso_functions=["lp", "re"],
                              l_pen_SGL=0.02, eta_sp_gp_l1=.9, l_pen_EN=0.02)
        learner.fit(X_lights_train, Y_lights_train, T_train, delta_train)
        prediction_times = data_lights_test[["T_max"]].values.flatten()
        lights_marker = learner.predict_marker(X_lights_test, Y_lights_test, prediction_times)
        lights_c_index = c_index_score(T_test, lights_marker, delta_test)
        lights_c_index = max(lights_c_index, 1 - lights_c_index)
        t.add_row(["lights", "%g" % lights_c_index, "%.3f" % (time() - start)])

        print(t)





