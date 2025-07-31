# %%
import os
import time

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from functools import partial
from multiprocessing import Pool

from scipy.optimize import brentq

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel as C, RBF, Matern
from sklearn.metrics import mean_squared_error, r2_score as r2
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict, cross_val_score

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

plt.rcParams['backend'] = 'tkagg'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times new roman'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.it'] = 'serif:italic'
plt.rcParams['mathtext.bf'] = 'serif:bold'
plt.rcParams['mathtext.fontset'] = 'custom'

module_path = os.path.realpath(__file__)
base_dir = os.path.dirname(module_path)
print("base_dir:", base_dir)

GRID_DENSITY = 100


def time_counter(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} execution time: {end - start:.6f} seconds")
        return result
    return wrapper


def _compute_intersections_for_stroke(args):
    stroke, model, poly_func, f_min, f_max, num_points = args

    intersections = []

    def diff_func(flowrate):
        head_gpr = model.predict(np.array([[flowrate, stroke]])).item()
        head_sys = poly_func(flowrate)
        return head_gpr - head_sys

    f_root = brentq(diff_func, f_min, f_max)
    h_root = model.predict(np.array([[f_root, stroke]])).item()
    intersections.append((f_root, stroke, h_root))

    return intersections

# %%
class FluidMachinery:
    def __init__(self, name, category, control_param='Stroke', auto_draw=True):
        self._machine_name = name
        self._control = control_param
        self._rated = (1185, 2560, 0)[category]
        self._rho = (1.225, 997, 0)[category]
        self._g = 9.81

        module_path = os.path.realpath(__file__)
        self.base_dir = os.path.dirname(module_path)
        self.save_npy()
        self.df = self.load_npy
        self.save_system_npy()
        df_system = self.load_system_npy

        self.flowrate = self.df['Flowrate']
        self.dP = self.df['dP']
        if 'Head' in self.df.dtype.names:
            self.Head = self.df['Head']
        else:
            self.pressure = self.df['Pressure']
        self.control = self.df[self._control]
        self.power = self.df['Power']
        self.efficiency = self.df['Efficiency']
        self.flowrate_system = df_system['Flowrate']
        self.Head_system = df_system['Head']


    def settings(self):
        print(f"{self._machine_name}가 기본 장비입니다")


    def save_npy(self):
        data_dir = os.path.join(self.base_dir, 'Data')
        csv_path = os.path.join(data_dir, f'{self._machine_name}_data.csv')
        npy_path = os.path.join(data_dir, f'{self._machine_name}_data.npy')
        df = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding='utf-8-sig')
        np.save(npy_path, df)


    def save_system_npy(self):
        data_dir = os.path.join(self.base_dir, 'Data')
        csv_path = os.path.join(data_dir, f'{self._machine_name}_system.csv')
        npy_path = os.path.join(data_dir, f'{self._machine_name}_system.npy')
        df_system = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding='utf-8-sig')
        np.save(npy_path, df_system)


    @property
    def load_npy(self):
        data_dir = os.path.join(self.base_dir, 'Data')
        npy_path = os.path.join(data_dir, f'{self._machine_name}_data.npy')
        df = np.load(npy_path, allow_pickle=True)
        return df


    @property
    def load_system_npy(self):
        data_dir = os.path.join(self.base_dir, 'Data')
        npy_path = os.path.join(data_dir, f'{self._machine_name}_system.npy')
        df_system = np.load(npy_path, allow_pickle=True)
        return df_system


    def cal_efficiency(self):
        pass

class Blower(FluidMachinery):
    def __init__(self, name):
        self._machine_name = name


    def settings(self):
        print(f"{self._machine_name}는 압축기입니다")


class Pump(FluidMachinery):
    def __init__(self, name, category, auto_draw=False):
        super().__init__(name, category=category, auto_draw=auto_draw)


    def fit_poly_through_zero(self, x, y):
        X = np.column_stack((x ** 2, x))
        a, b = np.linalg.inv(X.T @ X) @ X.T @ y
        return a, b


    def _poly_func(self, x_val, a, b):
        return a * x_val ** 2 + b * x_val


    def _fit_system_curve(self):
        q_system = self.flowrate_system
        pressure_system = self.Head_system
        a, b = self.fit_poly_through_zero(q_system, pressure_system)
        return a, b


    def cal_efficiency(self, q, h, p):
        eff = 100 * ((q / 60) * (h * self._rho * self._g) / 1000) / p
        return eff


    # def _fit_surface(self, x1, x2, y, cv_splits=5, verbose=True, stratify_by=None):
    #     if stratify_by is None:
    #         stratify_by = self.control
    #
    #     X = np.c_[x1, x2]
    #
    #     unique_vals, stratify_labels = np.unique(stratify_by, return_inverse=True)
    #     skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    #     scores = []
    #
    #     rbf_constant = 20 ##
    #     matern_constant = 100 ##
    #     rbf_flowrate_scale = 10000 #
    #     rbf_stroke_scale = 100 #
    #     rbf_flowrate_scale_bound = (10000, 15000) ##
    #     rbf_stroke_scale_bound = (100, 1000) ##
    #     matern_flowrate_scale = 100 #
    #     matern_storke_scale = 3000 #
    #     matern_flowrate_scale_bound = (100, 1000) ##
    #     matern_storke_scale_bound =  (3000, 5000) ##
    #     noise_level = 1e-4 ##
    #     noise_level_bound = (1e-6, 1e-4) ##
    #
    #     kernel = (
    #               C(rbf_constant, 'fixed')
    #               * RBF(length_scale=(rbf_flowrate_scale, rbf_stroke_scale),
    #                     length_scale_bounds=[rbf_flowrate_scale_bound, rbf_stroke_scale_bound]) +
    #               C(matern_constant, 'fixed')
    #               * Matern(length_scale=(matern_flowrate_scale, matern_storke_scale),
    #                        length_scale_bounds=[matern_flowrate_scale_bound, matern_storke_scale_bound],
    #                        nu=1.5)
    #              ) + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bound)
    #
    #     best_score = -np.inf
    #     best_kernel = None
    #
    #     for fold, (train_idx, test_idx) in enumerate(skf.split(X, stratify_labels)):
    #         X_train, X_test = X[train_idx], X[test_idx]
    #         y_train, y_test = y[train_idx], y[test_idx]
    #
    #         unique_vals, counts = np.unique(stratify_by[test_idx], return_counts=True)
    #         for label, count in zip(unique_vals, counts):
    #             print(f"Stratify label {label}: {count} samples in validation set")
    #
    #         model = GaussianProcessRegressor(
    #             kernel=kernel,
    #             alpha=1e-4,
    #             optimizer='fmin_l_bfgs_b'
    #         )
    #         model.fit(X_train, y_train.ravel())
    #         y_pred = model.predict(X_test)
    #         fold_r2 = r2(y_test, y_pred)
    #         scores.append(fold_r2)
    #
    #         print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    #         print(f"  R² Score: {fold_r2:.4f}")
    #         print(f"  Kernel: {model.kernel_}")
    #
    #         if fold_r2 > best_score:
    #             best_score = fold_r2
    #             best_kernel = model.kernel_
    #
    #     mean_cv_score = np.mean(scores)
    #     print(f"\nMean cross-validated R²: {mean_cv_score:.4f}")
    #
    #     print("\nFinal model training on full data with best kernel from CV:")
    #     final_model = GaussianProcessRegressor(
    #         kernel=best_kernel,
    #         alpha=1e-8,
    #         optimizer=None
    #     )
    #     final_model.fit(X, y.ravel())
    #     y_pred = final_model.predict(X)
    #     final_r2_score = r2(y, y_pred)
    #
    #     print(f"  Final model Kernel: {final_model.kernel_}")
    #     print(f"  Final model R² score on full data: {final_r2_score:.4f}")
    #
    #     return final_model, y_pred

    def _fit_surface(self, x1, x2, y, cv_splits=5, verbose=True, use_bayes=True, stratify_by=None):
        if stratify_by is None:
            stratify_by = self.control

        X = np.c_[x1, x2]
        unique_vals, stratify_labels = np.unique(stratify_by, return_inverse=True)
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scores = []

        rbf_constant = 20
        matern_constant = 100
        noise_level = 1e-4
        noise_level_bound = (1e-6, 1e-4)

        space = [
            Real(10000, 15000, name='rbf_flowrate_scale'),
            Real(100, 1000, name='rbf_stroke_scale'),
            Real(100, 1000, name='matern_flowrate_scale'),
            Real(3000, 5000, name='matern_stroke_scale')
        ]

        best_score = -np.inf
        best_kernel = None

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, stratify_labels)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            unique_vals, counts = np.unique(stratify_by[test_idx], return_counts=True)
            for label, count in zip(unique_vals, counts):
                print(f"Stratify label {label}: {count} samples in validation set")

            if use_bayes:
                def objective(params):
                    rbf_flow, rbf_stroke, mat_flow, mat_stroke = params
                    kernel = (
                                     C(rbf_constant, 'fixed') * RBF(length_scale=(rbf_flow, rbf_stroke)) +
                                     C(matern_constant, 'fixed') * Matern(length_scale=(mat_flow, mat_stroke), nu=1.5)
                             ) + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bound)

                    model = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, optimizer='fmin_l_bfgs_b')
                    model.fit(X_train, y_train.ravel())
                    y_pred = model.predict(X_test)
                    return -r2(y_test, y_pred)

                result = gp_minimize(objective, space, n_calls=20, random_state=42)
                opt_rbf_flow, opt_rbf_stroke, opt_mat_flow, opt_mat_stroke = result.x

                kernel = (
                                 C(rbf_constant, 'fixed') * RBF(length_scale=(opt_rbf_flow, opt_rbf_stroke)) +
                                 C(matern_constant, 'fixed') * Matern(length_scale=(opt_mat_flow, opt_mat_stroke),
                                                                      nu=1.5)
                         ) + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bound)

                print(f"  Bayesian best params: rbf=({opt_rbf_flow:.1f}, {opt_rbf_stroke:.1f}), "
                          f"matern=({opt_mat_flow:.1f}, {opt_mat_stroke:.1f})")
                print(f"  Best R² during BO (fold): {-result.fun:.4f}")
            else:
                kernel = (
                                 C(rbf_constant, 'fixed') * RBF(length_scale=(10000, 100),
                                                                length_scale_bounds=[(10000, 15000), (100, 1000)]) +
                                 C(matern_constant, 'fixed') * Matern(length_scale=(100, 3000),
                                                                      length_scale_bounds=[(100, 1000), (3000, 5000)],
                                                                      nu=1.5)
                         ) + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bound)

            model = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, optimizer='fmin_l_bfgs_b')
            model.fit(X_train, y_train.ravel())
            y_pred = model.predict(X_test)
            fold_r2 = r2(y_test, y_pred)
            scores.append(fold_r2)

            print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")
            print(f"  R² Score: {fold_r2:.4f}")
            print(f"  Kernel: {model.kernel_}")

            if fold_r2 > best_score:
                best_score = fold_r2
                best_kernel = model.kernel_

        mean_cv_score = np.mean(scores)
        print(f"\nMean cross-validated R²: {mean_cv_score:.4f}")
        print("\nFinal model training on full data with best kernel from CV:")

        final_model = GaussianProcessRegressor(
            kernel=best_kernel,
            alpha=1e-8,
            optimizer=None,
            normalize_y=True
        )
        final_model.fit(X, y.ravel())
        y_pred = final_model.predict(X)
        final_r2 = r2(y, y_pred)

        print(f"  Final model Kernel: {final_model.kernel_}")
        print(f"  Final model R² score on full data: {final_r2:.4f}")

        return final_model, y_pred


    def fit_head_and_power_surface(self, flowrate, stroke, head, power, cv_splits=5, use_bayes=True):
        model_head, head_pred = self._fit_surface(
            x1=flowrate,
            x2=stroke,
            y=head,
            stratify_by=stroke,
            cv_splits=cv_splits,
            verbose=False,
            # use_bayes=False
        )

        fit_power_model = partial(
            self._fit_surface,
            x1=flowrate,
            x2=head_pred,
            stratify_by=stroke,
            cv_splits=cv_splits,
            verbose=False,
            # use_bayes=False
        )

        power_model, power_pred = fit_power_model(y=power)

        return model_head, power_model, head_pred, power_pred


    def visualize_head_surface_with_model(self):
        x1 = self.flowrate
        x2 = self.control
        y = self.Head

        model, y_pred = self._fit_surface(x1, x2, y)
        # model, y_pred = self._fit_surface(x1, x2, y, use_bayes = False)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        grid_f = np.linspace(x1.min(), x1.max(), GRID_DENSITY)
        grid_s = np.linspace(x2.min(), x2.max(), GRID_DENSITY)
        F_grid, S_grid = np.meshgrid(grid_f, grid_s)
        X_grid = np.c_[F_grid.ravel(), S_grid.ravel()]
        H_pred_grid = model.predict(X_grid).reshape(F_grid.shape)

        ax.plot_surface(F_grid, S_grid, H_pred_grid, cmap='viridis', alpha=0.6)

        ax.scatter(x1, x2, y, c='blue', marker='^', label='Actual', s=40)

        ax.scatter(x1, x2, y_pred, c='green', marker='o', label='Predicted', s=40)

        for xi, si, yi, ypi in zip(x1, x2, y, y_pred):
            ax.plot([xi, xi], [si, si], [yi, ypi], color='red', linestyle='-', linewidth=3, alpha=1.0)

        ax.set_xlabel('Flowrate', fontsize=20)
        ax.set_ylabel('Stroke', fontsize=20)
        ax.set_zlabel('Head', fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='z', labelsize=20)
        ax.set_title('GPR Head Surface with Actual & Predicted', fontsize=30)
        ax.legend(fontsize=20)
        plt.tight_layout()
        plt.show()

        return model


    def visualize_head_surface_colored_by_power(self):
        flowrate = self.flowrate
        stroke = self.control
        head = self.Head
        power = self.power

        model_head, model_power, head_pred, power_pred = self.fit_head_and_power_surface(
            flowrate=flowrate,
            stroke=stroke,
            head=head,
            power=power,
            cv_splits=5,
            # use_bayes=False
        )

        f_min, f_max = flowrate.min(), flowrate.max()
        s_min, s_max = stroke.min(), stroke.max()
        grid_f = np.linspace(f_min, f_max, GRID_DENSITY)
        grid_s = np.linspace(s_min, s_max, GRID_DENSITY)
        F_grid, S_grid = np.meshgrid(grid_f, grid_s)
        grid_input = np.c_[F_grid.ravel(), S_grid.ravel()]

        head_grid_pred = model_head.predict(grid_input)
        X_power_grid = np.c_[grid_input[:, 0], head_grid_pred]
        power_grid_pred = model_power.predict(X_power_grid)
        head_grid_pred = head_grid_pred.reshape(F_grid.shape)
        power_grid_pred = power_grid_pred.reshape(F_grid.shape)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        vmin = np.percentile(power_grid_pred, 5)
        vmax = np.percentile(power_grid_pred, 95)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.viridis
        rgba = cmap(norm(power_grid_pred))
        rgba[..., -1] = 0.6

        surf = ax.plot_surface(
            F_grid, S_grid, head_grid_pred,
            facecolors=rgba,
            linewidth=0,
            shade=False
        )

        ax.scatter(flowrate, stroke, head, color='blue', marker='^', s=40, label='Actual Head')
        ax.scatter(flowrate, stroke, head_pred, color='green', marker='o', s=40, label='Predicted Head')

        for f, s, h_actual, h_pred in zip(flowrate, stroke, head, head_pred):
            ax.plot([f, f], [s, s], [h_actual, h_pred], color='red', linewidth=3, alpha=1.0)

        ax.set_xlabel('Flowrate', fontsize=20)
        ax.set_ylabel('Stroke', fontsize=20)
        ax.set_zlabel('Head', fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='z', labelsize=20)
        ax.set_title('Head Surface (Colored by Power)', fontsize=30)
        ax.legend(fontsize=20)

        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(power_grid_pred)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Predicted Power', fontsize=20)
        cbar.ax.tick_params(labelsize=20)

        plt.tight_layout()
        plt.show()

        return model_head, model_power


    def find_and_plot_intersections_2d_and_3d(self, num_points=1000, use_multiprocessing=True):
        x1 = self.flowrate
        x2 = self.control
        y = self.Head

        model, _ = self._fit_surface(x1, x2, y)
        # model, _ = self._fit_surface(x1, x2, y, use_bayes=False)
        a, b = self._fit_system_curve()

        self.poly_func = partial(self._poly_func, a=a, b=b)

        f_min, f_max = x1.min(), x1.max()
        s_min, s_max = x2.min(), x2.max()
        f_grid = np.linspace(f_min, f_max, num_points)
        s_grid = np.linspace(s_min, s_max, num_points)
        F, S = np.meshgrid(f_grid, s_grid)
        points = np.c_[F.ravel(), S.ravel()]

        target_strokes = np.unique(x2)
        plt.figure(figsize=(10, 6))
        all_intersections = []

        for stroke in target_strokes:
            args = (stroke, model, self.poly_func, f_min, f_max, num_points)
            intersections = _compute_intersections_for_stroke(args)

            gpr_vals = [model.predict(np.array([[f, stroke]])).item() for f in f_grid]
            sys_vals = self.poly_func(f_grid)

            intersections = np.array(intersections)
            all_intersections.extend(intersections)

            plt.plot(f_grid, gpr_vals, label=f'GPR (stroke={stroke:.1f})')
            plt.plot(f_grid, sys_vals, linestyle='--', label='System Curve' if stroke == target_strokes[0] else None)
            plt.scatter(intersections[:, 0], intersections[:, 2], color='red', s=40,
                        label=f'Intersections (stroke={stroke:.1f})')

            print(f"\nStroke = {stroke:.4f} 교점 좌표:")
            for pt in intersections:
                print(f"  Flowrate: {pt[0]:.6f}, Stroke: {pt[1]:.6f}, Head: {pt[2]:.6f}")

        plt.xlabel('Flowrate')
        plt.ylabel('Head')
        plt.title('Flowrate vs Head with Intersections for Specified Strokes')
        plt.legend()
        plt.grid(True)
        plt.show()

        H_gpr = np.array([model.predict(np.array([pt])).item() for pt in points]).reshape(F.shape)
        H_sys = self.poly_func(F)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(F, S, H_gpr, alpha=0.6, cmap='viridis')
        ax.plot_surface(F, S, H_sys, alpha=0.3, color='gray')

        all_intersections = np.array(all_intersections)
        ax.scatter(all_intersections[:, 0], all_intersections[:, 1], all_intersections[:, 2], color='red', s=50, label='Intersections')

        ax.set_xlabel('Flowrate')
        ax.set_ylabel('Stroke')
        ax.set_zlabel('Head')
        ax.set_title('3D GPR vs System Curve with Intersections')
        ax.legend()
        plt.show()

        args_list = [(stroke, model, self.poly_func, f_min, f_max, num_points) for stroke in s_grid]
        if use_multiprocessing:
            pool = Pool(10)
            results = pool.map(_compute_intersections_for_stroke, args_list)
            pool.close()
            pool.join()
        else:
            results = [_compute_intersections_for_stroke(args) for args in args_list]

        intersection_points = [pt for sublist in results for pt in sublist]
        intersection_points = np.array(intersection_points)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(F, S, H_gpr, cmap='viridis', alpha=0.6, edgecolor='none')
        ax.plot_surface(F, S, H_sys, color='gray', alpha=0.4)
        ax.scatter(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], color='red', s=40, label='Exact Intersections')

        print(f'전체 교점 좌표 : (Flowrate, Stroke, Head)\n', intersection_points)
        print(f'전체 교점 좌표 수 :', intersection_points.shape)

        ax.set_xlabel('Flowrate')
        ax.set_ylabel('Stroke')
        ax.set_zlabel('Head')
        ax.set_title('Exact Intersections: GPR Surface & System Curve (3D)')
        ax.legend()
        plt.show()


    def find_optimal_stroke_with_arange(self, stroke_step=0.1):
        model_head, model_power, _, _ = self.fit_head_and_power_surface(
            flowrate=self.flowrate,
            stroke=self.control,
            head=self.Head,
            power=self.power,
            # use_bayes=False
        )

        stroke_min, stroke_max = self.control.min(), self.control.max()
        stroke_candidates = np.arange(stroke_min, stroke_max + stroke_step, stroke_step)

        flow_input = input("\nFlowrate 값을 콤마(,)로 입력하세요 (예: 1.0, 2.0, 3.0): ")
        target_flowrates = [float(val.strip()) for val in flow_input.split(',')]

        for flowrate_val in target_flowrates:
            inputs_head = np.c_[[flowrate_val] * len(stroke_candidates), stroke_candidates]
            head_pred = model_head.predict(inputs_head)

            inputs_power = np.c_[[flowrate_val] * len(stroke_candidates), head_pred]
            power_pred = model_power.predict(inputs_power)

            idx_min = np.argmin(power_pred)
            opt_stroke = stroke_candidates[idx_min]
            min_power = power_pred[idx_min]

            print(f"Flowrate {flowrate_val:.3f} → 최소 power: {min_power:.3f} at stroke: {opt_stroke:.3f}")

            plt.figure(figsize=(6, 4))
            plt.plot(stroke_candidates, power_pred, label=f"Flowrate {flowrate_val}")
            plt.scatter(opt_stroke, min_power, color='red', label='min Power')
            plt.xlabel("Stroke")
            plt.ylabel("Predicted Power")
            plt.title("Power vs Stroke (GPR predict)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


    def visualize_gpr_surface_combinations(self):
        X = np.c_[self.flowrate, self.control]
        y = self.Head

        noise_levels = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4] # 1e-4
        combinations = noise_levels

        cols = 3
        rows = int(np.ceil(len(combinations) / cols))
        fig = plt.figure(figsize=(4 * cols, 3.5 * rows))

        x1 = X[:, 0]
        x2 = X[:, 1]
        x1_lin = np.linspace(x1.min(), x1.max(), GRID_DENSITY)
        x2_lin = np.linspace(x2.min(), x2.max(), GRID_DENSITY)
        xx1, xx2 = np.meshgrid(x1_lin, x2_lin)
        X_test = np.c_[xx1.ravel(), xx2.ravel()]

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        for idx, noise in enumerate(combinations):
            kernel = (
                        C(20) * RBF(length_scale=(10000, 100)) +
                        C(100) * Matern(length_scale=(100, 3000), nu=1.5)
                     ) + WhiteKernel(noise_level=noise)

            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-4,
                optimizer=None
            )

            r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            y_cv_pred = cross_val_predict(model, X, y, cv=cv)
            mse = mean_squared_error(y, y_cv_pred)
            r2_mean = np.mean(r2_scores)

            model.fit(X, y)
            y_pred_surface = model.predict(X_test).reshape(xx1.shape)

            ax = fig.add_subplot(rows, cols, idx+1, projection='3d')
            ax.plot_surface(xx1, xx2, y_pred_surface, cmap='viridis', alpha=0.8)
            ax.scatter(X[:, 0], X[:, 1], y, c='blue', marker='^', s=20, alpha=0.4)

            ax.set_title(f"noise={noise:.0e}\nR²={r2_mean:.3f}, MSE={mse:.2f}")

        plt.tight_layout()
        plt.show()

class Compressor(FluidMachinery):
    def __init__(self, name):
        self._machine_name = name


    def settings(self):
        print(f"{self._machine_name}는 압축기입니다")


def create_machine(category, name, auto_draw=True):
    machines = [Blower, Pump, Compressor]
    return machines[category](name, category, auto_draw=auto_draw)


if __name__ == "__main__":
    pump = create_machine(1, "pump", auto_draw=True)
    print(pump.df.dtype.names)

    pump.fit_head_and_power_surface(pump.flowrate, pump.control, pump.Head, pump.power)

    pump.visualize_head_surface_with_model()
    pump.visualize_head_surface_colored_by_power()
    pump.find_and_plot_intersections_2d_and_3d(use_multiprocessing=True)
    # pump.find_and_plot_intersections_2d_and_3d(use_multiprocessing=False)
    pump.find_optimal_stroke_with_arange()


    # pump.visualize_gpr_surface_combinations()