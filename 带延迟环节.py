import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize, differential_evolution
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class HeatingSystemWithDelay:
    """带延迟环节的加热炉控制系统分析"""

    def __init__(self):
        self.time = None
        self.temperature = None
        self.voltage = None
        self.identified_params = None
        self.transfer_function = None
        self.pid_params = None

    def load_data(self, csv_file='B 任务数据集.csv'):
        """加载数据"""
        try:
            data = pd.read_csv(csv_file)
            self.time = data['time'].values
            self.temperature = data['temperature'].values
            self.voltage = data['volte'].values

            print(f"数据加载成功!")
            print(f"数据点数: {len(self.time)}")
            print(f"时间范围: {self.time[0]:.1f} - {self.time[-1]:.1f} 秒")
            print(f"温度范围: {self.temperature.min():.2f} - {self.temperature.max():.2f} °C")
            print(f"电压: {self.voltage[0]:.2f} V (恒定)")
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    def identify_delay_parameters(self):
        """辨识带延迟的系统参数"""
        print("\n=== 带延迟环节的系统辨识 ===")

        initial_temp = self.temperature[0]
        final_temp = self.temperature[-1]
        input_voltage = self.voltage[0]

        # 寻找系统开始响应的时间点（延迟时间的估算）
        temp_change_threshold = 0.1  # 温度变化阈值
        delay_estimate = 0

        for i in range(1, min(len(self.temperature), 1000)):  # 只在前1000个点中找
            temp_change = abs(self.temperature[i] - initial_temp)
            if temp_change > temp_change_threshold:
                delay_estimate = self.time[i]
                break

        print(f"初始延迟估计: {delay_estimate:.1f} 秒")

        def delayed_first_order_response(t, K, T, tau):
            """带延迟的一阶系统阶跃响应"""
            response = np.zeros_like(t)
            delayed_indices = t >= tau
            if np.any(delayed_indices):
                t_delayed = t[delayed_indices] - tau
                response[delayed_indices] = initial_temp + K * input_voltage * (1 - np.exp(-t_delayed / T))
            # 延迟时间内保持初始温度
            response[~delayed_indices] = initial_temp
            return response

        def objective_function(params):
            """优化目标函数"""
            K, T, tau = params

            # 参数约束
            if K <= 0 or T <= 0 or tau < 0 or tau > 1000:
                return 1e10

            try:
                predicted = delayed_first_order_response(self.time, K, T, tau)
                mse = np.mean((self.temperature - predicted) ** 2)
                return mse
            except:
                return 1e10

        # 初始参数估计
        K_init = (final_temp - initial_temp) / input_voltage
        T_init = 2000  # 基于经验估计
        tau_init = max(0, delay_estimate)

        print(f"初始参数估计:")
        print(f"  K_init: {K_init:.3f}")
        print(f"  T_init: {T_init:.1f}")
        print(f"  tau_init: {tau_init:.1f}")

        # 使用多种优化方法确保全局最优
        bounds = [(0.1, 20),  # K bounds
                  (100, 5000),  # T bounds
                  (0, 500)]  # tau bounds

        # 方法1: 差分进化算法（全局优化）
        result_de = differential_evolution(objective_function, bounds,
                                           maxiter=100, popsize=10, seed=42)

        # 方法2: 局部优化改进
        result_local = minimize(objective_function, result_de.x,
                                method='L-BFGS-B', bounds=bounds)

        # 选择更好的结果
        if result_local.fun < result_de.fun:
            best_params = result_local.x
            best_cost = result_local.fun
        else:
            best_params = result_de.x
            best_cost = result_de.fun

        K_opt, T_opt, tau_opt = best_params

        self.identified_params = {
            'K': K_opt,
            'T': T_opt,
            'tau': tau_opt,
            'initial_temp': initial_temp,
            'final_temp': final_temp,
            'input_voltage': input_voltage,
            'optimization_cost': best_cost
        }

        print(f"\n优化辨识结果:")
        print(f"  静态增益 K: {K_opt:.3f} °C/V")
        print(f"  时间常数 T: {T_opt:.1f} s")
        print(f"  延迟时间 τ: {tau_opt:.1f} s")
        print(f"  优化目标函数值: {best_cost:.6f}")

        # 构造传递函数（用于后续PID设计）
        # 注意：scipy的TransferFunction不直接支持延迟，这里记录参数用于仿真
        num = [K_opt]
        den = [T_opt, 1]
        self.transfer_function = signal.TransferFunction(num, den)

        return self.identified_params

    def validate_identification_with_delay(self):
        """验证带延迟的辨识结果"""
        print("\n=== 验证带延迟的辨识结果 ===")

        K = self.identified_params['K']
        T = self.identified_params['T']
        tau = self.identified_params['tau']
        initial_temp = self.identified_params['initial_temp']
        input_voltage = self.identified_params['input_voltage']

        # 计算模型响应
        def delayed_response(t, K, T, tau):
            response = np.zeros_like(t)
            delayed_indices = t >= tau
            if np.any(delayed_indices):
                t_delayed = t[delayed_indices] - tau
                response[delayed_indices] = initial_temp + K * input_voltage * (1 - np.exp(-t_delayed / T))
            response[~delayed_indices] = initial_temp
            return response

        predicted_temp = delayed_response(self.time, K, T, tau)

        # 计算误差指标
        error = self.temperature - predicted_temp
        mse = np.mean(error ** 2)
        mae = np.mean(np.abs(error))
        max_error = np.max(np.abs(error))

        # 计算R²
        ss_res = np.sum(error ** 2)
        ss_tot = np.sum((self.temperature - np.mean(self.temperature)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"验证结果:")
        print(f"  均方误差 MSE: {mse:.4f} °C²")
        print(f"  平均绝对误差 MAE: {mae:.4f} °C")
        print(f"  最大误差: {max_error:.2f} °C")
        print(f"  拟合优度 R²: {r_squared:.4f}")

        # 绘制验证图
        plt.figure(figsize=(14, 10))

        plt.subplot(3, 1, 1)
        plt.plot(self.time, self.temperature, 'b-', label='实际温度', linewidth=1.5)
        plt.plot(self.time, predicted_temp, 'r--', label='辨识模型（带延迟）', linewidth=2)
        plt.axvline(x=tau, color='g', linestyle=':', linewidth=2,
                    label=f'延迟时间 τ={tau:.1f}s')
        plt.xlabel('时间 (秒)')
        plt.ylabel('温度 (°C)')
        plt.title(f'带延迟的系统辨识验证 - R² = {r_squared:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 2)
        plt.plot(self.time, error, 'g-', linewidth=1)
        plt.xlabel('时间 (秒)')
        plt.ylabel('误差 (°C)')
        plt.title(f'辨识误差 (MAE: {mae:.2f}°C)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        plt.subplot(3, 1, 3)
        plt.plot(self.time, self.voltage, 'orange', linewidth=2, label='输入电压')
        plt.xlabel('时间 (秒)')
        plt.ylabel('电压 (V)')
        plt.title('系统输入')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 更新参数
        self.identified_params.update({
            'validated_r2': r_squared,
            'mae': mae,
            'mse': mse,
            'max_error': max_error
        })

        return r_squared, mae

    def design_pid_with_delay_compensation(self, setpoint=35.0):
        """设计考虑延迟补偿的PID控制器"""
        print(f"\n=== 考虑延迟的PID控制器设计 (设定值: {setpoint}°C) ===")

        K = self.identified_params['K']
        T = self.identified_params['T']
        tau = self.identified_params['tau']

        print(f"系统参数: K={K:.3f}, T={T:.1f}s, τ={tau:.1f}s")

        # 基于延迟的PID整定经验公式 (Cohen-Coon方法改进)
        if tau > 0:
            # Cohen-Coon整定公式，适用于有延迟的系统
            alpha = tau / T

            if alpha < 0.1:  # 延迟较小
                kp_est = (1.2 * T) / (K * tau) * (1 + 0.2 * alpha)
                ti_est = tau * (2.5 - 2 * alpha) / (1 - 0.39 * alpha)
                td_est = tau * 0.37 / (1 - 0.81 * alpha)
            else:  # 延迟较大
                kp_est = (1.0 * T) / (K * tau)
                ti_est = tau * 2.5
                td_est = tau * 0.25

            ki_est = kp_est / ti_est
            kd_est = kp_est * td_est

            print(f"Cohen-Coon初始估计:")
            print(f"  Kp: {kp_est:.4f}")
            print(f"  Ki: {ki_est:.6f}")
            print(f"  Kd: {kd_est:.4f}")
        else:
            # 无延迟的经典整定
            kp_est = 0.6
            ki_est = 0.01
            kd_est = 0.1

        def pid_objective_with_delay(params):
            """考虑延迟的PID目标函数"""
            kp, ki, kd = params

            if kp <= 0 or ki < 0 or kd < 0:
                return 1000

            try:
                # 简化的闭环特征方程分析
                # 对于带延迟的系统，使用近似方法评估性能

                # 开环增益
                loop_gain = kp * K

                # 考虑延迟的稳定性边界
                stability_margin = 1.0 / (1 + tau * ki * 10)  # 延迟对积分的影响

                # 稳态误差 (考虑延迟的影响)
                steady_error = 1 / (1 + loop_gain) * 100

                # 超调量估算 (考虑延迟降低阻尼的影响)
                damping_estimate = (kp + kd * 2 / T) / (2 * np.sqrt(ki * K * stability_margin))

                if damping_estimate < 0.1:
                    overshoot = 50  # 延迟系统容易振荡
                elif damping_estimate < 1:
                    overshoot = np.exp(-np.pi * damping_estimate /
                                       np.sqrt(1 - damping_estimate ** 2 + 0.01)) * 100
                else:
                    overshoot = 0

                # 调节时间估算 (延迟增加调节时间)
                settling_time_factor = 1 + tau / T * 0.5

                # 综合目标函数
                cost = (steady_error * 2 +
                        overshoot * 1 +
                        settling_time_factor * 5 +
                        kd * 10)  # 限制微分增益避免噪声敏感

                return cost

            except:
                return 1000

        # 优化边界 (考虑延迟系统的特点)
        bounds = [(0.01, 5),  # Kp - 可以更大
                  (0.0001, 0.1),  # Ki - 较小避免积分饱和
                  (0, 2)]  # Kd - 适中避免噪声

        # 多起点优化
        best_result = None
        best_cost = float('inf')

        for _ in range(3):  # 多次优化取最好结果
            result = differential_evolution(pid_objective_with_delay, bounds,
                                            maxiter=50, popsize=10, seed=None)
            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result

        kp_opt, ki_opt, kd_opt = best_result.x

        self.pid_params = {
            'Kp': kp_opt,
            'Ki': ki_opt,
            'Kd': kd_opt,
            'optimization_cost': best_cost
        }

        print(f"\n优化的PID参数:")
        print(f"  Kp: {kp_opt:.4f}")
        print(f"  Ki: {ki_opt:.6f}")
        print(f"  Kd: {kd_opt:.4f}")
        print(f"  目标函数值: {best_cost:.4f}")

        return self.pid_params

    def simulate_control_with_delay(self, setpoint=35.0, sim_time=1500):
        """仿真考虑延迟的控制系统"""
        print(f"\n=== 带延迟的控制系统仿真 (设定值: {setpoint}°C) ===")

        if self.pid_params is None:
            print("请先设计PID控制器!")
            return

        # 仿真参数
        dt = 1.0
        t_sim = np.arange(0, sim_time, dt)
        n_steps = len(t_sim)

        # 系统参数
        K = self.identified_params['K']
        T = self.identified_params['T']
        tau = self.identified_params['tau']
        initial_temp = self.identified_params['initial_temp']

        # PID参数
        kp, ki, kd = self.pid_params['Kp'], self.pid_params['Ki'], self.pid_params['Kd']

        # 延迟处理：使用环形缓冲区存储控制信号
        delay_steps = int(tau / dt)
        control_buffer = np.zeros(delay_steps + 1)

        # 初始化数组
        temperature = np.zeros(n_steps)
        control_output = np.zeros(n_steps)
        delayed_control = np.zeros(n_steps)
        error_signal = np.zeros(n_steps)

        temperature[0] = initial_temp
        integral_error = 0
        prev_error = 0

        print(f"延迟步数: {delay_steps} (延迟时间: {tau:.1f}s)")

        # 数字仿真循环
        for i in range(1, n_steps):
            # 计算控制误差
            error = setpoint - temperature[i - 1]
            error_signal[i - 1] = error

            # PID控制算法
            integral_error += error * dt

            # 积分饱和限制
            integral_error = np.clip(integral_error, -100, 100)

            derivative_error = (error - prev_error) / dt

            # PID输出
            control_output[i - 1] = (kp * error +
                                     ki * integral_error +
                                     kd * derivative_error)

            # 限制控制输出范围
            control_output[i - 1] = np.clip(control_output[i - 1], 0, 10)

            # 延迟处理
            if delay_steps > 0:
                # 将当前控制信号存入缓冲区
                control_buffer[:-1] = control_buffer[1:]
                control_buffer[-1] = control_output[i - 1]

                # 取出延迟后的控制信号
                delayed_control[i - 1] = control_buffer[0]
            else:
                delayed_control[i - 1] = control_output[i - 1]

            # 一阶系统响应 (使用延迟后的控制信号)
            temp_change_rate = (K * delayed_control[i - 1] -
                                (temperature[i - 1] - initial_temp)) / T
            temperature[i] = temperature[i - 1] + temp_change_rate * dt

            prev_error = error

        # 性能指标计算
        steady_start_idx = max(n_steps - 100, n_steps // 2)
        steady_state = np.mean(temperature[steady_start_idx:])
        peak_temp = np.max(temperature[100:])  # 忽略初始100步

        # 超调量计算
        if steady_state > initial_temp:
            overshoot = max(0, (peak_temp - steady_state) /
                            (steady_state - initial_temp) * 100)
        else:
            overshoot = 0

        # 稳态误差
        steady_error = abs(setpoint - steady_state) / setpoint * 100

        # 调节时间 (2%误差带)
        settling_time = sim_time
        tolerance = 0.02 * abs(setpoint - initial_temp)
        for j in range(200, len(temperature)):
            if abs(temperature[j] - setpoint) <= tolerance:
                if all(abs(temperature[k] - setpoint) <= tolerance
                       for k in range(j, min(j + 50, len(temperature)))):
                    settling_time = t_sim[j]
                    break

        print(f"控制性能指标:")
        print(f"  稳态值: {steady_state:.2f}°C")
        print(f"  稳态误差: {steady_error:.2f}%")
        print(f"  超调量: {overshoot:.2f}%")
        print(f"  调节时间: {settling_time:.1f}s")

        # 绘制仿真结果
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        # 温度响应
        axes[0].plot(t_sim, temperature, 'b-', linewidth=2, label='系统响应')
        axes[0].axhline(y=setpoint, color='r', linestyle='--', linewidth=2,
                        label=f'设定值 ({setpoint}°C)')
        axes[0].axhline(y=steady_state, color='g', linestyle=':', alpha=0.7,
                        label=f'稳态值 ({steady_state:.1f}°C)')
        axes[0].set_ylabel('温度 (°C)')
        axes[0].set_title('带延迟的PID控制系统响应')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 控制误差
        axes[1].plot(t_sim[:-1], error_signal[:-1], 'g-', linewidth=1.5)
        axes[1].set_ylabel('控制误差 (°C)')
        axes[1].set_title('控制误差')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # 控制器输出
        axes[2].plot(t_sim[:-1], control_output[:-1], 'r-', linewidth=1.5,
                     label='PID输出')
        axes[2].set_ylabel('控制输出 (V)')
        axes[2].set_title('控制器输出')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # 延迟后的控制信号
        axes[3].plot(t_sim[:-1], delayed_control[:-1], 'orange', linewidth=1.5,
                     label=f'延迟后控制信号 (τ={tau:.1f}s)')
        axes[3].set_xlabel('时间 (秒)')
        axes[3].set_ylabel('延迟控制输出 (V)')
        axes[3].set_title('实际作用于系统的控制信号')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return {
            'steady_state': steady_state,
            'overshoot': overshoot,
            'settling_time': settling_time,
            'steady_error': steady_error,
            'peak_temp': peak_temp
        }

    def compare_with_without_delay(self):
        """对比有无延迟的辨识结果"""
        print("\n=== 对比分析：有无延迟的差异 ===")

        if self.identified_params is None:
            print("请先进行系统辨识!")
            return

        # 不考虑延迟的简单辨识
        initial_temp = self.temperature[0]
        final_temp = self.temperature[-1]
        input_voltage = self.voltage[0]

        K_simple = (final_temp - initial_temp) / input_voltage

        # 简单一阶拟合
        def simple_first_order(t, T):
            return initial_temp + K_simple * input_voltage * (1 - np.exp(-t / T))

        def simple_objective(T):
            if T <= 0:
                return 1e10
            try:
                predicted = simple_first_order(self.time, T)
                return np.mean((self.temperature - predicted) ** 2)
            except:
                return 1e10

        from scipy.optimize import minimize_scalar
        result_simple = minimize_scalar(simple_objective, bounds=(100, 5000),
                                        method='bounded')
        T_simple = result_simple.x

        # 计算两种模型的预测
        predicted_simple = simple_first_order(self.time, T_simple)

        K_delay = self.identified_params['K']
        T_delay = self.identified_params['T']
        tau_delay = self.identified_params['tau']

        # 带延迟的预测
        predicted_delay = np.zeros_like(self.time)
        delayed_indices = self.time >= tau_delay
        if np.any(delayed_indices):
            t_delayed = self.time[delayed_indices] - tau_delay
            predicted_delay[delayed_indices] = (initial_temp +
                                                K_delay * input_voltage * (1 - np.exp(-t_delayed / T_delay)))
        predicted_delay[~delayed_indices] = initial_temp

        # 计算拟合优度
        def calc_r2(actual, predicted):
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            return 1 - (ss_res / ss_tot)

        r2_simple = calc_r2(self.temperature, predicted_simple)
        r2_delay = calc_r2(self.temperature, predicted_delay)

        print(f"不带延迟模型:")
        print(f"  K = {K_simple:.3f} °C/V")
        print(f"  T = {T_simple:.1f} s")
        print(f"  R² = {r2_simple:.4f}")

        print(f"\n带延迟模型:")
        print(f"  K = {K_delay:.3f} °C/V")
        print(f"  T = {T_delay:.1f} s")
        print(f"  τ = {tau_delay:.1f} s")
        print(f"  R² = {r2_delay:.4f}")

        print(f"\n模型改进:")
        print(f"  R²提升: {r2_delay - r2_simple:.4f}")
        print(f"  相对改进: {(r2_delay - r2_simple) / r2_simple * 100:.2f}%")

        # 绘制对比图
        plt.figure(figsize=(14, 8))

        plt.subplot(2, 1, 1)
        plt.plot(self.time, self.temperature, 'b-', linewidth=2, label='实际数据')
        plt.plot(self.time, predicted_simple, 'g--', linewidth=2,
                 label=f'无延迟模型 (R²={r2_simple:.4f})')
        plt.plot(self.time, predicted_delay, 'r:', linewidth=2,
                 label=f'带延迟模型 (R²={r2_delay:.4f})')
        plt.axvline(x=tau_delay, color='orange', linestyle='-', alpha=0.7,
                    label=f'延迟时间 τ={tau_delay:.1f}s')
        plt.xlabel('时间 (秒)')
        plt.ylabel('温度 (°C)')
        plt.title('系统辨识模型对比')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        error_simple = self.temperature - predicted_simple
        error_delay = self.temperature - predicted_delay
        plt.plot(self.time, error_simple, 'g-', linewidth=1, label='无延迟模型误差')
        plt.plot(self.time, error_delay, 'r-', linewidth=1, label='带延迟模型误差')
        plt.xlabel('时间 (秒)')
        plt.ylabel('误差 (°C)')
        plt.title('辨识误差对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    def run_complete_analysis_with_delay(self):
        """运行完整的带延迟分析"""
        print("=" * 70)
        print("加热炉控制系统智能辨识与参数优化 - 带延迟环节版本")
        print("=" * 70)

        # 1. 加载数据
        if not self.load_data():
            return

        # 2. 辨识带延迟的系统参数
        self.identify_delay_parameters()

        # 3. 验证辨识结果
        r2, mae = self.validate_identification_with_delay()

        # 4. 对比有无延迟的差异
        self.compare_with_without_delay()

        # 5. 设计考虑延迟的PID控制器
        self.design_pid_with_delay_compensation(setpoint=35.0)

        # 6. 仿真带延迟的控制系统
        performance = self.simulate_control_with_delay(setpoint=35.0)

        # 7. 显示传递函数
        self.display_transfer_function()

        # 8. 总结
        print("\n" + "=" * 70)
        print("分析总结")
        print("=" * 70)
        print(
            f"辨识的传递函数: G(s) = {self.identified_params['K']:.3f}/({self.identified_params['T']:.1f}s + 1) * e^(-{self.identified_params['tau']:.1f}s)")
        print(f"辨识精度: R² = {r2:.4f}")
        print(f"延迟时间: τ = {self.identified_params['tau']:.1f} s")
        print(f"控制性能: 稳态误差 = {performance['steady_error']:.2f}%")
        print(f"调节时间: {performance['settling_time']:.1f} s")
        print("带延迟的系统分析完成!")

        return performance

    def display_transfer_function(self):
        """显示辨识得到的传递函数"""
        print("\n=== 辨识得到的传递函数 ===")

        K = self.identified_params['K']
        T = self.identified_params['T']
        tau = self.identified_params['tau']

        print(f"传递函数形式:")
        print(f"G(s) = K/(Ts + 1) * e^(-τs)")
        print(f"")
        print(f"参数值:")
        print(f"G(s) = {K:.3f}/({T:.1f}s + 1) * e^(-{tau:.1f}s)")
        print(f"")
        print(f"物理意义:")
        print(f"  K = {K:.3f} °C/V : 静态增益，表示稳态时每1V输入对应的温度变化")
        print(f"  T = {T:.1f} s    : 时间常数，表示系统响应的快慢")
        print(f"  τ = {tau:.1f} s     : 纯延迟时间，表示从输入到开始响应的延迟")

        # 与参考传递函数对比
        print(f"\n与任务给出的参考传递函数对比:")
        print(f"参考: G(s) = K/(Ts + 1) * e^(-τs)")
        print(f"辨识: G(s) = {K:.3f}/({T:.1f}s + 1) * e^(-{tau:.1f}s)")
        print(f"结构一致性: ✓ 完全符合参考传递函数形式")

    def analyze_delay_impact(self):
        """分析延迟对控制系统的影响"""
        print("\n=== 延迟对控制系统的影响分析 ===")

        if self.identified_params is None:
            print("请先进行系统辨识!")
            return

        K = self.identified_params['K']
        T = self.identified_params['T']
        tau = self.identified_params['tau']

        # 计算延迟比
        delay_ratio = tau / T

        print(f"延迟分析:")
        print(f"  延迟时间 τ = {tau:.1f} s")
        print(f"  时间常数 T = {T:.1f} s")
        print(f"  延迟比 τ/T = {delay_ratio:.3f}")

        # 延迟对控制的影响评估
        if delay_ratio < 0.1:
            impact_level = "轻微"
            recommendation = "可使用经典PID方法"
        elif delay_ratio < 0.5:
            impact_level = "中等"
            recommendation = "建议使用Smith预估器或IMC方法"
        else:
            impact_level = "严重"
            recommendation = "必须使用预测控制或鲁棒控制方法"

        print(f"  延迟影响程度: {impact_level}")
        print(f"  控制建议: {recommendation}")

        # 稳定性分析
        print(f"\n稳定性影响:")
        print(f"  延迟会降低系统的相位裕度")
        print(f"  延迟会限制闭环带宽")
        print(f"  延迟会增加超调和振荡的风险")


# 主程序
if __name__ == "__main__":
    system = HeatingSystemWithDelay()
    system.run_complete_analysis_with_delay()

    # 额外的延迟影响分析
    system.analyze_delay_impact()