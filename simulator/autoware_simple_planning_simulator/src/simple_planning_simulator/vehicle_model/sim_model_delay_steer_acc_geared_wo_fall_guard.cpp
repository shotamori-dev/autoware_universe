// Copyright 2025 The Autoware Foundation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/simple_planning_simulator/vehicle_model/sim_model_delay_steer_acc_geared_wo_fall_guard.hpp"

#include "autoware_vehicle_msgs/msg/gear_command.hpp"

#include <algorithm>

namespace autoware::simulator::simple_planning_simulator
{

SimModelDelaySteerAccGearedWoFallGuard::SimModelDelaySteerAccGearedWoFallGuard(
  double vx_lim, double steer_lim, double vx_rate_lim, double steer_rate_lim, double wheelbase,
  double dt, double acc_delay, double brake_delay, double acc_time_constant, double brake_time_constant,
  double brake_accuracy_error, double brake_hysteresis_width, double brake_jump_threshold, double brake_jump_value, double brake_resolution,
  double steer_delay,
  double steer_time_constant, double steer_dead_band, double steer_bias,
  double steer_accuracy_error, double steer_resolution, double steer_hysteresis_width,
  double vel_sensor_delay, double vel_sensor_resolution, double vel_sensor_noise_stddev, int vel_sensor_noise_seed,
  double debug_acc_scaling_factor, double debug_steer_scaling_factor)
: SimModelInterface(7 /* dim x */, 4 /* dim u */),
  MIN_TIME_CONSTANT(0.03),
  vx_lim_(vx_lim),
  vx_rate_lim_(vx_rate_lim),
  steer_lim_(steer_lim),
  steer_rate_lim_(steer_rate_lim),
  wheelbase_(wheelbase),
  acc_delay_(acc_delay),
  brake_delay_(brake_delay),
  acc_time_constant_(std::max(acc_time_constant, MIN_TIME_CONSTANT)),
  brake_time_constant_(std::max(brake_time_constant, MIN_TIME_CONSTANT)),
  brake_accuracy_error_(brake_accuracy_error),
  brake_hysteresis_width_(brake_hysteresis_width),
  brake_jump_threshold_(brake_jump_threshold),
  brake_jump_value_(brake_jump_value),
  brake_resolution_(brake_resolution),
  steer_delay_(steer_delay),
  steer_time_constant_(std::max(steer_time_constant, MIN_TIME_CONSTANT)),
  steer_dead_band_(steer_dead_band),
  steer_bias_(steer_bias),
  steer_accuracy_error_(steer_accuracy_error),
  steer_resolution_(steer_resolution),
  steer_hysteresis_width_(steer_hysteresis_width),
  vel_sensor_delay_(vel_sensor_delay),
  vel_sensor_resolution_(vel_sensor_resolution),
  vel_sensor_noise_stddev_(std::max(vel_sensor_noise_stddev, 0.0)),
  debug_acc_scaling_factor_(std::max(debug_acc_scaling_factor, 0.0)),
  debug_steer_scaling_factor_(std::max(debug_steer_scaling_factor, 0.0)),
  prev_brake_cmd_(0.0), // 初期化
  prev_steer_cmd_(0.0),
  delayed_vx_(0.0),
  vel_rng_(vel_sensor_noise_seed),
  vel_dist_(0.0, 1.0)
{
  initializeInputQueue(dt);
}

double SimModelDelaySteerAccGearedWoFallGuard::getX()
{
  return state_(IDX::X);
}
double SimModelDelaySteerAccGearedWoFallGuard::getY()
{
  return state_(IDX::Y);
}
double SimModelDelaySteerAccGearedWoFallGuard::getYaw()
{
  return state_(IDX::YAW);
}
double SimModelDelaySteerAccGearedWoFallGuard::getVx()
{
  // 1. 遅延適用済みの物理車速を取得
  double vx = delayed_vx_;

  // 2. ホワイトノイズの付与
  if (vel_sensor_noise_stddev_ > 1e-5) {
    vx += vel_dist_(vel_rng_) * vel_sensor_noise_stddev_;
  }

  // 3. 分解能（丸め）
  if (vel_sensor_resolution_ > 1e-5) {
    vx = std::round(vx / vel_sensor_resolution_) * vel_sensor_resolution_;
  }

  return vx;
}
double SimModelDelaySteerAccGearedWoFallGuard::getVy()
{
  return 0.0;
}
double SimModelDelaySteerAccGearedWoFallGuard::getAx()
{
  return state_(IDX::ACCX);
}
double SimModelDelaySteerAccGearedWoFallGuard::getWz()
{
  return state_(IDX::VX) * std::tan(state_(IDX::STEER)) / wheelbase_;
}
double SimModelDelaySteerAccGearedWoFallGuard::getSteer()
{
  // return measured values with bias added to actual values
  return state_(IDX::STEER) + steer_bias_;
}
void SimModelDelaySteerAccGearedWoFallGuard::update(const double & dt)
{
  Eigen::VectorXd delayed_input = Eigen::VectorXd::Zero(dim_u_);

  acc_input_queue_.push_back(input_(IDX_U::PEDAL_ACCX_DES));
  brake_input_queue_.push_back(input_(IDX_U::PEDAL_ACCX_DES));

  const double acc_delayed_val = acc_input_queue_.front();
  acc_input_queue_.pop_front();
  const double brake_delayed_val = brake_input_queue_.front();
  brake_input_queue_.pop_front();

  if (acc_delayed_val >= 0.0) {
    delayed_input(IDX_U::PEDAL_ACCX_DES) = acc_delayed_val;
  } else {
    delayed_input(IDX_U::PEDAL_ACCX_DES) = brake_delayed_val;
  }

  steer_input_queue_.push_back(input_(IDX_U::STEER_DES));
  delayed_input(IDX_U::STEER_DES) = steer_input_queue_.front();
  steer_input_queue_.pop_front();
  delayed_input(IDX_U::GEAR) = input_(IDX_U::GEAR);
  delayed_input(IDX_U::SLOPE_ACCX) = input_(IDX_U::SLOPE_ACCX);

  const auto prev_state = state_;
  updateEuler(dt, delayed_input);
  // we cannot use updateRungeKutta() because the differentiability or the continuity condition is
  // not satisfied, but we can use Runge-Kutta method with code reconstruction.

  // take velocity limit explicitly
  state_(IDX::VX) = std::max(-vx_lim_, std::min(state_(IDX::VX), vx_lim_));

  if (
    prev_state(IDX::VX) * state_(IDX::VX) <= 0.0 &&
    -state_(IDX::PEDAL_ACCX) >= std::abs(delayed_input(IDX_U::SLOPE_ACCX))) {
    // stop condition is satisfied
    state_(IDX::VX) = 0.0;
  }

  const auto apply_hsa_stop = [&]() {
    state_(IDX::VX) = 0.0;
    state_(IDX::X) = prev_state(IDX::X);
    state_(IDX::Y) = prev_state(IDX::Y);
    state_(IDX::YAW) = prev_state(IDX::YAW);
  };

  using autoware_vehicle_msgs::msg::GearCommand;
  const auto gear = delayed_input(IDX_U::GEAR);

  if (
    gear == GearCommand::DRIVE || gear == GearCommand::DRIVE_2 || gear == GearCommand::DRIVE_3 ||
    gear == GearCommand::DRIVE_4 || gear == GearCommand::DRIVE_5 || gear == GearCommand::DRIVE_6 ||
    gear == GearCommand::DRIVE_7 || gear == GearCommand::DRIVE_8 || gear == GearCommand::DRIVE_9 ||
    gear == GearCommand::DRIVE_10 || gear == GearCommand::DRIVE_11 ||
    gear == GearCommand::DRIVE_12 || gear == GearCommand::DRIVE_13 ||
    gear == GearCommand::DRIVE_14 || gear == GearCommand::DRIVE_15 ||
    gear == GearCommand::DRIVE_16 || gear == GearCommand::DRIVE_17 ||
    gear == GearCommand::DRIVE_18 || gear == GearCommand::LOW || gear == GearCommand::LOW_2) {
    if (state_(IDX::VX) < 0.0) { // Dギアなのに後ろに下がろうとしている
      apply_hsa_stop();
    }
  } else if (gear == GearCommand::REVERSE || gear == GearCommand::REVERSE_2) {
    if (state_(IDX::VX) > 0.0) { // Rギアなのに前に転がろうとしている
      apply_hsa_stop();
    }
  } else if (gear == GearCommand::PARK) {
    apply_hsa_stop(); // Pギアの時は動かさない
  }

  state_(IDX::ACCX) = (state_(IDX::VX) - prev_state(IDX::VX)) / dt;

  // ====== 追加：遅延バッファの更新 ======
  if (vel_history_queue_.empty()) {
    delayed_vx_ = state_(IDX::VX);
  } else {
    vel_history_queue_.push_back(state_(IDX::VX));
    delayed_vx_ = vel_history_queue_.front();
    vel_history_queue_.pop_front();
  }
}

void SimModelDelaySteerAccGearedWoFallGuard::initializeInputQueue(const double & dt)
{
  size_t acc_input_queue_size = static_cast<size_t>(round(acc_delay_ / dt));
  acc_input_queue_.resize(acc_input_queue_size);
  std::fill(acc_input_queue_.begin(), acc_input_queue_.end(), 0.0);

  size_t brake_input_queue_size = static_cast<size_t>(round(brake_delay_ / dt));
  brake_input_queue_.resize(brake_input_queue_size);
  std::fill(brake_input_queue_.begin(), brake_input_queue_.end(), 0.0);

  size_t steer_input_queue_size = static_cast<size_t>(round(steer_delay_ / dt));
  steer_input_queue_.resize(steer_input_queue_size);
  std::fill(steer_input_queue_.begin(), steer_input_queue_.end(), 0.0);

  size_t vel_input_queue_size = static_cast<size_t>(std::round(vel_sensor_delay_ / dt));
  vel_history_queue_.resize(vel_input_queue_size);
  std::fill(vel_history_queue_.begin(), vel_history_queue_.end(), 0.0);
}

Eigen::VectorXd SimModelDelaySteerAccGearedWoFallGuard::calcModel(
  const Eigen::VectorXd & state, const Eigen::VectorXd & input)
{
  auto sat = [](double val, double u, double l) { return std::max(std::min(val, u), l); };

  const double vel = sat(state(IDX::VX), vx_lim_, -vx_lim_);
  const double pedal_acc = sat(state(IDX::PEDAL_ACCX), vx_rate_lim_, -vx_rate_lim_);
  const double yaw = state(IDX::YAW);
  const double steer = state(IDX::STEER);
  double pedal_acc_des =
    sat(input(IDX_U::PEDAL_ACCX_DES), vx_rate_lim_, -vx_rate_lim_) * debug_acc_scaling_factor_;

  // =========================================================================
  if (pedal_acc_des < 0.0) { // ブレーキ指令の時だけ適用
    double brake_cmd = std::abs(pedal_acc_des); // 扱いやすいように絶対値（正の値）にする

    // 1. 精度（ゲイン誤差）
    brake_cmd = brake_cmd * (1.0 + brake_accuracy_error_);

    // 2. ヒステリシス（行きと帰りの差）
    double hist_cmd = brake_cmd;
    if (brake_cmd > prev_brake_cmd_ + 1e-5) {
      hist_cmd = std::max(0.0, brake_cmd - (brake_hysteresis_width_ / 2.0)); // 踏み増し時は効きにくい
    } else if (brake_cmd < prev_brake_cmd_ - 1e-5) {
      hist_cmd = brake_cmd + (brake_hysteresis_width_ / 2.0);                // 緩め時は抜けにくい
    }
    prev_brake_cmd_ = brake_cmd; // 次回のために記憶

    // 3. ジャンプ（最低作動圧・クラックプレッシャー）
    double jump_cmd = hist_cmd;
    if (hist_cmd < brake_jump_threshold_) {
      jump_cmd = 0.0; // 閾値まではバルブが開かない
    } else if (hist_cmd < brake_jump_value_) {
      jump_cmd = brake_jump_value_; // 開いた瞬間、最低でもこのGが出てしまう
    }

    // 4. 解像度（階段状の効き）
    double res_cmd = jump_cmd;
    if (brake_resolution_ > 1e-5) { // ゼロ割れ防止
      res_cmd = std::round(jump_cmd / brake_resolution_) * brake_resolution_;
    }

    // 計算したブレーキ力を元のマイナス符号に戻して書き換える
    pedal_acc_des = -res_cmd;
  } else {
    // アクセルの時は、ヒステリシスの記憶をリセットしておく（ブレーキを完全に離した状態）
    prev_brake_cmd_ = 0.0;
  }
  // =========================================================================

  const double current_tc = (pedal_acc_des < 0.0) ? brake_time_constant_ : acc_time_constant_;
  double steer_des =
    sat(input(IDX_U::STEER_DES), steer_lim_, -steer_lim_) * debug_steer_scaling_factor_;

  // ================= 操舵フィルター適用 =================
  // 1. 精度誤差
  steer_des *= (1.0 + steer_accuracy_error_);

  // 2. ヒステリシス（ガタ）
  double steer_hist = steer_des;
  if (steer_des > prev_steer_cmd_ + 1e-5) {
    steer_hist = steer_des - (steer_hysteresis_width_ / 2.0); // 右に切り増し時は少し遅れる
  } else if (steer_des < prev_steer_cmd_ - 1e-5) {
    steer_hist = steer_des + (steer_hysteresis_width_ / 2.0); // 左に戻し時は少し遅れる
  }
  prev_steer_cmd_ = steer_des;

  // 3. 分解能（カクつき）
  if (steer_resolution_ > 1e-5) {
    steer_hist = std::round(steer_hist / steer_resolution_) * steer_resolution_;
  }

  steer_des = steer_hist; // フィルター後の値を再代入
  // =====================================================

  // NOTE: `steer_des` is calculated by control from measured values. getSteer() also gets the
  // measured value. The steer_rate used in the motion calculation is obtained from these
  // differences.
  const double steer_diff = getSteer() - steer_des;
  const double steer_diff_with_dead_band = std::invoke([&]() {
    if (steer_diff > steer_dead_band_) {
      return steer_diff - steer_dead_band_;
    } else if (steer_diff < -steer_dead_band_) {
      return steer_diff + steer_dead_band_;
    } else {
      return 0.0;
    }
  });
  const double steer_rate =
    sat(-steer_diff_with_dead_band / steer_time_constant_, steer_rate_lim_, -steer_rate_lim_);

  Eigen::VectorXd d_state = Eigen::VectorXd::Zero(dim_x_);

  d_state(IDX::X) = vel * cos(yaw);
  d_state(IDX::Y) = vel * sin(yaw);
  d_state(IDX::YAW) = vel * std::tan(steer) / wheelbase_;
  d_state(IDX::VX) = [&] {
    if (pedal_acc >= 0.0) {
      using autoware_vehicle_msgs::msg::GearCommand;
      const auto gear = input(IDX_U::GEAR);
      if (gear == GearCommand::NONE || gear == GearCommand::PARK) {
        return 0.0;
      } else if (gear == GearCommand::NEUTRAL) {
        return input(IDX_U::SLOPE_ACCX);
      } else if (gear == GearCommand::REVERSE || gear == GearCommand::REVERSE_2) {
        return -pedal_acc + input(IDX_U::SLOPE_ACCX);
      } else {
        return pedal_acc + input(IDX_U::SLOPE_ACCX);
      }
    } else {
      if (vel > 0.0) {
        return pedal_acc + input(IDX_U::SLOPE_ACCX);
      } else if (vel < 0.0) {
        return -pedal_acc + input(IDX_U::SLOPE_ACCX);
      } else if (-pedal_acc >= std::abs(input(IDX_U::SLOPE_ACCX))) {
        return 0.0;
      } else {
        return input(IDX_U::SLOPE_ACCX);
      }
    }
  }();
  d_state(IDX::STEER) = steer_rate;
  d_state(IDX::PEDAL_ACCX) = -(pedal_acc - pedal_acc_des) / current_tc;

  return d_state;
}

}  // namespace autoware::simulator::simple_planning_simulator
