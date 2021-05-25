#pragma once

/**
 * This code is adapted from https://github.com/kazuotani14/iLQR
 */

#include <Eigen/Dense>

#include "common.hpp"
#include "math/eigen_algebra.hpp"
#include "model.hpp"


using namespace Eigen;

static const int maxIter = 1000;
static const double tolFun = 1e-6;
static const double tolGrad = 1e-6;
static double lambda = 50;
static double dlambda = 1;
static const double lambdaFactor = 1.2;
static const double lambdaMax = 1e11;
static const double lambdaMin = 1e-8;
static const double zMin = 0;

/**
 * Step sizes to use during line search
 */
static std::vector<double> alpha_vec = {1.0000, 0.5012, 0.2512, 0.1259,
                                        0.0631, 0.0316, 0.0158, 0.0079,
                                        0.0040, 0.0020, 0.0010};
static Eigen::Map<VectorXd> Alpha(alpha_vec.data(), alpha_vec.size());

class iLQR {
 public:
  iLQR(Model<tds::EigenAlgebra> *p_dyn, double timeDelta) : dt(timeDelta) {
    model.reset(p_dyn);

    Qx.resize(model->x_dims);
    Qu.resize(model->u_dims);
    Qxx.resize(model->x_dims, model->x_dims);
    Qux.resize(model->u_dims, model->x_dims);
    Quu.resize(model->u_dims, model->u_dims);

    k_i.resize(model->u_dims);
    K_i.resize(model->u_dims, model->x_dims);
    Qux_reg.resize(model->x_dims, model->u_dims);
    QuuF.resize(model->u_dims, model->u_dims);

    Eigen::initParallel();
  }
  iLQR() = default;

  std::shared_ptr<Model<tds::EigenAlgebra>> model;

  void generate_trajectory();
  void generate_trajectory(const VectorXd &x_0);  // warm-start
  void generate_trajectory(const VectorXd &x_0,
                           const VecOfVecXd &u0);  // fresh start

  void output_to_csv(const std::string filename);
  double init_traj(const VectorXd &x_0, const VecOfVecXd &u_0);

  const VecOfVecXd &get_xs() const { return xs; }
  const VecOfVecXd &get_us() const { return us; }
  const VecOfVecXd &get_ls() const { return ls; }

 private:
  double dt;
  int T;  // number of state transitions

  VectorXd x0;

  VecOfVecXd xs;  // s: "step". current working sequence
  VecOfVecXd us;
  VecOfVecXd ls;
  VecOfMatXd Ls;
  double cost_s;

  // n = dims(state), m = dims(control)
  MatrixXd du;     // m*T
  VecOfMatXd fx;   // n x n x (T+1)
  VecOfMatXd fu;   // n x m x (T+1)
  VecOfVecXd cx;   // n x (T+1)
  VecOfVecXd cu;   // m x (T+1)
  VecOfMatXd cxx;  // n x n x (T+1)
  VecOfMatXd cxu;  // n x m x (T+1)
  VecOfMatXd cuu;  // m x m x (T+1)

  Vector2d dV;     // 2x1
  VecOfVecXd Vx;   // n x (T+1)
  VecOfMatXd Vxx;  // n x n x (T+1)
  VecOfVecXd k;    // m x n x T
  VecOfMatXd K;    // m x T

  VectorXd Qx, Qu, k_i;
  MatrixXd Qxx, Qux, Quu, K_i, Qux_reg, QuuF;

  double forward_pass(const VectorXd &x0, const VecOfVecXd &u);
  int backward_pass();
  double get_gradient_norm(const VecOfVecXd &l, const VecOfVecXd &u);

  void compute_derivatives(const VecOfVecXd &x, const VecOfVecXd &u);
  void get_dynamics_derivatives(const VecOfVecXd &x, const VecOfVecXd &u,
                                VecOfMatXd &f_x, VecOfMatXd &f_u);
  void get_cost_derivatives(const VecOfVecXd &x, const VecOfVecXd &u,
                            VecOfVecXd &c_x, VecOfVecXd &c_u);
  void get_cost_2nd_derivatives(const VecOfVecXd &x, const VecOfVecXd &u,
                                VecOfMatXd &c_xx, VecOfMatXd &c_xu,
                                VecOfMatXd &c_uu);

  // multi-threaded version - about 2 to 3 times faster
  void get_cost_2nd_derivatives_mt(const VecOfVecXd &x, const VecOfVecXd &u,
                                   VecOfMatXd &c_xx, VecOfMatXd &c_xu,
                                   VecOfMatXd &c_uu, int n_threads_per);

  void calculate_cxx(const VecOfVecXd &x, const VecOfVecXd &u, VecOfMatXd &c_xx,
                     int start_T, int end_T);
  void calculate_cxu(const VecOfVecXd &x, const VecOfVecXd &u, VecOfMatXd &c_xu,
                     int start_T, int end_T);
  void calculate_cuu(const VecOfVecXd &x, const VecOfVecXd &u, VecOfMatXd &c_uu,
                     int start_T, int end_T);
};