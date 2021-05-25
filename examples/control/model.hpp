#pragma once

template <typename Algebra>
struct Model {
  using VectorX = typename Algebra::VectorX;

  virtual VectorX dynamics(const VectorX& x, const VectorX& u) = 0;
  virtual double cost(const VectorX& x, const VectorX& u) = 0;
  virtual double final_cost(const VectorX& x) = 0;

  virtual VectorX integrate_dynamics(const VectorX& x, const VectorX& u,
                                     double dt) {
    VectorX x1 = x + dynamics(x, u) * dt;
    return x1;
  }

  VectorX u_min, u_max;

  int x_dims;
  int u_dims;
};