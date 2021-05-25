#pragma once

#include "base.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/StdVector>
#include "eigen_helpers.hpp"


#define EIGEN_USE_NEW_STDVECTOR

using std::cout;
using std::endl;

using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::VectorXd;
typedef std::vector<VectorXd, Eigen::aligned_allocator<VectorXd>> VecOfVecXd;
typedef std::vector<MatrixXd, Eigen::aligned_allocator<MatrixXd>> VecOfMatXd;

//---------------------------------
// Math helper functions

const double pi = 3.141592653589793238462643383279;

template <typename T>
inline T sqr(const T &val) {
  return val * val;
}

template <typename T>
inline T cube(const T &val) {
  return val * val * val;
}

template <typename T>
inline int sgn(T &val) {
  return (T(0) < val) - (val < T(0));
}

inline double smooth_abs(double x, double alpha = 1.0) {
  // Differentiable "soft" absolute value function
  return sqrt(sqr(x) + sqr(alpha)) - alpha;
}

//---------------------------------
// Floating-point modulo
// The result (the remainder) has same sign as the divisor.
// Similar to matlab's mod(); Not similar to fmod() -   Mod(-3,4)= 1 fmod(-3,4)=
// -3 From
// http://stackoverflow.com/questions/4633177/c-how-to-wrap-a-float-to-the-interval-pi-pi
template <typename T>
inline T Mod(T x, T y) {
  // static_assert(!std::numeric_limits<T>::is_exact , "Mod: floating-point type
  // expected");
  if (0. == y) return x;
  double m = x - y * floor(x / y);
  // handle boundary cases resulted from floating-point cut off:
  if (y > 0) {   // modulo range: [0..y)
    if (m >= y)  // Mod(-1e-16, 360.): m= 360.
      return 0;

    if (m < 0) {
      if (y + m == y)
        return 0;  // just in case...
      else
        return y + m;  // Mod(106.81415022205296, _TWO_PI ): m= -1.421e-14
    }
  } else {       // modulo range: (y..0]
    if (m <= y)  // Mod(1e-16, -360.): m= -360.
      return 0;

    if (m > 0) {
      if (y + m == y)
        return 0;  // just in case...
      else
        return y + m;  // Mod(-106.81415022205296, -_TWO_PI): m= 1.421e-14
    }
  }

  return m;
}

inline double wrap_to_pi(double angle) { return Mod(angle + pi, 2 * pi) - pi; }