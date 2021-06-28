/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <stdexcept>
#include <vector>

#include "math/pose.hpp"
#include "sdf_utils.hpp"
#include "utils/conversion.hpp"
#include "utils/serialization.hpp"

namespace tds {
enum GeometryTypes {
  TINY_SPHERE_TYPE = 0,
  TINY_PLANE_TYPE,
  TINY_CAPSULE_TYPE,
  TINY_MESH_TYPE,     // only for visual shapes at the moment
  TINY_BOX_TYPE,      // only for visual shapes at the moment
  TINY_CYLINDER_TYPE, // unsupported
  TINY_MAX_GEOM_TYPE,
};

template <typename Algebra>
class Geometry : public Serializable<Algebra> {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using typename Serializable<Algebra>::Iter;
  using typename Serializable<Algebra>::ConstIter;

  int type;

public:
  explicit Geometry(int type) : type(type) {}
  virtual ~Geometry() = default;
  int get_type() const { return type; }

  // SDF related members
public:
  const Vector3 &get_max_boundaries() const { return max_boundaries; }
  const Vector3 &get_min_boundaries() const { return min_boundaries; }
  virtual typename Algebra::Scalar
  distance(const typename Algebra::Vector3 &point) const {
    std::string error_str(
        "SDF distance computation is not supported for this geometry type: ");
    error_str += std::to_string(type);
    throw std::runtime_error(error_str);
  }

protected:
  Vector3 max_boundaries;
  Vector3 min_boundaries;
  size_t serialization_size_(SerializationMode mode) const override;
  void serialize_(Iter& output, SerializationMode mode) const override;
  void deserialize_(ConstIter& input, SerializationMode mode) override;
};

template <typename Algebra> class Sphere : public Geometry<Algebra> {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  Scalar radius;

public:
  explicit Sphere(const Scalar &radius)
      : Geometry<Algebra>(TINY_SPHERE_TYPE), radius(radius) {
    this->max_boundaries =
        Vector3(Scalar(2) * radius, Scalar(2) * radius, Scalar(2) * radius);
    this->min_boundaries =
        Vector3(Scalar(-2) * radius, Scalar(-2) * radius, Scalar(-2) * radius);
  }

  template <typename AlgebraTo = Algebra> Sphere<AlgebraTo> clone() const {
    typedef Conversion<Algebra, AlgebraTo> C;
    return Sphere<AlgebraTo>(C::convert(radius));
  }

  const Scalar &get_radius() const { return radius; }
  void set_radius(const Scalar &radius) { this->radius = radius; }

  Vector3 compute_local_inertia(const Scalar &mass) const {
    Scalar elem = Algebra::fraction(4, 10) * mass * radius * radius;
    return Vector3(elem, elem, elem);
  }

  Scalar distance(const Vector3 &p) const override {
    return Algebra::norm(p) - radius;
  }
};

// capsule aligned with the Z axis
template <typename Algebra> class Capsule : public Geometry<Algebra> {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  Scalar radius;
  Scalar length;

public:
  explicit Capsule(const Scalar &radius, const Scalar &length)
      : Geometry<Algebra>(TINY_CAPSULE_TYPE), radius(radius), length(length) {
    Scalar bound = Scalar(1.2) * (radius + length / Scalar(2));
    this->max_boundaries = Vector3(bound, bound, bound);
    this->min_boundaries = Vector3(-bound, -bound, -bound);
  }

  template <typename AlgebraTo = Algebra> Capsule<AlgebraTo> clone() const {
    typedef Conversion<Algebra, AlgebraTo> C;
    return Capsule<AlgebraTo>(C::convert(radius), C::convert(length));
  }

  const Scalar &get_radius() const { return radius; }
  void set_radius(const Scalar &radius) { this->radius = radius; }

  const Scalar &get_length() const { return length; }
  void set_length(const Scalar &length) { this->length = length; }

  Vector3 compute_local_inertia(const Scalar &mass) const {
    Scalar lx = Algebra::fraction(2, 1) * (radius);
    Scalar ly = Algebra::fraction(2, 1) * (radius);
    Scalar lz = length + Algebra::fraction(2, 1) * (radius);
    Scalar x2 = lx * lx;
    Scalar y2 = ly * ly;
    Scalar z2 = lz * lz;
    Scalar scaledmass = mass * Algebra::fraction(1, 12);

    Vector3 inertia;
    inertia[0] = scaledmass * (y2 + z2);
    inertia[1] = scaledmass * (x2 + z2);
    inertia[2] = scaledmass * (x2 + y2);
    return inertia;
  }

  Scalar distance(const Vector3 &p) const override {
    Vector3 pt(p.x(), p.y(),
               p.z() - std::clamp(p.z(), -this->length / Scalar(2),
                                  this->length / Scalar(2)));
    return Algebra::norm(pt) - this->radius;
  }
};

template <typename Algebra> class Plane : public Geometry<Algebra> {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  Vector3 normal;
  Scalar constant;

public:
  Plane(const Vector3 &normal = Algebra::unit3_z(),
        const Scalar &constant = Algebra::zero(),
        const Scalar &bound = Scalar(500))
      : Geometry<Algebra>(TINY_PLANE_TYPE), normal(normal), constant(constant) {
    // TODO: Find a good boundary for redering planes
    // Scalar bound = constant > Scalar(5.0) ? Scalar(2) * constant :
    // Scalar(10.);
    this->max_boundaries = Vector3(bound, bound, bound);
    this->min_boundaries = Vector3(-bound, -bound, -bound);
  }

  template <typename AlgebraTo = Algebra> Plane<AlgebraTo> clone() const {
    typedef Conversion<Algebra, AlgebraTo> C;
    return Plane<AlgebraTo>(C::convert(normal), C::convert(constant));
  }

  const Vector3 &get_normal() const { return normal; }
  void set_normal(const Vector3 &normal) { this->normal = normal; }

  const Scalar &get_constant() const { return constant; }

  Scalar distance(const Vector3 &p) const override {
    return Algebra::dot(p, this->normal) / Algebra::norm(this->normal) -
           this->constant;
  }
  
  void set_constant(const Scalar &constant) { this->constant = constant; }
};

template <typename AlgebraFrom, typename AlgebraTo>
static TINY_INLINE Geometry<AlgebraTo> *clone(const Geometry<AlgebraFrom> *g) {
  switch (g->get_type()) {
  case TINY_SPHERE_TYPE:
    return new Sphere<AlgebraTo>(
        ((Sphere<AlgebraFrom> *)g)->template clone<AlgebraTo>());
  case TINY_CAPSULE_TYPE:
    return new Capsule<AlgebraTo>(
        ((Capsule<AlgebraFrom> *)g)->template clone<AlgebraTo>());
  case TINY_PLANE_TYPE:
    return new Plane<AlgebraTo>(
        ((Plane<AlgebraFrom> *)g)->template clone<AlgebraTo>());
  }
}

template <typename Algebra> class Cylinder : public Geometry<Algebra> {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  Scalar radius;
  Scalar length;

public:
  Cylinder(const Scalar &radius, const Scalar &length)
      : Geometry<Algebra>(TINY_CYLINDER_TYPE), radius(radius), length(length) {
    Scalar bound = (radius + length / Scalar(2));
    this->max_boundaries = Vector3(bound, bound, bound);
    this->min_boundaries = Vector3(-bound, -bound, -bound);
  }

  template <typename AlgebraTo = Algebra> Cylinder<AlgebraTo> clone() const {
    typedef Conversion<Algebra, AlgebraTo> C;
    return Cylinder<AlgebraTo>(C::convert(radius), C::convert(length));
  }

  const Scalar &get_radius() const { return radius; }
  const Scalar &get_length() const { return length; }

  Scalar distance(const Vector3 &p) const override {
    // The Marching Cubes algorithm is not very good at sharp edges
    // Define a small rounding radius to smooth the edge of the two faces
    Scalar rb = radius / Scalar(20); // Can be exposed to the users

    Scalar lxy = Algebra::sqrt(p.x() * p.x() + p.y() * p.y());
    Scalar lz = Algebra::abs(p.z());
    Scalar dx = lxy - radius + rb;
    Scalar dy = lz - length / Scalar(2) + rb;
    Scalar min_d = Algebra::min(Algebra::max(dx, dy), Algebra::zero());
    Scalar max_d = Algebra::sqrt(
        Algebra::max(dx, Algebra::zero()) * Algebra::max(dx, Algebra::zero()) +
        Algebra::max(dy, Algebra::zero()) * Algebra::max(dy, Algebra::zero()));
    return min_d + max_d - rb;
  }
};

template <typename Algebra>
size_t Geometry<Algebra>::serialization_size_(SerializationMode mode) const {
  if (mode | SERIALIZE_GEOMETRY) {
    switch (get_type()) {
      case TINY_SPHERE_TYPE:
        return 1;
      case TINY_CAPSULE_TYPE:
        return 2;
      case TINY_PLANE_TYPE:
        return 4;
    }
  }
  return 0;
}

template <typename Algebra>
void Geometry<Algebra>::serialize_(Iter& param_iter,
                                   SerializationMode mode) const {
  if (mode | SERIALIZE_GEOMETRY) {
    switch (get_type()) {
      case TINY_SPHERE_TYPE:
        *param_iter = ((Sphere<Algebra> *)this)->get_radius();
        param_iter = std::next(param_iter);
        break;
      case TINY_CAPSULE_TYPE:
        *param_iter = ((Capsule<Algebra> *)this)->get_radius();
        param_iter = std::next(param_iter);
        *param_iter = ((Capsule<Algebra> *)this)->get_length();
        param_iter = std::next(param_iter);
        break;
      case TINY_PLANE_TYPE:
        const Vector3 &normal = ((const Plane<Algebra> *)this)->get_normal();
        *param_iter = normal[0];
        param_iter = std::next(param_iter);
        *param_iter = normal[1];
        param_iter = std::next(param_iter);
        *param_iter = normal[2];
        param_iter = std::next(param_iter);
        *param_iter = ((const Plane<Algebra> *)this)->get_constant();
        param_iter = std::next(param_iter);
        break;
    }
  }
}

template <typename Algebra>
void Geometry<Algebra>::deserialize_(ConstIter& param_iter,
                                     SerializationMode mode) {
  if (mode | SERIALIZE_GEOMETRY) {
    switch (get_type()) {
      case TINY_SPHERE_TYPE:
        ((Sphere<Algebra> *)this)->set_radius(*param_iter);
        param_iter = std::next(param_iter);
        break;
      case TINY_CAPSULE_TYPE:
        ((Capsule<Algebra> *)this)->set_radius(*param_iter);
        param_iter = std::next(param_iter);
        ((Capsule<Algebra> *)this)->set_length(*param_iter);
        param_iter = std::next(param_iter);
        break;
      case TINY_PLANE_TYPE:
        Vector3 normal;
        normal[0] = *param_iter;
        param_iter = std::next(param_iter);
        normal[1] = *param_iter;
        param_iter = std::next(param_iter);
        normal[2] = *param_iter;
        param_iter = std::next(param_iter);
        ((Plane<Algebra> *)this)->set_normal(normal);
        ((Plane<Algebra> *)this)->set_constant(*param_iter);
        param_iter = std::next(param_iter);
        break;
    }
  }
}
}  // namespace tds
