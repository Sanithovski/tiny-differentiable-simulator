// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>
#include <thread>

#include "control/ilqr.h"
#include "control/model.hpp"
#include "dynamics/forward_dynamics.hpp"
#include "dynamics/integrator.hpp"
#include "math/eigen_algebra.hpp"
#include "math/tiny/tiny_double_utils.h"
#include "opengl_urdf_visualizer.h"
#include "tiny_visual_instance_generator.h"
#include "urdf/urdf_cache.hpp"
#include "urdf/urdf_parser.hpp"
#include "urdf/urdf_to_multi_body.hpp"
#include "utils/file_utils.hpp"

const double Pi = 3.141592653589793238462643383279;
const double HalfPi = Pi / 2.0;

/**
 * Uses a mechanism defined in a URDF file to create a controllable system where
 * [q, qd] is the state, and predefined actuation indices are used for the
 * control input.
 */
template <typename Algebra>
struct UrdfDynamics : public Model<Algebra> {
  using Model<Algebra>::u_min;
  using Model<Algebra>::u_max;
  using Model<Algebra>::x_dims;
  using Model<Algebra>::u_dims;

  using Scalar = typename Algebra::Scalar;
  using VectorX = typename Algebra::VectorX;

  tds::World<Algebra> world;
  tds::MultiBody<Algebra>* system;

  // make sure to use a thread-local cache to prevent any file system access
  static thread_local inline tds::UrdfCache<Algebra> cache{};

  UrdfDynamics(const std::string& urdf_filename,
               const std::vector<int>& control_indices = {},
               OpenGLUrdfVisualizer<tds::EigenAlgebra>* viz = nullptr) {
    std::string system_file_name;
    bool found = tds::FileUtils::find_file(urdf_filename, system_file_name);
    if (!found) {
      throw std::runtime_error("Could not find URDF file.");
    }
    tds::UrdfStructures<Algebra>& urdf_structures =
        cache.retrieve(system_file_name);
    system = world.create_multi_body();
    TinyVisualInstanceGenerator<Algebra>* vig = nullptr;
    if constexpr (std::is_same_v<Scalar, double>) {
      if (viz) {
        vig = new TinyVisualInstanceGenerator<Algebra>(*viz);
      }
    }
    tds::UrdfToMultiBody<Algebra>::convert_to_multi_body(urdf_structures, world,
                                                         *system, vig);
    system->initialize();
    if (!control_indices.empty()) {
      system->set_control_indices(control_indices);
    }

    x_dims = system->dof() + system->dof_qd();
    // assume q and qd have the same dimension,
    // we can modify the integration in iLQR otherwise
    assert(system->dof() == system->dof_qd());

    u_dims = system->dof_actuated();
    u_min = Algebra::zerox(u_dims);
    u_max = Algebra::zerox(u_dims);
    for (int i = 0; i < u_dims; ++i) {
      u_min[i] = Algebra::from_double(-100.);
      u_max[i] = Algebra::from_double(100.);
    }

    if constexpr (std::is_same_v<Scalar, double>) {
      if (viz) {
        char search_path[TINY_MAX_EXE_PATH_LEN];
        std::string texture_path = "";
        tds::FileUtils::extract_path(system_file_name.c_str(), search_path,
                                     TINY_MAX_EXE_PATH_LEN);
        viz->m_path_prefix = search_path;
        viz->convert_visuals(urdf_structures, texture_path);

        int num_total_threads = 1;
        std::vector<int> visual_instances;
        std::vector<int> num_instances;
        int num_base_instances;

        for (int t = 0; t < num_total_threads; t++) {
          ::TINY::TinyVector3f pos(0, 0, 0);
          ::TINY::TinyQuaternionf orn(0, 0, 0, 1);
          ::TINY::TinyVector3f scaling(1, 1, 1);
          int uid = urdf_structures.base_links[0]
                        .urdf_visual_shapes[0]
                        .visual_shape_uid;
          OpenGLUrdfVisualizer<Algebra>::TinyVisualLinkInfo& vis_link =
              viz->m_b2vis[uid];
          int instance = -1;
          int num_instances_per_link = 0;
          for (int v = 0; v < vis_link.visual_shape_uids.size(); v++) {
            int sphere_shape = vis_link.visual_shape_uids[v];
            ::TINY::TinyVector3f color(1, 1, 1);
            // viz->m_b2vis
            instance = viz->m_opengl_app.m_renderer->register_graphics_instance(
                sphere_shape, pos, orn, color, scaling);
            visual_instances.push_back(instance);
            num_instances_per_link++;
            system->visual_instance_uids().push_back(instance);
          }
          num_base_instances = num_instances_per_link;

          for (int i = 0; i < system->num_links(); ++i) {
            int uid =
                urdf_structures.links[i].urdf_visual_shapes[0].visual_shape_uid;
            OpenGLUrdfVisualizer<Algebra>::TinyVisualLinkInfo& vis_link =
                viz->m_b2vis[uid];
            int instance = -1;
            int num_instances_per_link = 0;
            for (int v = 0; v < vis_link.visual_shape_uids.size(); v++) {
              int sphere_shape = vis_link.visual_shape_uids[v];
              ::TINY::TinyVector3f color(1, 1, 1);
              // viz->m_b2vis
              instance =
                  viz->m_opengl_app.m_renderer->register_graphics_instance(
                      sphere_shape, pos, orn, color, scaling);
              visual_instances.push_back(instance);
              num_instances_per_link++;

              system->links_[i].visual_instance_uids.push_back(instance);
            }
            num_instances.push_back(num_instances_per_link);
          }
        }
      }
    }
  }

  VectorX dynamics(const VectorX& x, const VectorX& u) override {
    // allocate vector for the state differential
    static thread_local VectorX xd = Algebra::zerox(x_dims);

    system->clear_forces();
    int i = 0;
    for (int j = 0; j < system->dof(); ++j) {
      system->q(j) = x[i++];
    }
    for (int j = 0; j < system->dof_qd(); ++j) {
      system->qd(j) = x[i++];
    }
    i = 0;
    for (int j : system->control_indices()) {
      system->tau(j) = u[i++];
    }
    tds::forward_dynamics(*system, world.get_gravity());

    // return [qd, qdd] (change in state)
    for (int j = 0; j < system->dof_qd(); ++j) {
      xd[j] = system->qd(j);
      xd[j + system->dof_qd()] = system->qdd(j);
    }

    return xd;
  }

  void visualize_trajectory(OpenGLUrdfVisualizer<tds::EigenAlgebra>* viz,
                            const VecOfVecXd& xs, double dt = 0.01) {
    if constexpr (std::is_same_v<Scalar, double>) {
      for (const VectorX& x : xs) {
        for (int i = 0; i < system->dof(); ++i) {
          system->q(i) = x[i];
        }
        tds::forward_kinematics(*system);
        viz->sync_visual_transforms(system);
        viz->render();
        std::this_thread::sleep_for(
            std::chrono::duration<double>(Algebra::to_double(dt)));
      }
    }
  }
};

template <typename Algebra>
struct CartpoleDynamics : public UrdfDynamics<Algebra> {
  using Scalar = typename Algebra::Scalar;

  // can only control cart, i.e. actuate just first entry of tau
  CartpoleDynamics(OpenGLUrdfVisualizer<tds::EigenAlgebra>* viz = nullptr)
      : UrdfDynamics<Algebra>("cartpole.urdf", {0}, viz) {}

  double cost(const VectorX& x, const VectorX& u) override {
    // minimize control effort
    return 1e-3 * Algebra::sqnorm(u) + x[1] * x[1] + 1e-1 * x[0] * x[0] +
           1e-1 * x[2] * x[2] + 1e-1 * x[3] * x[3];
  }

  double final_cost(const VectorX& x) override {
    // achieve swing-up
    return Algebra::sqnorm(x) + x[1] * x[1] ;
  }
};

int main(int argc, char* argv[]) {
  using Algebra = typename tds::EigenAlgebra;
  using VectorX = typename Algebra::VectorX;

  // create graphics
  OpenGLUrdfVisualizer<Algebra> visualizer;

  CartpoleDynamics<Algebra> dynamics(&visualizer);

  double dt = 0.01;
  int num_time_steps = 250;

  iLQR ilqr(&dynamics, dt);

  VectorX x0 = Algebra::zerox(dynamics.x_dims);
  x0[1] = Pi;  // pole is all the way down
  VecOfVecXd u0(num_time_steps);
  for (int t = 0; t < num_time_steps; ++t) {
    u0[t] = Algebra::zerox(dynamics.u_dims);
    // initialize with sinusoidal force trajectory
    u0[t][0] = rand() * 2.0 / RAND_MAX - 1.0;
  }

  double c0 = ilqr.init_traj(x0, u0);
  std::cout << "Initial state cost: " << c0 << std::endl;
  dynamics.visualize_trajectory(&visualizer, ilqr.get_xs(), dt);

  ilqr.generate_trajectory(x0, u0);

  std::cout << "Solution controls:\n";
  for (const auto& u : ilqr.get_us()) {
    std::cout << "  " << u;
  }
  std::cout << std::endl;

  while (!visualizer.m_opengl_app.m_window->requested_exit()) {
    std::cout << "Rolling out solution..." << std::endl;
    dynamics.visualize_trajectory(&visualizer, ilqr.get_xs(), dt);
  }

  visualizer.delete_all();

  printf("finished\n");
  return EXIT_SUCCESS;
}
