/****************************************************************************
** 
**  Copyright (C) 2019-2021 Boris Krasnopolsky, Alexey Medvedev
**  Contact: xamg-test@imec.msu.ru
** 
**  This file is part of the XAMG library.
** 
**  Commercial License Usage
**  Licensees holding valid commercial XAMG licenses may use this file in
**  accordance with the terms of commercial license agreement.
**  The license terms and conditions are subject to mutual agreement
**  between Licensee and XAMG library authors signed by both parties
**  in a written form.
** 
**  GNU General Public License Usage
**  Alternatively, this file may be used under the terms of the GNU
**  General Public License, either version 3 of the License, or (at your
**  option) any later version. The license is as published by the Free 
**  Software Foundation and appearing in the file LICENSE.GPL3 included in
**  the packaging of this file. Please review the following information to
**  ensure the GNU General Public License requirements will be met:
**  https://www.gnu.org/licenses/gpl-3.0.html.
** 
****************************************************************************/

#pragma once

namespace XAMG {
namespace solver {

template <typename F, uint16_t NV>
struct MultiGrid : public base_solver<F, NV> {
    const uint16_t nvecs = 1;
    const uint16_t comm_size = 0;
    DECLARE_INHERITED_FROM_BASESOLVER(MultiGrid)
    std::shared_ptr<base_iface> coarse_grid_solver;
    std::vector<std::shared_ptr<base_iface>> pre_smooth;
    std::vector<std::shared_ptr<base_iface>> post_smooth;
    std::vector<matrix::mg_layer> mg_tree;
    virtual void setup(const params::global_param_list &params) override;
    virtual void renew_params(const params::global_param_list &params,
                              bool solver_mode = true) override;
    template <typename T1, typename T2>
    void MG_cycle(vector::vector &x, const vector::vector &b, uint8_t level, uint16_t mg_cycle,
                  const vector::vector &iconv);
};

template <typename F, uint16_t NV>
void MultiGrid<F, NV>::matrix_info() {
    assert(mg_tree.size() > 0);

    A.info.print_header();
    for (size_t lev = 0; lev < mg_tree.size(); ++lev) {
        mg_tree[lev].get_A().info.print("A[" + std::to_string(lev) + "]", true);
        // mg_tree[lev].get_A().plot("topo/mat_A-"+std::to_string(lev));
    }
    A.info.print_footer();

    mg_tree[0].get_R().info.print_header();
    for (size_t lev = 0; lev < mg_tree.size() - 1; ++lev) {
        mg_tree[lev].get_R().info.print("R[" + std::to_string(lev) + "]", true);
        // mg_tree[lev].get_R().plot("topo/mat_R-"+std::to_string(lev));
    }
    mg_tree[0].get_R().info.print_footer();

    mg_tree[0].get_P().info.print_header();
    for (size_t lev = 0; lev < mg_tree.size() - 1; ++lev) {
        mg_tree[lev].get_P().info.print("P[" + std::to_string(lev) + "]", true);
        // mg_tree[lev].get_P().plot("topo/mat_P-"+std::to_string(lev));
    }
    mg_tree[0].get_P().info.print_footer();
}

template <typename F, uint16_t NV>
void MultiGrid<F, NV>::renew_params(const params::global_param_list &params, bool solver_mode) {
    if (solver_mode) {
        base::renew_params(params, true);
    }
    assert(params.find("pre_smoother"));
    assert(params.find("post_smoother"));
    assert(params.find("coarse_grid_solver"));
    assert(coarse_grid_solver);

    {
        auto num_levels = pre_smooth.size();
        params::xamg_overrides_holder holder(num_levels);
        std::string override_key = "pre_smoother_override";
        if (params.find(override_key)) {
            const XAMG::params::param_list &overrides = params.get(override_key);
            holder.fill_in(overrides);
            for (size_t lev = 0; lev < num_levels; lev++) {
                if (holder.find(lev)) {
                    auto &per_level_overrides = holder.get(lev);
                    pre_smooth[lev]->renew_param_list(per_level_overrides);
                }
            }
        }
    }
    {
        auto num_levels = post_smooth.size();
        params::xamg_overrides_holder holder(num_levels);
        std::string override_key = "post_smoother_override";
        if (params.find(override_key)) {
            const XAMG::params::param_list &overrides = params.get(override_key);
            holder.fill_in(overrides);
            for (size_t lev = 0; lev < num_levels; lev++) {
                if (holder.find(lev)) {
                    auto &per_level_overrides = holder.get(lev);
                    post_smooth[lev]->renew_param_list(per_level_overrides);
                }
            }
        }
    }
    coarse_grid_solver->renew_param_list(params.get("coarse_grid_solver"));
}

template <typename F, uint16_t NV>
void MultiGrid<F, NV>::setup(const XAMG::params::global_param_list &params) {
    base::setup(params);

    std::string solver_type;
    const auto &list = params.get_if<std::string>({{"method", "MultiGrid"}}, solver_type);
    auto hypre_per_level_hierarchy = list.get_bool("hypre_per_level_hierarchy");

    if (hypre_per_level_hierarchy)
        hypre::get_per_level_hierarchy<F>(A, mg_tree, params);
    else
        hypre::get_full_hierarchy<F>(A, mg_tree, params);

    assert(params.find("pre_smoother"));
    assert(params.find("post_smoother"));
    assert(params.find("coarse_grid_solver"));
    for (uint8_t nl = 0; nl < mg_tree.size() - 1; ++nl) {
        if (mg_tree[nl].get_A().reduced_prec) {
            pre_smooth.push_back(construct_basic_solver<float32_t, NV>(
                params, params.get("pre_smoother", nl), mg_tree[nl].get_A(), "pre_smoother"));
            post_smooth.push_back(construct_basic_solver<float32_t, NV>(
                params, params.get("post_smoother", nl), mg_tree[nl].get_A(), "post_smoother"));
        } else {
            // F -> float64_t
            pre_smooth.push_back(construct_basic_solver<F, NV>(
                params, params.get("pre_smoother", nl), mg_tree[nl].get_A(), "pre_smoother"));
            post_smooth.push_back(construct_basic_solver<F, NV>(
                params, params.get("post_smoother", nl), mg_tree[nl].get_A(), "post_smoother"));
        }
    }
    if (mg_tree.back().get_A().reduced_prec)
        coarse_grid_solver = construct_basic_solver<float32_t, NV>(
            params, params.get("coarse_grid_solver"), mg_tree.back().get_A());
    else
        coarse_grid_solver = construct_basic_solver<F, NV>(
            params, params.get("coarse_grid_solver"), mg_tree.back().get_A(), "coarse_grid_solver");

    /////////

    uint16_t buffer_nvecs = 3;

    for (uint8_t nl = 0; nl < mg_tree.size(); ++nl) {
        uint32_t block_nrows = mg_tree[nl].get_A().row_part->numa_layer.block_size[id.nd_numa];

        if (nl == 0) {
            mg_tree[nl].buffer.emplace_back(vector::vector(mem::DISTRIBUTED));
            mg_tree[nl].buffer.back().alloc<F>(block_nrows, NV);

            if (mg_tree[nl].get_R().reduced_prec) {
                mg_tree[nl].buffer.emplace_back(vector::vector(mem::DISTRIBUTED));
                mg_tree[nl].buffer.back().alloc<float32_t>(block_nrows, NV);
            }
        } else {
            for (size_t i = 0; i < buffer_nvecs; ++i) {
                mg_tree[nl].buffer.emplace_back(vector::vector(mem::DISTRIBUTED));
                if (mg_tree[nl].get_A().reduced_prec) {
                    mg_tree[nl].buffer.back().alloc<float32_t>(block_nrows, NV);
                } else {
                    mg_tree[nl].buffer.back().alloc<F>(block_nrows, NV);
                }
            }
        }

        for (auto &buf : mg_tree[nl].buffer) {
            buf.set_part(mg_tree[nl].get_A().row_part);
        }
    }
}

template <typename F, uint16_t NV>
template <typename T1, typename T2>
void MultiGrid<F, NV>::MG_cycle(vector::vector &x, const vector::vector &b, uint8_t level,
                                uint16_t mg_cycle, const vector::vector &iconv) {
    monitor.start("alloc");
    const vector::vector &a0 = blas::ConstVectorsCache<T1>::get_zeroes_vec(NV);
    const vector::vector &a1 = blas::ConstVectorsCache<T1>::get_ones_vec(NV);
    const vector::vector &a_1 = blas::ConstVectorsCache<T1>::get_minus_ones_vec(NV);

    //////////

    vector::vector rho0, res;
    rho0.alloc<T1>(1, NV);
    res.alloc<T1>(1, NV);

    vector::vector iconv2;
    iconv2.alloc<T2>(1, NV);
    monitor.stop("alloc");

    /////////

    matrix::matrix &A = mg_tree[level].get_A();
    matrix::matrix &R = mg_tree[level].get_R();
    matrix::matrix &P = mg_tree[level].get_P();

    vector::vector &temp = mg_tree[level].buffer[0];
    vector::vector &b_H = mg_tree[level + 1].buffer[1];
    vector::vector &x_H = mg_tree[level + 1].buffer[2];

    /////////////////////

    auto pre_smoother = pre_smooth[level];
    if (mg_cycle != F_cycle)
        pre_smoother->solve(x, b, iconv);

    /////////

    blas2::Ax_y<T1, NV>(A, x, temp, NV);
    blas::axpby<T1, NV>(a1, b, a_1, temp);

    /////////
    // Restriction:

    if (typeid(T1).hash_code() != typeid(T2).hash_code()) {
        //    temp -> 32 bit
        auto &temp32 = mg_tree[level].buffer[1];
        vector::convert<T1, T2>(temp, temp32);
        blas2::Ax_y<T2, NV>(R, temp32, b_H, NV);
        vector::convert<T1, T2>(iconv, iconv2);
    } else {
        blas2::Ax_y<T2, NV>(R, temp, b_H, NV);
        iconv2 = iconv;
    }
    blas::set_const<T2, NV>(x_H, 0);

    if (level + 1 == (uint8_t)(mg_tree.size() - 1)) {
        coarse_grid_solver->solve(x_H, b_H, iconv2);
        // b_H.print<F>("CGS:B");
        // x_H.print<F>("CGS:X");

        // XAMG::out << XAMG::DBG;
        // XAMG::out.norm<T2, NV>(x_H, "CGS:x_H");
    } else {
        MG_cycle<T2, T2>(x_H, b_H, level + 1, mg_cycle, iconv2);

        if (mg_cycle == W_cycle)
            MG_cycle<T2, T2>(x_H, b_H, level + 1, W_cycle, iconv2);
        else if (mg_cycle == F_cycle)
            MG_cycle<T2, T2>(x_H, b_H, level + 1, V_cycle, iconv2);
    }

    ////

    // Prolongation:

    if (typeid(T1).hash_code() != typeid(T2).hash_code()) {
        auto &temp32 = mg_tree[level].buffer[1];
        blas2::Ax_y<T2, NV>(P, x_H, temp32, NV);
        // temp32.print<T2>("T32");
        // x.print<T1>("X_P");

        // T -> float64_t
        vector::convert<T2, T1>(temp32, temp);
        blas::axpby<T1, NV>(a1, temp, a1, x);

    } else {
        blas2::Axpy<T1, NV>(P, x_H, x, NV);
    }

    auto post_smoother = post_smooth[level];
    post_smoother->solve(x, b, iconv);

    //    XAMG::out << XAMG::DBG;
    //    XAMG::out.norm<T1, NV>(x, "x*");
}

template <typename F, uint16_t NV>
void MultiGrid<F, NV>::solve(const vector::vector &conv, XAMG::mpi::token &tok) {
    assert(buffer.size() >= (size_t)nvecs);
    assert(mg_tree.size() > 0);
    vector::vector &x = *this->x;
    const vector::vector &b = *this->b;

    monitor.start("params");
    auto max_iters = param_list.get_int("max_iters");
    auto mg_cycle = param_list.get_int("mg_cycle");
    monitor.stop("params");

    monitor.start("alloc");
    // inverted convergence flag; used to switch off updates to converged RHSs
    vector::vector iconv(conv);
    monitor.stop("alloc");

    vector::vector &r = buffer[0];
    assert(nvecs == 1);
    base::set_buffers_zero();

    stats.init(r, iconv);
    if (stats.is_converged_initial()) {
        return;
    }

    ////////////////////

    assert(mg_tree.size() > 0);
    perf.stop();

    for (bool last_iter = false; !last_iter;) {
        auto it = stats.increment_iters_counter();
        if (it >= max_iters)
            last_iter = true;
        if (it == 2)
            perf.start();

        if (mg_tree[1].get_A().reduced_prec)
            MG_cycle<F, float32_t>(x, b, 0, mg_cycle, iconv);
        else
            MG_cycle<F, F>(x, b, 0, mg_cycle, iconv);

        if (stats.is_converged()) {
            return;
        }
        perf.stop();
    }
    stats.print_residuals_footer();
}

} // namespace solver
} // namespace XAMG
