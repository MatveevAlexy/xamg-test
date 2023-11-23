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
struct IterativeRefinement : public base_solver<F, NV> {
    const uint16_t nvecs = 2;
    const uint16_t comm_size = 0;
    std::shared_ptr<matrix::matrix> A32;
    DECLARE_INHERITED_FROM_BASESOLVER(IterativeRefinement)
    std::shared_ptr<base_solver<F, NV>> internal_solver;
    std::shared_ptr<base_solver<float32_t, NV>> internal_solver_reduced;
    std::shared_ptr<vector::vector> internal_x;
    std::shared_ptr<vector::vector> internal_b;
    virtual void setup(const params::global_param_list &params) override;
    virtual void renew_params(const params::global_param_list &params,
                              bool solver_mode = true) override;
    virtual void init(const XAMG::params::param_list &list,
                      const std::string &solver_role_ = "meta_solver") override {
        base_solver<F, NV>::solver_role = solver_role_;
        base::init_base(list);
    }
};

template <typename F, uint16_t NV>
void IterativeRefinement<F, NV>::matrix_info() {
    auto reduced_precision = param_list.get_bool("subsolver_reduced_precision");
    if (reduced_precision) {
        internal_solver_reduced->matrix_info();
    } else {
        internal_solver->matrix_info();
    }
}

template <typename F, uint16_t NV>
void IterativeRefinement<F, NV>::setup(const XAMG::params::global_param_list &params) {
    base::setup(params);

    internal_x = std::make_shared<XAMG::vector::vector>(mem::DISTRIBUTED);
    internal_b = std::make_shared<XAMG::vector::vector>(mem::DISTRIBUTED);

    stats.set_sub_method(params.get("solver").get_string("method"));
    auto reduced_precision = param_list.get_bool("subsolver_reduced_precision");
    if (reduced_precision) {
        using matrix_t = XAMG::matrix::csr_matrix<F, uint32_t, uint32_t, uint32_t, uint32_t>;
        using matrix32_t =
            XAMG::matrix::csr_matrix<float32_t, uint32_t, uint32_t, uint32_t, uint32_t>;
        matrix_t csr;
        matrix32_t csr32;
        assert(A.row_part == A.col_part);

        matrix::unpack_to_csr(A, csr);
        matrix::convert(csr, csr32);

        A32 = std::make_shared<matrix::matrix>(A.alloc_mode);
        matrix::construct_distributed<matrix32_t>(A.row_part, csr32, *A32);
        internal_solver_reduced = construct_solver_hierarchy<float32_t, NV>(params, *A32, "solver");
        internal_x->alloc<float32_t>(A.row_part->numa_layer.block_size[id.nd_numa], NV);
        internal_b->alloc<float32_t>(A.row_part->numa_layer.block_size[id.nd_numa], NV);
    } else {
        internal_solver = construct_solver_hierarchy<F, NV>(params, A, "solver");
        internal_x->alloc<F>(A.row_part->numa_layer.block_size[id.nd_numa], NV);
        internal_b->alloc<F>(A.row_part->numa_layer.block_size[id.nd_numa], NV);
    }
    internal_x->set_part(A.row_part);
    internal_b->set_part(A.row_part);
}

template <typename F, uint16_t NV>
void IterativeRefinement<F, NV>::renew_params(const params::global_param_list &params,
                                              bool solver_mode) {
    auto reduced_precision = param_list.get_bool("subsolver_reduced_precision");
    base::renew_params(params, true);
    if (reduced_precision) {
        internal_solver_reduced->renew_params(params);
    } else {
        internal_solver->renew_params(params);
    }
}

template <typename F, uint16_t NV>
void IterativeRefinement<F, NV>::solve(const vector::vector &conv, XAMG::mpi::token &tok) {
    vector::vector &x = *this->x;
    const vector::vector &b = *this->b;

    auto max_iters = param_list.get_int("max_iters");
    auto reduced_precision = param_list.get_bool("subsolver_reduced_precision");

    // inverted convergence flag; used to switch off updates to converged RHSs
    //vector::vector iconv(conv);

    const vector::vector &a1 = blas::ConstVectorsCache<F>::get_ones_vec(NV);

    vector::vector &r = buffer[0];
    vector::vector &x_corr = buffer[1];
    assert(nvecs == 2);
    base::set_buffers_zero();

    ///////////////////

    stats.init(r);
    if (stats.is_converged_initial(r)) {
        return;
    }

    // blas::xdivy_z<F, NV>(a1, res0, inv_res);
    // blas::pwr<F, NV>(inv_res, 0.5);

    ////////////////////
    perf.stop();
    if (reduced_precision) {
        auto &internal_stats = internal_solver_reduced->get_stats();
        auto tol = stats.convergence_status.export_per_nv_tolerance();
        internal_stats.convergence_status.set_tolerance(tol);
        for (bool last_iter = false; !last_iter;) {
            auto it = stats.increment_iters_counter();
            if (it >= max_iters)
                last_iter = true;
            if (it == 2)
                perf.start();
            // blas::scal<float64_t, NV>(inv_res, r);
            // out.vector<float64_t>(inv_res, std::string("inv_r"));
            vector::convert<F, float32_t>(r, *internal_b);
            blas::set_const<float32_t, NV>(*internal_x, 0);

            internal_solver_reduced->solve(*internal_x, *internal_b);

            vector::convert<float32_t, F>(*internal_x, x_corr);
            // blas::scal<F, NV>(res, x_corr);
            blas::axpby<F, NV>(a1, x_corr, a1, x);
            stats.increment_sub_iters_counter(internal_stats.get_iters_counter());
            if (stats.is_converged(r)) {
                return;
            }
            perf.stop();
        }
        internal_stats.convergence_status.reset_tolerance();
    } else {
        auto &internal_stats = internal_solver->get_stats();
        auto tol = stats.convergence_status.export_per_nv_tolerance();
        internal_stats.convergence_status.set_tolerance(tol);
        for (bool last_iter = false; !last_iter;) {
            auto it = stats.increment_iters_counter();
            if (it >= max_iters)
                last_iter = true;
            if (it == 2)
                perf.start();
            blas::set_const<F, NV>(x_corr, 0);

            internal_solver->solve(x_corr, r);

            // blas::scal<F, NV>(res, x_corr);
            blas::axpby<F, NV>(a1, x_corr, a1, x);
            stats.increment_sub_iters_counter(internal_stats.get_iters_counter());
            if (stats.is_converged(r)) {
                return;
            }
            perf.stop();
        }
        internal_stats.convergence_status.reset_tolerance();
    }

    // base::get_residual(r, res);
    // blas::pwr<F, NV>(res, 0.5);
    // blas::xdivy_z<F, NV>(a1, res, inv_res);

    stats.print_residuals_footer();
}

} // namespace solver
} // namespace XAMG
