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

static inline void
params_handling_examples(XAMG::params::global_param_list &params,
                         std::shared_ptr<XAMG::solver::base_solver_interface> sol) {
    // NOTE: here are a few basic test cases for some solver parameters handling
    // functions

    // --- rel_tolerance reset example
    params.change_value<float32_t>("solver", "rel_tolerance", 1e-12);
    if (params.find_if<std::string>({{"method", "MultiGrid"}})) {
        params.forced_change_value<uint32_t>("preconditioner", "mg_cycle", 2);
    }
    sol->renew_params(params);

    // --- override mg_num_levels
    std::string solver_type;
    if (params.find_if<std::string>({{"method", "MultiGrid"}}, solver_type)) {
        params.forced_change_value<uint32_t>(solver_type, "mg_max_levels", 2);
        params.change_value_onlayer<uint32_t>("pre_smoother", "max_iters", 3, 3);
    }
    params.change_value_onlayer<uint32_t>("pre_smoother", "max_iters", 12, 3);
    sol->renew_params(params);
    params.print();

    // --- getting minmax diapason for a parameter example
    std::pair<uint32_t, uint32_t> minmax;
    if (params.get("solver").get_minmax<uint32_t>("mg_coarsening_type", minmax)) {
        std::cout << ">> *** mg_coarsening_type: {" << minmax.first << "," << minmax.second << "}"
                  << std::endl;
    }
    if (params.get("solver").get_minmax<uint32_t>("mg_interpolation_type", minmax)) {
        std::cout << ">> *** mg_interpolation_type: {" << minmax.first << "," << minmax.second
                  << "}" << std::endl;
    }
}

template <typename F>
void tst_solver_test(XAMG::matrix::matrix &m, XAMG::vector::vector &x,
                     const XAMG::vector::vector &y, XAMG::params::global_param_list &params,
                     uint64_t niters, tst_store_output &tst_output) {
    double dt = 0;
    constexpr uint64_t nwarmups = 1;
    niters = niters + nwarmups;

    params.print();

    // solving SLAE y = A*x
    auto sol = XAMG::solver::construct_solver_hierarchy<F, NV>(params, m, x, y);
    sol->matrix_info();

#if 0
    params_handling_examples(params, sol);
#endif

    // monitor.activate_group("main");
    monitor.activate_group("hsgs");
    monitor.activate_group("node_recv");

    std::vector<std::vector<double>> abs_res_history;
    abs_res_history.resize(NV);
    for (uint64_t it = 0; it < niters; ++it) {
        perf.reset();

        std::vector<F> a;
        for (size_t m = 0; m < NV; ++m)
            a.push_back(m);
        XAMG::blas::set_const<F, NV>(x, a);
        //XAMG::blas::set_rand<F, NV>(x, false);
        XAMG::out.norm<F, NV>(x, "x0");
        XAMG::out.norm<F, NV>(y, "y0");
        if (it == 0) {
            tst_output.store_xamg_vector_norm<F, NV>("solver", "X_initial_norm", x);
        }

        monitor.reset();
        monitor.enable();
#ifdef ITAC_TRACE
        VT_traceon();
#endif
        XAMG::mpi::barrier();
        double t1 = XAMG::sys::timer();
        sol->solve();
        XAMG::mpi::barrier();
        double t2 = XAMG::sys::timer();
        double local_dt = (t2 - t1);
        if (it >= nwarmups) {
            dt += local_dt;
        }
#ifdef ITAC_TRACE
        VT_traceoff();
#endif
        monitor.disable();

        perf.print();

        std::string method = sol->param_list.get_string("method");

        auto &stats = sol->get_stats();

        XAMG::out << XAMG::SUMMARY << "Solver: " << stats.get_full_method()
                  << "\tMatrix size: " << sol->A.info.nrows
                  << "\tMatrix nonzeros: " << sol->A.info.nonzeros << std::endl;
        XAMG::out << XAMG::SUMMARY << "Nvecs: " << NV << "   nprocs: " << id.gl_nprocs
                  << "   nnumas: " << id.nd_nnumas << "   ncores: " << id.nm_ncores
                  << "   Solver time: " << local_dt
                  << " sec \tIters: " << stats.get_full_iters_info() << std::endl;
        for (uint16_t nv = 0; nv < NV; ++nv) {
            XAMG::out << XAMG::SUMMARY << "Vec: " << nv << "\tConverged: " << stats.if_converged[nv]
                      << "\tAbs.residual: " << stats.abs_res[nv]
                      << "\tRel.residual: " << stats.rel_res[nv] << std::endl;
            abs_res_history[nv].push_back(stats.abs_res[nv]);
        }
        monitor.print();
    }
    bool compare_status = true;
    for (uint16_t nv = 0; nv < NV; ++nv) {
        auto &h = abs_res_history[nv];
        std::sort(h.begin(), h.end());
        if (h.front() != h.back()) {
            XAMG::out << XAMG::SUMMARY
                      << "ERROR: absolute residuals are different on solver runs: " << h.front()
                      << ", " << h.back() << " (for vector #" << nv << ")" << std::endl;
            compare_status = false;
        }
    }
    if (!compare_status) {
        XAMG::out << "FATAL: Residuals are not equal for identical solve() runs." << std::endl;
        return;
    }

    auto &stats = sol->get_stats();
    XAMG::out.norm<F, NV>(x, "X");
    tst_output.store_xamg_vector_norm<F, NV>("solver", "X_norm", x);
    tst_output.store_xamg_vector_norm<F, NV>("solver", "Y_norm", y);
    XAMG::out.vector<F>(stats.get_residual_norm(x, y), "||Y - A * X||");
    // Next line must do the same:
    //XAMG::out.vector<F>(stats.get_residual_norm(), "||Y - A * X||");

    tst_output.store_item("solver", "iters", stats.iters);
    tst_output.store_item("solver", "converged", stats.if_converged);
    tst_output.store_item("solver", "abs_residual", stats.abs_res);
    tst_output.store_item("solver", "rel_residual", stats.rel_res);
    tst_output.store_item("solver", "initial_residual", stats.init_res);

    tst_output.store_item("info", "nrows", sol->A.info.nrows);
    tst_output.store_item("info", "nonzeros", sol->A.info.nonzeros);
    tst_output.store_item("info", "NV", NV);
    tst_output.store_item("info", "nprocs", id.gl_nprocs);

    tst_output.store_item("topo", "nnumas", id.nd_nnumas);
    tst_output.store_item("topo", "ncores", id.nd_ncores);

    tst_output.store_item("timing", "solver", dt / (niters - nwarmups));
}
