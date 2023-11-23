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

#include "xamg_headers.h"
#include "xamg_types.h"

#include "primitives/matrix/matrix.h"
#include "primitives/vector/vector.h"

#include "blas/blas.h"
#include "blas/blas_merged.h"

#include "param/params.h"

#include "convergence.h"

#include <chrono>

namespace XAMG {
namespace solver {

template <typename F, uint16_t NV>
struct base_solver;

template <typename F, uint16_t NV>
struct solver_stats {
    uint16_t norm_type = L2_norm;
    XAMG::params::param_list residual_control_params;
    std::vector<bool> if_converged;
    std::vector<float64_t> abs_res;
    std::vector<float64_t> rel_res;
    std::vector<float64_t> init_res;
    vector::vector temp_norm;
    // --- residual_replacement
    vector::vector rr_temp_vector = mem::DISTRIBUTED;
    //---
    uint32_t iters = 0;
    uint32_t sub_iters = 0;
    std::string method;
    std::string sub_method;
    uint16_t conv_check = 0, conv_info = 0;
    bool conf_parsed = false, conf_abs_tol = false;
    // --- residual_replacement
    uint16_t residual_replacement = 0;
    //---
    float64_t abs_tol = 0.0, rel_tol = 0.0;
    base_solver<F, NV> *solver;
    convergence<F, NV> convergence_status;
    vector::vector *p_iconv = nullptr, *p_r0 = nullptr;
    solver_stats() {}
    void set_residual_control_list(const XAMG::params::param_list &list) {
        residual_control_params = list;
    }
    void configure(size_t nrows, std::shared_ptr<part::part> part) {
        update_conf();
        temp_norm.alloc<F>(1, NV);
        // --- residual_replacement
        if (residual_replacement) {
            rr_temp_vector.alloc<F>(nrows, NV);
            rr_temp_vector.set_part(part);
            blas::set_const<F, NV>(rr_temp_vector, 0.0);
        }
        // --- residual_replacement
        convergence_status.configure(residual_control_params, solver->solver_role);
    }
    void init(vector::vector &r0) {
        convergence_status.init();
        update_conf();
        p_r0 = &r0;
    }
    void init(vector::vector &r0, vector::vector &iconv) {
        convergence_status.init(iconv);
        p_iconv = &iconv;
        update_conf();
        p_r0 = &r0;
    }
    void set_top_solver_flag() { convergence_status.top_solver_flag = true; }
    void update_conf() {
        auto &param_list = solver->param_list;
        if (param_list.is_value_set("convergence_details") && !conf_parsed) {
            std::set<std::string> conv;
            misc::str_split(param_list.get_string("convergence_details"), ',', conv);
            if (conv.count("noprint")) {
                conv_info = 0;
            } else {
                conv_info = 1;
            }
            if (conv.count("nocheck")) {
                conv_check = 0;
            } else {
                if (conv.count("check-L2")) {
                    conv_check = 1;
                }
                if (conv.count("RR")) {
                    ASSERT(conv_check == 1 && "RR mode requires check.");
                    residual_replacement = 1;
                }
            }
            conf_parsed = true;
        }
        if (param_list.is_value_set("abs_tolerance") && param_list.is_value_set("rel_tolerance")) {
            abs_tol = param_list.get_float("abs_tolerance");
            rel_tol = param_list.get_float("rel_tolerance");
        }
        convergence_status.residual_replacement = residual_replacement;
        convergence_status.conv_check = conv_check;
        convergence_status.conv_info = conv_info;
        convergence_status.set_tolerance(abs_tol, rel_tol);
    }
    void update_results() {
        abs_res.resize(NV, 0.0);
        rel_res.resize(NV, 0.0);
        init_res.resize(NV, 0.0);
        iters = convergence_status.get_iters_counter();
        sub_iters = convergence_status.get_sub_iters_counter();
        if_converged = convergence_status.status;
        if (conv_check && convergence_status.res.size()) {
            for (uint16_t nv = 0; nv < NV; ++nv) {
                abs_res[nv] = std::sqrt(convergence_status.res[nv]);
                rel_res[nv] = std::sqrt(convergence_status.res[nv] / convergence_status.res0[nv]);
                init_res[nv] = std::sqrt(convergence_status.res0[nv]);
            }
        }
    }

    // --- residual_replacement
    bool rr_is_converged() {
        if (residual_replacement && convergence_status.rr_check()) {
            solver->get_residual(rr_temp_vector);
            blas::vector_norm<F, NV>(rr_temp_vector, norm_type, temp_norm);
            if (convergence_status.rr_deviation_check(temp_norm)) {
                return true;
            }
        }
        return false;
    }
    //--- residual_replacement

    bool internal_is_converged(const vector::vector &r_norm, vector::vector &iconv) {
        bool status = false;
        convergence_status.update_residual(r_norm);
        status = convergence_status.is_converged(r_norm, iconv);
        if (!status)
            status = rr_is_converged();
        return status;
    }

    bool internal_is_converged(const vector::vector &r_norm) {
        bool status = false;
        convergence_status.update_residual(r_norm);
        status = convergence_status.is_converged(r_norm);
        if (!status)
            status = rr_is_converged();
        return status;
    }

    std::vector<F> get_residual_norm() {
        assert(p_r0 != nullptr);
        solver->get_residual(*p_r0);
        blas::vector_norm<F, NV>(*p_r0, norm_type, temp_norm);
        return temp_norm.get_element<F>(0);
    }

    std::vector<F> get_residual_norm(const vector::vector &x, const vector::vector &b) {
        assert(p_r0 != nullptr);
        solver->get_residual(x, b, *p_r0);
        blas::vector_norm<F, NV>(*p_r0, norm_type, temp_norm);
        return temp_norm.get_element<F>(0);
    }

    bool is_converged_initial(vector::vector &r0, vector::vector &r_norm) {
        solver->get_residual(r0);
        blas::vector_norm<F, NV>(r0, norm_type, r_norm);
        if (!conv_check)
            return false;
        convergence_status.set_initial_residual(r_norm);
        if (p_iconv == nullptr) {
            return internal_is_converged(r_norm);
        } else {
            return internal_is_converged(r_norm, *p_iconv);
        }
    }

    bool is_converged_initial(vector::vector &r0) {
        solver->get_residual(r0);
        if (!conv_check)
            return false;
        blas::vector_norm<F, NV>(r0, norm_type, temp_norm);
        convergence_status.set_initial_residual(temp_norm);
        if (p_iconv == nullptr) {
            return internal_is_converged(temp_norm);
        } else {
            return internal_is_converged(temp_norm, *p_iconv);
        }
    }

    bool is_converged_initial() {
        if (!conv_check)
            return false;
        assert(p_r0 != nullptr);
        solver->get_residual(*p_r0);
        blas::vector_norm<F, NV>(*p_r0, norm_type, temp_norm);
        convergence_status.set_initial_residual(temp_norm);
        if (p_iconv == nullptr) {
            return internal_is_converged(temp_norm);
        } else {
            return internal_is_converged(temp_norm, *p_iconv);
        }
    }

    bool is_converged_simple(vector::vector &r_norm) {
        if (!conv_check) {
            return false;
        }
        if (p_iconv == nullptr) {
            return internal_is_converged(r_norm);
        } else {
            return internal_is_converged(r_norm, *p_iconv);
        }
    }

    bool is_converged(vector::vector &r, vector::vector &r_norm) {
        solver->get_residual(r);
        blas::vector_norm<F, NV>(r, norm_type, r_norm);
        if (!conv_check) {
            return false;
        }
        if (p_iconv == nullptr) {
            return internal_is_converged(r_norm);
        } else {
            return internal_is_converged(r_norm, *p_iconv);
        }
    }

    bool is_converged(vector::vector &r) {
        solver->get_residual(r);
        if (!conv_check) {
            return false;
        }
        blas::vector_norm<F, NV>(r, norm_type, temp_norm);
        if (p_iconv == nullptr) {
            return internal_is_converged(temp_norm);
        } else {
            return internal_is_converged(temp_norm, *p_iconv);
        }
    }

    bool is_converged() {
        assert(p_r0 != nullptr);
        if (!conv_check) {
            return false;
        }
        solver->get_residual(*p_r0);
        blas::vector_norm<F, NV>(*p_r0, norm_type, temp_norm);
        if (p_iconv == nullptr) {
            return internal_is_converged(temp_norm);
        } else {
            return internal_is_converged(temp_norm, *p_iconv);
        }
    }

    void print_residuals_footer() {
        if (conv_check && conv_info) {
            io::print_residuals_footer(NV);
        }
    }

    uint16_t get_iters_counter() const { return convergence_status.get_iters_counter(); }

    uint16_t get_sub_iters_counter() const { return convergence_status.get_sub_iters_counter(); }

    uint16_t increment_iters_counter(uint16_t n = 1) {
        return convergence_status.increment_iters_counter(n);
    }

    uint16_t increment_sub_iters_counter(uint16_t n = 1) {
        return convergence_status.increment_sub_iters_counter(n);
    }

    void set_method(const std::string &meth) { method = meth; }
    void set_sub_method(const std::string &meth) { sub_method = meth; }
    std::string get_iters_info() const { return std::to_string(iters); }
    std::string get_full_iters_info() const {
        if (sub_iters)
            return (std::to_string(iters) + "(" + std::to_string(sub_iters) + ")");
        else
            return std::to_string(iters);
    }
    std::string get_method() { return method; }
    std::string get_full_method() {
        if (sub_method != "")
            return (method + "(" + sub_method + ")");
        else
            return method;
    }
};

} // namespace solver
} // namespace XAMG
