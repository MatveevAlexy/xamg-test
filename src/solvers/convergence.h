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

#include "primitives/vector/vector.h"

#include "blas/blas.h"
#include "blas/blas_merged.h"

#include "param/params.h"

#include "io/io.h"

namespace XAMG {
namespace solver {

const float64_t ZERO = 1e-10;
std::map<std::string, std::string> solver_role_prefix{
    {"meta_solver", "META"},  {"solver", "SOLV"},        {"preconditioner", "PREC"},
    {"pre_smoother", "PRE_"}, {"post_smoother", "POST"}, {"coarse_grid_solver", "COAR"}};

template <typename F, uint16_t NV>
struct convergence {
  private:
    uint16_t iter = 0;
    uint16_t sub_iter = 0;

  public:
    std::vector<bool> status;
    std::vector<float64_t> res0;
    std::vector<float64_t> res;

    uint16_t conv_check = 0;
    uint16_t conv_info = 0;
    uint16_t residual_replacement = 0;
    float32_t rr_stepping = 0.0;
    float32_t rr_deviation = 0.0;
    float64_t rr_tol = 0.0;
    std::string solver_prefix;
    bool top_solver_flag = false;
    struct tolerance_holder {
        enum { PER_NV, CONST } type;
        virtual float64_t get(uint16_t nv) = 0;
        virtual void set(const std::vector<float64_t> &_tol) = 0;
        virtual ~tolerance_holder() {}
    };

    struct per_nv_tolerance : public tolerance_holder {
        using tolerance_holder::type;
        std::vector<float64_t> tol;
        per_nv_tolerance() { type = tolerance_holder::PER_NV; }
        virtual float64_t get(uint16_t nv) override {
            assert(tol.size() == NV);
            return tol[nv];
        }
        virtual void set(const std::vector<float64_t> &_tol) override {
            tol = _tol;
            assert(tol.size() == NV);
        }
        virtual ~per_nv_tolerance() {}
    };

    struct constant_tolerance : public tolerance_holder {
        using tolerance_holder::type;
        float64_t abs_tol = 0.0;
        float64_t rel_tol = 0.0;
        convergence<F, NV> *c;
        constant_tolerance(convergence<F, NV> *_c) : c(_c) { type = tolerance_holder::CONST; }
        virtual float64_t get(uint16_t nv) override {
            return std::max(abs_tol, rel_tol * std::sqrt(c->res0[nv]));
        }
        virtual void set(const std::vector<float64_t> &_tol) override {
            assert(_tol.size() == 2);
            abs_tol = _tol[0];
            rel_tol = _tol[1];
        }
        virtual ~constant_tolerance() {}
    };

    tolerance_holder *tolerance_values = nullptr;

    ~convergence() { reset_tolerance(); }

    void configure(const params::param_list &conv_param_list, const std::string &solver_role) {
        assert(solver_role_prefix.find(solver_role) != solver_role_prefix.end());
        solver_prefix = solver_role_prefix[solver_role];
        status.resize(NV);
        res0.resize(NV);
        res.resize(NV);
        // FIXME this doesnot allow to change these params after setup
        //--- residual_replacement
        if (residual_replacement) {
            rr_stepping = conv_param_list.get_float("RR_stepping");
            rr_deviation = conv_param_list.get_float("RR_deviation");
        }
        //--- residual_replacement
    }

    void set_tolerance(const std::vector<float64_t> &_tol) {
        if (tolerance_values) {
            if (tolerance_values->type != tolerance_holder::PER_NV) {
                return;
            }
        } else {
            tolerance_values = new per_nv_tolerance;
        }
        tolerance_values->set(_tol);
    }

    void set_tolerance(float64_t _abs_tol, float64_t _rel_tol) {
        if (tolerance_values) {
            if (tolerance_values->type != tolerance_holder::CONST) {
                return;
            }
        } else {
            tolerance_values = new constant_tolerance(this);
        }
        tolerance_values->set({_abs_tol, _rel_tol});
    }

    void reset_tolerance() {
        if (tolerance_values) {
            delete tolerance_values;
            tolerance_values = nullptr;
        }
    }

    std::vector<float64_t> export_per_nv_tolerance() {
        std::vector<float64_t> values;
        for (uint16_t i = 0; i < NV; ++i) {
            values.push_back(tolerance_values->get(i));
        }
        return values;
    }

    void init(const vector::vector &conv_vector) {
        assert(conv_vector.nv == NV);
        iter = sub_iter = 0;
        for (uint16_t nv = 0; nv < NV; ++nv) {
            res0[nv] = res[nv] = 0.0;
            status[nv] = (std::abs(1.0 - conv_vector.get_value<F>(nv)) > ZERO);
        }
        if (residual_replacement) {
            rr_tol = rr_stepping;
        }
    }

    void init() {
        iter = sub_iter = 0;
        for (uint16_t nv = 0; nv < NV; ++nv) {
            res0[nv] = res[nv] = 0.0;
            status[nv] = false;
        }
        if (residual_replacement) {
            rr_tol = rr_stepping;
        }
    }

    uint16_t increment_iters_counter(uint16_t n = 1) {
        iter += n;
        return iter;
    }

    uint16_t increment_sub_iters_counter(uint16_t n = 1) {
        sub_iter += n;
        return sub_iter;
    }

    uint16_t get_iters_counter() const { return iter; }

    uint16_t get_sub_iters_counter() const { return sub_iter; }

  public:
    void set_initial_residual(const vector::vector &res_vector) {
        for (uint32_t nv = 0; nv < NV; ++nv) {
            res0[nv] = res_vector.get_value<F>(nv);
        }
    }

    void update_residual(const vector::vector &res_vector) {
        for (uint32_t nv = 0; nv < NV; ++nv) {
            if (!status[nv]) {
                res[nv] = res_vector.get_value<F>(nv);
            }
        }
    }

  private:
    void set_convergence_status(vector::vector &conv_vector) const {
        for (uint32_t nv = 0; nv < NV; ++nv) {
            if (status[nv])
                conv_vector.set_value<F>(nv, 0);
        }
        conv_vector.if_zero = false;
    }

    bool check() {
        if (!conv_check)
            return false;
        bool convergence_flag = true;
        for (uint16_t nv = 0; nv < NV; ++nv) {
            if (status[nv])
                continue;
            float64_t tol = tolerance_values->get(nv);
            F abs_res = sqrt(res[nv]);
            if (abs_res > (F)tol) {
                convergence_flag = false;
            } else {
                status[nv] = true;
            }
        }
        return convergence_flag;
    }

  public:
    bool is_converged(const vector::vector &res_vector) {
        if (!conv_check)
            return false;
        if (conv_info) {
            if (!iter) {
                if (top_solver_flag)
                    io::print_residuals_header(0, NV);
                else
                    io::print_residuals_delimiter(NV);
            }
            io::print_residuals(solver_prefix, iter, res, res0, status);
        }
        if (check()) {
            if (conv_info)
                io::print_residuals_footer(NV);
            return true;
        }
        return false;
    }

    bool is_converged(const vector::vector &res_vector, vector::vector &conv) {
        bool retval = is_converged(res_vector);
        if (!retval)
            set_convergence_status(conv);
        return retval;
    }

    bool rr_check() {
        if (!conv_check)
            return false;

        bool flag = true;
        float64_t max_rel_tol = 0.0;
        for (uint16_t nv = 0; nv < NV; ++nv) {
            float64_t ratio = std::sqrt(res[nv] / res0[nv]);
            max_rel_tol = std::max(max_rel_tol, ratio);
        }

        if (max_rel_tol > rr_tol)
            flag = false;
        else
            rr_tol = max_rel_tol * rr_stepping;

        return flag;
    }

    bool rr_deviation_check(const vector::vector &true_res_vector) {
        for (uint16_t nv = 0; nv < NV; ++nv) {
            float64_t res_ratio = std::sqrt(res[nv] / true_res_vector.get_value<F>(nv));
            if ((res_ratio > rr_deviation) || (res_ratio < 1.0 / rr_deviation)) {
                if (conv_info) {
                    XAMG::out << XAMG::CONVERGENCE
                              << "Residuals deviation observed, restart required..." << std::endl;
                    io::print_residuals_footer(NV);
                }
                return true;
            }
        }
        //    XAMG::out << "Extending run..." << std::endl;
        return false;
    }
};

} // namespace solver
} // namespace XAMG
