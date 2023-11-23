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
struct Chebyshev : public base_solver<F, NV> {
    const uint16_t nvecs = 3;
    const uint16_t comm_size = 0;
    DECLARE_INHERITED_FROM_BASESOLVER(Chebyshev)
    virtual void init(const XAMG::params::param_list &list,
                      const std::string &solver_role_) override {
        base_solver<F, NV>::solver_role = solver_role_;
        base::init_base(list);

        auto &numa_layer = A.data_layer.find(segment::NUMA)->second;
        numa_layer.diag.get_inv_sqrt_diag();
        // A.get_eigenvalue_estimates();
    }
    struct Chebyshev_polynomial_coeffs {
        uint32_t poly_order = 0;
        float64_t spectrum_fraction = 0.0;
        std::vector<vector::vector> coeffs;
        void update(float64_t min_eig, float64_t max_eig, uint32_t _poly_order,
                    float64_t _spectrum_fraction) {
            if (_poly_order == poly_order && _spectrum_fraction == spectrum_fraction) {
                return;
            }
            coeffs.clear();
            poly_order = _poly_order;
            spectrum_fraction = _spectrum_fraction;
            make(min_eig, max_eig);
        }
        void make(float64_t min_eig, float64_t max_eig) {
            float64_t den;

            float64_t upper_bound = max_eig * 1.1;
            float64_t lower_bound = (upper_bound - min_eig) * spectrum_fraction + min_eig;

            float64_t theta = (upper_bound + lower_bound) / 2.;
            float64_t delta = (upper_bound - lower_bound) / 2.;

            ///////////////////

            coeffs.resize(poly_order, vector::vector());
            for (auto &coeff : coeffs)
                coeff.alloc<F>(1, NV);

            switch (poly_order) {
            case 1: {
                F cheby0 = 1. / theta;
                blas::set_const<F, NV>(coeffs[0], cheby0);

                break;
            }
            case 2: /* (  2*t - 4*th)/(del^2 - 2*th^2) */
            {
                den = delta * delta - 2 * theta * theta;
                F cheby0 = -4 * theta / den;
                F cheby1 = 2 / den;
                blas::set_const<F, NV>(coeffs[0], cheby0);
                blas::set_const<F, NV>(coeffs[1], cheby1);

                break;
            }
            case 3: /* (3*del^2 - 4*t^2 + 12*t*th - 12*th^2)/(3*del^2*th - 4*th^3)*/
            {
                den = 3 * (delta * delta) * theta - 4 * (theta * theta * theta);
                F cheby0 = (3 * delta * delta - 12 * theta * theta) / den;
                F cheby1 = 12 * theta / den;
                F cheby2 = -4 / den;

                blas::set_const<F, NV>(coeffs[0], cheby0);
                blas::set_const<F, NV>(coeffs[1], cheby1);
                blas::set_const<F, NV>(coeffs[2], cheby2);

                break;
            }
            case 4: /*(t*(8*del^2 - 48*th^2) - 16*del^2*th + 32*t^2*th - 8*t^3 + 32*th^3)/(del^4 -
                       8*del^2*th^2 + 8*th^4)*/
            {
                den =
                    std::pow(delta, 4) - 8 * delta * delta * theta * theta + 8 * std::pow(theta, 4);
                F cheby0 = (32 * std::pow(theta, 3) - 16 * delta * delta * theta) / den;
                F cheby1 = (8 * delta * delta - 48 * theta * theta) / den;
                F cheby2 = 32 * theta / den;
                F cheby3 = -8 / den;

                blas::set_const<F, NV>(coeffs[0], cheby0);
                blas::set_const<F, NV>(coeffs[1], cheby1);
                blas::set_const<F, NV>(coeffs[2], cheby2);
                blas::set_const<F, NV>(coeffs[3], cheby3);

                break;
            }
            default: {
                assert(0);
                break;
            }
            }
            for (auto &coeff : coeffs) {
                coeff.if_initialized = true;
                coeff.if_zero = false;
            }
        }
    };
    Chebyshev_polynomial_coeffs coeffs_holder;
};

template <typename F, uint16_t NV>
void Chebyshev<F, NV>::matrix_info() {
    A.info.print("A");
}

template <typename F, uint16_t NV>
void Chebyshev<F, NV>::solve(const vector::vector &conv, XAMG::mpi::token &tok) {
    vector::vector &x = *this->x;
    const vector::vector &b = *this->b;

    auto poly_order = param_list.get_int("polynomial_order");
    auto spec_fraction = param_list.get_float("spectrum_fraction");

    // inverted convergence flag; used to switch off updates to converged RHSs
    // vector::vector iconv(conv);

    // get cached constant vectors
    const vector::vector &a1 = blas::ConstVectorsCache<F>::get_ones_vec(NV);

    // get pre-calculated inv-sqrt-diag vector
    const vector::vector &inv_sqrt_diag = A.inv_sqrt_diag();

    // get pre-allocated buffers
    vector::vector &r = buffer[0];
    vector::vector &t1 = buffer[1];
    vector::vector &t2 = buffer[2];
    assert(nvecs == 3);
    base::set_buffers_zero();

    ///////////////////
    coeffs_holder.update(A.min_eig, A.max_eig, poly_order, spec_fraction);
    const auto &cheby_coeffs = coeffs_holder.coeffs;

    stats.init(r);
    if (stats.is_converged_initial(r)) {
        return;
    }
    blas::scal<F, NV>(inv_sqrt_diag, r);
    blas::ax_y<F, NV>(cheby_coeffs[poly_order - 1], r, t1);
    for (int i = poly_order - 2; i >= 0; --i) {
        blas::scal<F, NV>(inv_sqrt_diag, t1);
        blas2::Ax_y<F, NV>(A, t1, t2, NV);
        blas::scal<F, NV>(inv_sqrt_diag, t2);
        blas::axpby_z<F, NV>(cheby_coeffs[i], r, a1, t2, t1);
    }
    blas::scal<F, NV>(inv_sqrt_diag, t1);
    blas::axpby<F, NV>(a1, t1, a1, x);
    stats.is_converged();
}
} // namespace solver
} // namespace XAMG
