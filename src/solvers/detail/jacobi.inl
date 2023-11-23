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
struct Jacobi : public base_solver<F, NV> {
    const uint16_t nvecs = 1;
    const uint16_t comm_size = 0;
    DECLARE_INHERITED_FROM_BASESOLVER(Jacobi)
    virtual void init(const XAMG::params::param_list &list,
                      const std::string &solver_role_) override {
        base_solver<F, NV>::solver_role = solver_role_;
        base::init_base(list);

        auto &numa_layer = A.data_layer.find(segment::NUMA)->second;
        numa_layer.diag.get_inv_diag();
    }
};

template <typename F, uint16_t NV>
void Jacobi<F, NV>::matrix_info() {
    A.info.print("A");
}

template <typename F, uint16_t NV>
void Jacobi<F, NV>::solve(const vector::vector &conv, XAMG::mpi::token &tok) {
    vector::vector &x = *this->x;
    const vector::vector &b = *this->b;

    auto max_iters = param_list.get_int("max_iters");
    F relax_factor = param_list.get_float("relax_factor");

    // inverted convergence flag; used to switch off updates to converged RHSs
    vector::vector iconv(conv);

    const vector::vector &a1 = blas::ConstVectorsCache<F>::get_ones_vec(NV);

    const vector::vector &inv_diag = A.inv_diag();

    vector::vector relax_conv;
    relax_conv.alloc<F>(1, NV);
    blas::set_const<F, NV>(relax_conv, relax_factor);

    vector::vector &r = buffer[0];
    assert(nvecs == 1);
    base::set_buffers_zero();

    ///////////////////

    stats.init(r, iconv);
    if (stats.conv_check) {
        if (stats.is_converged_initial(r)) {
            return;
        }
        for (bool last_iter = false; !last_iter;) {
            auto it = stats.increment_iters_counter();
            if (it >= max_iters)
                last_iter = true;
            blas::scal<F, NV>(inv_diag, r);
            blas::scal<F, NV>(iconv, relax_conv);
            blas::axpby<F, NV>(relax_conv, r, a1, x);
            if (stats.is_converged(r)) {
                return;
            }
        }
        stats.print_residuals_footer();
    } else {
        for (bool last_iter = false; !last_iter;) {
            auto it = stats.increment_iters_counter();
            if (it >= max_iters)
                last_iter = true;
            blas2::Ax_y<F, NV>(A, x, r, NV);
            blas::specific::merged_jacobi<F, NV>(x, b, inv_diag, r, relax_conv);
        }
    }
}

} // namespace solver
} // namespace XAMG
