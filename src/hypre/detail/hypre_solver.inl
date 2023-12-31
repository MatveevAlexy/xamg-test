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
namespace hypre {

#ifndef XAMG_SEPARATE_OBJECT
void solve(matrix::matrix &m, vector::vector &x, const vector::vector &y,
           const params::global_param_list &params) {

    assert(m.row_part == m.col_part);
    hypre_base hypre_obj(m.row_part);

    hypre_obj.assemble_matrix(m);
    hypre_obj.assemble_vector(hypre_obj.hb, y);
    hypre_obj.assemble_vector(hypre_obj.hx, x);

    hypre_obj.get_objects();

    ///////////////////

    bool precond_flag = params.find("preconditioner");

    std::string prec_method;
    auto &solver_params = params.get("solver");
    std::string sol_method = solver_params.get_string("method");

    if (precond_flag) {
        auto &precond_params = params.get("preconditioner");
        prec_method = precond_params.get_string("method");
    }

    if ((sol_method == "BiCGStab") || (sol_method == "PBiCGStab")) {
        hypre_obj.create_bicgstab_solver(hypre_obj.solver, params);

        if (precond_flag) {
            if ((sol_method == "PBiCGStab") && (prec_method == "MultiGrid")) {
                hypre_obj.create_multigrid_solver(hypre_obj.precond, params);
                hypre_obj.mg_precond_flag = true;
            }
        }

        hypre_obj.setup_bicgstab_solver();

        hypre_obj.bicgstab_solver();

        hypre_obj.destroy_bicgstab_solver();
    } else if (sol_method == "MultiGrid") {

        hypre_obj.create_multigrid_solver(hypre_obj.solver, params);

        hypre_obj.setup_multigrid_solver();

        hypre_obj.multigrid_solver();

        hypre_obj.destroy_multigrid_solver();
    }

    hypre_obj.get_vector_data(hypre_obj.hx, x);
    hypre_obj.destroy_objects();
}
/*
std::vector<float64_t> get_residual_norm(const matrix::matrix &m, const vector::vector &x, 
                                         const vector::vector &y) {
	vector::vector r = mem::LOCAL;
	r.alloc(x.size, 1);
	blas::set_const<float64_t, 1>(r, 0);
	blas2::Ax_y<float64_t, 1>(A, x, r, NV);
    blas::axpby<float64_t, 1>(1.0, b, -1.0, r);	
	return r.get_element<float64_t>(0);
}
*/
#endif

} // namespace hypre
} // namespace XAMG
