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
#include "xamg_macros.h"
#include <sstream>

struct xamg_params_details {
    using my_dictionary = params::dictionary<xamg_params_details>;
    using my_list = params::list<xamg_params_details>;

    // Which key in the list represents the family of the list by its value.
    // For XAMG, all aolver parameter lists are characterized and distinguished
    // by solver method.
    static std::string get_family_key() { return "method"; }

    // The keyword for parameter layers: it is used in command line to setup
    // per-layer overrides and is used when printing out a table
    static std::string get_layer_prefix() { return "lev"; }

    // How many layers reserves overrides_holder for internal operation
    static uint16_t get_nlayers() { return 50; }

    // Each table portion is printed out with this function
    static void print_stream(const std::stringstream &ss) {
        std::stringstream my_ss(ss.str());
        std::string line;
        while (std::getline(my_ss, line)) {
            XAMG::out << XAMG::PARAMS << line << std::endl;
        }
    }

    // Printing out the whole table
    static void print_table(const my_dictionary &params) {
        my_list::print_line_delimiter();
        my_list::print_header("XAMG library");
        my_list::print_header(std::string("Build: ") + std::string(XAMG_REVISION), 0);
        if (params.find("meta_solver")) {
            params.print_list("meta_solver", "META_SOLVER");
        }
        params.print_list("solver", "SOLVER");
        if (params.find("residual_control")) {
            params.print_list("residual_control", "RESIDUAL CONTROL");
        }
        bool mg_flag = params.find_if<std::string>({{"method", "MultiGrid"}});
        if (params.find("preconditioner")) {
            params.print_list("preconditioner", "PRECONDITIONER");
        }
        if (mg_flag) {
            params.print_list("pre_smoother", "PRE_SMOOTHER");
            params.print_list("post_smoother", "POST_SMOOTHER");
            params.print_list("coarse_grid_solver", "COARSE GRID SOLVER");
        }
    }

#define ALLFAMILIES                                                                                \
    {}
#define NOMINMAX                                                                                   \
    {}
#define ALLALLOWED                                                                                 \
    {}
#define NOSPECIALSOLVERS                                                                           \
    { "!Identity", "!Direct", "!ResidualReplacement" }

    static const params::expected_params_t &get_expected_params() {
        using namespace params;
        static const expected_params_t expected_params = {
            {"method", {value::S, false, ALLFAMILIES, NOMINMAX, ALLALLOWED}},
            {"polynomial_order", {value::I, false, {"Chebyshev"}, {"1", "4"}, ALLALLOWED}},
            {"spectrum_fraction", {value::F, false, {"Chebyshev"}, {"0.0", "1.0"}, ALLALLOWED}},
            {"max_iters",
             {value::I,
              true,
              {"!Identity", "!Direct", "!ResidualReplacement", "!Chebyshev"},
              {"0", "inf"},
              ALLALLOWED}},
            {"subsolver_reduced_precision",
             {value::B, false, {"IterativeRefinement"}, NOMINMAX, ALLALLOWED}},
            {"convergence_details", {value::S, false, NOSPECIALSOLVERS, NOMINMAX, ALLALLOWED}},
            {"abs_tolerance", {value::F, true, NOSPECIALSOLVERS, {"0.0", "inf"}, ALLALLOWED}},
            {"rel_tolerance", {value::F, true, NOSPECIALSOLVERS, {"0.0", "inf"}, ALLALLOWED}},
            {"RR_deviation", {value::F, true, {"ResidualReplacement"}, {"0.0", "inf"}, ALLALLOWED}},
            {"RR_stepping", {value::F, true, {"ResidualReplacement"}, {"0.0", "inf"}, ALLALLOWED}},
            {"relax_factor", {value::F, true, {"Jacobi", "HSGS"}, NOMINMAX, ALLALLOWED}},
            {"hypre_log", {value::B, false, {"MultiGrid"}, NOMINMAX, ALLALLOWED}},
            {"hypre_per_level_hierarchy", {value::B, false, {"MultiGrid"}, NOMINMAX, ALLALLOWED}},
            {"mg_reduced_precision", {value::B, false, {"MultiGrid"}, NOMINMAX, ALLALLOWED}},
            {"mg_cycle", {value::I, false, {"MultiGrid"}, NOMINMAX, {"0", "1", "2"}}},
            {"mg_max_levels", {value::I, false, {"MultiGrid"}, {"1", "50"}, ALLALLOWED}},
            {"mg_coarse_matrix_size", {value::I, false, {"MultiGrid"}, {"1", "2000"}, ALLALLOWED}},
            {"mg_strength_threshold", {value::F, false, {"MultiGrid"}, {"0.0", "1.0"}, ALLALLOWED}},
            {"mg_trunc_factor", {value::F, false, {"MultiGrid"}, {"0.0", "1.0"}, ALLALLOWED}},
            {"mg_num_paths", {value::I, false, {"MultiGrid"}, {"1", "10"}, ALLALLOWED}},
            {"mg_coarsening_type",
             {value::I,
              false,
              {"MultiGrid"},
              {"0", "22"},
              {"0", "1", "3", "6", "7", "8", "9", "10", "11", "21", "22"}}},
            {"mg_interpolation_type", {value::I, false, {"MultiGrid"}, {"0", "14"}, ALLALLOWED}},
            {"mg_Pmax_elements", {value::I, false, {"MultiGrid"}, {"0", "inf"}, ALLALLOWED}},
            {"mg_max_row_sum", {value::F, false, {"MultiGrid"}, {"0.0", "1.0"}, ALLALLOWED}},
            {"mg_nonGalerkin_tol", {value::F, false, {"MultiGrid"}, {"0.0", "1.0"}, ALLALLOWED}},
            {"mg_agg_num_levels", {value::I, false, {"MultiGrid"}, {"0", "50"}, ALLALLOWED}},
            {"mg_agg_interpolation_type", {value::I, false, {"MultiGrid"}, {"1", "4"}, ALLALLOWED}},
            {"mg_agg_trunc_factor", {value::F, false, {"MultiGrid"}, {"0.0", "1.0"}, ALLALLOWED}},
            {"mg_agg_P12_trunc_factor",
             {value::F, false, {"MultiGrid"}, {"0.0", "1.0"}, ALLALLOWED}},
            {"mg_agg_Pmax_elements", {value::I, false, {"MultiGrid"}, {"0", "inf"}, ALLALLOWED}},
            {"mg_agg_P12max_elements", {value::I, false, {"MultiGrid"}, {"0", "inf"}, ALLALLOWED}}};
        return expected_params;
    }

#undef ALLFAMILIES
#undef NOMINMAX
#undef ALLALLOWED
#undef NOSPECIALSOLVERS

    // Is called before starting the defaults processing for each list
    static void set_dictionary_defaults(my_dictionary &params) {
        bool mg_flag = params.find_if<std::string>({{"method", "MultiGrid"}});
        if (mg_flag) {
            if (!params.find("pre_smoother")) {
                params.add("pre_smoother", {"method", "Jacobi"});
            }
            if (!params.find("post_smoother")) {
                params.add("post_smoother", {"method", "Jacobi"});
            }
            if (!params.find("coarse_grid_solver")) {
                params.add("coarse_grid_solver", {"method", "Direct"});
            }
        }
        if (params.find("meta_solver")) {
            auto &list = params.get("meta_solver");
            list.set_value_if_missing<std::string>("convergence_details", "check-L2,print");
        }
        if (params.find("solver")) {
            auto &list = params.get("solver");
            list.set_value_if_missing<std::string>("convergence_details", "check-L2,print");
            std::set<std::string> convdetails;
            misc::str_split(list.get_string("convergence_details"), ',', convdetails);
            if (convdetails.count("RR") && !params.find("residual_control")) {
                ASSERT(0 && "residual_control parameters block must present if RR convergence mode "
                            "is used.");
            }
        }
    }

    static void set_family_defaults(my_list &list, const std::string &method,
                                    const std::string &solver_role) {
        if ((method == "Identity") || (method == "Direct")) {
            return;
        }

        if (method == "IterativeRefinement") {
            ASSERT(solver_role == "meta_solver");
            list.set_value_if_missing<bool>("subsolver_reduced_precision", false);
            //set_family_defaults(list, "PBiCGStab", solver_role);
            return;
        }

        if (method == "ResidualReplacement") {
            ASSERT(solver_role == "residual_control");
            list.set_value_if_missing<float64_t>("RR_deviation", 2.0);
            list.set_value_if_missing<float64_t>("RR_stepping", 0.1);
            return;
        }

        if (method == "Chebyshev") {
            list.set_value_if_missing<std::string>("convergence_details", "nocheck,noprint");
            list.set_value_if_missing<uint32_t>("polynomial_order", 2);
            list.set_value_if_missing<float64_t>("spectrum_fraction", 0.3);
            list.set_value_if_missing<float64_t>("abs_tolerance", 0);
            list.set_value_if_missing<float64_t>("rel_tolerance", 0);
            return;
        }

        uint32_t iters = 0;
        if (solver_role == "solver" || solver_role == "meta_solver") {
            list.set_value_if_missing<float64_t>("abs_tolerance", 1.e-8);
            list.set_value_if_missing<float64_t>("rel_tolerance", 1.e-8);
            list.set_value_if_missing<std::string>("convergence_details", "check-L2,print");
        } else {
            list.set_value_if_missing<float64_t>("abs_tolerance", 0);
            list.set_value_if_missing<float64_t>("rel_tolerance", 0);
            list.set_value_if_missing<std::string>("convergence_details", "nocheck,noprint");
        }

        std::set<std::string> convdetails;
        misc::str_split(list.get_string("convergence_details"), ',', convdetails);
        if (solver_role != "solver") {
            if (convdetails.count("RR")) {
                ASSERT(0 && "RR mode for convergence is available for main solver only.");
            }
        } else {
            if (convdetails.count("RR")) {
                if (method == "Identity" || method == "Direct" || method == "Chebyshev" ||
                    method == "IterativeRefinement") {
                    ASSERT(0 && "RR mode for convergence is not available for this solver method.");
                }
            }
        }

        if ((method == "Jacobi") || (method == "HSGS")) {
            if (solver_role == "solver")
                iters = 100;
            else
                iters = 2;

            list.set_value_if_missing<float64_t>("relax_factor", 0.99);
        } else if (method == "MultiGrid") {
            if (solver_role == "solver")
                iters = 30;
            else
                iters = 1;

            list.set_value_if_missing<bool>("hypre_log", true);
            list.set_value_if_missing<bool>("hypre_per_level_hierarchy", false);

            list.set_value_if_missing<bool>("mg_reduced_precision", false);
            list.set_value_if_missing<uint32_t>("mg_cycle", V_cycle);
            list.set_value_if_missing<uint32_t>("mg_max_levels", 25);
            list.set_value_if_missing<uint32_t>("mg_coarse_matrix_size", 100);

            list.set_value_if_missing<float64_t>("mg_strength_threshold", 0.5);
            list.set_value_if_missing<float64_t>("mg_trunc_factor", 0.25);

            list.set_value_if_missing<uint32_t>("mg_num_paths", 3);

            list.set_value_if_missing<uint32_t>("mg_coarsening_type", 6);
            list.set_value_if_missing<uint32_t>("mg_interpolation_type", 0);
            // list.set_value_if_missing<uint32_t>("mg_coarsening_type", 8);
            // list.set_value_if_missing<uint32_t>("mg_interpolation_type", 6);

            list.set_value_if_missing<float64_t>("mg_max_row_sum", 0.9);
            list.set_value_if_missing<uint32_t>("mg_Pmax_elements", 4);
            list.set_value_if_missing<float64_t>("mg_nonGalerkin_tol", 0.0);

            list.set_value_if_missing<uint32_t>("mg_agg_num_levels", 10);
            list.set_value_if_missing<uint32_t>("mg_agg_interpolation_type", 4);

            list.set_value_if_missing<float64_t>("mg_agg_trunc_factor", 0.25);
            list.set_value_if_missing<float64_t>("mg_agg_P12_trunc_factor", 0.0);

            list.set_value_if_missing<uint32_t>("mg_agg_Pmax_elements", 4);
            list.set_value_if_missing<uint32_t>("mg_agg_P12max_elements", 4);
        } else if (method == "IterativeRefinement") {
            ASSERT(solver_role == "meta_solver");
            iters = 20;
        } else if ((method == "PCG") || (method == "BiCGStab") || (method == "MergedBiCGStab") ||
                   (method == "PBiCGStab") || (method == "MergedPBiCGStab") ||
                   (method == "RBiCGStab") || (method == "MergedRBiCGStab")
#ifdef XAMG_EXPERIMENTAL_SOLVERS
                   || (method == "PipeBiCGStab") || (method == "MergedPipeBiCGStab") ||
                   (method == "IBiCGStab") || (method == "MergedIBiCGStab") ||
                   (method == "PPipeBiCGStab") || (method == "MergedPPipeBiCGStab")
#endif
        ) {
            if (solver_role == "solver")
                iters = 50;
            else
                iters = 1;
        } else {
            ASSERT(0 && "method not found");
        }
        list.set_value_if_missing("max_iters", iters);
    }
};
