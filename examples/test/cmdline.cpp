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

// FIXME Think of more elegant way to do this
#define XAMG_SEPARATE_OBJECT

#include <iostream>
#include <fstream>
#include <regex>
#include <argsparser.h>
#include <xamg_headers.h>
#include <xamg_types.h>

#include <param/params.h>

#undef XAMG_SEPARATE_OBJECT

#include "cmdline.h"
#include "../common/cmdline/cmdline.h"

extern ID id;

void parser_add_specific(args_parser &parser, execution_mode_t &execution_mode,
                         XAMG::params::global_param_list &params) {
    parser.add<std::string>("mode", "solver").set_caption(MODE_CAPTION).set_description(MODE_DESCR);
    parser.add_flag("graph_reordering")
        .set_caption(GRAPH_REORDERING_CAPTION)
        .set_description(GRAPH_REORDERING_DESCR);
    parser.add_flag("save_pattern")
        .set_caption(SAVE_PATTERN_CAPTION)
        .set_description(SAVE_PATTERN_DESCR);
    parser.add<std::string>("result", "")
        .set_caption(" - The YAML file to store summary test results in.");
    parser.add<int>("test_iters", 0).set_caption("- Number of time-measured iterations to make.");
}

parse_result_t parse_cmdline_specific(args_parser &parser, execution_mode_t &execution_mode,
                                      XAMG::params::global_param_list &params) {
    auto cmdline_mode = parser.get<std::string>("mode");
    execution_mode = str_to_execution_mode(cmdline_mode);
    if (execution_mode == execution_mode_t::none) {
        std::cerr << "Wrong execution mode: " << cmdline_mode << std::endl;
        return PARSE_FATAL_FAILURE;
    }
    return PARSE_OK;
}

parse_result_t parse_cmdline(args_parser &parser, const std::vector<std::string> &solver_roles,
                             execution_mode_t &execution_mode,
                             XAMG::params::global_param_list &params) {
    parser_add_common(parser, solver_roles, params);
    parser_add_specific(parser, execution_mode, params);

    parse_result_t res;
    res = parse_cmdline_common(parser, solver_roles, params);
    if (res != PARSE_OK)
        return res;
    res = parse_cmdline_specific(parser, execution_mode, params);
    if (res != PARSE_OK)
        return res;

    return PARSE_OK;
}
