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

#include "param/params.h"

template <typename T>
void set_param_if_missing(XAMG::params::param_list &list, const std::string &key, T val) {
    list.set_value_if_missing(key, val);
}

static inline bool parse_params(const std::map<std::string, std::string> kvmap,
                                XAMG::params::param_list &list) {
    for (auto &kv : kvmap) {
        list.parse_and_set_value(kv.first, kv.second);
    }
    return true;
}

#define DUMP_CAPTION "<config.yaml> - Dump the config to file. See detailed help."
#define DUMP_DESCR                                                                                 \
    "Saves all the parameters after applying defaults and parsing the given command line\n"        \
    "as a YAML-file. This config can be edited then and given to the option -load for the\n"       \
    "next time. See also -load option description for reference."

#define LOAD_CAPTION "<config.yaml> - Load the config from file. See detailed help."
#define LOAD_DESCR                                                                                 \
    "Read all the config parameters from a given YAML-file, then handle the command line.\n"       \
    "This means, command-line has a priority over a file. Use -dump option to initially\n"         \
    "make a config file holding a set of default parameters or the parameters, given\n"            \
    "in a command line."

#define SLEEP_CAPTION "<N> - Sleep for N seconds before action. See detailed help."
#define SLEEP_DESCR                                                                                \
    "This option instructs the example to wait for <N> seconds before calling XAMG::init()\n"      \
    "and continuing execution. You may use this pause to attach debugger to just "                 \
    "spawned\nMPI-processes."

#define MATRIX_CAPTION                                                                             \
    "<matrix.csr> - CSR matrix file name or \"generate\" to use internal matrix generation"
#define MATRIX_DESCR ""

static std::map<std::string, std::string> SOLVER_PARAMS_CAPTIONS = {
    {"meta_solver", "<param=value:...>  - Parameters for meta_solver. See detailed help."},
    {"solver", "<param=value:...>  - Parameters for solver. See detailed help."},
    {"preconditioner", "<param=value:...>  - Parameters for preconditioner. See detailed help."},
    {"pre_smoother", "<param=value:...>  - Parameters for pre_smoother. See detailed help."},
    {"post_smoother", "<param=value:...>  - Parameters for post_smoother. See detailed help."},
    {"coarse_grid_solver",
     "<param=value:...>  - Parameters for coarse_grid_solver. See detailed help."}};

static std::map<std::string, std::string> SOLVER_PARAMS_DESCRS = {
    {"meta_solver", "The set of colon-separated pairs: param=value. The allowed parameters are:\n"
                    "    method=IterativeRefinement\n"
                    "    max_iters=UINT\n"
                    "    abs_tolerance=FLOAT\n"
                    "    rel_tolerance=FLOAT\n"
                    "    convergence_details=kyword,keyword,...\n"
                    "    no default value; if the section is missing, the meta-solver\n"
                    "    functionality is ignored"},
    {"solver", "The set of colon-separated pairs: param=value. The default method is "
               "\"BiCGStab\". Additional method-specific parameters can also be included. \n"
               "NOTE: The default value is applied only when the option is missing"},
    {"preconditioner", "The set of colon-separated pairs: param=value. This section has no default "
                       "method. The preconditioner must be specified by the user.\n"},
    {"pre_smoother", "The set of colon-separated pairs: param=value. The default method is "
                     "\"Jacobi\". Additional method-specific parameters can also be included. \n"
                     "NOTE: The default value is applied only when the option is missing"},
    {"post_smoother", "The set of colon-separated pairs: param=value. The default method is "
                      "\"Jacobi\". Additional method-specific parameters can also be included. \n"
                      "NOTE: The default value is applied only when the option is missing"},
    {"coarse_grid_solver",
     "The set of colon-separated pairs: param=value. The default method is "
     "\"Direct\". Additional method-specific parameters can also be included. \n"
     "NOTE: The default value is applied only when the option is missing"}

};

#define PARAMS_OVERRIDE_CAPTION                                                                    \
    "<param=value@levX[-Y]:...> - Override solver parameters for some MultiGrid levels. See "      \
    "detailed help."
#define PARAMS_OVERRIDE_DESCR                                                                      \
    "The set of colon-separated pairs: param=value in the same form as for other solver "          \
    "parameters,\n"                                                                                \
    "but the suffix in form: @levX[-Y] must be added to each pair. See the set of allowed "        \
    "parameters in detailed\n"                                                                     \
    "help for corresponding solver parameters. Suffix sets up the MultiGrid level numbers this "   \
    "override\n"                                                                                   \
    "is made for. Both X and Y are positive integers which denote starting and ending number of "  \
    "level\n"                                                                                      \
    "for this override, but Y can be also set to a special value: \"E\" which means: the last "    \
    "level of \n"                                                                                  \
    "the hierarchy. Missing Y value means override for a single level only. The option can be "    \
    "repeated\n"                                                                                   \
    "several times for different parameters and level diapasons.\n\n"                              \
    "Default is: \n"                                                                               \
    "    no overrides"

#define CONVERGENCE_PARAMS_CAPTION "----"
#define CONVERGENCE_PARAMS_DESCR "----"

#define GENERATOR_PARAMS_CAPTION                                                                   \
    "<param=value:...> - Parameters for internal matrix generator. See detailed help."
#define GENERATOR_PARAMS_DESCR                                                                     \
    "The set of colon-separated pairs: param=value. The allowed parameters are:\n"                 \
    "    case=cube|channel_with_cube\n"                                                            \
    "      For the cube case the allowed params are (config parameter specifies the anisotropic "  \
    "matrix configuration):\n"                                                                     \
    "        nx=UINT\n"                                                                            \
    "        ny=UINT\n"                                                                            \
    "        nz=UINT\n"                                                                            \
    "        config=0..4\n"                                                                        \
    "      For the channel_with_cube case the allowed params are (3 predefined grids of 2.3, 9.7 " \
    "and 32M cells from CFD):\n"                                                                   \
    "        scale=1..3\n"                                                                         \
    "        const_rhs=0..1\n"                                                                     \
    "Default is:\n"                                                                                \
    "    case=cube:nx=10:ny=10:nz=1:config=0; scale=1:const_rhs=1 is default for the "             \
    "channel_with_cube case\n"                                                                     \
    "NOTE: Default value is applied only when the option is missing"

#define NODE_CONFIG_CAPTION                                                                        \
    "<...> - Parameters for node config (cores, numas and gpus). See detailed help."
#define NODE_CONFIG_DESCR                                                                          \
    "The parameters can be given in one of the forms:\n"                                           \
    "1) empty line (default): number of cores and gpus are detected automatically at\n"            \
    "   runtime. Number of numas is assumed to be 1.\n"                                            \
    "2) The set of colon-separated pairs: param=value. The allowed parameters are:\n"              \
    "    nnumas=UINT\n"                                                                            \
    "    ncores=UINT\n"                                                                            \
    "    ngpus=UINT\n"                                                                             \
    "NOTE: Default values for ncores and nnumas are equal to 1, for gpus to 0\n"                   \
    "3) The precise core-numa-gpu mapping in the form: \n"                                         \
    "       core1,core2,...,coreN@gpu1,gpu2,...,gpuM;...\n"                                        \
    "   The mapping consists of blocks separated by semicolon.\n"                                  \
    "   Each block corresponds to a single NUMA node. It consists of all cores\n"                  \
    "   which belong to the node, and gpus which are local to the node.\n"                         \
    "   Alternative form allows bitmaps instead of comma-separated core lists.\n"                  \
    "   Examples:\n"                                                                               \
    "       0,1,2,3,4,5,6,7@0;8,9,10,11,12,13,14,15@1\n"                                           \
    "       0xff@0;0xff00@1\n"                                                                     \
    "Default is:\n"                                                                                \
    "   (empty line)"

#define RESULT_CAPTION "<result.yaml> - The YAML file to store computation results in"
#define RESULT_DESCR                                                                               \
    "The computation results are stored in a given file. The structures YAML format\n"             \
    "is used to hold common info on the running configuration, the timing results,\n"              \
    "residual norm and other important data.\n\n"                                                  \
    "Default is:\n"                                                                                \
    "    (no result output)"

#define LOGFILE_CAPTION "<[outfile]> - Output file to put messages instead of stdout"
#define LOGFILE_DESCR                                                                              \
    "If the outfile name is given, all output is redirected to this file. When an empty\n"         \
    "string as an outfile parameter, the output is not redirected and printed to stdout\n"         \
    "as usual. If the logopts parameter specified, it sets up the logging options as\n"            \
    "a set of log message types with are allowed or supressed in the output. See documentation\n"  \
    "for details."

enum parse_result_t { PARSE_OK, PARSE_FATAL_FAILURE, PARSE_HELP_PRINTED };
void parser_add_common(args_parser &parser, const std::vector<std::string> &solver_roles,
                       XAMG::params::global_param_list &params);

parse_result_t parse_cmdline_common(args_parser &parser,
                                    const std::vector<std::string> &solver_roles,
                                    XAMG::params::global_param_list &params);
