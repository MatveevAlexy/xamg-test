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

#include <iostream>

namespace XAMG {

#ifndef XAMG_SEPARATE_OBJECT

namespace io {
namespace eps {
const double eps_point_size = 0.2;

void plot_header(const int x_0, const int y_0, std::ofstream &f) {
    f << std::fixed << std::setprecision(1);
    f << "%%!PS" << std::endl;
    f << "%%BoundingBox: 0 0 " << x_0 * eps_point_size << " " << y_0 * eps_point_size << std::endl
      << std::endl;

    f << "\n/box_crs  {" << eps_point_size << " 0 rlineto 0 " << eps_point_size << " rlineto -"
      << eps_point_size << " 0 rlineto closepath 0 0 0 setrgbcolor fill} def" << std::endl;

    f << "/bnd_hline {" << y_0 * eps_point_size << " 0 rlineto " << eps_point_size
      << "setlinewidth 0 0 1 setrgbcolor stroke} def" << std::endl;
    f << "/bnd_hline {0 " << x_0 * eps_point_size << " rlineto " << eps_point_size
      << "setlinewidth 0 0 1 setrgbcolor stroke} def" << std::endl
      << std::endl;
}

void plot_row(const int i, const std::vector<uint64_t> &col, const int x0, const int y0,
              std::ofstream &f) {
    for (const auto j : col) {
        f << "newpath"
          << " " << j * eps_point_size << " " << (y0 - i) * eps_point_size << " moveto box_crs"
          << std::endl;
    }
}

void plot_footer(std::ofstream &f) {
    f << "showpage" << std::endl;
}
} // namespace eps
} // namespace io

#endif

} // namespace XAMG
