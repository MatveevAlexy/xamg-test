#############################################################################
## 
##  Copyright (C) 2019-2021 Boris Krasnopolsky, Alexey Medvedev
##  Contact: xamg-test@imec.msu.ru
## 
##  This file is part of the XAMG library.
## 
##  Commercial License Usage
##  Licensees holding valid commercial XAMG licenses may use this file in
##  accordance with the terms of commercial license agreement.
##  The license terms and conditions are subject to mutual agreement
##  between Licensee and XAMG library authors signed by both parties
##  in a written form.
## 
##  GNU General Public License Usage
##  Alternatively, this file may be used under the terms of the GNU
##  General Public License, either version 3 of the License, or (at your
##  option) any later version. The license is as published by the Free 
##  Software Foundation and appearing in the file LICENSE.GPL3 included in
##  the packaging of this file. Please review the following information to
##  ensure the GNU General Public License requirements will be met:
##  https://www.gnu.org/licenses/gpl-3.0.html.
## 
#############################################################################

#!/bin/bash

set -eu

[ -f ../env.sh ] && source ../env.sh || echo "WARNING: no environment file ../env.sh!"

BSCRIPTSDIR=../tools/dbscripts

source $BSCRIPTSDIR/base.inc
source $BSCRIPTSDIR/funcs.inc
source $BSCRIPTSDIR/compchk.inc
source $BSCRIPTSDIR/envchk.inc
source $BSCRIPTSDIR/db.inc
source $BSCRIPTSDIR/apps.inc

####

PACKAGES="hypre yaml-cpp argsparser cppcgen numactl scotch"
VERSIONS="hypre:2.20.0 yaml-cpp:0.6.3 argsparser:HEAD cppcgen:0.0.1 numactl:1.0.2 scotch:6.1.0"
TARGET_DIRS=""

started=$(date "+%s")
echo "Download and build started at timestamp: $started."
environment_check_main || fatal "Environment is not supported, exiting"
cd "$INSTALL_DIR"
dubi_main "$*"
finished=$(date "+%s")
echo "----------"
echo "Full operation time: $(expr $finished - $started) seconds."
