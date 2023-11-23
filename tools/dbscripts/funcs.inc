# BSD 3-Clause License
#
# Copyright (c) 2019-2021, Boris Krasnopolsky, Alexey Medvedev
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function find_including_paths() {
    for i in $(seq 1 $#); do
        local p=$(eval echo \"\$$i\")
        [ -e $p ] && find $p/ -type f
    done
}

function find_excluding_paths() {
    local exclude_paths=""
    for i in $(seq 1 $#); do
        local p=$(eval echo \"\$$i\")
        exclude_paths=$(echo "$exclude_paths" -not -path "$p")
    done
    set -f
    find . $exclude_paths -type f
    set +f
}

function is_elf() {
    local f="$1"
    [ $(file -b "$f" | awk '{print $1}') == "ELF" ] && return 0
    return 1
}

function ls_with_time() {
    for i in $*; do 
        ls -Bl1 --time-style="+%s" $i
    done
}

function print_all_exec_files() {
    local f=""
    set +u
    local paths="$1"
    local started_time="$2"
    set -u
    if [ -z "$paths" ]; then
        f=$(find_excluding_paths '*.src/*' './_*')
    else
        f=$(find_including_paths $paths)
    fi
    local allf=""
    [ -z "$f" ] && echo "" && return
    for i in $f; do is_elf $i && allf="$allf $i"; done
    ls_with_time $allf | awk -v T="$started_time" '{ if (T!="" && $6>=T) ISNEW=">>"; else ISNEW="--"; printf ISNEW " " $6 "\\t" "%15d" "\\t" $7 "\\n", $5 }'
}

function construct_replacement_command() {
    local from="$1"
    local to="$2"
    echo "s!$from!$to!;"
}

function template_to_psubmitopts {
    local dir=$1
    local suffix="$2"
    local repl=""
    set +u
    repl=$(echo "${repl}" $(construct_replacement_command "__QUEUE__" "$PSUBMIT_OPTS_QUEUE_NAME"))
    repl=$(echo "${repl}" $(construct_replacement_command "__QUEUE_SUFFIX__" "$PSUBMIT_OPTS_QUEUE_SUFFIX"))
    repl=$(echo "${repl}" $(construct_replacement_command "__NODETYPE__" "$PSUBMIT_OPTS_NODETYPE"))
    repl=$(echo "${repl}" $(construct_replacement_command "__INIT_COMMANDS__" "$PSUBMIT_OPTS_INIT_COMMANDS"))
    repl=$(echo "${repl}" $(construct_replacement_command "__INJOB_INIT_COMMANDS__" "$PSUBMIT_OPTS_INJOB_INIT_COMMANDS"))
    repl=$(echo "${repl}" $(construct_replacement_command "__MPI_SCRIPT__" "$PSUBMIT_OPTS_MPI_SCRIPT"))
    repl=$(echo "${repl}" $(construct_replacement_command "__BATCH_SCRIPT__" "$PSUBMIT_OPTS_BATCH_SCRIPT"))
    if [ "$suffix" == "" ]; then
        cat psubmit.opt.TEMPLATE | sed "$repl" > $dir/psubmit.opt
    else
        cat psubmit_${suffix}.opt.TEMPLATE | sed "$repl" > $dir/psubmit_${suffix}.opt
    fi
    set -u
}

function expand_mode_string {
    local allargs="$1"
    local dflt="$2"
    local m="$3"
    local final=""
    [ -z "$m" ] && m=":dubi"
    [ -z "$allargs" ] && allargs="$m"
    [ ":" != $(get_first_char "$m") ] && fatal "wrong mode string"
    if [ ":" == $(get_first_char "$allargs") ]; then
        [ ":" == $(get_first_char "$allargs") ] && m="$allargs"
        for p in $dflt; do
            is_any_char_present "$p" ":" && final="$final $p" && continue
            final="$final ${p}${m}"
        done
        echo $final
        return
    fi
    for a in $allargs; do
        [ ":" == $(get_first_char "$a") ] && fatal "wrong mode string"
        is_any_char_present "$a" ":" && final="$final $a" && continue
        final="$final ${a}${m}"
    done
    echo "$final"
}

function mode_for_pkg {
    local pkg="$1"
    local bld="$2"
    for p in $bld; do
        if [ "$pkg" == $(get_field "$p" 1) ]; then
            get_field "$p" 2
            return
        fi
    done
    echo ""
}

function version_for_pkg() {
    local pkg="$1"
    local defvers="$2"
    local cfgvers="$3"
    local bld="$4"
    local defm=$(mode_for_pkg ${pkg} "$defvers")
    local cfgm=$(mode_for_pkg ${pkg} "$cfgvers")
    local cmdm=""
    for p in $bld; do
        if [ "$pkg" == $(get_field "$p" 1) ]; then
            cmdm=$(get_field "$p" 3)
        fi
    done
    local version=""
    [ -z "$defm" ] && fatal "no default version for project: $pkg"
    [ -z "$cfgm" ] && cfgm="$defm"
    [ -z "$cmdm" ] && cmdm="$cfgm"
    version="$cmdm"
    echo "$version"
    return 0
}

function check_bash_func_declared() {
    if [ `type -t $1`"" != 'function' ]; then
        return 1
    fi
    return 0
}

