function i_save_this_pkg_record() {
    local pkg="$1"
    local V="$2"
    local mode="$3"
    [ -z "$m" ] && return
    this_mode_is_set 'i' "$m" || return
    local pkg_as_variable=$(echo "$pkg" | sed 's/-/_/g')_VERSION
    local record="local $pkg_as_variable=$V"
    grep -q "$record" .saved_pkg_records || echo "local $pkg_as_variable=$V" >> .saved_pkg_records
}

function check_this_pkg_record() {
    local pkg="$1"
    local V="$2"
    local mode="$3"
    [ -z "$m" ] && return
    set +u
    [ -z "$DNB_SUPERIOR" ] && return
    set -u
    [ -e "$DNB_SUPERIOR/.saved_pkg_records" ] || return
    source "$DNB_SUPERIOR/.saved_pkg_records"
    local pkg_as_variable=$(echo "$pkg" | sed 's/-/_/g')_VERSION
    SAVED_VERSION=$(eval "echo \$$pkg_as_variable")
    if [ "$SAVED_VERSION" == "$V" ]; then
        if this_mode_is_set 'i' "$m"; then
        [ -e "$pkg.bin" ] && rm "$pkg.bin"
            ln -s "$DNB_SUPERIOR/$pkg.bin" "$pkg.bin"
            local record="local $pkg_as_variable=$V"
            grep -q "$record" .saved_pkg_records || echo "local $pkg_as_variable=$V" >> .saved_pkg_records
        fi
        echo "cached"
        return
    fi
}

function dnb_psubmit() {
    local pkg="psubmit"
    environment_check_specific "$pkg" || fatal "pkg: environment check failed"
    local m=$(get_field "$1" 2 "=")
    local V=$(get_field "$2" 2 "=")
    [ "$(check_this_pkg_record $pkg $V $m)" == "cached" ] && return 0
    du_github "a-v-medvedev" "psubmit" "v" "$V" "$m"
    if this_mode_is_set "i" "$m"; then
        local FILES=""
        cd ${pkg}-${V}.src
        FILES=$(ls -1 *.sh)
        cd $INSTALL_DIR
        i_direct_copy "$pkg" "$V" "$FILES" "$m"
        i_make_binary_symlink "$pkg" "${V}" "$m"
        i_save_this_pkg_record "$pkg" "${V}" "$m"
    fi
    return 0
}

function dnb_yaml-cpp() {
    local pkg="yaml-cpp"
    environment_check_specific "$pkg" || fatal "$pkg: environment check failed"
    local m=$(get_field "$1" 2 "=")
    local V=$(get_field "$2" 2 "=")
    [ "$(check_this_pkg_record $pkg $V $m)" == "cached" ] && return 0
    du_github "jbeder" "yaml-cpp" "yaml-cpp-" "$V" "$m"
    local OPTS=""
    OPTS="$OPTS -DYAML_BUILD_SHARED_LIBS=ON"
    OPTS="$OPTS -DYAML_CPP_BUILD_TESTS=OFF"
    OPTS="$OPTS -DYAML_CPP_BUILD_TOOLS=OFF"
    OPTS="$OPTS -DYAML_CPP_BUILD_CONTRIB=OFF"
    bi_cmake "$pkg" "$V" ".." "$OPTS" "$m"
    i_make_binary_symlink "$pkg" "${V}" "$m"
    i_save_this_pkg_record "$pkg" "${V}" "$m"
    return 0
}

function dnb_argsparser() {
    local pkg="argsparser"
    environment_check_specific "$pkg" || fatal "$pkg: environment check failed"
    local m=$(get_field "$1" 2 "=")
    local V=$(get_field "$2" 2 "=")
    [ "$(check_this_pkg_record $pkg $V $m)" == "cached" ] && return 0
    du_github "a-v-medvedev" "argsparser" "v" "$V" "$m"
    if any_mode_is_set "bi" "$m"; then
        [ -f "$INSTALL_DIR/yaml-cpp.bin/include/yaml-cpp/yaml.h" ] || fatal "$pkg: installed yaml-cpp is required to build"
    fi
    local COMMANDS=""
    PARAMS="YAML_DIR=$INSTALL_DIR/yaml-cpp.bin"
    b_make "$pkg" "$V" "$COMMANDS" "clean" "$m"
    b_make "$pkg" "$V" "$COMMANDS" "$PARAMS" "$m"
    local FILES="argsparser/include/argsparser.h argsparser/libargsparser.so argsparser/libargsparser.a"
    this_mode_is_set "i" "$m" && i_direct_copy "$pkg" "$V" "$FILES" "$m"
    FILES="extensions"
    i_direct_copy "$pkg" "$V" "$FILES" "$m"
    i_make_binary_symlink "$pkg" "${V}" "$m"
    i_save_this_pkg_record "$pkg" "${V}" "$m"
    return 0
}

function dnb_hypre() {
    local pkg="hypre"
    environment_check_specific "$pkg" || fatal "${pkg}: environment check failed"
    local m=$(get_field "$1" 2 "=")
    local V=$(get_field "$2" 2 "=")
    du_github "hypre-space" "hypre" "v" "$V" "$m"
    local OPTS=""
    OPTS="$OPTS CC=$MPICC"
    OPTS="$OPTS CFLAGS=-O3"
    OPTS="$OPTS CXX=$MPICXX"
    OPTS="$OPTS CXXFLAGS=-O3"
    OPTS="$OPTS --with-timing"
    OPTS="$OPTS --enable-shared"
    OPTS="$OPTS --without-openmp"
    OPTS="$OPTS --without-fei"
    OPTS="$OPTS --without-mli"
    OPTS="$OPTS --disable-fortran"
    bi_autoconf_make "$pkg" "$V" "cd src" "$OPTS" "$m"
    i_make_binary_symlink "$pkg" "${V}" "$m"
    return 0
}

function dnb_cppcgen() {
    local pkg="cppcgen"
    environment_check_specific "$pkg" || fatal "$pkg: environment check failed"
    local m=$(get_field "$1" 2 "=")
    local V=$(get_field "$2" 2 "=")
    du_github "a-v-medvedev" "cppcgen" "" "$V" "$m"
    local COMMANDS=""
    local PARAMS="clean all"
    b_make "$pkg" "$V" "$COMMANDS" "$PARAMS" "$m"
    local FILES="distr/include distr/lib"
    i_direct_copy "$pkg" "$V" "$FILES" "$m"
    i_make_binary_symlink "$pkg" "${V}"  "$m"
    return 0
}

function dnb_numactl() {
    local pkg="numactl"
    environment_check_specific "$pkg" || fatal "$pkg: environment check failed"
    local m=$(get_field "$1" 2 "=")
    local V=$(get_field "$2" 2 "=")
    du_github "numactl" "numactl" "v" "$V" "$m"
    local COMMANDS=""
    local PARAMS="clean all"
    b_make "$pkg" "$V" "$COMMANDS" "$PARAMS" "$m"
    local FILES="libnuma.so.1 libnuma.so numa.h numaif.h"
    i_direct_copy "$pkg" "$V" "$FILES" "$m"
    i_make_binary_symlink "$pkg" "${V}"  "$m"
    return 0
}

function dnb_scotch() {
    local pkg="scotch"
    environment_check_specific "$pkg" || fatal "${pkg}: environment check failed"
    local m=$(get_field "$1" 2 "=")
    local V=$(get_field "$2" 2 "=")
    du_gitlab "$pkg" "$pkg" "${pkg}-" "v$V" "$m" "gitlab.inria.fr"
    if this_mode_is_set "u" "$m"; then
        [ -e "${pkg}-${V}.src" ] && rm -rf "${pkg}-${V}.src"
        mv ${pkg}-v${V}.src ${pkg}-${V}.src
        cd ${pkg}-${V}.src
        cd src
        [ ! -e Makefile.inc ] && ln -s Make.inc/Makefile.inc.i686_pc_linux2.shlib Makefile.inc
        sed -i 's/LDFLAGS.*/& -lrt/;s/-DSCOTCH_PTHREAD//;s/-DCOMMON_PTHREAD/-DSCOTCH_DETERMINISTIC/' Makefile.inc
        cd $INSTALL_DIR
    fi
    local COMMANDS=""
    COMMANDS="cd src"
    local PARAMS=""
    PARAMS="$PARAMS CCS=$CC CCP=$MPICC CCD=$MPICC AR=$CC"
    b_make "scotch" "${V}" "$COMMANDS" "clean $PARAMS" "$m"
    b_make "scotch" "${V}" "$COMMANDS" "ptscotch $PARAMS" "$m"
    PARAMS="install installstub prefix=$INSTALL_DIR/${pkg}-${V}"
    i_make "scotch" "${V}" "$COMMANDS" "$PARAMS" "$m"
    i_make_binary_symlink "$pkg" "${V}" "$m"
    this_mode_is_set "i" "$m" && cp "${pkg}.bin/include_stub"/*.h "${pkg}.bin/include"
    return 0
}

function dnb_CGNS() {
    local pkg="CGNS"
    environment_check_specific "$pkg" || fatal "${pkg}: environment check failed"
    local m=$(get_field "$1" 2 "=")
    local V=$(get_field "$2" 2 "=")
    du_github CGNS CGNS v "$V" "$m"
    local OPTS=""
    bi_cmake "$pkg" "$V" ".." "$OPTS" "$m"
    i_make_binary_symlink "$pkg" "${V}" "$m"
    return 0
}

function dnb_silo() {
    local pkg="silo"
    environment_check_specific "$pkg" || fatal "${pkg}: environment check failed"
    local m=$(get_field "$1" 2 "=")
    local V=$(get_field "$2" 2 "=")
    local archive=${pkg}.dwn/${pkg}-${V}.tar.gz
    if this_mode_is_set "d" "$m"; then
        local dwnld_user="visit-dav"
        local dwnld_url="third-party"
        local visit_tag=$(github_get_latest_version_tag "$dwnld_user" "$dwnld_url")
        [ -e "$archive" ] && rm -rf "$archive"
        [ ! -d "${pkg}.dwn" -a -e "${pkg}.dwn" ] && rm "${pkg}.dwn"
        mkdir -p "${pkg}.dwn"
        wget -nv -O "$archive" https://github.com/$dwnld_user/$dwnld_url/releases/download/$visit_tag/${pkg}-${V}.tar.gz
    fi
    if this_mode_is_set "u" "$m"; then
        [ -e "$archive" ] || fatal "${pkg}: no downloaded archive file: $archive"
        [ -d ${pkg}-${V}.src ] && rm -rf ${pkg}-${V}.src
        [ -d ${pkg}-${V} ] && rm -rf ${pkg}-${V}
        tar xzf "$archive"
        mv ${pkg}-${V} ${pkg}-${V}.src
        local host_name=`hostname -s`
        rm -f ${pkg}-${V}.src/config-site/${host_name}.conf
    fi
    local COMMANDS=""
    local OPTS="--enable-shared --disable-fortran"
    bi_autoconf_make "$pkg" "$V" "$COMMANDS" "$OPTS" "$m"
    i_make_binary_symlink "$pkg" "${V}" "$m"
    return 0
}

