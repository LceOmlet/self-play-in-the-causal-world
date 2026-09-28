#!/bin/bash
# Run with sudo once. No model code is accepted by this installer.
set -euo pipefail
base=$(realpath -- "${1:?Pass the trusted build/source directory}")
install -d -m 0755 /opt/cpt-numerics/bin
install -m 0755 "$base/nsjail-source/nsjail" /opt/cpt-numerics/bin/nsjail
cat > /etc/apparmor.d/cpt-numerics-nsjail <<'PROFILE'
abi <abi/4.0>,
include <tunables/global>
profile cpt-numerics-nsjail /opt/cpt-numerics/bin/nsjail flags=(unconfined) {
  userns,
}
PROFILE
apparmor_parser -r /etc/apparmor.d/cpt-numerics-nsjail
sha256sum /opt/cpt-numerics/bin/nsjail
stat -c '%U:%G %a %n' /opt/cpt-numerics/bin/nsjail
sysctl kernel.apparmor_restrict_unprivileged_userns
