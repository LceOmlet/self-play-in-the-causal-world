#!/bin/bash
set -euo pipefail
base=$(realpath -- "${1:?Pass the trusted build/source directory}")
root=/opt/cpt-numerics/rootfs
test -d "$root/opt/numerics"
install -d -m 0755 "$root/opt/bridge" "$root/opt/kernels/numerics"
install -m 0644 "$base/sandbox_bridge.py" "$base/tool_mode.py" "$root/opt/bridge/"
cat > "$root/opt/kernels/numerics/kernel.json" <<'JSON'
{"argv":["/opt/numerics/bin/python","-I","-m","ipykernel_launcher","-f","{connection_file}","--HistoryManager.hist_file=:memory:"],"display_name":"NumPy SciPy","language":"python"}
JSON
chown -R root:root "$root/opt/bridge" "$root/opt/kernels"
chmod -R go-w "$root/opt/bridge" "$root/opt/kernels"
sha256sum "$root/opt/bridge/"*.py "$root/opt/kernels/numerics/kernel.json"
