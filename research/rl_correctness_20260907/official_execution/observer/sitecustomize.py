"""Enable an isolated read-only audit, never a replacement training path."""

import os

if os.environ.get("CPT_DAPO_OBSERVATION_DIR"):
    import execution_observer

    execution_observer.install()
