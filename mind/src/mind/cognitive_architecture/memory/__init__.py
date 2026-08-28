"""Memory subsystem for the cognitive architecture.

Exports the pure data models only. VectorDBMemory is deliberately NOT re-exported
here: importing a submodule runs this file, so an eager storage import at the
package level pulled chromadb and sentence_transformers into every consumer of
anything in the package - including PipelineState, via state.py's `Memory`
import, and including the scoring module whose whole point is to have no storage
dependency. Import it from its own module: `from .vector_db_memory import
VectorDBMemory`, which is what every call site already did.
"""

from .models import Memory, VectorDBMetadata

__all__ = ["Memory", "VectorDBMetadata"]
