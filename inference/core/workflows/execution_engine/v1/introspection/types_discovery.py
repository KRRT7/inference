from typing import Dict, List, Set, Union

from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_all_defined_kinds,
)
from inference.core.workflows.execution_engine.v1.introspection.kinds_schemas_register import (
    KIND_TO_SCHEMA_REGISTER,
)


def discover_kinds_typing_hints(kinds_names: Set[str]) -> Dict[str, str]:
    all_defined_kinds = load_all_defined_kinds()
    return {
        kind.name: kind.serialised_data_type
        for kind in all_defined_kinds
        if kind.serialised_data_type is not None and kind.name in kinds_names
    }


def discover_kinds_schemas(kinds_names: Set[str]) -> Dict[str, Union[dict, List[dict]]]:
    kinds_to_schema = {}
    # Since names are unique, use set intersection for fast lookup
    present_kinds = (
        kinds_names & KIND_TO_SCHEMA_REGISTER.keys()
        if not isinstance(KIND_TO_SCHEMA_REGISTER, dict)
        else kinds_names & set(KIND_TO_SCHEMA_REGISTER)
    )
    for name in present_kinds:
        kinds_to_schema[name] = KIND_TO_SCHEMA_REGISTER[name]
    return kinds_to_schema
