from copy import deepcopy

def resolve_yaml_definitions(root: dict, definitions_key: str = 'definitions'):
    """
    Resolves all variable definitions in the provided YAML dictionary.

    Variables to be resolved should start with one '$' character and should be defined
    in a single dictionary of the yaml file at `definitions_key`. Variables may contain
    other variables in their definitions, but the onus is on the user to avoid infinite
    recursion!
    """
    # Resolve definitions within definitions (i love recursion)
    definitions_unresolved: dict[str] = deepcopy(root[definitions_key])
    definitions_prefixed = {
        f"${key}" : val
        for key, val in definitions_unresolved.items()
    }
    # Format properly
    definitions = _resolve_yaml_definitions_recursive(definitions_unresolved, definitions_prefixed, [f"/{definitions_key}"], True)
    definitions = {
        f"${key}" : val
        for key, val in definitions.items()
    }

    # Resolve everything else
    root_new = deepcopy(root)
    del root_new[definitions_key]
    root_new = _resolve_yaml_definitions_recursive(root_new, definitions, [], False)
    root_new[definitions_key] = definitions
    return root_new


def _resolve_yaml_definitions_recursive(tree: dict | str, definitions: dict[str], path: list[str], is_defs: bool) -> dict | str:
    # Base case: str
    if isinstance(tree, str):
        if tree == "$$SELF":
            if is_defs:
                return path[1].removeprefix('/')
            else:
                raise ValueError(f"{''.join(path)}: '$$SELF' may only be used within the definitions section of the yaml file")
        elif tree.startswith('$') and not tree.startswith('$$'):
            if tree not in definitions:
                raise ValueError(f"{''.join(path)}: Key '{tree}' not found in definitions")
            return _resolve_yaml_definitions_recursive(definitions[tree], definitions, path, is_defs)
        else:
            return tree
    
    # Base case: numeric
    if isinstance(tree, (int, float)):
        return tree
    
    # Case: list
    if isinstance(tree, list):
        return [
            _resolve_yaml_definitions_recursive(item, definitions, path + [f"[{i}]"], is_defs)
            for i, item in enumerate(tree)
        ]

    # Case: set
    if isinstance(tree, set):
        return {
            _resolve_yaml_definitions_recursive(item, definitions, path, is_defs)
            for item in tree
        }

    # Case: dict
    if isinstance(tree, dict):
        return {
            _resolve_yaml_definitions_recursive(key, definitions, path, is_defs) :
            _resolve_yaml_definitions_recursive(val, definitions, path + [f"/{key}"], is_defs)
            for key, val in tree.items()
        }

    # Any other dtypes not allowed!
    raise ValueError(f"{path}: Invalid data type '{type(tree)}'")
