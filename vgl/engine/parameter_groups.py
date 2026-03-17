class LayerwiseLrDecay:
    def __init__(self, module_name_groups, lr_decay=0.5, include_rest=True):
        if lr_decay <= 0.0 or lr_decay > 1.0:
            raise ValueError("lr_decay must be in (0, 1]")
        self.module_name_groups = self._normalize_module_name_groups(module_name_groups)
        self.lr_decay = float(lr_decay)
        self.include_rest = bool(include_rest)

    def _normalize_module_name_groups(self, module_name_groups):
        if isinstance(module_name_groups, str):
            module_name_groups = [module_name_groups]
        groups = []
        for group in module_name_groups:
            if isinstance(group, str):
                group = (group,)
            else:
                group = tuple(group)
            if not group:
                raise ValueError("module_name_groups must not be empty")
            for name in group:
                if not isinstance(name, str) or not name:
                    raise ValueError("module_name_groups entries must be non-empty strings")
            groups.append(group)
        if not groups:
            raise ValueError("module_name_groups must not be empty")
        return tuple(groups)

    def __call__(self, model, lr):
        base_lr = float(lr)
        named_params = list(model.named_parameters())
        param_lookup = dict(named_params)
        group_entries = []
        ownership = {}

        for group_index, group in enumerate(self.module_name_groups):
            group_names = []
            for prefix in group:
                matches = [
                    name
                    for name, _ in named_params
                    if name == prefix or name.startswith(f"{prefix}.")
                ]
                if not matches:
                    raise ValueError(f"module prefix '{prefix}' matched no parameters")
                for name in matches:
                    if name in ownership and ownership[name] != group_index:
                        raise ValueError(f"module_name_groups overlap on parameter '{name}'")
                    ownership[name] = group_index
                    if name not in group_names:
                        group_names.append(name)

            group_entries.append(
                {
                    "params": [param_lookup[name] for name in group_names],
                    "lr": base_lr * (self.lr_decay**group_index),
                    "group_name": "|".join(group),
                    "param_names": list(group_names),
                }
            )

        remaining_names = [name for name, _ in named_params if name not in ownership]
        if remaining_names:
            if not self.include_rest:
                raise ValueError("some parameters were not assigned to any layer-wise LR group")
            group_entries.append(
                {
                    "params": [param_lookup[name] for name in remaining_names],
                    "lr": base_lr * (self.lr_decay ** len(self.module_name_groups)),
                    "group_name": "__rest__",
                    "param_names": list(remaining_names),
                }
            )

        return group_entries


__all__ = ["LayerwiseLrDecay"]
