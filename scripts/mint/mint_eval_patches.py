#!/usr/bin/env python
"""
Draccus/lerobot compatibility patches for MINT evaluation with local checkpoints.

These patches fix two issues in the lerobot 0.4.3 + draccus 0.8.0 ecosystem:

1. PreTrainedConfig.from_pretrained() line 219: config.pop("type") raises KeyError
   when 'type' is not in config.json. Fixed by using pop("type", None).

2. decode_choice_class(): when parsing a ChoiceRegistry type (like PreTrainedConfig)
   from a JSON file where the 'type' key is missing, draccus falls through to
   ParsingError. This happens because:
   - PreTrainedConfig is a ChoiceRegistry (base class for all policy configs)
   - get_choice_name(PreTrainedConfig) raises ValueError (it's the base, not a subclass)
   - default_choice_name() returns None
   - With no 'type' key in the dict, there is nothing to fall back to

   Fixed by patching decode_choice_class to auto-detect the registered subclass
   by field coverage when 'type' is missing.
"""

import sys

def apply():
    """Apply all compatibility patches. Call this BEFORE importing lerobot/mint."""
    import draccus
    from draccus.parsers import decoding

    # ── Patch 1: decode_choice_class ──────────────────────────────────────────
    # When 'type' key is absent from the raw dict, try to infer the subclass
    # by checking which registered choice class has ALL its fields present.
    original_decode_choice_class = decoding.decode_choice_class

    def patched_decode_choice_class(cls, raw_value, path):
        import dataclasses
        from draccus.choice_types import CHOICE_TYPE_KEY, ChoiceType
        from draccus.parsers.decoding import decode_dataclass, ParsingError

        assert issubclass(cls, ChoiceType)

        try:
            cls.get_choice_name(cls)
            return decode_dataclass(cls, raw_value)
        except ValueError:
            pass

        if not isinstance(raw_value, dict):
            raise ParsingError(f"Expected a dict for a choice class, got {raw_value}")

        if CHOICE_TYPE_KEY not in raw_value:
            raw_keys = set(raw_value.keys())
            # Try to find a registered choice class where raw_keys ⊆ choice_fields
            for choice_name, choice_cls in cls._choice_registry.items():
                if not dataclasses.is_dataclass(choice_cls):
                    continue
                choice_fields = {f.name for f in dataclasses.fields(choice_cls)}
                if raw_keys.issubset(choice_fields):
                    return decode_dataclass(choice_cls, raw_value, path)
            default = cls.default_choice_name()
            if default is None:
                raise ParsingError(
                    f"Expected a dict with a '{CHOICE_TYPE_KEY}' key for {cls}, "
                    f"got {raw_value}"
                )
            choice_type = default
        else:
            choice_type = raw_value[CHOICE_TYPE_KEY]

        try:
            subcls = cls.get_choice_class(choice_type)
        except KeyError as e:
            raise decoding.DecodingError(
                path, f"Couldn't find a choice class for '{choice_type}' in {cls}"
            ) from e

        raw_value = raw_value.copy()
        if CHOICE_TYPE_KEY in raw_value:
            raw_value.pop(CHOICE_TYPE_KEY)
        return decode_dataclass(subcls, raw_value, path)

    decoding.decode_choice_class = patched_decode_choice_class

    # Clear the LRU cache so patched function is used
    decoding.get_decoding_fn.cache_clear()


# ── Apply patches when this module is imported ─────────────────────────────────
apply()
