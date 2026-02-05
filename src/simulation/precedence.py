"""Precedence constraints for block assembly.

Blocks must obey precedence relations specified by their `predecessors` lists.
This module provides helper functions to check whether predecessors have
completed their tasks or been placed on the dock before a dependent block
may proceed.
"""

from __future__ import annotations

from typing import List, Dict

from .entities import Block, BlockStatus


def is_predecessor_complete(block: Block, block_lookup: Dict[str, Block]) -> bool:
    """Return True if all predecessors of `block` are placed on the dock.

    Parameters
    ----------
    block : Block
        The block whose predecessors are being checked.
    block_lookup : dict
        Mapping from block IDs to Block objects. Each predecessor ID in
        `block.predecessors` must exist in this mapping.
    """
    for pred_id in block.predecessors:
        pred = block_lookup.get(pred_id)
        if pred is None or pred.status != BlockStatus.PLACED_ON_DOCK:
            return False
    return True