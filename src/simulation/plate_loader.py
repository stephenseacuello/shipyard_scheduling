"""Plate-level decomposition data loader.

Loads partner's 3D ship model decomposition data (plates within blocks)
from JSON files and applies it to Block entities. Also provides synthetic
plate generation for when real decomposition data is unavailable.

Expected JSON format from partner's 3D decomposition scripts::

    {
      "ship_id": "HN2900",
      "ship_type": "lng_carrier",
      "decomposition_version": "1.0",
      "source_model": "3d_cad_export_v2",
      "blocks": [
        {
          "block_id": "B00001",
          "block_type": "flat_bottom",
          "erection_sequence": 1,
          "predecessors": [],
          "size_m": [20.0, 18.0, 12.0],
          "plates": [
            {
              "plate_id": "P00_001_001",
              "plate_type": "flat",
              "length_mm": 12000,
              "width_mm": 3200,
              "thickness_mm": 22,
              "material_grade": "AH36",
              "has_stiffeners": true,
              "n_stiffeners": 4,
              "stiffener_spacing_mm": 600,
              "curvature_radius_mm": 0
            }
          ]
        }
      ]
    }
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .entities import Block, BlockType, Plate, PlateType

logger = logging.getLogger(__name__)

# Map string plate_type values to PlateType enum
_PLATE_TYPE_MAP = {
    "flat": PlateType.FLAT,
    "curved": PlateType.CURVED,
    "stiffened": PlateType.STIFFENED,
    "bracket": PlateType.BRACKET,
    "bulkhead": PlateType.BULKHEAD,
    "shell": PlateType.SHELL,
}

# Synthetic plate generation parameters per block type
_SYNTHETIC_PARAMS = {
    BlockType.FLAT_BOTTOM: {"type_factor": 1.0, "curved_pct": 0.0, "stiffened_pct": 0.55},
    BlockType.FLAT_SIDE: {"type_factor": 1.0, "curved_pct": 0.05, "stiffened_pct": 0.50},
    BlockType.DECK: {"type_factor": 0.9, "curved_pct": 0.0, "stiffened_pct": 0.45},
    BlockType.CARGO_TANK_SUPPORT: {"type_factor": 1.1, "curved_pct": 0.10, "stiffened_pct": 0.60},
    BlockType.ENGINE_ROOM: {"type_factor": 1.3, "curved_pct": 0.05, "stiffened_pct": 0.50},
    BlockType.CURVED_BOW: {"type_factor": 0.7, "curved_pct": 0.45, "stiffened_pct": 0.40},
    BlockType.CURVED_STERN: {"type_factor": 0.75, "curved_pct": 0.40, "stiffened_pct": 0.40},
    BlockType.ACCOMMODATION: {"type_factor": 0.85, "curved_pct": 0.0, "stiffened_pct": 0.35},
}

REQUIRED_BLOCK_FIELDS = {"block_id", "plates"}
REQUIRED_PLATE_FIELDS = {"plate_id"}


def load_ship_decomposition(filepath: str) -> Dict[str, List[Dict]]:
    """Load partner's decomposed ship data from JSON.

    Parameters
    ----------
    filepath : str
        Path to the JSON decomposition file.

    Returns
    -------
    dict
        Mapping of ship_id -> list of block dicts with plate data.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Decomposition file not found: {filepath}")

    with open(path) as f:
        data = json.load(f)

    warnings = validate_decomposition(data)
    if warnings:
        for w in warnings:
            logger.warning("Decomposition validation: %s", w)

    ship_id = data.get("ship_id", path.stem)
    return {ship_id: data.get("blocks", [])}


def apply_decomposition_to_blocks(
    blocks: List[Block],
    decomposition: Dict[str, List[Dict]],
    ship_id_map: Optional[Dict[str, str]] = None,
) -> int:
    """Apply plate data from decomposition to existing Block objects.

    Matches by block_id or by (ship_id, erection_sequence).

    Parameters
    ----------
    blocks : list of Block
        Block objects to update.
    decomposition : dict
        Mapping of ship_id -> list of block dicts from load_ship_decomposition().
    ship_id_map : dict, optional
        Maps decomposition ship_id -> env ship_id if naming differs.

    Returns
    -------
    int
        Count of blocks updated.
    """
    # Build lookup by block_id and by (ship_id, erection_sequence)
    block_data_by_id: Dict[str, Dict] = {}
    block_data_by_seq: Dict[tuple, Dict] = {}

    for ship_id, block_list in decomposition.items():
        mapped_ship = ship_id_map.get(ship_id, ship_id) if ship_id_map else ship_id
        for bd in block_list:
            bid = bd.get("block_id", "")
            if bid:
                block_data_by_id[bid] = bd
            seq = bd.get("erection_sequence")
            if seq is not None:
                block_data_by_seq[(mapped_ship, seq)] = bd

    updated = 0
    for block in blocks:
        # Try matching by block_id first
        bd = block_data_by_id.get(block.id)
        if bd is None:
            bd = block_data_by_seq.get((block.ship_id, block.erection_sequence))
        if bd is None:
            continue

        # Apply plate data
        plates = []
        for pd in bd.get("plates", []):
            plate_type_str = pd.get("plate_type", "flat")
            plate = Plate(
                id=pd.get("plate_id", f"P_{block.id}_{len(plates)}"),
                plate_type=_PLATE_TYPE_MAP.get(plate_type_str, PlateType.FLAT),
                length_mm=float(pd.get("length_mm", 12000)),
                width_mm=float(pd.get("width_mm", 3000)),
                thickness_mm=float(pd.get("thickness_mm", 20)),
                weight_kg=float(pd.get("weight_kg", 0)),
                material_grade=pd.get("material_grade", "AH36"),
                has_stiffeners=pd.get("has_stiffeners", False),
                n_stiffeners=int(pd.get("n_stiffeners", 0)),
                stiffener_spacing_mm=float(pd.get("stiffener_spacing_mm", 600)),
                curvature_radius_mm=float(pd.get("curvature_radius_mm", 0)),
            )
            plates.append(plate)

        block.plates = plates
        block.compute_plate_stats()

        # Update predecessors if provided in decomposition
        if "predecessors" in bd and bd["predecessors"]:
            block.predecessors = list(bd["predecessors"])

        updated += 1

    return updated


def generate_synthetic_plates(
    block: Block,
    rng: np.random.Generator,
) -> List[Plate]:
    """Generate synthetic plate data for a block when no real decomposition exists.

    Uses physically-motivated formulas based on block weight, type, and size.
    Typical LNG carrier plate: ~600kg, 12m x 3m, 20mm thick.

    Parameters
    ----------
    block : Block
        The block to generate plates for.
    rng : numpy Generator
        Random number generator for reproducibility.

    Returns
    -------
    list of Plate
        Generated synthetic plates.
    """
    params = _SYNTHETIC_PARAMS.get(block.block_type, {"type_factor": 1.0, "curved_pct": 0.0, "stiffened_pct": 0.5})
    type_factor = params["type_factor"]
    curved_pct = params["curved_pct"]
    stiffened_pct = params["stiffened_pct"]

    # Estimate plate count: typical block has 15-60 plates depending on size/type.
    # A 350t flat bottom block has ~30 plates, a 200t accommodation ~15.
    # Formula: weight_tons * type_factor / ~12 tons_per_plate (avg after stiffeners)
    n_plates = max(3, int(math.ceil(block.weight / 12.0 * type_factor)))

    plates: List[Plate] = []
    for i in range(n_plates):
        is_curved = rng.random() < curved_pct
        is_stiffened = (not is_curved) and (rng.random() < stiffened_pct)

        length_mm = float(rng.uniform(6000, 14000))
        width_mm = float(rng.uniform(2000, 3500))
        # Thickness scales mildly with block weight (heavier blocks = thicker plates)
        base_thickness = 16.0 + (block.weight / 400.0) * 12.0  # 16-28mm range
        thickness_mm = float(rng.uniform(base_thickness * 0.85, base_thickness * 1.15))

        n_stiffeners = 0
        if is_stiffened:
            n_stiffeners = max(1, int(length_mm / 600.0))  # ~600mm spacing

        plate_type = PlateType.CURVED if is_curved else (PlateType.STIFFENED if is_stiffened else PlateType.FLAT)

        plate = Plate(
            id=f"P{block.id[1:]}_{i:03d}" if len(block.id) > 1 else f"P_{i:03d}",
            plate_type=plate_type,
            length_mm=length_mm,
            width_mm=width_mm,
            thickness_mm=thickness_mm,
            material_grade="AH36" if not is_curved else "DH36",
            has_stiffeners=is_stiffened,
            n_stiffeners=n_stiffeners,
            curvature_radius_mm=float(rng.uniform(5000, 20000)) if is_curved else 0.0,
        )
        plates.append(plate)

    return plates


def validate_decomposition(data: Dict[str, Any]) -> List[str]:
    """Validate decomposition JSON against expected schema.

    Parameters
    ----------
    data : dict
        Loaded JSON data.

    Returns
    -------
    list of str
        List of warnings/errors (empty if valid).
    """
    warnings: List[str] = []

    if "ship_id" not in data:
        warnings.append("Missing 'ship_id' field")
    if "blocks" not in data:
        warnings.append("Missing 'blocks' field")
        return warnings

    blocks = data["blocks"]
    if not isinstance(blocks, list):
        warnings.append("'blocks' should be a list")
        return warnings

    for i, block in enumerate(blocks):
        for field in REQUIRED_BLOCK_FIELDS:
            if field not in block:
                warnings.append(f"Block {i}: missing required field '{field}'")

        plates = block.get("plates", [])
        if not plates:
            warnings.append(f"Block {i} ({block.get('block_id', '?')}): empty plates list")
        for j, plate in enumerate(plates):
            for field in REQUIRED_PLATE_FIELDS:
                if field not in plate:
                    warnings.append(f"Block {i}, plate {j}: missing '{field}'")

    return warnings
