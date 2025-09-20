"""Static metadata describing whether a marine species is invasive in South Florida."""
from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class SpeciesRecord:
    key: str
    common_name: str
    scientific_name: str
    is_invasive: bool
    notes: str = ""


_SPECIES: Dict[str, SpeciesRecord] = {
    # High-priority invasive species
    "red lionfish": SpeciesRecord(
        key="red lionfish",
        common_name="Red Lionfish",
        scientific_name="Pterois volitans",
        is_invasive=True,
        notes="Aggressive invasive predator known to disrupt reef ecosystems."
    ),
    "devil firefish": SpeciesRecord(
        key="devil firefish",
        common_name="Devil Firefish",
        scientific_name="Pterois miles",
        is_invasive=True,
        notes="Close relative of red lionfish; both treated as invasive in South Florida."
    ),
    "green iguana": SpeciesRecord(
        key="green iguana",
        common_name="Green Iguana",
        scientific_name="Iguana iguana",
        is_invasive=True,
        notes="Terrestrial invasive reptile often spotted near coastal areas."
    ),
    "brown anole": SpeciesRecord(
        key="brown anole",
        common_name="Brown Anole",
        scientific_name="Anolis sagrei",
        is_invasive=True,
        notes="Competes with native lizards; included for completeness."
    ),
    # Common native reference species
    "queen angelfish": SpeciesRecord(
        key="queen angelfish",
        common_name="Queen Angelfish",
        scientific_name="Holacanthus ciliaris",
        is_invasive=False,
        notes="Native reef species."
    ),
    "spiny lobster": SpeciesRecord(
        key="spiny lobster",
        common_name="Caribbean Spiny Lobster",
        scientific_name="Panulirus argus",
        is_invasive=False,
    ),
    "nurse shark": SpeciesRecord(
        key="nurse shark",
        common_name="Nurse Shark",
        scientific_name="Ginglymostoma cirratum",
        is_invasive=False,
    ),
}

# Aliases help map multiple model labels to the same species record.
_ALIASES: Dict[str, str] = {
    "lionfish": "red lionfish",
    "pterois volitans": "red lionfish",
    "pterois miles": "devil firefish",
    "common lionfish": "red lionfish",
    "poison lionfish": "red lionfish",
}


def available_species() -> Iterable[SpeciesRecord]:
    return _SPECIES.values()


def lookup_species(label: str) -> Optional[SpeciesRecord]:
    """Return the closest metadata match for the provided model label."""
    key = label.strip().lower()
    if key in _SPECIES:
        return _SPECIES[key]
    alias = _ALIASES.get(key)
    if alias:
        return _SPECIES.get(alias)
    return None
