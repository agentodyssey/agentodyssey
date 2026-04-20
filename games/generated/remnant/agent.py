from typing import Dict, List, Optional, Any
import inspect
import sys

from games.generated.remnant.world import Container
from dataclasses import dataclass
import games.generated.remnant.rules.action_rules as action_rules
from games.generated.remnant.rule import BaseActionRule


class Inventory:
    _type: str = "inventory"

    def __init__(self, id: str, name: str, container: Optional[Container] = None):
        self.id = id
        self.name = name
        self._container: Optional[Container] = container

    @property
    def type(self) -> str:
        return self._type

    # attachment
    @property
    def container(self) -> Optional[Container]:
        return self._container
    
    # setter of the container
    @container.setter
    def container(self, container: Container) -> None:
        self._container = container

    # helper: ensure we’re attached before accessing state
    def _require_container(self) -> Container:
        if self._container is None:
            raise RuntimeError("Inventory is detached (container=None).")
        return self._container

    # aliases to keep with the previous interface
    @property
    def items(self) -> Dict[str, int]:
        return self._require_container().inventory

    @items.setter
    def items(self, new_items: Dict[str, int]) -> None:
        self._require_container().inventory = new_items

    @property
    def capacity(self) -> Optional[int]:
        return self._require_container().capacity
    
    @capacity.setter
    def capacity(self, cap: Optional[int]) -> None:
        self._require_container().capacity = cap
    
    def to_dict(self) -> Dict:
        if self._container is None:
            return {
                "id": self.id,
                "name": self.name,
                "container": None,
            }
        
        return {
            "id": self.id,
            "name": self.name,
            "container": self._container.id,
        }

    def from_dict(self, data: Dict, container_instances: dict[str, Container]) -> None:
        self.id = data["id"]
        self.name = data["name"]
        if data["container"] is None:
            self._container = None
        else:
            self._container = container_instances[data["container"]]

@dataclass
class Action:
    name: str
    verb: str
    params: List[str]
    description: Optional[str] = None
    _type: str = "action"

    @property
    def type(self):
        return self._type

class Agent:
    id: str
    name: str
    inventory: Inventory
    available_actions: List[Action]
    equipped_items_in_limb: Dict[str, int] = {}
    items_in_hands: Dict[str, int] = {}
    max_hp: int = 100
    min_attack: int = 10
    xp_per_level: int = 100  # XP required to level up
    hp: int = max_hp  # health
    attack: int = min_attack
    xp: int = 0  # experience
    level: int = 1
    defense: int = 0
    _type: str = "agent"
    _action_rules_module: Any = None  # can be set by environment to use game-specific action rules
    _base_action_rule_class: Any = None  # can be set by environment to use game-specific BaseActionRule
    
    @property
    def type(self):
        return self._type
    
    @property
    def available_actions(self):
        # Use game-specific action rules if set, otherwise use base action rules
        rules_module = self._action_rules_module if self._action_rules_module is not None else action_rules
        base_action_rule = self._base_action_rule_class if self._base_action_rule_class is not None else BaseActionRule
        actions: list[BaseActionRule] = [
            cls for _, cls in inspect.getmembers(rules_module, inspect.isclass)
            if issubclass(cls, base_action_rule) and cls is not base_action_rule and not inspect.isabstract(cls)
        ]
        actions_info = [
            {"name": cls.name, 
             "verb": cls.verb, 
             "params": cls.params, 
             "description": cls.description} 
             for cls in actions
        ]
        return [Action(**action) for action in actions_info]

    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.inventory = Inventory(id=f"inventory_{id}", name=f"{name}'s Inventory", container=None)
        self.equipped_items_in_limb = {}
        self.items_in_hands = {}
        self.max_hp = 100
        self.min_attack = 10
        self.xp_per_level = 100
        self.hp = self.max_hp
        self.attack = self.min_attack
        self.xp = 0
        self.level = 1
        self.defense = 0

    def gain_xp(self, amount: int) -> int:
        self.xp += amount
        levels_gained = 0
        
        while self.xp >= self.xp_per_level:
            self.xp -= self.xp_per_level
            self.level += 1
            levels_gained += 1
            
            hp_increase = 20
            self.max_hp += hp_increase
            self.hp += hp_increase
            
            self.min_attack += 5
        
        return levels_gained

    def act(self, obs):
        action = self._act(obs)
        return action

    def _act(self, obs):
		# override
        pass

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "inventory": self.inventory.to_dict(),
            "equipped_items_in_limb": self.equipped_items_in_limb,
            "items_in_hands": self.items_in_hands,
            "max_hp": self.max_hp,
            "hp": self.hp,
            "xp": self.xp,
            "level": self.level,
            "min_attack": self.min_attack,
            "attack": self.attack,
            "defense": self.defense,
            "xp_per_level": self.xp_per_level,
            "available_actions": [
                {
                    "name": action.name,
                    "verb": action.verb,
                    "params": action.params,
                    "description": action.description,
                }
                for action in self.available_actions
            ],
        }

    def from_dict(self, data: Dict, container_instances: dict[str, Container]) -> None:
        self.inventory.from_dict(data["inventory"], container_instances)
        self.equipped_items_in_limb = data["equipped_items_in_limb"]
        self.items_in_hands = data["items_in_hands"]
        self.max_hp = data["max_hp"]
        self.hp = data["hp"]
        self.xp = data["xp"]
        self.attack = data["attack"]
        self.defense = data["defense"]
        self.xp_per_level = data["xp_per_level"]
        self.level = data.get("level", 1)
        self.min_attack = data["min_attack"]