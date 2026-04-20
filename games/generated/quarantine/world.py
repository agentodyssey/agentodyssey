from __future__ import annotations
from dataclasses import dataclass, field, asdict
import math
from typing import ClassVar, Dict, List, Optional, Set, Tuple, Any
import random
import json
import numpy as np
import copy
from tools.logger import get_logger
from utils import *


@dataclass
class Object:
    # required
    type: str
    id: str
    name: str
    category: str
    usage: str
    # old interfaces are kept for backward compatibility
    value: Optional[int]
    size: int
    description: str
    text: str
    attack: int
    defense: int
    level: int
    quest: bool = False
    areas: List[str] = field(default_factory=list)
    craft_ingredients: Dict[str, float] = field(default_factory=dict)
    craft_dependencies: List[str] = field(default_factory=list)
    # arbitrary json-defined fields
    extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __getattr__(self, key: str) -> Any:
        extra = self.__dict__.get("extra", None)
        if extra is not None and key in extra:
            return extra[key]
        raise AttributeError(f"{type(self).__name__} has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        fields = getattr(self, "__dataclass_fields__", {})
        if key in fields:
            object.__setattr__(self, key, value)
            return

        extra = self.__dict__.get("extra", None)
        if extra is None:
            object.__setattr__(self, "extra", {key: value})
        else:
            extra[key] = value

    def create_instance(self, index: int) -> "Object":
        new_object = copy.deepcopy(self)
        new_object.name = f"{self.name}_{index}"
        new_object.id = f"{self.id}_{index}"
        return new_object

@dataclass
class Container(Object):
    capacity: int = 0
    inventory: Dict[str, int] = field(default_factory=dict)

    def current_load(self, objects: Dict[str, Object]) -> int:
        load = 0
        for oid, count in self.inventory.items():
            if oid in objects:
                load += objects[get_def_id(oid)].size * count
        return load

@dataclass
class Writable(Object):
    max_text_length: int = 0

@dataclass
class NPC:
    type: str
    id: str
    name: str
    enemy: bool
    unique: bool
    role: str
    level: int
    description: str
    attack_power: int
    hp: int  # hp and attack power will act as base stats for prototype NPC instances
    coins: int
    slope_hp: int  # default attack power
    slope_attack_power: int
    quest: bool = False  # True for quest-only NPCs
    dialogue: str = ""
    inventory: Dict[str, int] = field(default_factory=dict)
    # combat_pattern: list of actions ("attack", "defend", "wait") that enemy NPCs cycle through during combat
    combat_pattern: List[str] = field(default_factory=list)

    DEFAULT_COMBAT_PATTERN: ClassVar[List[str]] = ["attack", "defend", "attack", "wait"]
    
    def __setattr__(self, name, value):
        if name in ["slope_hp", "slope_attack_power"] and hasattr(self, name):
            raise AttributeError(f"{name} is read-only after initialization")
        super().__setattr__(name, value)

    def create_instance(self, index: int, level: int, objects: Dict[str, Object], 
                        rng: random.Random) -> Tuple[NPC, Dict[str, int], Dict[str, int]]:
        new_npc = copy.deepcopy(self)
        new_npc.name = f"{self.name}_{index}" if index is not None else self.name
        new_npc.id = f"{self.id}_{index}" if index is not None else self.id
        new_npc.level = level
        new_npc.hp = self.hp + self.slope_hp * (level - 1)
        new_npc.attack_power = self.attack_power + self.slope_attack_power * (level - 1)
        new_npc.coins = rng.randint(20 * level, 50 * level)
        for oid in self.inventory.keys():
            if oid in objects:
                count = rng.randint(0, 2 * level)
                if count > 0:
                  new_npc.inventory[oid] = count
                    
        return new_npc

@dataclass
class Path:
    to_id: str
    locked: bool
    object_to_unlock: Optional[str] = None
    _type: str = "path"

    @property
    def type(self):
        return self._type

@dataclass
class Area:
    type: str
    id: str
    name: str
    light: bool
    level: int
    objects: Dict[str, float] = field(default_factory=dict)
    npcs: list[str] = field(default_factory=list)
    neighbors: Dict[str, Path] = field(default_factory=dict)

@dataclass
class Place:
    type: str
    id: str
    name: str
    unlocked: bool
    areas: List[str] = field(default_factory=list)
    neighbors: List[str] = field(default_factory=list)

@dataclass
class World:
    world_definition: Dict = field(default_factory=dict)
    objects: Dict[str, Object] = field(default_factory=dict)
    npcs: Dict[str, NPC] = field(default_factory=dict)
    npc_instances: Dict[str, NPC] = field(default_factory=dict)
    container_instances: Dict[str, Container] = field(default_factory=dict)
    writable_instances: Dict[str, Writable] = field(default_factory=dict)
    place_instances: Dict[str, Place] = field(default_factory=dict)
    area_instances: Dict[str, Area] = field(default_factory=dict)
    auxiliary: Dict = field(default_factory=dict)
    _type: str = "world"

    @property
    def type(self):
        return self._type

    def to_dict(self) -> Dict[str, Any]:
        logger = get_logger("WorldLogger")
        data = asdict(self)
        if "entities" in data["world_definition"]:
            del data["world_definition"]["entities"]  # avoid duplications to clean up
        for oid, obj in data["objects"].items():
            extra = obj.pop("extra", {})
            for k, v in extra.items():
                # don't override official fields if collision
                obj.setdefault(k, v)

            if "craft" in obj:
                continue
            if "craft_ingredients" in obj and "craft_dependencies" in obj:
                obj["craft"] = {
                    "ingredients": obj["craft_ingredients"],
                    "dependencies": obj["craft_dependencies"],
                }
                del obj["craft_ingredients"]
                del obj["craft_dependencies"]
            else:
                logger.warning(f"Object {oid} missing crafting information during serialization.")
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "World":
        logger = get_logger("WorldLogger")
        try:
            world_definition = data.get("world_definition", {})
            objects: Dict[str, Object] = {
                oid: World._parse_object(obj_data)
                for oid, obj_data in data.get("objects", {}).items()
            }
            npcs: Dict[str, NPC] = {
                nid: World._parse_npc(npc_data)
                for nid, npc_data in data.get("npcs", {}).items()
            }
            npc_instances: Dict[str, NPC] = {
                iid: World._parse_npc(npc_data)
                for iid, npc_data in data.get("npc_instances", {}).items()
            }
            writable_instances: Dict[str, Writable] = {}
            for wid, writable_data in data.get("writable_instances", {}).items():
                writable_obj = World._parse_object(writable_data)
                if not isinstance(writable_obj, Writable):
                    logger.error(
                        f"❌ writable_instances['{wid}'] is not a Writable "
                        f"(got {type(writable_obj).__name__})"
                    )
                    continue
                writable_instances[wid] = writable_obj
            container_instances: Dict[str, Container] = {}
            for cid, cont_data in data.get("container_instances", {}).items():
                cont_obj = World._parse_object(cont_data)
                if not isinstance(cont_obj, Container):
                    logger.error(
                        f"❌ container_instances['{cid}'] is not a Container "
                        f"(got {type(cont_obj).__name__})"
                    )
                    continue
                container_instances[cid] = cont_obj
            area_instances: Dict[str, Area] = {}
            for aid, area_data in data.get("area_instances", {}).items():
                neighbors = {
                    nid: Path(**path_data)
                    for nid, path_data in area_data.get("neighbors", {}).items()
                }
                area_instances[aid] = Area(
                    type=area_data["type"],
                    id=area_data["id"],
                    name=area_data["name"],
                    light=area_data["light"],
                    objects=area_data.get("objects", {}),
                    npcs=area_data.get("npcs", []),
                    neighbors=neighbors,
                    level=area_data.get("level", 1),
                )
            place_instances: Dict[str, Place] = {}
            for pid, place_data in data.get("place_instances", {}).items():
                place = Place(
                    type=place_data["type"],
                    id=place_data["id"],
                    name=place_data["name"],
                    unlocked=place_data["unlocked"],
                    areas=place_data["areas"],
                    neighbors=place_data.get("neighbors", []),
                )
                place_instances[pid] = place

            auxiliary: Dict[str, Any] = data.get("auxiliary", {})
            world = World(
                world_definition=world_definition,
                objects=objects,
                npcs=npcs,
                npc_instances=npc_instances,
                container_instances=container_instances,
                writable_instances=writable_instances,
                place_instances=place_instances,
                area_instances=area_instances,
                auxiliary=auxiliary,
            )

            if "_type" in data:
                world._type = data["_type"]

            return world

        except KeyError as e:
            logger.error(f"❌ Error reconstructing World from config: {e}")
            raise

    @staticmethod
    def generate(world_definition: dict, seed: Optional[int] = None) -> World:
        logger = get_logger("WorldLogger")
        logger.info(f"🚀 Generating the game world using seed {seed}")

        # We generate different random generators because call sequence of the single rng will change the randomness
        base_rng = random.Random(seed)
        seed_npcs = base_rng.randint(0, 10**9)
        seed_places = base_rng.randint(0, 10**9)
        seed_connections = base_rng.randint(0, 10**9)
        seed_objects_dist = base_rng.randint(0, 10**9)
        seed_npc_assign  = base_rng.randint(0, 10**9)
        rng_npcs = random.Random(seed_npcs)
        rng_places = random.Random(seed_places)
        rng_connections = random.Random(seed_connections)
        rng_objects_dist = random.Random(seed_objects_dist)
        rng_npc_assign = random.Random(seed_npc_assign)

        try:
            place_instances, area_instances, area_to_place = World._instantiate_places(world_definition, rng_places)
            objects, obj_name_to_id, ing_to_obj_map = World._synthesize_objects(world_definition, area_instances)
            npcs = World._synthesize_npcs(world_definition, rng_npcs, list(objects.keys()))
            if not World._load_predefined_graph(world_definition, area_instances):
                World._connect_intraplace_graphs(rng_connections, place_instances, area_instances, objects)
                World._connect_interplace_graphs(rng_connections, place_instances, area_instances, objects)
            # need a global array of containers to keep track of individual container's items
            container_instances, container_id_to_count, writable_instances, writable_id_to_count, obj_name_to_id = World._distribute_objects(world_definition, rng_objects_dist, area_instances, objects, obj_name_to_id)
            npc_instances, npc_name_to_id, npc_id_to_count, npc_name_to_count = World._assign_npcs_to_areas(world_definition, rng_npc_assign, place_instances, area_instances, npcs, objects)
            logger.info("✅ World generation completed")

            return World(world_definition=world_definition, 
                        objects=objects, 
                        npcs=npcs, 
                        container_instances=container_instances, 
                        writable_instances=writable_instances,
                        npc_instances=npc_instances,
                        place_instances=place_instances, 
                        area_instances=area_instances,
                        auxiliary={"obj_name_to_id": obj_name_to_id,
                                   "ing_to_obj_map": ing_to_obj_map,
                                   "container_id_to_count": container_id_to_count,
                                   "writable_id_to_count": writable_id_to_count,
                                   "npc_name_to_id": npc_name_to_id,
                                   "npc_id_to_count": npc_id_to_count,
                                   "npc_name_to_count": npc_name_to_count,
                                   "area_to_place": area_to_place,
                                   })
        except Exception as e:
            logger.error(f"❌ Error during world generation: {e}")
            raise
    
    @staticmethod
    def _collect_extra_fields(spec: dict, reserved: set[str]) -> Dict[str, Any]:
        extra = {}
        for k, v in spec.items():
            if k in reserved or k == "craft":
                continue
            extra[k] = v
        return extra

    @staticmethod
    def _parse_object(object: dict, area_instances: dict[str, Area] = None) -> Object:
        logger = get_logger("WorldLogger")
        if area_instances and object.get("places", None):
            for place in object["places"]:
                if place not in area_instances:
                    logger.warning(f"Object {object['id']} specifies unknown place {place}")
        
        reserved = {
            "type","id","name","category","usage", "description",
            "value","size","text","attack","defense","level","quest",
            "areas","craft","craft_ingredients","craft_dependencies",
            "capacity","inventory","max_text_length", "places",
        }
        extra = World._collect_extra_fields(object, reserved)
        
        # checking for container
        if object["category"] == "container":
            return Container(
                type=object.get("type", "object"),
                id=object["id"],
                name=object["name"],
                size=object["size"],
                category=object["category"],
                usage=object["usage"],
                capacity=object["capacity"],
                level=object["level"],
                description=object.get("description", ""),
                value=object.get("value", None),
                text=object.get("text", ""),
                attack=object.get("attack", 0),
                defense=object.get("defense", 0),
                quest=object.get("quest", False),
                craft_ingredients=object.get("craft", {}).get("ingredients", {}),
                craft_dependencies=object.get("craft", {}).get("dependencies", []),
                inventory=object.get("inventory", {}),
                areas=object.get("areas", []),
                extra=extra,
            )
        
        if object["usage"] == "writable":
            max_text_length = len(object["text"]) if object["id"].startswith("obj_given_") else object["max_text_length"]
            return Writable(
                type=object.get("type", "object"),
                id=object["id"],
                name=object["name"],
                size=object["size"],
                category=object["category"],
                usage=object["usage"],
                level=object.get("level", 1),
                description=object.get("description", ""),
                text=object.get("text", ""),
                value=object.get("value", None),
                max_text_length=max_text_length,
                attack=object.get("attack", 0),
                defense=object.get("defense", 0),
                quest=object.get("quest", False),
                craft_ingredients=object.get("craft", {}).get("ingredients", {}),
                craft_dependencies=object.get("craft", {}).get("dependencies", []),
                areas=object.get("areas", []),
                extra=extra,
            )

        return Object(
            type=object.get("type", "object"),
            id=object["id"],
            name=object["name"],
            size=object["size"],
            level=object.get("level", 1),
            text=object.get("text", ""),
            description=object.get("description", ""),
            category=object.get("category", "misc"),
            usage=object.get("usage", ""),
            value=object.get("value", None),
            attack=object.get("attack", 0),
            defense=object.get("defense", 0),
            quest=object.get("quest", False),
            craft_ingredients=object.get("craft", {}).get("ingredients", {}),
            craft_dependencies=object.get("craft", {}).get("dependencies", []),
            areas=object.get("areas", []),
            extra=extra,
        )
    
    @staticmethod
    def _parse_npc(npc: dict, rng: random.Random = None, objects: Dict[str, Object] = None) -> NPC:
        if rng is None:
            return NPC(
                type=npc.get("type", "npc"),
                id=npc["id"],
                name=npc["name"],
                hp=npc["hp"],
                enemy=npc["enemy"],
                unique=npc["unique"],
                role=npc["role"],
                description=npc["description"],
                inventory=npc["inventory"],
                coins=npc["coins"],
                attack_power=npc["attack_power"],
                slope_hp=npc["slope_hp"],
                slope_attack_power=npc["slope_attack_power"],
                level=npc["level"],
                quest=npc.get("quest", False),
                dialogue=npc.get("dialogue", ""),
                combat_pattern=npc.get("combat_pattern", NPC.DEFAULT_COMBAT_PATTERN if npc["enemy"] else []),
            )
                
        enemy = bool(npc.get("enemy", False))
        unique = bool(npc.get("unique", True))
        hp = npc.get("base_hp", 60) if enemy else npc.get("base_hp", 1000)
        attack_power = npc.get("base_attack_power", 15) if enemy else npc.get("base_attack_power", 10)
        slope_hp = npc.get("slope_hp", 20) if enemy else npc.get("slope_hp", 50)
        slope_attack_power = npc.get("slope_attack_power", 15) if enemy else npc.get("slope_attack_power", 5)

        inventory: Dict[str, int] = {}
        npc_objects = npc.get("objects", [])
        for obj_id in npc_objects:
            inventory[obj_id] = inventory.get(obj_id, 0) + 1

        return NPC(
            type=npc.get("type", "npc"),
            id=npc["id"],
            name=npc["name"],
            enemy=enemy,
            unique=unique,
            role=npc["role"],
            hp=hp,
            level=1,  # a placeholder, will be adjusted when real NPC instances are created
            description=npc.get("description", ""),
            inventory=inventory,  # a placeholder, will be adjusted when real NPC instances are created
            coins=0,
            attack_power=attack_power,
            slope_hp=slope_hp,
            slope_attack_power=slope_attack_power,
            quest=npc.get("quest", False),
            combat_pattern=npc.get("combat_pattern", NPC.DEFAULT_COMBAT_PATTERN if enemy else []),
        )

    @staticmethod
    def _synthesize_objects(world_definition: dict, area_instances: dict[str, Area] = None) -> Tuple[Dict[str, Object], Dict[str, str], Dict[str, List[str]]]:
        logger = get_logger("WorldLogger")
        objects: Dict[str, Object] = {}
        obj_name_to_id: Dict[str, str] = {}
        ing_to_obj_map: Dict[str, List[str]] = {}

        ing_to_obj_map_set: Dict[str, Set[str]] = {}
        for spec in world_definition["entities"].get("objects", []):
            objects[spec["id"]] = World._parse_object(spec, area_instances)
            obj_name_to_id[spec["name"]] = spec["id"]
            # logger.info(f"✅ Synthesized Object {obj_map[spec['id']].name} (ID: {obj_map[spec['id']].id}, Category: {obj_map[spec['id']].category}, Size: {obj_map[spec['id']].size})")

            for oid, obj in objects.items():
                if not hasattr(obj, "craft_ingredients"):
                    continue
                for ing_id in getattr(obj, "craft_ingredients", {}).keys():
                    ing_to_obj_map_set.setdefault(ing_id, set()).add(oid)

        ing_to_obj_map = {k: list(v) for k, v in ing_to_obj_map_set.items()}

        # crafting_chart = World._setup_crafting_chart(world_definition, obj_map)
        # obj_map[crafting_chart.id] = crafting_chart
        # obj_name_to_id[crafting_chart.name] = crafting_chart.id
        logger.info(f"✅ Synthesized Objects (total: {len(objects)})")

        return objects, obj_name_to_id, ing_to_obj_map

    @staticmethod
    def _synthesize_npcs(world_definition: dict, rng: random.Random, object_ids: List[str]) -> Dict[str, list[NPC]]:
        logger = get_logger("WorldLogger")
        npcs: Dict[str, list[NPC]] = {}
        for spec in world_definition["entities"].get("npcs", []):
            npcs[spec["id"]] = World._parse_npc(spec, rng, object_ids)
            # logger.info(f"✅ Synthesized NPC {npcs[spec['id']][0].name} (ID: {npcs[spec['id']][0].id}, Role: {npcs[spec['id']][0].role})")
        logger.info(f"✅ Synthesized NPCs (total types: {len(npcs)})")
        return npcs

    @staticmethod
    def _instantiate_places(world_definition: dict, rng: random.Random) -> Tuple[Dict[str, Place], Dict[str, Area], Dict[str, str]]:
        logger = get_logger("WorldLogger")
        place_instances: Dict[str, Place] = {}
        area_instances: Dict[str, Area] = {}
        area_to_place: Dict[str, str] = {}
        spawn_area = world_definition["initializations"].get("spawn", {}).get("area", None)
        for p in world_definition["entities"].get("places", []):
            areas: Dict[str, Area] = {}
            for a in p.get("areas", []):
                level = a["level"] if a["id"] != spawn_area else 1
                is_spawn = a["id"] == spawn_area
                areas[a["id"]] = Area(
                    type=a["type"],
                    id=a["id"],
                    name=a["name"],
                    light=True if is_spawn else a.get("light", bool(rng.random() < 0.75)),
                    level=level,
                )
                area_to_place[a["id"]] = p["id"]
            place_instances[p["id"]] = Place(
                type=p["type"],
                id=p["id"], 
                name=p["name"], 
                unlocked=p["unlocked"], 
                areas=list(areas.keys()), 
                neighbors=[]
            )
            area_instances.update(areas)
            logger.info(f"✅ Instantiated Place {place_instances[p['id']].name} (ID: {place_instances[p['id']].id}, Unlocked: {place_instances[p['id']].unlocked}) with {len(areas)} areas")
        return place_instances, area_instances, area_to_place

    @staticmethod
    def _load_predefined_graph(world_definition: dict, area_instances: Dict[str, Area]) -> bool:
        logger = get_logger("WorldLogger")
        graph_def = world_definition.get("graph")
        if not graph_def:
            return False
        connections = graph_def.get("connections", [])
        if not connections:
            logger.warning("Graph definition found but no connections specified")
            return False

        for area in area_instances.values():
            area.neighbors = {}

        for conn in connections:
            from_id = conn.get("from")
            to_id = conn.get("to")
            locked = conn.get("locked", False)
            key = conn.get("key")
            if from_id not in area_instances:
                logger.warning(f"Graph connection references unknown area: {from_id}")
                continue
            if to_id not in area_instances:
                logger.warning(f"Graph connection references unknown area: {to_id}")
                continue
            from_area = area_instances[from_id]
            to_area = area_instances[to_id]
            from_area.neighbors[to_id] = Path(to_id=to_id, locked=locked, object_to_unlock=key)
            to_area.neighbors[from_id] = Path(to_id=from_id, locked=locked, object_to_unlock=key)

        logger.info(f"✅ Loaded predefined graph with {len(connections)} connections")
        return True

    @staticmethod
    def _connect_intraplace_graphs(rng: random.Random, place_instances: Dict[str, Place], area_instances: Dict[str, Area], objects: Dict[str, Object]) -> None:
        logger = get_logger("WorldLogger")
        unlockables = [oid for oid, obj in objects.items() if obj.usage == "unlock"]
        for place in place_instances.values():
            area_ids = place.areas.copy()
            if len(area_ids) <= 1:
                continue
            rng.shuffle(area_ids)
            for i in range(len(area_ids) - 1):
                a = area_instances[area_ids[i]]
                b = area_instances[area_ids[i + 1]]
                locked = False
                key = None
                if not place.unlocked and unlockables:
                    if rng.random() < 0.6:
                        locked = True
                        key = rng.choice(unlockables)
                a.neighbors[b.id] = Path(to_id=b.id, locked=locked, object_to_unlock=key)
                b.neighbors[a.id] = Path(to_id=a.id, locked=locked, object_to_unlock=key)
        logger.info("✅ Connected areas within each place")

    @staticmethod
    def _connect_interplace_graphs(rng: random.Random, place_instances: Dict[str, Place], area_instances: Dict[str, Area], objects: Dict[str, Object]) -> None:
        if len(place_instances) <= 1:
            return
        logger = get_logger("WorldLogger")
        unlockables = [oid for oid, obj in objects.items() if obj.usage == "unlock"]
        place_ids = list(place_instances.keys())
        rng.shuffle(place_ids)
        for i in range(len(place_ids) - 1):
            u_id, v_id = place_ids[i], place_ids[i + 1]
            u_place, v_place = place_instances[u_id], place_instances[v_id]
            u_area_id = rng.choice(u_place.areas)
            v_area_id = rng.choice(v_place.areas)
            u_area = area_instances[u_area_id]
            v_area = area_instances[v_area_id]

            if u_place.unlocked and v_place.unlocked:
                locked = False
                key = None
            elif u_place.unlocked != v_place.unlocked:
                locked = True
                key = rng.choice(unlockables) if unlockables else None
            else:
                locked = rng.random() < 0.5 if unlockables else False
                key = rng.choice(unlockables) if locked else None

            u_area.neighbors[v_area_id] = Path(to_id=v_area_id, locked=locked, object_to_unlock=key)
            v_area.neighbors[u_area_id] = Path(to_id=u_area_id, locked=locked, object_to_unlock=key)
        logger.info("✅ Connected places together")

    @staticmethod
    def _distribute_objects(world_definition: dict, rng: random.Random, area_instances: Dict[str, Area],
                            objects: Dict[str, Object], obj_name_to_id: Dict[str, str]) -> Tuple[Dict[str, Container], Dict[str, int], Dict[str, str]]:
        area_list: List[Area] = list(area_instances.values())
        object_pool = list(objects.keys())
        undistributable_objects = world_definition["initializations"].get("undistributable_objects", [])
        logger = get_logger("WorldLogger")

        container_instances: dict[str, Container] = {}  # a separate dict to store all created containers with their ids as keys
        container_id_to_count: dict[str, int] = {oid: 0 for oid, obj in objects.items() if obj.category == "container"}

        writable_instances: Dict[str, Writable] = {}
        writable_id_to_count: Dict[str, int] = {oid: 0 for oid, obj in objects.items() if obj.usage == "writable"}

        non_comp_craft_pool = [oid for oid in object_pool if objects[oid].areas and not objects[oid].craft_ingredients
                       and not oid.startswith("obj_given_") and oid not in undistributable_objects]
        comp_craft_pool = [oid for oid in object_pool if objects[oid].craft_ingredients and oid not in undistributable_objects
                                    and not oid.startswith("obj_given_") and objects[oid].areas]
        global_pool = [oid for oid in object_pool if not objects[oid].craft_ingredients and not objects[oid].areas
                       and oid not in undistributable_objects]
        global_pool += [oid for oid in object_pool if objects[oid].usage == "writable" and not objects[oid].craft_dependencies
                        and not objects[oid].category == "currency" and oid not in undistributable_objects]
        station_pool = [oid for oid in object_pool if objects[oid].category == "station" and oid not in undistributable_objects]
        unlock_pool = [oid for oid in object_pool if (objects[oid].usage == "unlock" or objects[oid].usage == "lockpick")
                       and oid not in undistributable_objects]

        spawn_info = world_definition["initializations"].get("spawn", None)
        distributable_areas = area_list if not spawn_info else [a for a in area_list if a.id != spawn_info["area"]]
        area_levels = set(area.level for area in distributable_areas)
        base_distribution_k = 5
        distribution_k_by_area_level = {i : base_distribution_k + (max(area_levels) - i) for i in area_levels}

        for area in distributable_areas:
            level = area.level
            # only distribute materials and unlockables for now
            non_comp_craft_materials = [oid for oid in non_comp_craft_pool if area.id in objects[oid].areas]
            non_comp_craft_material_weights = []
            for oid in non_comp_craft_materials:
                weight = 0.8 if objects[oid].level == level else 0.4 if level - 1 <= objects[oid].level <= level + 1 else 0.1
                non_comp_craft_material_weights.append(weight)
            comp_craft_materials = [oid for oid in comp_craft_pool if area.id in objects[oid].areas]
            comp_craft_material_weights = []
            for oid in comp_craft_materials:
                weight = 0.4 if objects[oid].level == level else 0.2 if level - 1 <= objects[oid].level <= level + 1 else 0.05
                comp_craft_material_weights.append(weight)
            # guarantee every area-specific material appears at least once
            for oid in non_comp_craft_materials:
                area.objects[oid] = area.objects.get(oid, 0) + max(1, level)
            for oid in comp_craft_materials:
                area.objects[oid] = area.objects.get(oid, 0) + 1
            materials = non_comp_craft_materials + comp_craft_materials + global_pool + unlock_pool
            material_weights = non_comp_craft_material_weights + comp_craft_material_weights + [0.03] * len(global_pool) + [0.001] * len(unlock_pool)
            # random.choices() samples with replacement
            # any material count greater than `distribution_k_by_area_level[level]` will be awarded an expectation of < 1.5 per area at min
            # any material count greater than `distribution_k_by_area_level[level]` will be awarded an expectation of < 4 per area at max
            k_materials = rng.randint(min(int(len(materials) * 2), int(distribution_k_by_area_level[level] * 2)), min(int(len(materials) * 4), int(distribution_k_by_area_level[level] * 4)))
            chosen = rng.choices(materials, weights=material_weights, k=k_materials)
            for oid in chosen:
                if objects[oid].category in ["station"] and oid in area.objects and area.objects[oid] > 0:
                    continue  # only one station per area
                
                if objects[oid].usage == "writable":
                    new_writable: Writable = objects[oid].create_instance(writable_id_to_count[oid])
                    writable_instances[new_writable.id] = new_writable
                    area.objects[new_writable.id] = 1
                    obj_name_to_id[new_writable.name] = new_writable.id
                    writable_id_to_count[oid] += 1
                    continue
                
                area.objects[oid] = area.objects.get(oid, 0) + 1
        for area in area_list:
            coin_obj_ids = [oid for oid, obj in objects.items() if obj.category == "currency"]
            if coin_obj_ids and rng.random() < 0.5:
                coin_id = rng.choice(coin_obj_ids)
                qty = rng.randint(1, area.level * 2)
                area.objects[coin_id] = area.objects.get(coin_id, 0) + qty
                
        # execute predefined initializations
        if spawn_info:
            spawn_area = spawn_info["area"]
            spawn_objects = spawn_info.get("objects", {})
            area_instance = area_instances[spawn_area]
            area_instance.objects = {}  # clear the randomly spawned objects

            # find the object that has category container
            for (oid, count) in spawn_objects.copy().items():
                if oid not in obj_name_to_id.values():
                    logger.warning(f"❌ Object {oid} specified in spawn initialization not found in object definitions. Skipping...")
                    continue
                
                # containers and writables have separate instances for the same object type for storing unique information like texts
                if objects[oid].category == "container":
                    del spawn_objects[oid]
                    for i in range(count):
                        # we can safely use i for indexing because containers have dependencies to craft and so will not be auto distributed
                        # the container index in the name/id starts from 0
                        new_container: Container = objects[oid].create_instance(i)
                        container_instances[new_container.id] = new_container
                        area_instance.objects[new_container.id] = 1
                        obj_name_to_id[new_container.name] = new_container.id
                        container_id_to_count[oid] = container_id_to_count.get(oid, 0) + 1

                elif objects[oid].usage == "writable" and not objects[oid].text:  # for non-given writables
                    for _ in range(count):
                        new_writable: Writable = objects[oid].create_instance(writable_id_to_count[oid])
                        writable_instances[new_writable.id] = new_writable
                        area_instance.objects[new_writable.id] = 1
                        obj_name_to_id[new_writable.name] = new_writable.id
                        writable_id_to_count[oid] += 1
                
                else:
                    area_instance.objects[oid] = count

            # area_instance.objects["obj_crafting_chart"] = 1
        logger.info("✅ Distributed objects to areas")
        
        return container_instances, container_id_to_count, writable_instances, writable_id_to_count, obj_name_to_id

    @staticmethod
    def _setup_crafting_chart(world_definition: dict, objects: Dict[str, Object]) -> Object:
        # generate a writable that provides each object and its corresponding crafting recipe
        crafting_chart = Object(
            type="object",
            id="obj_crafting_chart",
            name="crafting_chart",
            category="tool",
            usage="writable",
            max_text_length=0,
            size=1,
            attack_power=10,
            defense=0,
            level=1,
            craft_dependencies=[],
            craft_ingredients={},
            text=""
        )
        lines = ["\n====== Crafting Chart ======\n"]
        for obj in objects.values():
            if obj.craft_ingredients:
                ingredient_str = f"{obj.name}: "
                for ingredient_id, qty in obj.craft_ingredients.items():
                    ingredient_name = objects[ingredient_id].name if ingredient_id in objects else "Unknown"
                    ingredient_str += f"{qty} {ingredient_name}, "
                lines.append(ingredient_str.strip(", "))  # Remove trailing comma and space
                if obj.craft_dependencies:
                    dep_names = [objects[dep_id].name for dep_id in obj.craft_dependencies if dep_id in objects]
                    lines.append(f"  - requires: {', '.join(dep_names)}")
                lines.append("")
        lines[-1] = "============================"
        crafting_chart.text = "\n".join(lines)
        crafting_chart.max_text_length = len(crafting_chart.text)
        return crafting_chart

    @staticmethod
    def _assign_npcs_to_areas(world_definition: dict, rng: random.Random, places: Dict[str, Place], area_instances: Dict[str, Area], 
                              npcs: Dict[str, NPC], objects: Dict[str, Object]) -> Tuple[Dict[str, NPC], Dict[str, str], Dict[str, int]]:
        spawn_info = world_definition["initializations"].get("spawn", None)
        distributable_area_list: List[Area] = [a for a in area_instances.values() if not (spawn_info and a.id == spawn_info["area"])]
        if not distributable_area_list or not npcs:
            return
        logger = get_logger("WorldLogger")
        unique_npcs = [npc for npc in npcs.values() if npc.unique]
        non_unique_npcs = [npc for npc in npcs.values() if not npc.unique]
        npc_id_to_count = {npc: 0 for npc in npcs.keys()}
        npc_name_to_count: Dict[str, int] = {}
        npc_instances: Dict[str, NPC] = {}
        area_levels = set(area.level for area in distributable_area_list)
        base_distribution_prob = 0.6
        distribution_prob_by_area_level = {i : base_distribution_prob + (i - 1) * (1 - base_distribution_prob) / max(area_levels) for i in area_levels}
        npc_name_to_id = {}
        for npc in unique_npcs:
            if npc.id in spawn_info.get("npcs", {}):
                continue
            if getattr(npc, "quest", False):
                continue
            home = rng.choice(distributable_area_list)
            new_npc_levels = [home.level - 1, home.level, home.level + 1] if home.level > 1 else [home.level, home.level + 1]
            new_npc_levels_weights = [0.2, 0.6, 0.2] if home.level > 1 else [0.8, 0.2]
            new_npc_level = rng.choices(new_npc_levels, weights=new_npc_levels_weights, k=1)[0]
            new_npc = npc.create_instance(None, new_npc_level, objects, rng)
            npc_instances[new_npc.id] = new_npc
            npc_name_to_id[new_npc.name] = new_npc.id
            home.npcs.append(new_npc.id)
        for home in distributable_area_list:
            enemy_count = rng.randint(math.floor(home.level / 2), home.level)  # make enemy count progress with area level
            for npc in non_unique_npcs:
                if getattr(npc, "quest", False):
                    continue
                if rng.random() < 1.0 - distribution_prob_by_area_level[home.level]:
                    continue
                if enemy_count <= 0:
                    break
                new_npc_levels = [home.level - 1, home.level, home.level + 1] if home.level > 1 else [home.level, home.level + 1]
                new_npc_levels_weights = [0.2, 0.6, 0.2] if home.level > 1 else [0.8, 0.2]
                new_npc_level = rng.choices(new_npc_levels, weights=new_npc_levels_weights, k=1)[0]
                new_npc = npc.create_instance(npc_id_to_count[npc.id], new_npc_level, objects, rng)
                if new_npc.enemy:
                    enemy_count -= 1
                npc_id_to_count[npc.id] += 1
                npc_name_to_count[npc.name] = npc_name_to_count.get(npc.name, 0) + 1
                npc_instances[new_npc.id] = new_npc
                npc_name_to_id[new_npc.name] = new_npc.id
                home.npcs.append(new_npc.id)
        # execute predefined initializations
        if spawn_info:
            spawn_area = spawn_info["area"]
            spawn_npcs = spawn_info.get("npcs", {})
            area_instances[spawn_area].npcs = []
            for npc_id, npc_count in spawn_npcs.items():
                if npc_id not in npcs.keys():
                    continue
                npc = npcs[npc_id]
                for _ in range(npc_count):
                    new_npc_levels = [home.level - 1, home.level, home.level + 1] if home.level > 1 else [home.level, home.level + 1]
                    new_npc_levels_weights = [0.2, 0.6, 0.2] if home.level > 1 else [0.8, 0.2]
                    new_npc_level = rng.choices(new_npc_levels, weights=new_npc_levels_weights, k=1)[0]
                    new_npc = npc.create_instance(npc_id_to_count[npc.id], new_npc_level, objects, rng)
                    npc_id_to_count[npc.id] += 1
                    npc_name_to_count[npc.name] = npc_name_to_count.get(npc.name, 0) + 1
                    npc_instances[new_npc.id] = new_npc
                    npc_name_to_id[new_npc.name] = new_npc.id
                    area_instances[spawn_area].npcs.append(new_npc.id)

        logger.info("✅ Assigned NPCs to areas")

        return npc_instances, npc_name_to_id, npc_id_to_count, npc_name_to_count

    def _find_furthest_area(
        self, spawn_area_id: Optional[str], candidate_ids: set
    ) -> str:
        from collections import deque

        if not spawn_area_id or spawn_area_id not in self.area_instances:
            # can't BFS — just return any candidate
            return next(iter(candidate_ids))

        visited: Dict[str, int] = {spawn_area_id: 0}
        queue: deque = deque([spawn_area_id])

        while queue:
            current = queue.popleft()
            current_area = self.area_instances.get(current)
            if current_area is None:
                continue
            for neighbor_id in current_area.neighbors:
                if neighbor_id not in visited and neighbor_id in self.area_instances:
                    visited[neighbor_id] = visited[current] + 1
                    queue.append(neighbor_id)

        best_id, best_dist = None, -1
        for cid in candidate_ids:
            dist = visited.get(cid, -1)
            if dist > best_dist:
                best_dist = dist
                best_id = cid

        return best_id if best_id is not None else next(iter(candidate_ids))

    def expand(self, expansion_def: dict, seed: int = None) -> Tuple[List[str], List[str]]:
        logger = get_logger("WorldLogger")
        rng = random.Random(seed)

        new_place_names: List[str] = []
        new_area_ids: List[str] = []
        new_area_names: List[str] = []

        # --- 1. add new object definitions ---
        added_objects = []
        for obj_spec in expansion_def.get("objects", []):
            obj_id = obj_spec.get("id")
            if not obj_id or obj_id in self.objects:
                continue
            try:
                obj = World._parse_object(obj_spec)
                self.objects[obj.id] = obj
                self.auxiliary["obj_name_to_id"][obj.name] = obj.id
                for ing_id in getattr(obj, "craft_ingredients", {}).keys():
                    bucket = self.auxiliary.setdefault("ing_to_obj_map", {}).setdefault(ing_id, [])
                    if obj.id not in bucket:
                        bucket.append(obj.id)
                added_objects.append(obj.name)
            except Exception as e:
                logger.warning(f"\u26a0\ufe0f Expansion: Failed to parse object {obj_id}: {e}")
        if added_objects:
            logger.info(f"\u2705 Expansion: Added {len(added_objects)} objects: {', '.join(added_objects)}")

        # --- 2. add new NPC prototypes ---
        added_npcs = []
        for npc_spec in expansion_def.get("npcs", []):
            npc_id = npc_spec.get("id")
            if not npc_id or npc_id in self.npcs:
                continue
            try:
                npc = World._parse_npc(npc_spec, rng, list(self.objects.keys()))
                self.npcs[npc.id] = npc
                self.auxiliary.setdefault("npc_id_to_count", {})[npc.id] = 0
                added_npcs.append(npc.name)
            except Exception as e:
                logger.warning(f"\u26a0\ufe0f Expansion: Failed to parse NPC {npc_id}: {e}")
        if added_npcs:
            logger.info(f"\u2705 Expansion: Added {len(added_npcs)} NPCs: {', '.join(added_npcs)}")

        # --- 3. instantiate new places and areas ---
        for place_spec in expansion_def.get("places", []):
            place_id = place_spec.get("id")
            if not place_id or place_id in self.place_instances:
                continue

            areas: Dict[str, Area] = {}
            for area_spec in place_spec.get("areas", []):
                area_id = area_spec.get("id")
                if not area_id or area_id in self.area_instances:
                    continue
                area = Area(
                    type=area_spec.get("type", "area"),
                    id=area_id,
                    name=area_spec["name"],
                    light=area_spec.get("light", rng.random() < 0.75),
                    level=area_spec.get("level", 1),
                )
                areas[area.id] = area
                self.area_instances[area.id] = area
                self.auxiliary["area_to_place"][area.id] = place_id
                new_area_ids.append(area.id)
                new_area_names.append(area.name)

            if not areas:
                continue

            place = Place(
                type=place_spec.get("type", "place"),
                id=place_id,
                name=place_spec["name"],
                unlocked=place_spec.get("unlocked", True),
                areas=list(areas.keys()),
                neighbors=[],
            )
            self.place_instances[place.id] = place
            new_place_names.append(place.name)

            # connect areas within the new place (chain)
            area_id_list = list(areas.keys())
            rng.shuffle(area_id_list)
            for i in range(len(area_id_list) - 1):
                a = self.area_instances[area_id_list[i]]
                b = self.area_instances[area_id_list[i + 1]]
                a.neighbors[b.id] = Path(to_id=b.id, locked=False)
                b.neighbors[a.id] = Path(to_id=a.id, locked=False)

        if new_place_names:
            place_detail = "; ".join(
                f"{name} ({len([a for a in self.place_instances[pid].areas])} areas)"
                for pid, name in zip(
                    [p.get("id") for p in expansion_def.get("places", []) if p.get("id") in self.place_instances],
                    new_place_names,
                )
            )
            logger.info(f"\u2705 Expansion: Added {len(new_place_names)} places: {place_detail}")

        # --- 4. bridge each new place to the reachable graph ---
        if new_area_ids:
            existing_area_ids = set(
                aid for aid in self.area_instances if aid not in new_area_ids
            )
            spawn_area_id = (
                self.world_definition.get("initializations", {})
                .get("spawn", {})
                .get("area")
            )

            # Group new area IDs by their parent place
            new_place_ids_ordered = []
            place_to_new_areas: Dict[str, List[str]] = {}
            for place_spec in expansion_def.get("places", []):
                pid = place_spec.get("id")
                if pid and pid in self.place_instances:
                    area_ids_in_place = [
                        a.get("id") for a in place_spec.get("areas", [])
                        if a.get("id") in self.area_instances
                    ]
                    if area_ids_in_place:
                        new_place_ids_ordered.append(pid)
                        place_to_new_areas[pid] = area_ids_in_place

            # Reachable set starts as all existing areas
            reachable = set(existing_area_ids)
            bridge_log = []

            for pid in new_place_ids_ordered:
                areas_in_place = place_to_new_areas[pid]
                # Pick the bridge target: furthest reachable area from spawn
                bridge_existing = self._find_furthest_area(
                    spawn_area_id, reachable
                )
                # Pick one area from this new place as bridge endpoint
                bridge_new = rng.choice(areas_in_place)

                self.area_instances[bridge_existing].neighbors[bridge_new] = Path(
                    to_id=bridge_new, locked=False
                )
                self.area_instances[bridge_new].neighbors[bridge_existing] = Path(
                    to_id=bridge_existing, locked=False
                )

                # Update place neighbour lists
                ep_id = self.auxiliary["area_to_place"][bridge_existing]
                if pid not in self.place_instances[ep_id].neighbors:
                    self.place_instances[ep_id].neighbors.append(pid)
                if ep_id not in self.place_instances[pid].neighbors:
                    self.place_instances[pid].neighbors.append(ep_id)

                bridge_log.append(
                    f"{self.area_instances[bridge_new].name} \u2194 "
                    f"{self.area_instances[bridge_existing].name}"
                )

                # Add this place's areas to the reachable set for chaining
                reachable.update(areas_in_place)

            if bridge_log:
                logger.info(
                    f"\u2705 Expansion: Bridged {len(bridge_log)} connections: "
                    + ", ".join(bridge_log)
                )

        # --- 5. populate new areas with objects and NPCs ---
        self._populate_expansion_areas(rng, new_area_ids, expansion_def)

        wd_entities = self.world_definition.setdefault("entities", {})
        wd_entities.setdefault("objects", []).extend(expansion_def.get("objects", []))
        wd_entities.setdefault("npcs", []).extend(expansion_def.get("npcs", []))
        wd_entities.setdefault("places", []).extend(expansion_def.get("places", []))

        logger.info(
            f"\u2705 Expansion complete: +{len(new_place_names)} places, "
            f"+{len(new_area_ids)} areas"
        )
        return new_place_names, new_area_names

    def _populate_expansion_areas(
        self, rng: random.Random, area_ids: List[str],
        expansion_def: Optional[dict] = None,
    ) -> None:
        logger = get_logger("WorldLogger")
        undistributable = set(
            self.world_definition.get("initializations", {})
            .get("undistributable_objects", [])
        )

        # build a mapping: area_id -> LLM-assigned existing object IDs
        llm_assigned: Dict[str, List[str]] = {}
        if expansion_def:
            for place_spec in expansion_def.get("places", []):
                for area_spec in place_spec.get("areas", []):
                    aid = area_spec.get("id")
                    if aid:
                        raw_ids = area_spec.get("existing_objects", [])
                        # Keep only IDs that actually exist and are distributable
                        llm_assigned[aid] = [
                            oid for oid in raw_ids
                            if oid in self.objects
                            and oid not in undistributable
                            and self.objects[oid].category not in ("currency",)
                            and getattr(self.objects[oid], "areas", [])
                        ]

        # fallback pool: only objects that were originally area-distributed
        fallback_pool = [
            oid for oid, obj in self.objects.items()
            if oid not in undistributable
            and obj.category not in ("currency", "container")
            and obj.usage != "writable"
            and getattr(obj, "areas", [])
        ]

        for area_id in area_ids:
            area = self.area_instances[area_id]
            level = area.level

            # -- 1. place LLM-assigned existing objects --
            assigned = llm_assigned.get(area_id, [])
            for oid in assigned:
                obj = self.objects[oid]
                if obj.category == "station" and oid in area.objects:
                    continue  # only one station per area
                area.objects[oid] = area.objects.get(oid, 0) + 1

            # -- 2. guarantee at least 3 existing objects via random fallback --
            existing_count = sum(
                cnt for oid, cnt in area.objects.items()
                if oid in self.objects
            )
            if existing_count < 3 and fallback_pool:
                deficit = 3 - existing_count
                # Weight by level proximity
                weights = [
                    0.6 if abs(self.objects[oid].level - level) <= 1 else 0.1
                    for oid in fallback_pool
                ]
                extra = rng.choices(fallback_pool, weights=weights, k=deficit)
                for oid in extra:
                    if (
                        self.objects[oid].category == "station"
                        and oid in area.objects
                    ):
                        continue
                    area.objects[oid] = area.objects.get(oid, 0) + 1

            # -- 3. place new expansion objects assigned to this area --
            new_area_craft = [
                oid for oid, obj in self.objects.items()
                if area_id in getattr(obj, "areas", [])
                and oid not in undistributable
            ]
            for oid in new_area_craft:
                area.objects[oid] = area.objects.get(oid, 0) + rng.randint(1, 2)

            # -- 4. coins (conservative) --
            coin_ids = [
                oid for oid, obj in self.objects.items()
                if obj.category == "currency"
            ]
            if coin_ids and rng.random() < 0.4:
                cid = rng.choice(coin_ids)
                area.objects[cid] = area.objects.get(cid, 0) + rng.randint(
                    1, max(1, level)
                )

            # -- npc assignment --
            non_unique_npcs = [
                npc
                for npc in self.npcs.values()
                if not npc.unique and not getattr(npc, "quest", False)
            ]
            npc_id_to_count = self.auxiliary.get("npc_id_to_count", {})
            enemy_budget = rng.randint(max(1, level // 2), level + 1)

            for npc_proto in non_unique_npcs:
                if enemy_budget <= 0:
                    break
                if rng.random() < 0.4:
                    continue
                npc_level = rng.choice(
                    [max(1, level - 1), level, min(level + 1, 10)]
                )
                count = npc_id_to_count.get(npc_proto.id, 0)
                new_npc = npc_proto.create_instance(
                    count, npc_level, self.objects, rng
                )
                npc_id_to_count[npc_proto.id] = count + 1
                self.npc_instances[new_npc.id] = new_npc
                self.auxiliary.setdefault("npc_name_to_id", {})[
                    new_npc.name
                ] = new_npc.id
                area.npcs.append(new_npc.id)
                if new_npc.enemy:
                    enemy_budget -= 1

            self.auxiliary["npc_id_to_count"] = npc_id_to_count

        logger.info(
            f"expansion: populated {len(area_ids)} new areas "
            "with objects and NPCs"
        )


if __name__ == "__main__":
    world_definition = json.load(open("assets/world_definitions/generated/quarantine/default.json"))
    world = World.generate(world_definition, seed=random.randint(0, 10000))
