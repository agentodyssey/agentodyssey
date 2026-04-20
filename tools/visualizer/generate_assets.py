import argparse
import json
import os
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

import torch
from diffusers import ZImagePipeline

@dataclass
class EntityInfo:
    id: str
    name: str
    entity_type: str
    description: str = ""
    category: str = ""
    extra_context: str = ""


class AssetGenerator:
    def __init__(
        self,
        output_dir: str,
        icon_size: int = 128,
        use_diffusion: bool = True,
        model_name: str = "Tongyi-MAI/Z-Image-Turbo",
        seed: int = 42
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.icon_size = icon_size
        self.seed = seed
        self.use_diffusion = use_diffusion
        self.pipe = None
        
        if self.use_diffusion:
            self._init_pipeline(model_name)
    
    def _init_pipeline(self, model_name: str):
        print(f"Loading diffusion model: {model_name}")
        try:
            self.pipe = ZImagePipeline.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            if torch.cuda.is_available():
                self.pipe.to("cuda")
            print("Diffusion model loaded successfully!")
        except Exception as e:
            print(f"Failed to load diffusion model: {e}")
            self.use_diffusion = False
            self.pipe = None
    
    def _normalize_id(self, entity_id: str) -> str:
        return re.sub(r'_\d+$', '', entity_id)
    
    def _stable_hash(self, *parts: str) -> int:
        h = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
        return int(h[:8], 16)
    
    def _build_prompt(self, entity: EntityInfo) -> str:
        name = entity.name.replace("_", " ")
        # For areas, include the place name for better context
        area_label = f"{entity.extra_context}, {name}" if entity.extra_context else name

        base_prompts = {
            'area': f"fantasy RPG game scene of {area_label}, centered composition suitable for circular crop, atmospheric lighting, detailed environment, game location, no characters, immersive scenery, square format",
            'place': f"fantasy RPG game establishing shot of {name}, centered composition suitable for circular crop, atmospheric, detailed architecture and environment, square format",
            'object': f"pixel art game item icon of a {name}, fantasy RPG inventory style, clean background, 64x64 icon",
            'npc': f"pixel art game character portrait of a {name}, fantasy RPG style, detailed face, 64x64 icon",
        }
        
        base = base_prompts.get(entity.entity_type, f"pixel art icon of {name}, 64x64")
        
        if entity.category:
            category_hints = {
                'weapon': ", weapon, combat item",
                'armor': ", protective gear, defense item",
                'tool': ", utility item, equipment",
                'food': ", consumable, healing item",
                'material': ", crafting material, resource",
                'container': ", storage item, bag",
                'currency': ", gold coin, money",
                'station': ", crafting station, workbench",
            }
            base += category_hints.get(entity.category, "")
        
        if entity.description:
            desc_short = entity.description[:100]
            base += f", {desc_short}"
        
        return base
    
    def _generate_placeholder(self, entity: EntityInfo, output_path: Path) -> bool:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            print("PIL not available for placeholder generation")
            return False
        
        size = self.icon_size
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        seed = self._stable_hash(entity.id, entity.entity_type)
        
        type_colors = {
            'area': (70, 130, 180),
            'place': (100, 149, 237),
            'object': (144, 238, 144),
            'npc': (255, 182, 193),
        }
        
        base_color = type_colors.get(entity.entity_type, (128, 128, 128))
        
        r = min(255, max(0, base_color[0] + (seed % 60) - 30))
        g = min(255, max(0, base_color[1] + ((seed // 7) % 60) - 30))
        b = min(255, max(0, base_color[2] + ((seed // 13) % 60) - 30))
        
        margin = 4
        draw.rounded_rectangle(
            [margin, margin, size - margin, size - margin],
            radius=12,
            fill=(r, g, b, 255),
            outline=(min(255, r + 40), min(255, g + 40), min(255, b + 40)),
            width=2
        )
        
        type_symbols = {
            'area': '🏞️',
            'place': '🏰',
            'object': '📦',
            'npc': '👤',
        }
        symbol = type_symbols.get(entity.entity_type, '?')
        
        glyph = (entity.name[:1] or '?').upper()
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size // 3)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size // 6)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        bbox = draw.textbbox((0, 0), glyph, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size - text_width) // 2
        y = (size - text_height) // 2 - 8
        
        draw.text((x, y), glyph, fill=(30, 30, 30), font=font)
        
        label = entity.entity_type[:6]
        bbox = draw.textbbox((0, 0), label, font=small_font)
        label_width = bbox[2] - bbox[0]
        draw.text(
            ((size - label_width) // 2, size - 24),
            label,
            fill=(30, 30, 30),
            font=small_font
        )
        
        img.save(output_path)
        return True

    def _generate_diffusion_icon(self, entity: EntityInfo, output_path: Path) -> bool:
        if not self.pipe:
            return False

        prompt = self._build_prompt(entity)

        try:
            generator = torch.Generator("cuda").manual_seed(self.seed + self._stable_hash(entity.id))

            # Use square dimensions for areas/places (better for circular cropping)
            if entity.entity_type in ('area', 'place'):
                gen_height = 768
                gen_width = 768
                output_size = (512, 512)
            else:
                gen_height = 512
                gen_width = 512
                output_size = (self.icon_size, self.icon_size)

            image = self.pipe(
                prompt=prompt,
                height=gen_height,
                width=gen_width,
                num_inference_steps=9,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]

            image = image.resize(output_size, Image.LANCZOS)
            image.save(output_path)
            return True

        except Exception as e:
            print(f"Diffusion generation failed for {entity.id}: {e}")
            return False
    
    def generate_icon(self, entity: EntityInfo, force: bool = False) -> Path:
        normalized_id = self._normalize_id(entity.id)
        output_path = self.output_dir / f"{normalized_id}.png"

        if output_path.exists() and not force:
            print(f"  Icon exists: {normalized_id}")
            return output_path
        
        print(f"  Generating icon: {normalized_id} ({entity.entity_type})")
        
        success = False
        if self.use_diffusion:
            success = self._generate_diffusion_icon(entity, output_path)
        
        if not success:
            success = self._generate_placeholder(entity, output_path)
        
        if success:
            print(f"    ✓ Saved to {output_path}")
        else:
            print(f"    ✗ Failed to generate icon")
        
        return output_path if success else None
    
    def generate_from_world_definition(self, world_definition: Dict[str, Any], force: bool = False, custom_events: List[str] = None) -> Dict[str, Path]:
        generated = {}
        entities = world_definition.get("entities", {})

        if custom_events is None:
            custom_events = world_definition.get("custom_events", [])

        print("\nGenerating area icons...")
        for place in entities.get("places", []):
            place_name = place["name"]
            for area in place.get("areas", []):
                area_entity = EntityInfo(
                    id=area["id"],
                    name=area["name"],
                    entity_type="area",
                    extra_context=place_name,
                )
                path = self.generate_icon(area_entity, force)
                if path:
                    generated[area["id"]] = path

        if 'tutorial' in custom_events:
            tutorial_entity = EntityInfo(
                id="area_tutorial_room",
                name="Tutorial Room",
                entity_type="area",
                description="A tutorial room for new players to learn the basics of the game",
                extra_context="Tutorial",
            )
            path = self.generate_icon(tutorial_entity, force)
            if path:
                generated["area_tutorial_room"] = path

        return generated


def load_world_definition(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate visual assets for the Agent Odyssey HTML visualizer"
    )
    parser.add_argument(
        "--world_definition_path", 
        type=str, 
        default=None,
    )
    parser.add_argument(
        "--game_name",
        type=str,
        default="remnant",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="assets/visualizer_icons",
    )
    parser.add_argument(
        "--icon_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Tongyi-MAI/Z-Image-Turbo",
    )
    parser.add_argument(
        "--no_diffusion",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.world_definition_path:
        world_def_path = args.world_definition_path
    else:
        world_def_path = f"assets/world_definitions/generated/{args.game_name}/default.json"

    if not os.path.exists(world_def_path):
        print(f"Error: World definition not found at {world_def_path}")
        return 1

    print(f"Loading world definition from: {world_def_path}")
    world_definition = load_world_definition(world_def_path)

    print(f"Output directory: {args.output_dir}")
    print(f"Icon size: {args.icon_size}x{args.icon_size}")
    print(f"Using diffusion: {not args.no_diffusion}")

    generator = AssetGenerator(
        output_dir=args.output_dir,
        icon_size=args.icon_size,
        use_diffusion=not args.no_diffusion,
        model_name=args.model,
        seed=args.seed,
    )

    generated = generator.generate_from_world_definition(world_definition, force=args.force)

    print(f"\nGenerated {len(generated)} icons")
    print(f"Icons saved to: {args.output_dir}")
    
    manifest_path = Path(args.output_dir) / "manifest.json"
    manifest = {
        "source": world_def_path,
        "icon_size": args.icon_size,
        "icons": {k: str(v) for k, v in generated.items()}
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to: {manifest_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
