import random
import re
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple

import gem
import gymnasium as gym
import numpy as np
from gem import Env
from minihack import LevelGenerator, MiniHackNavigation, MiniHackSkill
from minihack.level_generator import TRAP_NAMES
from minihack.tiles import GlyphMapper

from roll.pipeline.agentic.env.minihack.utils import (
    collect_symbol_descriptions_from_obs,
    get_common_monsters,
    get_common_objects,
    get_common_traps,
    get_terrain_description,
    get_wall_description,
    minihack_terrains,
    minihack_walls,
    render_glyphs,
)
from roll.pipeline.agentic.env.parse_action_utils import default_parser_action_func

VALID_TRAPS = set(TRAP_NAMES)


@contextmanager
def _temporary_seed(seed: Optional[int]):
    if seed is None:
        yield
        return

    random_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_state)


class MiniHackEnv(Env):
    """
    MiniHack environment wrapper.

    Supports both navigation and skill-based MiniHack environments.
    """

    def __init__(
        self,
        env_type: str = "navigation",
        env_name: Optional[str] = None,
        des_file: Optional[str] = None,
        max_steps: int = 100,
        observation_keys: Tuple[str] = (
            "glyphs",
            "blstats",
            "message",
            "inv_strs",
        ),
        render_mode: str = "text",
        action_pattern: str = "<answer>(.*?)</answer>",
        special_token_list: Tuple[str] = (
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "<answer>",
            "</think>",
            "</answer>",
        ),
        format_penalty: float = 0.0,
        use_skill_env: bool = False,
        env_instruction: Optional[str] = None,
        crop_to_bounds: bool = False,
        # Random DES generation parameters
        random_des: bool = False,
        map_width: int = 9,
        map_height: int = 9,
        random_seed: Optional[int] = None,
        # Terrain and wall configuration
        wall_type: str = "|",
        obstacle_density: float = 0.15,
        obstacle_terrain_types: Optional[List[str]] = None,
        obstacle_terrain_weights: Optional[List[float]] = None,
        # Monster, object, and trap configuration
        enable_monsters: bool = False,
        monster_density: float = 0.1,
        monster_types: Optional[List[str]] = None,
        enable_objects: bool = False,
        object_density: float = 0.05,
        object_types: Optional[List[str]] = None,
        enable_traps: bool = False,
        trap_density: float = 0.05,
        trap_types: Optional[List[str]] = None,
        lit: bool = False,
        premapped: bool = True,
        **kwargs,
    ):
        """
        Initialize MiniHack environment.

        Args:
            env_type: Type of environment ("navigation" or "skill")
            env_name: Predefined MiniHack environment name (e.g., "MiniHack-River-v0",
                "MiniHack-Corridor-v0"). If provided, this takes precedence over des_file
                and random_des. Use this to leverage MiniHack's built-in registered environments.
            des_file: MiniHack level description file content. If None and random_des=True,
                a random DES will be generated based on random_seed.
            max_steps: Maximum steps per episode
            observation_keys: Which observation keys to include
            render_mode: How to render observations ("text" or "rgb_array")
            action_pattern: Pattern for parsing actions from text
            special_token_list: Special tokens to handle in action parsing
            format_penalty: Penalty for invalid action format
            use_skill_env: Whether to use MiniHackSkill instead of MiniHackNavigation
                (ignored if env_name is provided)
            env_instruction: Custom environment instruction
            crop_to_bounds: Whether to manually crop observations (deprecated - MiniHack provides
                chars_crop and glyphs_crop natively, which are automatically included in observation_keys)
            random_des: If True and des_file is None, generate a random DES file
                (ignored if env_name is provided)
            map_width: Width of randomly generated map (used when random_des=True)
            map_height: Height of randomly generated map (used when random_des=True)
            obstacle_density: Density of obstacles (0.0-1.0) in random map (used when random_des=True)
            random_seed: Seed for random DES generation. If None, uses a default seed.
                Note: This seed is only used at initialization. For deterministic resets,
                use the seed parameter in reset() method.
            wall_type: Type of wall to use for outer walls ("|", "-", "#", " ")
            obstacle_terrain_types: List of terrain types to use as obstacles (e.g., ["{", "T", "L"]).
                If None, defaults to ["{"] (fountains).
            obstacle_terrain_weights: Weights for selecting obstacle terrain types (must match length of obstacle_terrain_types).
                If None, uniform weights are used.
            enable_monsters: Whether to add monsters to randomly generated levels
            monster_density: Density of monsters (0.0-1.0) relative to interior cells
            monster_types: List of monster names to use. If None, uses common monsters.
            enable_objects: Whether to add objects to randomly generated levels
            object_density: Density of objects (0.0-1.0) relative to interior cells
            object_types: List of object names to use. If None, uses common objects.
            enable_traps: Whether to add traps to randomly generated levels
            trap_density: Density of traps (0.0-1.0) relative to interior cells
            trap_types: List of trap names to use. If None, uses common traps.
            lit: Whether to light the generated level (default False)
            premapped: Whether to add the premapped flag so the full map is revealed
            **kwargs: Additional arguments passed to MiniHack environment
        """
        self.env_type = env_type
        self.env_name = env_name
        observation_keys = tuple(observation_keys)
        verbose = kwargs.pop("verbose", False)

        # Always include chars for rendering fallback
        if "chars" not in observation_keys:
            observation_keys = observation_keys + ("chars",)

        # Automatically add crop versions if not present - MiniHack provides these natively
        if "chars_crop" not in observation_keys:
            observation_keys = observation_keys + ("chars_crop",)
        if "glyphs_crop" not in observation_keys:
            observation_keys = observation_keys + ("glyphs_crop",)

        self.render_mode = render_mode
        self.observation_keys = observation_keys
        self.format_penalty = format_penalty
        self.action_pattern = action_pattern
        self.special_token_list = special_token_list
        self.max_steps = max_steps
        self.crop_to_bounds = crop_to_bounds
        self._turn_count = 0
        self._last_obs = None
        self._last_info: Dict = {}
        self._last_text_obs: str = ""
        self._glyph_mapper: Optional[GlyphMapper] = None

        # Store random DES generation parameters
        self.random_des = random_des or (des_file is None and env_name is None)
        self.map_width = map_width
        self.map_height = map_height
        self.obstacle_density = obstacle_density
        self._init_random_seed = random_seed

        # Store terrain and wall configuration
        self.wall_type = wall_type if wall_type in minihack_walls() else "|"
        self.obstacle_terrain_types = obstacle_terrain_types or ["{"]
        self.obstacle_terrain_weights = obstacle_terrain_weights
        if self.obstacle_terrain_weights is None:
            self.obstacle_terrain_weights = [1.0 / len(self.obstacle_terrain_types)] * len(self.obstacle_terrain_types)
        elif len(self.obstacle_terrain_weights) != len(self.obstacle_terrain_types):
            # Normalize weights if lengths don't match
            self.obstacle_terrain_weights = [1.0 / len(self.obstacle_terrain_types)] * len(self.obstacle_terrain_types)

        # Store monster, object, and trap configuration
        self.enable_monsters = enable_monsters
        self.monster_density = monster_density
        self.monster_types = monster_types or get_common_monsters()
        self.enable_objects = enable_objects
        self.object_density = object_density
        self.object_types = object_types or get_common_objects()
        self.enable_traps = enable_traps
        self.trap_density = trap_density
        self.trap_types = trap_types or get_common_traps()
        self.lit = lit
        self.premapped = premapped

        # Create MiniHack environment
        if env_name is not None:
            # Prepare kwargs for gym.make
            gym_kwargs = dict(kwargs)
            gym_kwargs["max_episode_steps"] = max_steps
            gym_kwargs["observation_keys"] = self.observation_keys

            # Remove des_file from kwargs if present (not applicable for registered envs)
            gym_kwargs.pop("des_file", None)
            gym_kwargs.pop("enable_thinking", None)

            self.minihack_env = gym.make(env_name, **gym_kwargs)

            # Unwrap TimeLimit wrapper to access the underlying MiniHack environment
            # This is needed to access the 'actions' attribute
            if hasattr(self.minihack_env, "unwrapped"):
                self._base_env = self.minihack_env.unwrapped
            elif hasattr(self.minihack_env, "env"):
                # Try to access wrapped environment
                self._base_env = self.minihack_env.env
                # Keep unwrapping if needed
                while hasattr(self._base_env, "env") and not hasattr(self._base_env, "actions"):
                    self._base_env = self._base_env.env
            else:
                self._base_env = self.minihack_env

            # Flag to skip random DES generation on reset
            self.random_des = False
        else:
            # Generate random DES if requested and des_file is None
            if self.random_des:
                gen_seed = random_seed if random_seed is not None else 0
                des_file = self._generate_random_des(gen_seed)

            # Create custom environment using base classes
            env_class = MiniHackSkill if use_skill_env else MiniHackNavigation
            env_kwargs = dict(kwargs)
            # Remove explicit None des_file coming from configuration
            if env_kwargs.get("des_file") is None:
                env_kwargs.pop("des_file", None)
            if des_file is not None:
                env_kwargs["des_file"] = self._normalize_des_file(des_file)
                # env_kwargs["des_file"] = des_file

            if verbose:
                print(f"DES File: {env_kwargs['des_file']}")
            self.minihack_env = env_class(
                max_episode_steps=max_steps,
                observation_keys=self.observation_keys,
                **env_kwargs,
            )
            # For custom environments, the base env is the same as minihack_env
            self._base_env = self.minihack_env

        # Copy environment properties
        self.action_space = self.minihack_env.action_space
        self.observation_space = self.minihack_env.observation_space
        self.spec = getattr(self.minihack_env, "spec", None)

        # Build action lookup (primary names and aliases)
        (
            self.ACTION_LOOKUP,
            self._action_alias_to_index,
        ) = self._build_action_lookup()

        # Default environment instruction
        self.env_instruction = env_instruction or self._default_instruction()

    def get_instructions(self) -> str:
        symbol_vocab: Dict[str, str] = {}

        # for terrain in minihack_terrains():
        #     symbol_vocab[terrain] = get_terrain_description(terrain)
        # for wall in minihack_walls():
        #     symbol_vocab.setdefault(wall, get_wall_description(wall))

        # grid_vocab_str = "\nThe meaning of common symbols in the dungeon view is:\n" + ", ".join(
        #     [f"{k}: {v}" for k, v in symbol_vocab.items()]
        # )
        action_lookup_str = "\nYour available actions are:\n" + ", ".join(
            [f"{action}" for i, action in self.ACTION_LOOKUP.items()]
        )
        # return self.env_instruction + grid_vocab_str + action_lookup_str
        return self.env_instruction + action_lookup_str

    def reset(self, seed: Optional[int] = None):
        """Reset the environment and return initial observation."""
        Env.reset(self, seed)

        if self.random_des:
            reset_seed = seed if seed is not None else 0
            new_des = self._generate_random_des(reset_seed)
            self.minihack_env.update(new_des)

        with _temporary_seed(seed):
            if seed is not None:
                # For gymnasium wrapped environments, seed both the wrapper and base env
                # First try seeding the wrapped environment
                seed_fn = getattr(self.minihack_env, "seed", None)
                if callable(seed_fn):
                    seed_fn(seed)

                # Also seed the base environment directly for predefined envs
                # This ensures deterministic behavior through wrapper layers
                if hasattr(self, "_base_env") and self._base_env is not self.minihack_env:
                    base_seed_fn = getattr(self._base_env, "seed", None)
                    if callable(base_seed_fn):
                        base_seed_fn(seed)

                    # For NLE-based environments, also seed the underlying NetHack game
                    # This is critical for deterministic episode generation
                    if hasattr(self._base_env, "_seed"):
                        self._base_env._seed = seed
                    if hasattr(self._base_env, "env") and hasattr(self._base_env.env, "seed"):
                        self._base_env.env.seed(seed)

            obs, info = self.minihack_env.reset(seed=seed)

        self._turn_count = 0
        info = dict(info or {})
        info["raw_obs"] = obs
        text_obs = self._update_cached_observation(obs, info)
        info["env_instruction"] = self.get_instructions()

        return text_obs, info

    def step(self, action: str):
        """Execute one step in the environment."""
        metrics_agg_mode = {
            "action_is_effective": "mean",
            "action_is_valid": "mean",
            "success": "last",
            "format_penalty": "mean",
        }

        # Parse action from text
        action_info = self.parse_action(action)

        current_turn = self._turn_count

        if action_info["action"] is None:
            # Invalid action format - return cached observation with penalty
            env_feedback = f"At turn {current_turn}, you did not provide a valid action."
            text_obs = self._last_text_obs
            text_obs = text_obs + env_feedback
            reward = self.format_penalty
            terminated = False
            truncated = False
            info = {}

            metrics = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
                "format_penalty": self.format_penalty,
            }
            info.update(action_info)
            info.update(
                {
                    "metrics": metrics,
                    "metrics_agg_mode": metrics_agg_mode,
                }
            )
            return text_obs, reward, terminated, truncated, info

        # Execute valid action
        action_id = action_info["action"]
        prev_obs = self._last_obs or {}
        prev_glyphs = prev_obs["glyphs"].copy() if "glyphs" in prev_obs else None

        obs, reward, terminated, truncated, info = self.minihack_env.step(action_id)
        info = dict(info or {})
        info["raw_obs"] = obs
        text_obs = self._update_cached_observation(obs, info)

        action_is_valid = action_is_effective = action_id in self.ACTION_LOOKUP.keys()
        if not action_is_valid:
            action_desc = f"At turn {current_turn}, you tried to perform {action_info['action_content']}, which is not a valid action."
        else:
            action_desc = ""

        action_lookup_str = "\nYour available actions are:\n" + ", ".join(
            [f"{action}" for i, action in self.ACTION_LOOKUP.items()]
        )

        text_obs = text_obs + action_desc + action_lookup_str

        # Check success condition (reached goal or positive reward)
        success = reward > 0

        metrics = {
            "action_is_effective": action_is_effective,
            "action_is_valid": action_is_valid,
            "success": success,
            "format_penalty": 0,
        }
        info.update(
            {
                "metrics": metrics,
                "metrics_agg_mode": metrics_agg_mode,
            }
        )
        info.update(action_info)

        return text_obs, reward, terminated, truncated, info

    def parse_action(self, text: str) -> Dict:
        """Parse action from text response."""
        action_info = default_parser_action_func(
            text,
            self.action_pattern,
            self.ACTION_LOOKUP,
            self.special_token_list,
        )
        raw_action = action_info.get("action")
        if isinstance(raw_action, str):
            stripped_action = raw_action.strip()
            if stripped_action.isdigit():
                numeric_action = int(stripped_action)
                if numeric_action in self.ACTION_LOOKUP:
                    action_info["action"] = numeric_action
                else:
                    action_info["action"] = None
        if action_info["action"] is None and action_info["action_content"]:
            candidate_raw = action_info["action_content"]
            candidate = candidate_raw.strip().lower()
            action_idx = self._action_alias_to_index.get(candidate)
            if action_idx is None:
                if candidate.isdigit():
                    numeric_candidate = int(candidate)
                    if numeric_candidate in self.ACTION_LOOKUP:
                        action_idx = numeric_candidate
                else:
                    digit_match = re.search(r"\d+", candidate)
                    if digit_match:
                        numeric_candidate = int(digit_match.group())
                        if numeric_candidate in self.ACTION_LOOKUP:
                            action_idx = numeric_candidate
            if action_idx is not None:
                action_info["action"] = action_idx
        return action_info

    def _render_observation(self, obs: Dict) -> str | np.ndarray:
        """Convert MiniHack observation to text format."""
        if self.render_mode == "text":
            return self._obs_to_text(obs)
        elif self.render_mode == "rgb_array":
            image: Optional[np.ndarray] = None

            image_fn = getattr(self.minihack_env, "get_image", None)
            if callable(image_fn):
                try:
                    image = image_fn(mode="rgb_array", tile_size=16)
                except (TypeError, AttributeError):
                    try:
                        image = image_fn()
                    except Exception:
                        pass

            # If get_image failed, try render method
            if not isinstance(image, np.ndarray):
                render_fn = getattr(self.minihack_env, "render", None)
                if callable(render_fn):
                    try:
                        # Try new gymnasium API (no arguments)
                        result = render_fn()
                        if isinstance(result, np.ndarray):
                            image = result
                    except (TypeError, AssertionError):
                        # Try old API with mode argument
                        try:
                            result = render_fn(mode="rgb_array")
                            if isinstance(result, np.ndarray):
                                image = result
                        except (TypeError, AssertionError):
                            pass

            # Fallback: render from glyphs using GlyphMapper
            if not isinstance(image, np.ndarray):
                grid = obs.get("glyphs")
                if grid is not None:
                    if self._glyph_mapper is None:
                        self._glyph_mapper = GlyphMapper()
                    try:
                        image = self._glyph_mapper.to_rgb(grid)
                    except Exception as exc:  # pragma: no cover - defensive
                        raise AttributeError(
                            "MiniHack environment does not support rgb_array rendering. "
                            "Glyph-based rendering also failed."
                        ) from exc

            if not isinstance(image, np.ndarray):
                raise AttributeError(
                    "MiniHack environment does not support rgb_array rendering. " "All rendering methods failed."
                )

            # Prefer crop versions from MiniHack if available
            if "chars_crop" in obs and obs["chars_crop"] is not None:
                # Already cropped by MiniHack, no need for additional cropping
                return image
            elif "glyphs_crop" in obs and obs["glyphs_crop"] is not None:
                # Already cropped by MiniHack, no need for additional cropping
                return image
            elif self.crop_to_bounds:
                # Manual cropping fallback only if crop versions unavailable
                grid = obs.get("chars") if "chars" in obs else obs.get("glyphs")
                if grid is not None:
                    _, bounds = render_glyphs(
                        grid,
                        crop_to_bounds=True,
                        return_bounds=True,
                    )
                    image = self._crop_rgb_array(image, grid, bounds)
            return image
        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")

    def _obs_to_text(self, obs: Dict) -> str:
        """Convert observation dict to text description."""
        text_parts = []

        # Message from the game
        if "message" in obs:
            message = self._decode_bytes(obs["message"]).strip()
            if message:
                text_parts.append(f"Message: {message}")

        # Player stats and location
        if "blstats" in obs:
            blstats = obs["blstats"]
            # Basic stats: [0: xl, 1: str, 2: dex, 3: con, 4: int, 5: wis, 6: cha]
            # Location: [20: x, 21: y]
            x, y = blstats[20], blstats[21]
            hp, hp_max = blstats[24], blstats[25]
            # Ensure hp_max is at least hp to avoid confusing displays like "1/0"
            if hp_max <= 0:
                hp_max = max(hp, 1)
            text_parts.append(f"Position: ({x}, {y})")
            text_parts.append(f"Health: {hp}/{hp_max}")

        # Render the dungeon view
        # Prefer crop versions (MiniHack provides these natively)
        # Only use manual cropping if crop versions aren't available
        dungeon_view = ""
        if "chars_crop" in obs and obs["chars_crop"] is not None:
            grid = obs["chars_crop"]
            # Use crop version directly without additional cropping
            dungeon_view = render_glyphs(grid, crop_to_bounds=False)
        elif "glyphs_crop" in obs and obs["glyphs_crop"] is not None:
            grid = obs["glyphs_crop"]
            # Use crop version directly without additional cropping
            dungeon_view = render_glyphs(grid, crop_to_bounds=False)
        elif "chars" in obs:
            grid = obs["chars"]
            # Fallback to manual cropping only if crop versions unavailable
            dungeon_view = render_glyphs(grid, crop_to_bounds=self.crop_to_bounds)
        elif "glyphs" in obs:
            grid = obs["glyphs"]
            # Fallback to manual cropping only if crop versions unavailable
            dungeon_view = render_glyphs(grid, crop_to_bounds=self.crop_to_bounds)

        if dungeon_view and dungeon_view.strip():
            text_parts.append("\nDungeon view:")
            text_parts.append(dungeon_view)

        if self._last_info is not None:
            raw_obs = self._last_info.get("raw_obs", None)
            symbolto_descriptions = collect_symbol_descriptions_from_obs(raw_obs)
            symbol_test = (
                "The meaning of common symbols in the dungeon view is (same symbol may have different descriptions):\n"
            )
            for key, value in symbolto_descriptions.items():
                symbol_test += f"{key}: {value}\n"
            text_parts.append(symbol_test)

        # Inventory
        if "inv_strs" in obs:
            inventory = self._decode_inventory(obs["inv_strs"])
            if inventory:
                text_parts.append(f"\nInventory:\n{inventory}")

        return "\n".join(text_parts)

    def _decode_bytes(self, data: np.ndarray) -> str:
        """Decode byte array to string."""
        try:
            return bytes(data).decode("utf-8", errors="ignore").rstrip("\x00")
        except (UnicodeDecodeError, AttributeError):
            return ""

    def _decode_inventory(self, inv_strs: np.ndarray) -> str:
        """Decode inventory strings."""
        items = []
        for inv_str in inv_strs:
            decoded = self._decode_bytes(inv_str).strip()
            if decoded:
                items.append(decoded)
        return "\n".join(items) if items else ""

    def sample_random_action(self) -> str:
        """Sample a random action name."""
        return random.choice(list(self.ACTION_LOOKUP.values()))

    def render(self, mode: Optional[str] = None):
        """Render the environment."""
        render_mode = mode if mode is not None else self.render_mode
        if render_mode == "rgb_array":
            # Try multiple methods to get RGB array
            render_fn = getattr(self.minihack_env, "render", None)
            if render_fn is not None:
                try:
                    # Try new gymnasium API (no arguments)
                    result = render_fn()
                    if isinstance(result, np.ndarray):
                        return self._maybe_crop_render_image(result)
                except (TypeError, AssertionError):
                    # TypeError: old API that needs mode argument
                    # AssertionError: unsupported render mode
                    pass

                try:
                    # Try old API with mode argument
                    result = render_fn(mode="rgb_array")
                    if isinstance(result, np.ndarray):
                        return self._maybe_crop_render_image(result)
                except (TypeError, AssertionError):
                    pass

            # Try get_image method
            image_fn = getattr(self.minihack_env, "get_image", None)
            if image_fn is not None:
                try:
                    result = image_fn(mode="rgb_array", tile_size=16)
                    if isinstance(result, np.ndarray):
                        return self._maybe_crop_render_image(result)
                except (TypeError, AttributeError):
                    pass

            # Fallback: render from glyphs using GlyphMapper
            if self._last_obs is not None and "glyphs" in self._last_obs:
                try:
                    if self._glyph_mapper is None:
                        self._glyph_mapper = GlyphMapper()
                    rgb = self._glyph_mapper.to_rgb(self._last_obs["glyphs"])
                    if isinstance(rgb, np.ndarray):
                        grid_override = (
                            self._last_obs["chars"] if "chars" in self._last_obs else self._last_obs["glyphs"]
                        )
                        return self._maybe_crop_render_image(rgb, grid_override=grid_override)
                except Exception:
                    pass

            # If all methods fail, raise error
            raise AttributeError(
                "MiniHack environment does not support rgb_array rendering. "
                "The environment may only support 'human', 'ansi', or 'full' modes. "
                "Falling back to glyph-based rendering failed."
            )
        elif render_mode == "human":
            return self.minihack_env.render()
        else:
            if self._last_obs is None:
                raise RuntimeError("MiniHackEnv.render called before reset has been performed.")
            return self._last_text_obs

    def close(self):
        """Close the environment."""
        if hasattr(self, "minihack_env"):
            self.minihack_env.close()

    def _update_cached_observation(self, obs: Dict, info: Optional[Dict]) -> str:
        """Cache the latest observation and return its text rendering."""
        self._last_obs = obs
        self._last_info = dict(info or {})
        self._last_text_obs = self._render_observation(obs)
        return self._last_text_obs

    def _maybe_crop_render_image(self, image: np.ndarray, grid_override: Optional[np.ndarray] = None) -> np.ndarray:
        """Crop rendered RGB array to active area if enabled."""
        # Prefer crop versions from MiniHack if available
        if self._last_obs is not None:
            if "chars_crop" in self._last_obs and self._last_obs["chars_crop"] is not None:
                # Already cropped by MiniHack, return as-is
                return image
            elif "glyphs_crop" in self._last_obs and self._last_obs["glyphs_crop"] is not None:
                # Already cropped by MiniHack, return as-is
                return image

        # Manual cropping fallback only if crop versions unavailable
        if not self.crop_to_bounds:
            return image

        grid = grid_override
        if grid is None and self._last_obs is not None:
            if "chars" in self._last_obs:
                grid = self._last_obs["chars"]
            else:
                grid = self._last_obs.get("glyphs")

        if grid is None:
            return image

        _, bounds = render_glyphs(grid, crop_to_bounds=True, return_bounds=True)
        return self._crop_rgb_array(image, grid, bounds)

    def _crop_rgb_array(
        self,
        image: np.ndarray,
        grid: np.ndarray,
        bounds: Optional[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Crop an RGB array using bounds computed from glyph/char grid."""
        if bounds is None:
            return image

        top, bottom, left, right = bounds
        grid_rows, grid_cols = grid.shape

        if top == 0 and bottom == grid_rows and left == 0 and right == grid_cols:
            return image

        if grid_rows == 0 or grid_cols == 0:
            return image

        height, width = image.shape[:2]
        tile_h = height // grid_rows if grid_rows else 0
        tile_w = width // grid_cols if grid_cols else 0

        if tile_h <= 0 or tile_w <= 0:
            return image

        return image[top * tile_h : bottom * tile_h, left * tile_w : right * tile_w]

    def _default_instruction(self) -> str:
        """Generate the default instruction string."""
        return (
            "You are playing NetHack, a roguelike game. "
            "Navigate through the dungeon, avoid monsters, and reach the goal. "
            "Think carefully, then respond with exactly one action per turn using "
            f"the format <think>[Your thoughts]</think><answer>[One of the actions in the action list]</answer>."
        )

    def _build_action_lookup(self) -> Tuple[Dict[int, str], Dict[str, int]]:
        """Build primary action names and alias mapping."""
        lookup: Dict[int, str] = {}
        alias_lookup: Dict[str, int] = {}
        used_names: Dict[str, int] = {}

        # Use _base_env to access actions (handles wrapped environments)
        env_for_actions = getattr(self, "_base_env", self.minihack_env)
        for idx, action in enumerate(env_for_actions.actions):
            primary, aliases = self._canonical_action_name(action)
            normalized_primary = primary.lower()
            if normalized_primary in used_names:
                # Ensure uniqueness by appending suffix.
                suffix = used_names[normalized_primary] + 2
                unique_primary = f"{primary}_{suffix}"
                used_names[normalized_primary] += 1
                primary = unique_primary
                normalized_primary = primary.lower()
            else:
                used_names[normalized_primary] = 1

            lookup[idx] = primary
            alias_lookup[normalized_primary] = idx
            for alias in aliases:
                alias_lookup.setdefault(alias.lower(), idx)

        return lookup, alias_lookup

    def _generate_random_des(self, seed: int) -> str:
        """
        Generate a random DES file deterministically based on seed.

        Args:
            seed: Random seed for deterministic generation

        Returns:
            DES file string
        """
        rng = random.Random(seed)
        np_rng_state = np.random.get_state()
        np.random.seed(seed)

        try:
            lg = LevelGenerator(
                w=self.map_width,
                h=self.map_height,
                lit=self.lit,
                flags=(("hardfloor", "premapped") if self.premapped else ("hardfloor",)),
            )

            # Add outer walls with configured wall type
            # Note: lg.map is indexed as [row, col] which is [y, x]
            # lg.y is height (number of rows), lg.x is width (number of cols)
            for y in range(lg.y):
                for x in range(lg.x):
                    if x == 0 or x == lg.x - 1 or y == 0 or y == lg.y - 1:
                        lg.map[y, x] = self.wall_type
                    else:
                        # Ensure interior tiles default to floor
                        lg.map[y, x] = "."

            # Calculate number of obstacles based on density
            interior_cells = [(x, y) for y in range(1, lg.y - 1) for x in range(1, lg.x - 1)]
            num_obstacles = int(len(interior_cells) * self.obstacle_density)

            # Shuffle and place obstacles with weighted terrain type selection
            rng.shuffle(interior_cells)
            obstacle_positions = interior_cells[:num_obstacles]

            # Set start and goal positions (avoid obstacles)
            start_pos = (1, 1)
            goal_pos = (lg.x - 2, lg.y - 2)

            # Remove start/goal from obstacle positions if present
            obstacle_positions = [pos for pos in obstacle_positions if pos != start_pos and pos != goal_pos]

            # Place obstacles with weighted terrain type selection
            for x, y in obstacle_positions:
                terrain_type = rng.choices(
                    self.obstacle_terrain_types,
                    weights=self.obstacle_terrain_weights,
                    k=1,
                )[0]
                lg.map[y, x] = terrain_type

            # Set start and goal
            lg.set_start_pos(start_pos)
            lg.add_stair_down(goal_pos)

            # Get base DES file
            des_lines = lg.get_des().split("\n")

            # Prepare lists for monsters, objects, and traps
            monster_lines = []
            object_lines = []
            trap_lines = []

            # Calculate positions for monsters, objects, and traps
            available_positions = [
                pos for pos in interior_cells if pos not in obstacle_positions and pos != start_pos and pos != goal_pos
            ]
            rng.shuffle(available_positions)

            # Add monsters if enabled
            if self.enable_monsters and available_positions:
                num_monsters = int(len(available_positions) * self.monster_density)
                monster_positions = available_positions[:num_monsters]
                available_positions = available_positions[num_monsters:]

                for x, y in monster_positions:
                    monster_name = rng.choice(self.monster_types)
                    monster_lines.append(f'MONSTER: "{monster_name}", ({x},{y}), hostile')

            # Add objects if enabled
            if self.enable_objects and available_positions:
                num_objects = int(len(available_positions) * self.object_density)
                object_positions = available_positions[:num_objects]
                available_positions = available_positions[num_objects:]

                for x, y in object_positions:
                    obj_entry = rng.choice(self.object_types)
                    if isinstance(obj_entry, tuple) and len(obj_entry) == 2:
                        symbol, name = obj_entry
                        object_lines.append(f"OBJECT:('{symbol}',\"{name}\"),({x},{y})")
                    else:
                        # Fallback for legacy string format
                        object_lines.append(f'OBJECT: "{obj_entry}", ({x},{y})')

            # Add traps if enabled
            if self.enable_traps and available_positions:
                num_traps = int(len(available_positions) * self.trap_density)
                trap_positions = available_positions[:num_traps]

                for x, y in trap_positions:
                    trap_name = rng.choice(self.trap_types)
                    if trap_name not in VALID_TRAPS:
                        continue
                    trap_lines.append(f'TRAP: "{trap_name}", ({x},{y})')

            # Combine all parts
            result_lines = des_lines.copy()

            # Insert monsters, objects, and traps after ENDMAP (outside MAP block)
            insert_idx = len(result_lines)
            for i, line in enumerate(result_lines):
                if line.strip().upper() == "ENDMAP":
                    insert_idx = i + 1
                    break

            # Insert new content after ENDMAP
            if monster_lines or object_lines or trap_lines:
                insertion_lines: List[str] = [""]

                # Insert monsters
                if monster_lines:
                    insertion_lines.extend(monster_lines)

                # Insert objects
                if object_lines:
                    insertion_lines.extend(object_lines)

                # Insert traps (already filtered to valid names)
                if trap_lines:
                    insertion_lines.extend(trap_lines)

                result_lines[insert_idx:insert_idx] = insertion_lines

            return "\n".join(result_lines)
        finally:
            np.random.set_state(np_rng_state)

    @staticmethod
    def _normalize_des_file(des_file: str) -> str:
        """
        Normalize DES file strings to avoid common formatting issues.

        Handles conversion of MAZE/LEVEL lines with flags= syntax to proper FLAGS: lines.
        Note: MAZE lines with a second quoted parameter (e.g., MAZE: "name", "hardfloor")
        are valid DES syntax and should NOT be modified.
        """
        if not isinstance(des_file, str):
            return des_file

        def _replace_flags(match: re.Match) -> str:
            prefix = match.group(1)
            flags_value = match.group(2).strip()
            return f"{prefix}, ' '\nFLAGS: {flags_value}"

        # Only match flags= syntax, not MAZE: "name", "hardfloor" syntax
        # The pattern specifically looks for "flags=" to avoid false matches
        pattern = re.compile(
            r'((?:MAZE|LEVEL)\s*:\s*"[^"]*?"),\s*flags\s*=\s*([^\n\r]+)',
            flags=re.IGNORECASE,
        )
        normalized = pattern.sub(_replace_flags, des_file)
        return normalized

    def _canonical_action_name(self, action: object) -> Tuple[str, Iterable[str]]:
        """Return canonical name and aliases for a given MiniHack action."""
        enum_name = type(action).__name__
        name = getattr(action, "name", str(action))
        aliases: List[str] = []

        if enum_name == "CompassDirection":
            mapping = {
                "N": "north",
                "E": "east",
                "S": "south",
                "W": "west",
                "NE": "northeast",
                "SE": "southeast",
                "SW": "southwest",
                "NW": "northwest",
            }
            canonical = mapping.get(name, name.lower())
            aliases.extend({name.lower(), name[0].lower()})
        elif enum_name == "CompassDirectionLonger":
            mapping = {
                "N": "north_run",
                "E": "east_run",
                "S": "south_run",
                "W": "west_run",
                "NE": "northeast_run",
                "SE": "southeast_run",
                "SW": "southwest_run",
                "NW": "northwest_run",
            }
            canonical = mapping.get(name, f"{name.lower()}_run")
            base = canonical.replace("_run", "")
            aliases.extend(
                {
                    canonical,
                    f"{base} run",
                    f"{base}_fast",
                    f"{base} fast",
                    name.lower(),
                }
            )
        elif enum_name == "MiscDirection":
            mapping = {
                "UP": "upstairs",
                "DOWN": "downstairs",
                "WAIT": "wait",
            }
            canonical = mapping.get(name, name.lower())
            if name == "WAIT":
                aliases.extend(["rest", "skip", "."])
            elif name == "UP":
                aliases.extend(["up", "<"])
            elif name == "DOWN":
                aliases.extend(["down", ">"])
        elif enum_name == "MiscAction":
            canonical = "more"
            aliases.extend(["continue", "next"])
        elif enum_name == "TextCharacters":
            char = chr(int(action))
            if char == " ":
                canonical = "space"
                aliases.extend([" ", "blank"])
            elif char == '"':
                canonical = "quote"
                aliases.extend(['"', "double_quote"])
            elif char == "'":
                canonical = "apostrophe"
                aliases.extend(["'", "apos"])
            elif char == "+":
                canonical = "+"
                aliases.extend(["plus", "spell_list"])
            elif char == "$":
                canonical = "$"
                aliases.extend(["dollar", "gold"])
            elif char == "-":
                canonical = "-"
                aliases.append("minus")
            else:
                canonical = char
        elif enum_name == "Command":
            canonical = self._command_to_canonical(name)
            aliases.extend(self._command_aliases(name))
        else:
            canonical = name.lower()

        aliases.append(canonical)
        return canonical, aliases

    @staticmethod
    def _command_to_canonical(name: str) -> str:
        """Convert NetHack command enum names to readable action names."""
        # Insert underscores for camel case transitions where possible.
        canonical = re.sub(r"(?<!^)(?=[A-Z][a-z])", "_", name).lower()
        # Handle fully uppercase sequences (e.g., TAKEOFFALL) manually.
        substitutions = {
            "takeoffall": "takeoff_all",
            "seeamulet": "see_amulet",
            "seegold": "see_gold",
            "seearmor": "see_armor",
            "seerings": "see_rings",
            "seespells": "see_spells",
            "seetools": "see_tools",
            "seetrap": "see_trap",
            "seeweapon": "see_weapon",
            "takeoff": "take_off",
            "pickup": "pick_up",
            "movefar": "move_far",
            "rush2": "rush_until_interesting",
            "rush": "rush_until_interesting",
            "whatdoes": "what_does",
            "whatis": "what_is",
            "extcmd": "extended_command",
            "extlist": "extended_command_list",
            "vershort": "version_short",
            "inventorytype": "inventory_type",
        }
        canonical = substitutions.get(canonical, canonical)
        if canonical == "esc":
            canonical = "escape"
        return canonical

    @staticmethod
    def _command_aliases(name: str) -> Iterable[str]:
        """Provide additional aliases for selected commands."""
        canonical = name.lower()
        alias_map = {
            "apply": ["use", "use_tool"],
            "quaff": ["drink"],
            "pray": ["prayer"],
            "engrave": ["write"],
            "inventory": ["show_inventory", "inv"],
            "inventorytype": ["inventory_type"],
            "takeoff": ["take_off"],
            "takeoffall": ["take_off_all"],
            "pickup": ["pick_up"],
            "whatdoes": ["what_does"],
            "whatis": ["what_is"],
            "teleport": ["teleport_self"],
            "extcmd": ["extended_command"],
            "extlist": ["extended_command_list"],
            "movefar": ["run_no_pickup"],
            "rush": ["run"],
            "rush2": ["run_fast"],
            "look": ["examine"],
            "fight": ["force_fight"],
            "more": ["continue"],
            "sit": ["sit_down"],
            "wear": ["equip_armor"],
            "wield": ["equip_weapon"],
            "zap": ["use_wand"],
            "quiver": ["select_quiver"],
            "throw": ["toss"],
            "dip": ["dip_item"],
            "drop": ["drop_item"],
            "droptype": ["drop_type"],
            "eat": ["consume_food"],
            "drink": ["quaff"],
            "kick": ["kick_something"],
            "call": ["name_object"],
            "read": ["read_item"],
            "open": ["open_door"],
            "close": ["close_door"],
            "pay": ["pay_bill"],
            "offer": ["offer_sacrifice"],
            "remove": ["remove_accessory"],
            "puton": ["put_on"],
            "rub": ["rub_item"],
            "search": ["search_area"],
            "zap": ["cast_wand"],
            "travel": ["auto_travel"],
            "jump": ["jump_move"],
            "esc": ["esc"],
        }
        return alias_map.get(canonical, [])
