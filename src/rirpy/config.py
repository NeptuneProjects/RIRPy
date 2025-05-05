# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass, field
import tomllib
from typing import Any

# import numpy as np
import torch

from rirpy.models_np import Models


@dataclass
class SimulationConfig:
    """Configuration interface impulse response computation."""

    # Model type
    models: list[str] = field(default_factory=lambda: ["freq"])

    # Source and receiver positions
    r_source: list[float] = field(default_factory=lambda: [2.5, 2.5, 1.0])
    r_receiver: list[float] = field(default_factory=lambda: [7.5, 7.5, 1.0])

    # Frequency range
    omega_start: float = 100.0
    omega_end: float = 1000.0
    omega_points: int = 1000

    # Tank dimensions
    Lx: float = 10.0
    Ly: float = 10.0
    Lz: float = 2.0

    # Physical parameters
    sound_speed: float = 1500.0
    beta_wall: float = 0.9
    beta_surface: float = 0.8
    cutoff_time: float = 0.1

    # Computation parameters
    use_gpu: bool = True
    batch_size: int = 1000
    output_file: str = "greens_function_results.pt"

    @property
    def omega(self) -> torch.Tensor:
        """Generate the frequency array based on config parameters."""
        return torch.linspace(self.omega_start, self.omega_end, self.omega_points)

    @property
    def device(self) -> torch.device:
        """Determine the appropriate device."""
        if self.use_gpu and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @classmethod
    def from_argparse(cls) -> "SimulationConfig":
        """Create a configuration from command-line arguments."""
        parser = argparse.ArgumentParser(description="Impulse response modeling.")

        # Model type
        parser.add_argument(
            "--models",
            type=str,
            nargs="+",
            default=["freq"],
            help="List of model types to use (e.g., freq, time)",
        )

        # Source and receiver positions
        parser.add_argument(
            "--source",
            type=float,
            nargs=3,
            default=[0.5, 0.5, 0.5],
            help="Source position [x, y, z] in meters",
        )
        parser.add_argument(
            "--receiver",
            type=float,
            nargs=3,
            default=[1.5, 1.0, 0.8],
            help="Receiver position [x, y, z] in meters",
        )

        # Frequency range
        parser.add_argument(
            "--omega-start",
            type=float,
            default=100.0,
            help="Starting angular frequency (rad/s)",
        )
        parser.add_argument(
            "--omega-end",
            type=float,
            default=1000.0,
            help="Ending angular frequency (rad/s)",
        )
        parser.add_argument(
            "--omega-points", type=int, default=100, help="Number of frequency points"
        )

        # Tank dimensions
        parser.add_argument(
            "--Lx", type=float, default=3.0, help="Tank length in x-direction (m)"
        )
        parser.add_argument(
            "--Ly", type=float, default=2.0, help="Tank length in y-direction (m)"
        )
        parser.add_argument(
            "--Lz", type=float, default=1.5, help="Tank length in z-direction (m)"
        )

        # Physical parameters
        parser.add_argument(
            "--sound-speed",
            type=float,
            default=1500.0,
            help="Sound speed in the medium (m/s)",
        )
        parser.add_argument(
            "--beta-wall",
            type=float,
            default=0.9,
            help="Reflection coefficient for the walls",
        )
        parser.add_argument(
            "--beta-surface",
            type=float,
            default=0.8,
            help="Reflection coefficient for the surface",
        )
        parser.add_argument(
            "--cutoff-time",
            type=float,
            default=0.01,
            help="Time cutoff for reflected paths (s)",
        )

        # Computation parameters
        parser.add_argument(
            "--use-gpu", action="store_true", help="Use GPU acceleration if available"
        )
        parser.add_argument(
            "--no-gpu",
            dest="use_gpu",
            action="store_false",
            help="Force CPU computation",
        )
        parser.add_argument(
            "--batch-size", type=int, default=1000, help="Batch size for computation"
        )
        parser.add_argument(
            "--output-file",
            type=str,
            default="greens_function_results.pt",
            help="Output file for results",
        )

        # Config file
        parser.add_argument(
            "--config", type=str, help="Path to TOML configuration file"
        )

        args = parser.parse_args()

        # If config file is specified, load from there
        if args.config:
            config = cls.from_toml(args.config)
            # Override with any command-line args that were explicitly provided
            arg_dict = {
                k: v
                for k, v in vars(args).items()
                if k != "config" and v is not parser.get_default(k)
            }
            for k, v in arg_dict.items():
                setattr(config, k, v)
            return config

        # Otherwise create from command-line args
        return cls(
            models=args.models,
            r_source=args.source,
            r_receiver=args.receiver,
            omega_start=args.omega_start,
            omega_end=args.omega_end,
            omega_points=args.omega_points,
            Lx=args.Lx,
            Ly=args.Ly,
            Lz=args.Lz,
            sound_speed=args.sound_speed,
            beta_wall=args.beta_wall,
            beta_surface=args.beta_surface,
            cutoff_time=args.cutoff_time,
            use_gpu=args.use_gpu,
            batch_size=args.batch_size,
            output_file=args.output_file,
        )

    @classmethod
    def from_toml(cls, config_path: str) -> "SimulationConfig":
        """Create a configuration from a TOML file using Python's built-in tomllib."""
        try:
            with open(config_path, "rb") as f:
                config_data = tomllib.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except Exception as e:
            raise ValueError(f"Error parsing TOML file: {e}")

        # Extract configuration sections
        sim_params = config_data.get("simulation", { })
        room_params = config_data.get("room", {})
        compute_params = config_data.get("computation", {})

        # Create configuration object
        return cls(
            models=sim_params.get("models", ["freq"]),
            r_source=sim_params.get("source", [2.0, 2.0, 1.0]),
            r_receiver=sim_params.get("receiver", [5.0, 5.0, 1.0]),
            omega_start=sim_params.get("omega_start", 100.0),
            omega_end=sim_params.get("omega_end", 1000.0),
            omega_points=sim_params.get("omega_points", 100),
            Lx=room_params.get("Lx", 10.0),
            Ly=room_params.get("Ly", 10.0),
            Lz=room_params.get("Lz", 2.0),
            sound_speed=sim_params.get("sound_speed", 1500.0),
            beta_wall=room_params.get("beta_wall", 0.9),
            beta_surface=room_params.get("beta_surface", 0.8),
            cutoff_time=sim_params.get("cutoff_time", 0.01),
            use_gpu=compute_params.get("use_gpu", True),
            batch_size=compute_params.get("batch_size", 1000),
            output_file=compute_params.get("output_file", "greens_function_results.pt"),
        )

    def validate(self) -> None:
        """Validate configuration parameters."""
        # Ensure model types are valid
        valid_models = {model.value for model in Models}
        for model in self.models:
            if model not in valid_models:
                raise ValueError(
                    f"Invalid model type: {model}. Valid options are: {valid_models}"
                )

        # Ensure positions are valid
        if len(self.r_source) != 3 or len(self.r_receiver) != 3:
            raise ValueError("Source and receiver positions must be 3D vectors")

        # Ensure frequencies are valid
        if self.omega_start <= 0 or self.omega_end <= 0:
            raise ValueError("Frequencies must be positive")
        if self.omega_start >= self.omega_end:
            raise ValueError("Start frequency must be less than end frequency")
        if self.omega_points <= 0:
            raise ValueError("Number of frequency points must be positive")

        # Ensure room dimensions are valid
        if self.Lx <= 0 or self.Ly <= 0 or self.Lz <= 0:
            raise ValueError("Tank dimensions must be positive")

        # Ensure physical parameters are valid
        if self.sound_speed <= 0:
            raise ValueError("Sound speed must be positive")
        if not 0 <= self.beta_wall <= 1:
            raise ValueError("Wall reflection coefficient must be between 0 and 1")
        if not 0 <= self.beta_surface <= 1:
            raise ValueError("Surface reflection coefficient must be between 0 and 1")
        if self.cutoff_time <= 0:
            raise ValueError("Cutoff time must be positive")

        # Ensure batch size is valid
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        # Check if GPU is requested but not available
        if self.use_gpu and not torch.backends.mps.is_available():
            print("Warning: GPU requested but MPS not available. Falling back to CPU.")
            self.use_gpu = False

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "simulation": {
                "models": self.models,
                "source": self.r_source,
                "receiver": self.r_receiver,
                "omega_start": self.omega_start,
                "omega_end": self.omega_end,
                "omega_points": self.omega_points,
                "sound_speed": self.sound_speed,
                "cutoff_time": self.cutoff_time,
            },
            "room": {
                "Lx": self.Lx,
                "Ly": self.Ly,
                "Lz": self.Lz,
                "beta_wall": self.beta_wall,
                "beta_surface": self.beta_surface,
            },
            "computation": {
                "use_gpu": self.use_gpu,
                "batch_size": self.batch_size,
                "output_file": self.output_file,
            },
        }
