#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import timedelta
import logging
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from rirpy.config import SimulationConfig
import rirpy.models as models

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    models.Models.FREQUENCY_DOMAIN: models.impulse_response_freq_domain,
}


def model_factory(model_name: str) -> callable:
    model_type = models.Models(model_name)
    if not model_type in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not registered or supported.")
    return MODEL_REGISTRY[model_type]


def main():
    """Main entry point for the rirtorch command-line interface."""
    print("-" * 60)
    print("🔊 RIRTorch: Room Impulse Response Modeling with Python")
    print("-" * 60)

    # Load configuration (either from command line or TOML file)
    config = SimulationConfig.from_argparse()

    # Validate configuration
    config.validate()

    # Print configuration summary
    logging.debug(config)

    # Verify device availability
    # if config.device.type == "mps":
    #     logging.info("Using Apple Silicon GPU acceleration (MPS)")
    # else:
    #     logging.info("Using CPU for computation")

    # Prepare tensors for computation
    # r_source = torch.tensor(config.r_source, dtype=torch.float32, device=config.device)
    r_source = np.array(config.r_source, dtype=np.float64)
    # r_receiver = torch.tensor(
    #     config.r_receiver, dtype=torch.float32, device=config.device
    # )
    r_receiver = np.array(config.r_receiver, dtype=np.float64)
    omega = config.omega

    logging.info("Starting Green's function computation...")
    logging.info(f"Calculating for {len(omega)} frequency points")
    # Time the computation
    start_time = time.time()

    # Run the computation with the optimized function
    g_tank = models.impulse_response_freq_domain(
        r_source=r_source,
        r_receiver=r_receiver,
        omega=omega,
        Lx=config.Lx,
        Ly=config.Ly,
        Lz=config.Lz,
        c=config.sound_speed,
        beta_wall=config.beta_wall,
        beta_surface=config.beta_surface,
        cutoff_time=config.cutoff_time,
        batch_size=config.batch_size,
        # device=config.device,
    )

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed_time)))

    logging.info(f"Computation completed in {elapsed_str} ({elapsed_time:.2f} seconds)")
    logging.info(f"Green's function shape: {g_tank.shape}")

    # Create output directory if needed
    output_path = Path(config.output_file)
    output_dir = output_path.parent
    if output_dir != Path(".") and not output_dir.exists():
        logging.info(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    logging.info(f"Saving results to {config.output_file}...")

    # Move tensor to CPU for saving
    # g_tank_cpu = g_tank.cpu()

    # # Save both the Green's function and the configuration parameters
    # torch.save(
    #     {
    #         "g_tank": g_tank_cpu,
    #         "omega": omega.cpu(),
    #         "config": config.to_dict(),
    #         "computation_time": elapsed_time,
    #     },
    #     config.output_file,
    # )

    logging.info(f"Results successfully saved to {config.output_file}")


if __name__ == "__main__":
    main()
