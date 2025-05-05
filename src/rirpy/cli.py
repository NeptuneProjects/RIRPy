#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import asdict
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
    """Main entry point for the rirpy command-line interface."""
    print("-" * 60)
    print("🔊 RIRPy: Room Impulse Response Modeling with Python (Numba-accelerated)")
    print("-" * 60)

    # Load configuration (either from command line or TOML file)
    config = SimulationConfig.from_argparse()

    # Validate configuration
    config.validate()

    # Print configuration summary
    logging.debug(config)

    # Configure Numba
    # config.configure_numba()

    logging.info("Starting Green's function computation...")
    logging.info(f"Calculating for {config.omega_points} frequency points")

    # Time the computation
    start_time = time.time()

    # Process each requested model
    results = {}

    for model_name in config.models:
        logging.info(f"Running model: {model_name}")

        # Get the appropriate model function
        omega = config.omega
        model_func = model_factory(model_name)
        g_tank = model_func(omega=omega, **asdict(config))
        results[model_name] = g_tank

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logging.info(f"Computation completed in {elapsed_time:.2f} seconds.")
    for model_name, result in results.items():
        logging.info(f"[{model_name}] Result shape: {result.shape}")

    # Create output directory if needed
    output_path = Path(config.output_file)
    output_dir = output_path.parent
    if output_dir != Path(".") and not output_dir.exists():
        logging.info(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    logging.info(f"Saving results to {config.output_file}...")

    # Save data in NumPy format
    np.save(
        config.output_file,
        {
            "results": results,
            "omega": config.omega,
            "config": asdict(config),
            "computation_time": elapsed_time,
        },
        allow_pickle=True,
    )

    logging.info(f"Results successfully saved to {config.output_file}")


if __name__ == "__main__":
    main()
