from setuptools import setup, find_packages

setup(
    name="inversion_fbpic",
    version="0.2",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "attrs>=23.2",
        "numpy>=1.24",
        "scipy>=1.10",
        "fbpic",
        "matplotlib>=3.7",
        "mpi4py",
        "h5py",
        "openpmd-viewer>=1.10",
        "opencv-python",
        "lasy",
        "periodictable",
    ],
    extras_require={
        "dev": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            # Ebeam analysis scripts
            "plot-ebeam-analysis=inversion_fbpic.scripts.ebeam.plot_ebeam_analysis:main",
            "visualize-ebeamparams-vs-scan=inversion_fbpic.scripts.ebeam.visualize_ebeamparams_vs_scan:main",
            "summarize-scan-sensitivities=inversion_fbpic.scripts.ebeam.summarize_scan_sensitivities:main",
            "simulate-magspec-scan=inversion_fbpic.scripts.ebeam.simulate_magspec_scan:main",
            "plot-slice-emittance=inversion_fbpic.scripts.ebeam.plot_slice_emittance:main",
            "plot-slice-energy-spread=inversion_fbpic.scripts.ebeam.plot_slice_energy_spread:main",
            "plot-dispersion=inversion_fbpic.scripts.ebeam.plot_dispersion:main",
            "plot-phase-space-moments=inversion_fbpic.scripts.ebeam.plot_phase_space_moments:main",
            "energy-at-peak-current=inversion_fbpic.scripts.ebeam.energy_at_peak_current:main",
            "energy-at-peak-current-multibunch=inversion_fbpic.scripts.ebeam.energy_at_peak_current_multibunch:main",
            # Charge density scripts
            "slideshow-from-npy=inversion_fbpic.scripts.charge_density.slideshow_from_npy:main",
            # Data extraction scripts
            "extract-hdf5-field=inversion_fbpic.scripts.data_parsing.extract_hdf5_field:main",
            "extract-hdf5-particles=inversion_fbpic.scripts.data_parsing.extract_hdf5_particles:main",
            "build-dataset=inversion_fbpic.scripts.data_parsing.build_dataset:main",
            # Calculation scripts
            "fbpic-calc-nr=inversion_fbpic.scripts.calculations.fbpic_calc_nr:main",
            # Laser analysis scripts
            "analyze-laser-evolution=inversion_fbpic.scripts.laser.analyze_laser_evolution:main",
            "lasy-propagation=inversion_fbpic.scripts.laser.lasy_propagation:main",
        ],
    },
)
