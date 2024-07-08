# FDMCoreHaloRelation

This repository provides a tool for predicting the core-halo mass relation in fuzzy dark matter (FDM).

## Usage

To run the script, use the following command in the terminal:

```
python your_script_name.py --halo_mass HALO_MASS --redshift REDSHIFT --particle_mass PARTICLE_MASS
```

### Command-Line Arguments
`--halo_mass` : The mass of the halo. Default is 1e12.
`--redshift` : The redshift value. Default is 0.
`--particle_mass` : The mass of the particle. Default is 2e-23.

The script will output the current cosmology and the predicted core masses for both the revised model and the Schive2014 model.