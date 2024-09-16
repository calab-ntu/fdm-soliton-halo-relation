# FDMCoreHaloRelation

This repository provides a tool for predicting the core-halo mass relation in fuzzy dark matter (FDM).

## Installation

### Using `requirements.txt`

To install the necessary dependencies, run:

```sh
pip install -r requirements.txt
```

The cosmological model and halo concentration are based on the Colossus package, which can be found here:
[Colossus Documentation](https://bdiemer.bitbucket.io/colossus/)

## Usage

To run the script, use the following command in the terminal:

```sh
python your_script_name.py --halo_mass HALO_MASS --redshift REDSHIFT --particle_mass PARTICLE_MASS
```

### Command-Line Arguments
`--halo_mass` : The mass of the halo in Msun. Default is `1e12`.\
`--redshift` : The redshift value. Default is 0.\
`--particle_mass` : The mass of the particle in eV. Default is `2e-23`.

The script will output the current cosmology and the predicted core masses for both the revised model and the Schive2014 model.