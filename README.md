# FDM Soliton-Halo Relation

This repository provides a tool for predicting the soliton-halo mass relation in fuzzy dark matter (FDM).
[Paper link]()

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
python SHR.py --halo_mass HALO_MASS --redshift REDSHIFT --m22 PARTICLE_MASS
```

### Command-Line Arguments
`-hm` or `--halo_mass` : The mass of the halo in Msun. Default is `1e12`.\
`-z` or `--redshift` : The redshift value. Default is 0.\
`-m22` or `--m22` : The mass of the particle in 1e-22 eV. Default is `2e-1`.

The script will output the current cosmology and the predicted core masses for both the revised model and the Schive2014 model.
