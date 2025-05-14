<p align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/slmsuite/slmsuite/main/docs/source/static/slmsuite-dark.svg">
<img alt="slmsuite" src="https://raw.githubusercontent.com/slmsuite/slmsuite/main/docs/source/static/slmsuite.svg" width="256">
</picture>
</p>

<h2 align="center">High-Performance Spatial Light Modulator Control and Holography</h2>

<p align="center">
<a href="https://slmsuite.readthedocs.io/en/latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/slmsuite/badge/?version=latest"></a>
<a href="https://pypi.org/project/slmsuite/"><img alt="PyPi Package" src="https://img.shields.io/badge/pypi-v0.3.0-3776AB.svg"></a>
<a href="https://github.com/slmsuite/slmsuite/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/slmsuite/slmsuite?color=purple"></a>
<!--<a href="https://pepy.tech/project/slmsuite"><img alt="Downloads" src="https://pepy.tech/badge/slmsuite"></a>-->
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://arxiv.org/abs/2505.08777"><img alt="Citation" src="https://img.shields.io/badge/cite-arXiv%3A2505.08777-B31B1B.svg"></a>
</p>

`slmsuite` combines GPU-accelerated beamforming algorithms with optimized hardware control, automated calibration, and user-friendly scripting to enable high-performance programmable optics with modern spatial light modulators.

## Key Features
- [GPU-accelerated iterative phase retrieval  algorithms](https://slmsuite.readthedocs.io/en/latest/_examples/computational_holography.html#Computational-Holography) (e.g. Gerchberg-Saxton, weighted GS, or phase-stationary WGS)
- [A simple hardware-control interface](https://slmsuite.readthedocs.io/en/latest/_examples/experimental_holography.html#Loading-Hardware) for working with various SLMs and cameras
- [Automated Fourier- to image-space coordinate transformations](https://slmsuite.readthedocs.io/en/latest/_examples/experimental_holography.html#Fourier-Calibration): choose how much light goes to which camera pixels; `slmsuite` takes care of the rest!
- [Automated wavefront calibration](https://slmsuite.readthedocs.io/en/latest/_examples/wavefront_calibration.html) to improve manufacturer-supplied flatness maps or compensate for additional aberrations along the SLM imaging train
- Optimized [optical focus/spot arrays](https://slmsuite.readthedocs.io/en/latest/_examples/computational_holography.html#Spot-Arrays) using [camera feedback](https://slmsuite.readthedocs.io/en/latest/_examples/experimental_holography.html#A-Uniform-Square-Array), automated statistics, and numerous analysis routines
- [Mixed region amplitude freedom](https://slmsuite.readthedocs.io/en/latest/_autosummary/slmsuite.holography.algorithms.Hologram.html#slmsuite.holography.algorithms.Hologram.optimize), which ignores unused far-field regions in favor of optimized hologram performance in high-interest areas.
- [Toolboxes for structured light](https://slmsuite.readthedocs.io/en/latest/_examples/structured_light.html#), imprinting sectioned phase masks, SLM unit conversion, padding and unpadding data, and more
- A fully-featured [example library](https://slmsuite.readthedocs.io/en/latest/examples.html) that demonstrates these and other features

## Installation

Install the stable version of `slmsuite` from [PyPI](https://pypi.org/project/slmsuite/) using:

```console
pip install slmsuite
```

Install the latest version of `slmsuite` from GitHub using:

```console
pip install git+https://github.com/slmsuite/slmsuite
```

## Documentation and Examples

Extensive
[documentation](https://slmsuite.readthedocs.io/en/latest/)
and
[API reference](https://slmsuite.readthedocs.io/en/latest/api.html)
are available through readthedocs.

Examples can be found embedded in
[documentation](https://slmsuite.readthedocs.io/en/latest/examples.html),
live through
[nbviewer](https://nbviewer.org/github/slmsuite/slmsuite-examples/tree/main/examples/),
or directly in
[source](https://github.com/slmsuite/slmsuite-examples).

## Citation

`slmsuite` is built upon the work of many others. If you used any technique in particular,
be sure to check our [documentation](https://slmsuite.readthedocs.io/en/latest/) for links to relevant papers.
For acknowledgement of the package or unique features such as
aberration-space holography, please use the following citation:

```
@misc{
    christen2025slmsuite,
    Author = {Ian Christen and Christopher Panuski and Thomas Propson and Dirk Englund},
    Title = {Full-volume aberration-space holography},
    Year = {2025},
    Eprint = {arXiv:2505.08777},
}
```

<p align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/slmsuite/slmsuite-examples/main/examples/ex-zernike-spots-dark.gif">
<img alt="slmsuite" src="https://raw.githubusercontent.com/slmsuite/slmsuite-examples/main/examples/ex-zernike-spots.gif" width="512">
</picture>
</p>
