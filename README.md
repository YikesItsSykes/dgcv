# **dgcv**: Differential Geometry with Complex Variables

**dgcv** is an open-source Python package providing basic tools for differential geometry integrated with systematic organization of structures naturally accompanying complex variables, in short, Differential Geometry with Complex Variables. The library is perfectly suitable for DG work independent of complex structures, and CV is emphasized in the name because **dgcv** is designed from the ground up to work well in holomorphic settings.

At its core are symbolic representations of standard DG objects such as tensor fields (e.g., vector fields and differential forms) and algebras, and tools for doing standard computations with them. There are coordinate free representations, and representations defined relative to coordinate systems falling into two general categories:

- **standard** - basic systems that can represent real or complex coordinates, sufficient for applications that do not require **dgcv**'s complex variable handling features.
- **complex** - richer systems for representing complex coordinate patches that interact with **dgcv**'s complex variable handling features. These are comprised of holomorphic coordinate functions, their conjugates, and their real and imaginary parts (e.g., $\\{z_j,\overline{z_j},x_j,y_j\\}$).

**dgcv** functions account for coordinate types dynamically when operating on objects built from such coordinate systems. The package has a substantial library for coordinate-free representations as well, and tools for converting between the two paradigms.

As systems of differential geometric objects constructed from complex variables inherit natural relationships from the underlying complex structure, **dgcv** objects track these relationships across the constructions. This system enables smooth switching between real and holomorphic coordinate representations of mathematical objects. In computations, **dgcv** objects dynamically manage this format switching on their own so that typical complex variables formulas can be written plainly and will simply work. Some examples of this: In coordinates $z_j = x_j + iy_j$, expressions such as $\frac{\partial}{\partial x_j}|z_j|^2$ or $d z_j \wedge d \overline{z_j} \left( \frac{\partial}{\partial z_j}, \frac{\partial}{\partial y_j} \right)$ are correctly parsed without needing to convert everything to a uniform variable format. Retrieving objects' complex structure-related attributes, like the holomorphic part of a vector field or pluriharmonic terms from a polynomial is straightforward. Complexified cotangent bundles and their exterior algebras are easily decomposed into components from the Dolbeault complex and Dolbeault operators themselves can be applied to functions and k-forms in either coordinate format.

**dgcv** is tested on Python 3.13.


## Documentation

**dgcv** documentation is hosted at [https://www.realandimaginary.com/dgcv/](https://www.realandimaginary.com/dgcv/), with documentation pages for individual functions in the library and more. 

## Tutorials 

Many tutorials for using **dgcv** are available at [https://www.realandimaginary.com/dgcv/tutorials/](https://www.realandimaginary.com/dgcv/tutorials/).

## Installation

**dgcv** can be installed directly from PyPI with pip:

```bash
pip install dgcv
```

Optional supporting libraries can be installed simultaneously as needed:

```bash
pip install dgcv[sympy] # simultaneously install SymPy
```

```bash
pip install dgcv[ipython] # simultaneously install IPython
```

```bash
pip install dgcv[recommended] # simultaneously installs both IPython and SymPy
```

Depending on Python install configurations, the above command can vary. The key is to have the relevant Python environment active so that the package manager `pip` sources from the right location (suggested to use virtual environments: [Getting started with virtual environments](https://docs.python.org/3/library/venv.html)).

## AI Disclosure

**dgcv** development is cautiosly incorporating AI assistance for coding, always followed by human review. The first significant use was in the developement of version 0.5.0, where AI orchestrated extensive performance profiling and identified important refinements to when and where **dgcv** algorithms apply CAS simplify calls. Ongoing roles are described further at [dgcv AI policies](https://www.realandimaginary.com/dgcv/resources/dgcv-ai-policies). Considerations behind the decision to allow AI tools in **dgcv** development are given there as well. Core mathematical algorithms and logic within **dgcv** remain human-designed, with one exception: Starting with version 0.6.0, **dgcv** includes a subpackage that is AI-generated, namely `dgcv._symbolic_scalars`, which supplies the built-in [symbolic engine](https://www.realandimaginary.com/dgcv/resources/symbolic-engines). This is the default engine today, but one can also completely avoid it and use SymPy or SageMath alternatives with **dgcv** settings configurations (see [`default_engine`](https://www.realandimaginary.com/dgcv/parameters/default_engine/?in=set_dgcv_settings)).

The subpackage has been carefully reviewed by the author and passed extensive testing. But such review does not rigorously certify correctness, so **dgcv** will always support compatibility with alternative open source CAS libraries.


## License

**dgcv** is licensed under an Apache 2.0 License. See the [`LICENSE`](https://github.com/YikesItsSykes/dgcv/blob/main/LICENSE) and [`NOTICE`](https://github.com/YikesItsSykes/dgcv/blob/main/NOTICE) files for more information.

## Author

**dgcv** was created and is maintained by [David Sykes](https://www.realandimaginary.com).

---

### Development Notes

**dgcv** is always growing and updated regularly. I often make new functions for personal projects, and add the ones with general utility for others to the public versions of the library. Contributions, suggestions for additions to the library, and feedback from anyone interested are very much welcome. –D.S.
