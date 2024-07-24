Welcome to the documentation of the MISO2 model
================================================

The MISO2 model (Material Inputs, Stocks, and Outputs) is a dynamic, inflow-driven material stock-flow model,
which covers 14 supply chain processes from raw material extraction and processing, stock-building and stock dynamics, to end-of-life flows and waste management. For detailed information and documentation, we refer to the peer-reviewed, open-access publication (Wiedenhofer et al. 2024).

With the MISO2 model, we also provide a global, country-level application which covers 23 raw
materials and 20 stock-building materials across 177 countries from 1900 to 2016. The
MAT_STOCKS database version 1.0 can be `found on Zenodo <https://zenodo.org/records/12794253>`_.

**Key Features**:

- Economy-wide material stock-flow model and analysis 
- Fully consistent with system boundaries and definitions established in economy-wide material flow accounting
- Comprehensive coverage of raw materials, stock-building materials, end-uses and stock dynamics, as well as end-of-life and waste flows
- Global, country-level application from 1900 to 2016, with a spin-up period from 1820 to 1900.

The documentation #LINK provides an overview of the software, guides on setup and usage, as well as detailed API references. We give working examples of how to process this input data, run the model, and access the results. Input data for two countries, United States of America and United Kingdom, is provided to test the model.

MISO2 wraps functionality from the `ODYM package <https://github.com/IndEcol/ODYM>`_ and contains a slightly modified version of the ODYM v1 release adapted to our needs.

Authors
---------------

The MISO2 software was developed by a collaborative team from BOKU University:

- Jan Streeck
- Benedikt Grammer
- Hanspeter Wieland
- Dominik Wiedenhofer

Contact
---------------

For domain-related questions and collaborations, contact:
dominik.wiedenhofer@boku.ac.at

For technical questions and bug reports, please open a GitHub issue or contact:
benedikt.grammer@boku.ac.at

License
---------------

This project is released under the GNU GPL-3.0 License.

How to cite
---------------

If you use MISO2 in your research, please cite the following article:

Wiedenhofer, D. Streeck, J. Wieland, H. Grammer, B. Baumgart, A. Plank, B. Helbig, C. Pauliuk, S. Haberl, H. and Krausmann, F.
“From Extraction to End-uses and Waste Management: Modelling Economy-wide Material Cycles and Stock Dynamics Around the World,”
SSRN Electronic Journal, January 2024. [DOI: 10.2139/ssrn.4794611] `Link to paper <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4794611>`_

Acknowledgments
---------------

This work was supported by the European Research Council (ERC) under the European Union’s Horizon 2020 research and
innovation programme (MAT_STOCKS, grant agreement No 741950), and the European Union's Horizon Europe programme
(CircEUlar, grant agreement No 101056810). Funded by the European Union. Views and opinions expressed are however those
of the author(s) only and do not necessarily reflect those of the European Union or granting authorities.

.. image:: figures/EU_flag.jpg
   :width: 25%

.. image:: figures/BOKU_Hauptlogo_RGB.png
   :width: 35%

.. toctree::
   :maxdepth: 1
   :hidden:

   background/about
   setup/setup
   usage/usage
   modules/API_reference

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`