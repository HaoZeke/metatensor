.. _devdoc-faq:

Developer FAQ
=============

This section addresses common questions and hurdles encountered when developing 
with the ecosystem.

How do I run the full CI suite locally?
---------------------------------------

You can use ``gh act`` to emulate GitHub Actions on your machine. This is 
useful for catching environment-specific bugs before pushing code. 
See :ref:`devdoc-local-ci` for setup and usage instructions.

How do I run just the NumPy < 2.0 tests?
----------------------------------------

When using ``gh act``, you can isolate specific matrix configurations by 
specifying the matrix keys. To run the legacy NumPy tests:

.. code-block:: bash

    gh act -j python-tests \
      --matrix os:ubuntu-24.04 \
      --matrix python-version:3.10 \
      --matrix torch-version:2.1 \
      --matrix numpy-version-pin:"<2.0"

How do I run a single tox environment in CI?
--------------------------------------------

If you are using ``act`` and want to skip the full suite in favor of a 
single environment (e.g., ``core-tests``), pass the ``TOXENV`` 
variable:

.. code-block:: bash

    gh act -j python-tests --env TOXENV=core-tests

How do I make a new release?
----------------------------

Follow a release PR. Find more `here <https://github.com/metatensor/metatensor/pulls?q=is%3Apr+release+is%3Aclosed>`_.

.. note::

  The changelog files are managed by hand.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Component
     - Release PR example
   * - **metatensor-core**
     - `v0.1.18 <https://github.com/metatensor/metatensor/pull/1013>`_
   * - **metatensor-rust**
     - `v0.2.3 <https://github.com/metatensor/metatensor/pull/1015>`_
   * - **metatensor-torch**
     - `v0.8.4 <https://github.com/metatensor/metatensor/pull/1042>`_
   * - **metatensor-operations**
     - `v0.4.0 <https://github.com/metatensor/metatensor/pull/1002>`_
   * - **metatensor-learn**
     - `v0.4.0 <https://github.com/metatensor/metatensor/pull/1003>`_

How do I handle new PyTorch versions?
-------------------------------------

Follow a PyTorch upgrade PR, . Then make a new release.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Complexity
     - Example
   * - Baseline version changes
     - `v2.9 <https://github.com/metatensor/metatensor/pull/1005>`_
   * - Handling deprecations
     - `v2.10 <https://github.com/metatensor/metatensor/pull/1041>`_
