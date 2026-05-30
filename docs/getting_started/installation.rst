Installation
============

Using ``uv`` (recommended):

.. code-block:: bash

   uv sync

Using ``pip``:

.. code-block:: bash

   pip install tf_gnns

Developer setup:

.. code-block:: bash

   uv sync --group dev
   uv run pytest -v

Torch backend smoke test:

.. code-block:: bash

   pip install "torch==2.11.0"
   KERAS_BACKEND=torch pytest -q tests
