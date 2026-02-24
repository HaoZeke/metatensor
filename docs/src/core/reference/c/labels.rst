Labels
======

.. doxygenstruct:: mts_labels_t
    :members:

The following functions operate on :c:type:`mts_labels_t`:

- :c:func:`mts_labels_create`: create the Rust-side data for the labels
- :c:func:`mts_labels_clone`: increment the reference count of the Rust-side data
- :c:func:`mts_labels_free`: decrement the reference count of the Rust-side data,
  and free the data when it reaches 0
- :c:func:`mts_labels_position`: get the position of an entry in the labels
- :c:func:`mts_labels_union`: get the union of two labels
- :c:func:`mts_labels_intersection`: get the intersection of two labels
- :c:func:`mts_labels_difference`: get the set difference of two labels
- :c:func:`mts_labels_select`: select entries in labels that match a selection
- :c:func:`mts_labels_values_array`: get the backing values array
- :c:func:`mts_labels_values`: get the CPU values, materializing if needed

Additionally, there are a few advanced functions which may be used if the user
can guarantee that the data used for labels is already unique.

- :c:func:`mts_labels_create_assume_unique`: create the Rust-side data for labels without verifying uniqueness
- :c:func:`mts_labels_create_from_array`: create labels from an ``mts_array_t`` (CPU only, checks uniqueness)
- :c:func:`mts_labels_create_from_array_assume_unique`: create labels from an ``mts_array_t`` on any device without checking uniqueness
- :c:func:`mts_labels_set_cached_values`: pre-fill cached CPU values without materializing from the backing array

--------------------------------------------------------------------------------

.. doxygenfunction:: mts_labels_create_assume_unique

.. doxygenfunction:: mts_labels_create

.. doxygenfunction:: mts_labels_create_from_array

.. doxygenfunction:: mts_labels_create_from_array_assume_unique

.. doxygenfunction:: mts_labels_clone

.. doxygenfunction:: mts_labels_free

.. doxygenfunction:: mts_labels_position

.. doxygenfunction:: mts_labels_values_array

.. doxygenfunction:: mts_labels_values

.. doxygenfunction:: mts_labels_union

.. doxygenfunction:: mts_labels_intersection

.. doxygenfunction:: mts_labels_difference

.. doxygenfunction:: mts_labels_select

.. doxygenfunction:: mts_labels_set_cached_values
