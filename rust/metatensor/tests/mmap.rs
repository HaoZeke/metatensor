mod tensor_mmap {
    const DATA_PATH: &str = "../../metatensor-core/tests/data.mts";

    #[test]
    fn load_mmap_matches_regular_load() {
        let regular = metatensor::io::load(DATA_PATH).unwrap();
        let mmap = metatensor::io::load_mmap(DATA_PATH).unwrap();

        // keys should match
        assert_eq!(regular.keys().names(), mmap.keys().names());
        assert_eq!(regular.keys().count(), mmap.keys().count());

        // check each block's shapes and labels
        for i in 0..regular.keys().count() {
            let regular_block = regular.block_by_id(i);
            let mmap_block = mmap.block_by_id(i);

            // value shapes
            let rv = regular_block.values();
            let mv = mmap_block.values();
            let regular_shape = rv.as_raw().shape().unwrap();
            let mmap_shape = mv.as_raw().shape().unwrap();
            assert_eq!(regular_shape, mmap_shape, "block {i} values shape mismatch");

            // labels
            assert_eq!(regular_block.samples().names(), mmap_block.samples().names());
            assert_eq!(regular_block.samples().count(), mmap_block.samples().count());
            assert_eq!(regular_block.properties().names(), mmap_block.properties().names());
            assert_eq!(regular_block.properties().count(), mmap_block.properties().count());

            // components
            assert_eq!(regular_block.components().len(), mmap_block.components().len());
            for (rc, mc) in regular_block.components().iter().zip(mmap_block.components().iter()) {
                assert_eq!(rc.names(), mc.names());
                assert_eq!(rc.count(), mc.count());
            }

            // gradients
            assert_eq!(regular_block.gradient_list(), mmap_block.gradient_list());
            for param in regular_block.gradient_list() {
                let rg = regular_block.gradient(param).unwrap();
                let mg = mmap_block.gradient(param).unwrap();

                let rgv = rg.values();
                let mgv = mg.values();
                let rg_shape = rgv.as_raw().shape().unwrap();
                let mg_shape = mgv.as_raw().shape().unwrap();
                assert_eq!(rg_shape, mg_shape, "block {i} gradient '{param}' shape mismatch");

                assert_eq!(rg.samples().names(), mg.samples().names());
                assert_eq!(rg.samples().count(), mg.samples().count());
            }
        }
    }

    #[test]
    fn load_mmap_keys_metadata() {
        let mmap = metatensor::io::load_mmap(DATA_PATH).unwrap();
        assert_eq!(mmap.keys().names(), ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);
        assert_eq!(mmap.keys().count(), 27);
    }

    #[test]
    fn load_mmap_block_details() {
        let mmap = metatensor::io::load_mmap(DATA_PATH).unwrap();
        let block = mmap.block_by_id(13);

        let values = block.values();
        let shape = values.as_raw().shape().unwrap();
        assert_eq!(shape, [9, 3, 3]);
        assert_eq!(block.samples().names(), ["system", "atom"]);
        assert_eq!(block.components().len(), 1);
        assert_eq!(block.components()[0].names(), ["o3_mu"]);
        assert_eq!(block.properties().names(), ["n"]);

        assert_eq!(block.gradient_list(), ["positions"]);
        let gradient = block.gradient("positions").unwrap();
        let gv = gradient.values();
        let g_shape = gv.as_raw().shape().unwrap();
        assert_eq!(g_shape, [27, 3, 3, 3]);
        assert_eq!(gradient.samples().names(), ["sample", "system", "atom"]);
        assert_eq!(gradient.components().len(), 2);
        assert_eq!(gradient.components()[0].names(), ["xyz"]);
        assert_eq!(gradient.components()[1].names(), ["o3_mu"]);
        assert_eq!(gradient.properties().names(), ["n"]);
    }
}

mod operations_mmap {
    use metatensor::Labels;

    const DATA_PATH: &str = "../../metatensor-core/tests/data.mts";

    /// keys_to_properties on an mmap-loaded TensorMap should produce the same
    /// result as on a regularly loaded TensorMap (checking shapes and labels).
    #[test]
    fn keys_to_properties() {
        let regular = metatensor::io::load(DATA_PATH).unwrap();
        let mmap = metatensor::io::load_mmap(DATA_PATH).unwrap();

        let keys_to_move = Labels::empty(vec!["o3_sigma"]);

        let regular_result = regular.keys_to_properties(&keys_to_move, true).unwrap();
        let mmap_result = mmap.keys_to_properties(&keys_to_move, true).unwrap();

        assert_eq!(regular_result.keys().names(), mmap_result.keys().names());
        assert_eq!(regular_result.keys().count(), mmap_result.keys().count());

        for i in 0..regular_result.keys().count() {
            let rb = regular_result.block_by_id(i);
            let mb = mmap_result.block_by_id(i);

            let rv = rb.values();
            let mv = mb.values();
            assert_eq!(
                rv.as_raw().shape().unwrap(),
                mv.as_raw().shape().unwrap(),
                "keys_to_properties block {i} shape mismatch"
            );

            assert_eq!(rb.samples().count(), mb.samples().count());
            assert_eq!(rb.properties().names(), mb.properties().names());
            assert_eq!(rb.properties().count(), mb.properties().count());
        }
    }

    /// keys_to_samples on an mmap-loaded TensorMap should produce the same
    /// result as on a regularly loaded TensorMap.
    #[test]
    fn keys_to_samples() {
        let regular = metatensor::io::load(DATA_PATH).unwrap();
        let mmap = metatensor::io::load_mmap(DATA_PATH).unwrap();

        let keys_to_move = Labels::empty(vec!["center_type"]);

        let regular_result = regular.keys_to_samples(&keys_to_move, true).unwrap();
        let mmap_result = mmap.keys_to_samples(&keys_to_move, true).unwrap();

        assert_eq!(regular_result.keys().names(), mmap_result.keys().names());
        assert_eq!(regular_result.keys().count(), mmap_result.keys().count());

        for i in 0..regular_result.keys().count() {
            let rb = regular_result.block_by_id(i);
            let mb = mmap_result.block_by_id(i);

            let rv = rb.values();
            let mv = mb.values();
            assert_eq!(
                rv.as_raw().shape().unwrap(),
                mv.as_raw().shape().unwrap(),
                "keys_to_samples block {i} shape mismatch"
            );

            assert_eq!(rb.samples().names(), mb.samples().names());
            assert_eq!(rb.samples().count(), mb.samples().count());
            assert_eq!(rb.properties().count(), mb.properties().count());
        }
    }

    /// components_to_properties on an mmap-loaded TensorMap should work.
    #[test]
    fn components_to_properties() {
        let regular = metatensor::io::load(DATA_PATH).unwrap();
        let mmap = metatensor::io::load_mmap(DATA_PATH).unwrap();

        let regular_result = regular.components_to_properties(&["o3_mu"]).unwrap();
        let mmap_result = mmap.components_to_properties(&["o3_mu"]).unwrap();

        assert_eq!(regular_result.keys().count(), mmap_result.keys().count());

        for i in 0..regular_result.keys().count() {
            let rb = regular_result.block_by_id(i);
            let mb = mmap_result.block_by_id(i);

            let rv = rb.values();
            let mv = mb.values();
            assert_eq!(
                rv.as_raw().shape().unwrap(),
                mv.as_raw().shape().unwrap(),
                "components_to_properties block {i} shape mismatch"
            );

            assert_eq!(rb.samples().count(), mb.samples().count());
            assert_eq!(rb.properties().names(), mb.properties().names());
            assert_eq!(rb.properties().count(), mb.properties().count());
            assert_eq!(rb.components().len(), mb.components().len());
        }
    }

    /// Copying an mmap-loaded TensorMap should produce a valid copy.
    #[test]
    fn copy_mmap_tensor() {
        let mmap = metatensor::io::load_mmap(DATA_PATH).unwrap();
        let copied = mmap.try_clone().unwrap();

        assert_eq!(mmap.keys().count(), copied.keys().count());
        assert_eq!(mmap.keys().names(), copied.keys().names());

        for i in 0..mmap.keys().count() {
            let orig = mmap.block_by_id(i);
            let copy = copied.block_by_id(i);

            let ov = orig.values();
            let cv = copy.values();
            assert_eq!(
                ov.as_raw().shape().unwrap(),
                cv.as_raw().shape().unwrap(),
                "copy block {i} shape mismatch"
            );

            assert_eq!(orig.samples().count(), copy.samples().count());
            assert_eq!(orig.properties().count(), copy.properties().count());
        }
    }

    /// Saving an mmap-loaded TensorMap to a buffer should produce
    /// identical bytes as saving a regularly loaded one.
    #[test]
    fn save_mmap_roundtrip() {
        let regular = metatensor::io::load(DATA_PATH).unwrap();
        let mmap = metatensor::io::load_mmap(DATA_PATH).unwrap();

        let mut regular_buf = Vec::new();
        metatensor::io::save_buffer(&regular, &mut regular_buf).unwrap();

        let mut mmap_buf = Vec::new();
        metatensor::io::save_buffer(&mmap, &mut mmap_buf).unwrap();

        assert_eq!(regular_buf.len(), mmap_buf.len());
        assert_eq!(regular_buf, mmap_buf);
    }
}

mod partial_mmap {
    use metatensor::{Labels, LabelsBuilder};

    const DATA_PATH: &str = "../../metatensor-core/tests/data.mts";

    /// Build Labels from dynamic name list and selected rows of an existing Labels.
    fn labels_from_rows(source: &Labels, row_indices: &[usize]) -> Labels {
        let names: Vec<&str> = source.names().iter().map(|s| &**s).collect();
        let mut builder = LabelsBuilder::new(names);
        for &idx in row_indices {
            builder.add(&source[idx]);
        }
        builder.finish()
    }

    /// Passing None for all selections should produce the same result as load_mmap.
    #[test]
    fn full_match_equals_load_mmap() {
        let full = metatensor::io::load_mmap(DATA_PATH).unwrap();
        let partial = metatensor::io::load_mmap_partial(DATA_PATH, None, None, None).unwrap();

        assert_eq!(full.keys().names(), partial.keys().names());
        assert_eq!(full.keys().count(), partial.keys().count());

        for i in 0..full.keys().count() {
            let fb = full.block_by_id(i);
            let pb = partial.block_by_id(i);

            let fv = fb.values();
            let pv = pb.values();
            assert_eq!(
                fv.as_raw().shape().unwrap(),
                pv.as_raw().shape().unwrap(),
                "block {i} shape mismatch"
            );

            assert_eq!(fb.samples().count(), pb.samples().count());
            assert_eq!(fb.properties().count(), pb.properties().count());

            // check gradients
            assert_eq!(fb.gradient_list(), pb.gradient_list());
            for param in fb.gradient_list() {
                let fg = fb.gradient(param).unwrap();
                let pg = pb.gradient(param).unwrap();

                let fgv = fg.values();
                let pgv = pg.values();
                assert_eq!(
                    fgv.as_raw().shape().unwrap(),
                    pgv.as_raw().shape().unwrap(),
                    "block {i} gradient '{param}' shape mismatch"
                );
                assert_eq!(fg.samples().count(), pg.samples().count());
            }
        }
    }

    /// Filter by keys only: select specific blocks.
    #[test]
    fn key_filtering() {
        let full = metatensor::io::load_mmap(DATA_PATH).unwrap();
        let all_keys = full.keys();

        // Select the first 3 keys
        let selection_keys = labels_from_rows(&all_keys, &[0, 1, 2]);

        let partial = metatensor::io::load_mmap_partial(
            DATA_PATH,
            Some(&selection_keys),
            None,
            None,
        ).unwrap();

        assert_eq!(partial.keys().count(), 3);
        assert_eq!(partial.keys().names(), all_keys.names());

        // Each block should have the same data as the corresponding full block
        for i in 0..3 {
            let fb = full.block_by_id(i);
            let pb = partial.block_by_id(i);

            let fv = fb.values();
            let pv = pb.values();
            assert_eq!(
                fv.as_raw().shape().unwrap(),
                pv.as_raw().shape().unwrap(),
            );
            assert_eq!(fb.samples().count(), pb.samples().count());
            assert_eq!(fb.properties().count(), pb.properties().count());
        }
    }

    /// Filter by samples only: keep a subset of rows.
    #[test]
    fn sample_filtering() {
        let full = metatensor::io::load_mmap(DATA_PATH).unwrap();

        // Select only samples where system==0
        let sample_sel = Labels::new(["system"], &[[0_i32]]);

        let partial = metatensor::io::load_mmap_partial(
            DATA_PATH,
            None,
            Some(&sample_sel),
            None,
        ).unwrap();

        // Should have same number of blocks
        assert_eq!(full.keys().count(), partial.keys().count());

        for i in 0..partial.keys().count() {
            let fb = full.block_by_id(i);
            let pb = partial.block_by_id(i);

            // Partial should have <= samples
            assert!(pb.samples().count() <= fb.samples().count());

            // All partial samples should have system==0
            for entry in pb.samples().iter() {
                assert_eq!(entry[0].i32(), 0, "block {i}: expected system==0");
            }

            // Properties should be unchanged
            assert_eq!(fb.properties().count(), pb.properties().count());
        }
    }

    /// Filter by properties only: keep a subset of columns.
    #[test]
    fn property_filtering() {
        let full = metatensor::io::load_mmap(DATA_PATH).unwrap();

        // Select only properties where n==0
        let prop_sel = Labels::new(["n"], &[[0_i32]]);

        let partial = metatensor::io::load_mmap_partial(
            DATA_PATH,
            None,
            None,
            Some(&prop_sel),
        ).unwrap();

        assert_eq!(full.keys().count(), partial.keys().count());

        for i in 0..partial.keys().count() {
            let fb = full.block_by_id(i);
            let pb = partial.block_by_id(i);

            // Samples should be unchanged
            assert_eq!(fb.samples().count(), pb.samples().count());

            // Partial should have <= properties
            assert!(pb.properties().count() <= fb.properties().count());

            // All partial properties should have n==0
            for entry in pb.properties().iter() {
                assert_eq!(entry[0].i32(), 0, "block {i}: expected n==0");
            }

            // Check gradient properties also filtered
            for param in pb.gradient_list() {
                let pg = pb.gradient(param).unwrap();
                for entry in pg.properties().iter() {
                    assert_eq!(entry[0].i32(), 0, "block {i} gradient '{param}': expected n==0");
                }
            }
        }
    }

    /// Combined key + sample + property filtering.
    #[test]
    fn combined_filtering() {
        let full = metatensor::io::load_mmap(DATA_PATH).unwrap();
        let all_keys = full.keys();

        // Select first 5 keys
        let key_sel = labels_from_rows(&all_keys, &[0, 1, 2, 3, 4]);

        let sample_sel = Labels::new(["system"], &[[0_i32]]);
        let prop_sel = Labels::new(["n"], &[[0_i32]]);

        let partial = metatensor::io::load_mmap_partial(
            DATA_PATH,
            Some(&key_sel),
            Some(&sample_sel),
            Some(&prop_sel),
        ).unwrap();

        assert_eq!(partial.keys().count(), 5);

        for i in 0..5 {
            let pb = partial.block_by_id(i);
            for entry in pb.samples().iter() {
                assert_eq!(entry[0].i32(), 0);
            }
            for entry in pb.properties().iter() {
                assert_eq!(entry[0].i32(), 0);
            }
        }
    }

    /// Gradient sample[0] should be correctly reindexed after sample filtering.
    #[test]
    fn gradient_reindexing() {
        let sample_sel = Labels::new(["system"], &[[0_i32]]);

        let partial = metatensor::io::load_mmap_partial(
            DATA_PATH,
            None,
            Some(&sample_sel),
            None,
        ).unwrap();

        for i in 0..partial.keys().count() {
            let pb = partial.block_by_id(i);
            let n_samples = pb.samples().count();

            for param in pb.gradient_list() {
                let pg = pb.gradient(param).unwrap();
                for entry in pg.samples().iter() {
                    let parent_idx = entry[0].i32();
                    assert!(
                        (parent_idx as usize) < n_samples,
                        "block {i} gradient '{param}': parent sample index {parent_idx} >= n_samples {n_samples}"
                    );
                }
            }
        }
    }

    /// Selecting keys that don't exist in the file should produce an empty TensorMap.
    #[test]
    fn empty_key_match() {
        let key_sel = Labels::new(
            ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"],
            &[[999_i32, 999, 999, 999]],
        );

        let partial = metatensor::io::load_mmap_partial(
            DATA_PATH,
            Some(&key_sel),
            None,
            None,
        ).unwrap();

        assert_eq!(partial.keys().count(), 0);
    }
}

mod block_mmap {
    const DATA_PATH: &str = "../../metatensor-core/tests/block.mts";

    #[test]
    fn load_block_mmap() {
        let block = metatensor::io::load_block_mmap(DATA_PATH).unwrap();

        let values = block.values();
        let shape = values.as_raw().shape().unwrap();
        assert_eq!(shape, [9, 5, 3]);
        assert_eq!(block.samples().names(), ["system", "atom"]);
        assert_eq!(block.components().len(), 1);
        assert_eq!(block.components()[0].names(), ["o3_mu"]);
        assert_eq!(block.properties().names(), ["n"]);

        let block_ref = block.as_ref();
        assert_eq!(block_ref.gradient_list(), ["positions"]);
        let gradient = block_ref.gradient("positions").unwrap();
        let gv = gradient.values();
        let g_shape = gv.as_raw().shape().unwrap();
        assert_eq!(g_shape, [59, 3, 5, 3]);
        assert_eq!(gradient.samples().names(), ["sample", "system", "atom"]);
        assert_eq!(gradient.components().len(), 2);
        assert_eq!(gradient.components()[0].names(), ["xyz"]);
        assert_eq!(gradient.components()[1].names(), ["o3_mu"]);
        assert_eq!(gradient.properties().names(), ["n"]);
    }

    #[test]
    fn load_block_mmap_matches_regular() {
        let regular = metatensor::io::load_block(DATA_PATH).unwrap();
        let mmap = metatensor::io::load_block_mmap(DATA_PATH).unwrap();

        let rv = regular.values();
        let mv = mmap.values();
        let regular_shape = rv.as_raw().shape().unwrap();
        let mmap_shape = mv.as_raw().shape().unwrap();
        assert_eq!(regular_shape, mmap_shape);

        assert_eq!(regular.samples().count(), mmap.samples().count());
        assert_eq!(regular.properties().count(), mmap.properties().count());

        let rr = regular.as_ref();
        let mr = mmap.as_ref();
        assert_eq!(rr.gradient_list(), mr.gradient_list());
    }

    /// Saving an mmap-loaded block to a buffer should produce
    /// identical bytes as saving a regularly loaded one.
    #[test]
    fn save_block_mmap_roundtrip() {
        let regular = metatensor::io::load_block(DATA_PATH).unwrap();
        let mmap = metatensor::io::load_block_mmap(DATA_PATH).unwrap();

        let mut regular_buf = Vec::new();
        metatensor::io::save_block_buffer(regular.as_ref(), &mut regular_buf).unwrap();

        let mut mmap_buf = Vec::new();
        metatensor::io::save_block_buffer(mmap.as_ref(), &mut mmap_buf).unwrap();

        assert_eq!(regular_buf.len(), mmap_buf.len());
        assert_eq!(regular_buf, mmap_buf);
    }
}
