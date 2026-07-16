use std::vec::Vec;

use bytesize::ByteSize;
use lumberjack_model::{
    Model,
    model::{Node, iter_trees},
};

use crate::compiler::tree_max_depth;

/// Perform an analysis of the model, and output some useful metrics to stdout.
///
/// The function takes a number of cells as an optional parameter. If set,
/// will also info about the cell utilization.
pub fn analyze(model: &Model) {
    println!("--- Lumberjack model analysis ---");
    if let Err(e) = model.verify_acyclic() {
        panic!("Model is malformed: {e:?}")
    }

    println!(
        "Random forest model with:\n\t- {} trees\n\t- {} features\n\t- {} targets\n\t- {} cells",
        model.num_trees(),
        model.num_features(),
        model.num_targets(),
        model.num_cells(),
    );

    let size_bytes = ByteSize::b(size_of_val(model.nodes()) as u64);
    println!("Total size: {} nodes ({size_bytes})", model.nodes().len(),);

    struct TreeMetadata {
        size: usize,
        depth: usize,
        paired: usize,
        even_idx: usize,
        odd_idx: usize,
        padding: usize,
    }

    let mut tree_metadata = Vec::new();

    for (i, header_idx) in model.tree_headers().enumerate() {
        let (header, nodes) = model.get_tree(header_idx);

        assert!(header_idx.is_multiple_of(2));

        let mut paired = 0;
        let mut padding = 0;
        let mut odd_idx = 0;
        let mut even_idx = 0;
        for (i, node) in nodes.iter().enumerate() {
            if node.is_padding() {
                padding += 1;
            }

            // Check pair in header
            if i == 0 && node.as_header().first_node_idx() == 1 {
                paired += 1;
            }
            if !i.is_multiple_of(2) {
                odd_idx += 1;
                continue;
            }

            even_idx += 1;

            let node = node.as_branch();

            if !node.flags().left_prediction() && node.left_ptr().get() as usize == i + 1 {
                paired += 1;
            }

            if !node.flags().right_prediction() && node.right_ptr().get() as usize == i + 1 {
                paired += 1;
            }
        }

        let is_cell_header = header.cache_metadata().is_cell_header();
        let max_depth = tree_max_depth(nodes, header.first_node_idx() as usize);

        let size_bytes = ByteSize::b(size_of_val(nodes) as u64);
        print!(
            "/ Tree {i} / Size: {len} nodes ({size_bytes}) / Pair utilization: {utilization:.1}% / Max depth: {max_depth}",
            len = nodes.len(),
            utilization = paired as f32 / even_idx as f32 * 100.0,
        );
        if is_cell_header {
            println!(" / cell_header: {is_cell_header}");
        } else {
            println!();
        }

        tree_metadata.push(TreeMetadata {
            size: nodes.len(),
            depth: max_depth,
            paired,
            even_idx,
            odd_idx,
            padding,
        });
    }

    let max_size = tree_metadata.iter().map(|m| m.size).max().unwrap();
    let max_depth = tree_metadata.iter().map(|m| m.depth).max().unwrap();

    let size_bytes = ByteSize::b((max_size * size_of::<Node>()) as u64);
    println!();
    println!("Max tree size: {max_size} nodes ({size_bytes})",);
    println!("Max tree depth: {max_depth}\n");

    println!("/ Cell analysis /");

    let mut tree_idx = 0;
    for (cell_idx, start_node_idx) in model.cache_headers().enumerate() {
        let all_nodes = model.nodes();

        let cell_header = all_nodes[start_node_idx].as_header();
        let cache_metadata = cell_header.cache_metadata();
        let num_trees = cache_metadata
            .get_num_trees()
            .expect("Header is a cell header");

        let headers = iter_trees(&all_nodes[start_node_idx..]).take(num_trees as usize);

        let mut max_depth = 0;
        let mut num_trees = 0;
        let mut required_capacity = 0;
        let mut paired = 0;
        let mut even = 0;
        let mut odd = 0;
        let mut padding = 0;
        for _ in headers {
            max_depth += tree_metadata[tree_idx].depth;
            num_trees += 1;
            required_capacity += tree_metadata[tree_idx].size;
            paired += tree_metadata[tree_idx].paired;
            even += tree_metadata[tree_idx].even_idx;
            odd += tree_metadata[tree_idx].odd_idx;
            padding += tree_metadata[tree_idx].padding;

            tree_idx += 1;
        }

        let size_bytes = ByteSize((required_capacity * size_of::<Node>()) as u64);
        println!(
            "  Cell {cell_idx}:\n\t{num_trees} trees\n\t{required_capacity} nodes ({size_bytes})\n\tMax depth: {max_depth}\n\t{paired} Paired nodes ({utilization:.1}% utilization)\n\tEven: {even}, Odd: {odd}, padding: {padding}",
            utilization = paired as f32 / even as f32 * 100.0
        );
    }

    println!("------");
}
