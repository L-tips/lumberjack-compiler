use std::vec::Vec;

use bytesize::ByteSize;
use lumberjack_model::{
    Model,
    model::{Node, iter_trees},
};

use crate::ir::tree_max_depth;

/// Perform an analysis of the model, and output some useful metrics to stdout.
///
/// The function takes a number of cells as an optional parameter. If set,
/// will also info about the cell utilization.
pub fn analyze(model: &Model) {
    println!("--- Lumberjack model analysis ---");
    model
        .verify_linear()
        .unwrap_or_else(|e| panic!("Could not verify forest: {e:?}"));

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
    }

    let mut tree_metadata = Vec::new();

    for (i, header_idx) in model.tree_headers().enumerate() {
        let (header, nodes) = model.get_tree(header_idx);
        let is_cell_header = header.cache_metadata().is_cell_header();
        let max_depth = tree_max_depth(nodes, header.first_node_idx() as usize);

        let size_bytes = ByteSize::b(size_of_val(nodes) as u64);
        print!(
            "/ Tree {i} / Size: {} nodes ({size_bytes}) / Max depth: {max_depth}",
            nodes.len(),
        );
        if is_cell_header {
            println!(" / cell_header: {is_cell_header}");
        } else {
            println!();
        }

        tree_metadata.push(TreeMetadata {
            size: nodes.len(),
            depth: max_depth,
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
        for _ in headers {
            max_depth += tree_metadata[tree_idx].depth;
            num_trees += 1;
            required_capacity += tree_metadata[tree_idx].size;

            tree_idx += 1;
        }

        let size_bytes = ByteSize((required_capacity * size_of::<Node>()) as u64);
        println!(
            "  Cell {cell_idx}:\n\t{num_trees} trees\n\t{required_capacity} nodes ({size_bytes})\n\tMax depth: {max_depth}"
        );
    }

    println!("------");
}
