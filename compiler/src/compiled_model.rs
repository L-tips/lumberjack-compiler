use std::{
    mem::{size_of, size_of_val},
    vec::Vec,
};

use bytesize::ByteSize;
use lumberjack_model::{
    Model,
    model::{Node, iter_trees},
};
use serde::Serialize;

use crate::compiler::tree_max_depth;

#[derive(Serialize, Clone, Debug)]
struct TreeMetadata {
    index: usize,
    size: usize,
    depth: usize,
    good_pairs: usize,
    guaranteed: usize,
    cache_lines: usize,
    padding: usize,
    cell_header: bool,
}

#[derive(Serialize, Clone, Debug)]
struct CellMetadata {
    index: usize,
    num_trees: usize,
    required_capacity: usize,
    max_depth: usize,
    good_pairs: usize,
    guaranteed_pairs: usize,
    cache_lines: usize,
    padding: usize,
}

#[derive(Serialize, Clone, Debug)]
pub struct AnalysisResults {
    num_trees: u32,
    num_features: u16,
    num_targets: u16,
    used_cells: u8,
    total_nodes: usize,
    total_size_bytes: usize,
    max_tree_size: usize,
    max_tree_depth: usize,
    trees: Vec<TreeMetadata>,
    cells: Vec<CellMetadata>,
}

/// Perform an analysis of the model, and output some useful metrics to stdout.
///
/// The function takes a number of cells as an optional parameter. If set,
/// will also info about the cell utilization.
pub fn analyze(model: &Model) -> AnalysisResults {
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

    let mut tree_metadata = Vec::new();

    for (i, header_idx) in model.tree_headers().enumerate() {
        let (header, nodes) = model.get_tree(header_idx);

        assert!(header_idx.is_multiple_of(2));

        let mut good_pairs = 0;
        let mut guaranteed = 0;
        let mut padding = 0;
        let mut cache_lines = 0;
        for (i, node) in nodes.iter().enumerate() {
            if node.is_padding() {
                padding += 1;
            }

            // Check pair in header
            if i == 0 && node.as_header().first_node_idx() == 1 {
                good_pairs += 1;
                guaranteed += 1;
            }
            if !i.is_multiple_of(2) {
                continue;
            }

            cache_lines += 1;

            let node = node.as_branch();

            if !node.flags().left_prediction() && node.left_ptr().get() as usize == i + 1 {
                good_pairs += 1;

                if node.flags().right_prediction() {
                    guaranteed += 1;
                }
            }

            if !node.flags().right_prediction() && node.right_ptr().get() as usize == i + 1 {
                good_pairs += 1;

                if node.flags().left_prediction() {
                    guaranteed += 1;
                }
            }
        }

        let is_cell_header = header.cache_metadata().is_cell_header();
        let max_depth = tree_max_depth(nodes, header.first_node_idx() as usize);

        let size_bytes = ByteSize::b(size_of_val(nodes) as u64);
        if is_cell_header {
            println!(" [CELL START]");
        }
        println!(
            "Tree {i} / Size: {len} nodes ({size_bytes}) / Pair utilization: {utilization:.1}% / Guaranteed 1-cycle lines: {guaranteed_utilization:.1}% / Max depth: {max_depth}",
            len = nodes.len(),
            utilization = good_pairs as f32 / cache_lines as f32 * 100.0,
            guaranteed_utilization = guaranteed as f32 / cache_lines as f32 * 100.0
        );

        tree_metadata.push(TreeMetadata {
            index: i,
            size: nodes.len(),
            depth: max_depth,
            good_pairs,
            guaranteed,
            cache_lines,
            padding,
            cell_header: is_cell_header,
        });
    }

    let max_size = tree_metadata.iter().map(|m| m.size).max().unwrap();
    let max_depth = tree_metadata.iter().map(|m| m.depth).max().unwrap();

    let size_bytes = ByteSize::b((max_size * size_of::<Node>()) as u64);
    println!();
    println!("Max tree size: {max_size} nodes ({size_bytes})",);
    println!("Max tree depth: {max_depth}\n");

    println!("/ Cell analysis /");

    let mut cell_metadata = Vec::new();
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
        let mut good_pairs = 0;
        let mut guaranteed_pairs = 0;
        let mut cache_lines = 0;
        let mut padding = 0;
        for _ in headers {
            max_depth += tree_metadata[tree_idx].depth;
            num_trees += 1;
            required_capacity += tree_metadata[tree_idx].size;
            good_pairs += tree_metadata[tree_idx].good_pairs;
            guaranteed_pairs += tree_metadata[tree_idx].guaranteed;
            cache_lines += tree_metadata[tree_idx].cache_lines;
            padding += tree_metadata[tree_idx].padding;

            tree_idx += 1;
        }

        let size_bytes = ByteSize((required_capacity * size_of::<Node>()) as u64);
        println!(
            r#"
  Cell {cell_idx}:
    {num_trees} trees
    {required_capacity} nodes ({size_bytes})
    Max depth: {max_depth}
    Cache lines: {cache_lines}, padding: {padding}
    {good_pairs} good paired nodes ({utilization:.1}% utilization)
    {guaranteed_pairs} guaranteed 1-cycle lines ({guaranteed_utilization:.1}% utilization)"#,
            utilization = good_pairs as f32 / cache_lines as f32 * 100.0,
            guaranteed_utilization = guaranteed_pairs as f32 / cache_lines as f32 * 100.0
        );

        cell_metadata.push(CellMetadata {
            index: cell_idx,
            num_trees,
            required_capacity,
            max_depth,
            good_pairs,
            guaranteed_pairs,
            cache_lines,
            padding,
        });
    }

    println!("------");

    AnalysisResults {
        num_trees: model.num_trees(),
        num_features: model.num_features().get(),
        num_targets: model.num_targets().get(),
        used_cells: model.num_cells(),
        total_nodes: model.nodes().len(),
        total_size_bytes: size_of_val(model.nodes()),
        max_tree_size: max_size,
        max_tree_depth: max_depth,
        trees: tree_metadata,
        cells: cell_metadata,
    }
}
