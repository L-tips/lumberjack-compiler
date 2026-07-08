use std::vec::Vec;

use bytesize::ByteSize;
use lumberjack_model::{
    Model,
    model::{HeadersIterator, Node},
};

// Split the model as evenly as possible between the provided number of
/// cells. The node slices can then be written to the cells' caches.
pub fn split<'a>(model: &'a Model, num_cells: usize) -> Vec<&'a [Node]> {
    let mut cells = Vec::new();

    let num_trees = model.num_trees().get() as usize;

    let base = num_trees / num_cells;
    let extra = num_trees % num_cells;

    let mut trees_iter = model.tree_headers().enumerate();

    for cell_idx in 0..num_cells {
        // Lowest cache indices get the extra trees.
        let num_trees = base + usize::from(cell_idx < extra);
        let mut headers = trees_iter.by_ref().take(num_trees).peekable();

        let start_idx = {
            let (_, start_idx) = headers.by_ref().peek().unwrap();
            *start_idx
        };
        let mut num_nodes = 0;

        for (_, header_idx) in headers {
            let slice_len = model.nodes()[header_idx].as_header().tree_len() as usize;
            num_nodes += slice_len;
        }

        cells.push(&model.nodes()[start_idx..start_idx + num_nodes]);
    }

    cells
}

/// Perform an analysis of the model, and output some useful metrics to stdout.
///
/// The function takes a number of cells as an optional parameter. If set,
/// will also info about the cell utilization.
pub fn analyze(model: &Model, num_cells: Option<usize>) {
    println!("--- Lumberjack model analysis ---");
    model
        .verify()
        .unwrap_or_else(|e| panic!("Could not verify forest: {e:?}"));

    println!(
        "Random forest model with:\n\t- {} trees\n\t- {} features\n\t- {} targets",
        model.num_trees(),
        model.num_features(),
        model.num_targets()
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
        let max_depth = tree_max_depth(nodes, header.first_node_idx() as usize);

        let size_bytes = ByteSize::b(size_of_val(nodes) as u64);
        println!(
            "/ Tree {i} / Size: {} nodes ({size_bytes}) / Max depth: {max_depth}",
            nodes.len(),
        );

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

    if let Some(c) = num_cells {
        println!("/ Cell analysis /");

        let mut tree_idx = 0;
        for (cell_idx, cell_nodes) in split(model, c).iter().enumerate() {
            let headers = HeadersIterator::new(cell_nodes);

            let mut max_depth = 0;
            let mut num_trees = 0;
            for _ in headers {
                max_depth += tree_metadata[tree_idx].depth;
                num_trees += 1;
                tree_idx += 1;
            }

            let required_capacity = cell_nodes.len();
            let size_bytes = ByteSize(std::mem::size_of_val(*cell_nodes) as u64);
            println!(
                "  Cell {cell_idx}:\n\t{num_trees} trees\n\t{required_capacity} nodes ({size_bytes})\n\tMax depth: {max_depth}"
            );
        }
    }

    println!("------");
}

/// Maximum depth of a single tree.
fn tree_max_depth(tree_nodes: &[Node], root_idx: usize) -> usize {
    let branch = tree_nodes[root_idx].as_branch();

    let left_depth = if branch.flags().left_prediction() {
        1
    } else {
        1 + tree_max_depth(tree_nodes, branch.left_ptr().get() as usize)
    };

    let right_depth = if branch.flags().right_prediction() {
        1
    } else {
        1 + tree_max_depth(tree_nodes, branch.right_ptr().get() as usize)
    };

    left_depth.max(right_depth)
}
