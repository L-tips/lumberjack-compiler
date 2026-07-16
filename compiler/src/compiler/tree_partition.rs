use std::fmt;

use lumberjack_model::model::Node;
use rand::seq::SliceRandom;

/// Cell partitioning strategy
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    clap::ValueEnum,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum PartitionStrategy {
    #[default]
    EqualRandom,
    EqualSorted,
    Greedy,
}

impl PartitionStrategy {
    fn partition_fn(&self) -> impl Fn(&[Vec<Node>], u8) -> Vec<Vec<Vec<Node>>> {
        match self {
            Self::Greedy => partition_greedy_search,
            Self::EqualRandom => partition_equal_random,
            Self::EqualSorted => partition_equal_sorted,
        }
    }

    pub(crate) fn partition(&self, forest: &[Vec<Node>], num_cells: u8) -> Vec<Vec<Vec<Node>>> {
        self.partition_fn()(forest, num_cells)
    }
}

impl fmt::Display for PartitionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PartitionStrategy::EqualRandom => "equal-random",
            PartitionStrategy::EqualSorted => "equal-sorted",
            PartitionStrategy::Greedy => "greedy",
        };
        f.write_str(s)
    }
}

pub fn partition_equal_random(forest: &[Vec<Node>], num_cells: u8) -> Vec<Vec<Vec<Node>>> {
    let num_cells = num_cells as usize;

    // Avoid division by zero: first cache holds all the trees
    if num_cells <= 1 {
        return vec![forest.to_vec()];
    }

    // Sort trees in random order
    let mut forest = forest.to_vec();
    let mut rng = rand::rng();
    forest.shuffle(&mut rng);

    let n = forest.len();
    let base = n / num_cells;
    let extra = n % num_cells;

    let mut result = Vec::with_capacity(num_cells);
    let mut idx = 0;

    for i in 0..num_cells {
        let size = base + usize::from(i < extra);
        result.push(forest[idx..idx + size].to_vec());
        idx += size;
    }

    result
}

pub fn partition_equal_sorted(forest: &[Vec<Node>], num_cells: u8) -> Vec<Vec<Vec<Node>>> {
    let num_cells = num_cells as usize;

    // Avoid division by zero: first cache holds all the trees
    if num_cells <= 1 {
        return vec![forest.to_vec()];
    }

    // Sort trees by depth ascending
    let mut indexed: Vec<(usize, usize)> = forest
        .iter()
        .enumerate()
        // First node in tree is at index 2
        .map(|(i, tree)| {
            let first_node_idx = tree[0].as_header().first_node_idx();
            (i, tree_max_depth(tree.as_slice(), first_node_idx as usize))
        })
        .collect();
    indexed.sort_unstable_by_key(|(_, depth)| *depth);

    let n = forest.len();
    let base = n / num_cells;
    let extra = n % num_cells;

    let mut result = Vec::with_capacity(num_cells);
    let mut idx = 0;

    for i in 0..num_cells {
        let size = base + usize::from(i < extra);
        result.push(forest[idx..idx + size].to_vec());
        idx += size;
    }

    result
}

pub fn partition_greedy_search(forest: &[Vec<Node>], num_cells: u8) -> Vec<Vec<Vec<Node>>> {
    let num_cells = num_cells as usize;

    // Avoid division by zero: first cache holds all the trees
    if num_cells <= 1 {
        return vec![forest.to_vec()];
    }

    // Sort trees by depth descending for tighter greedy packing
    let mut indexed: Vec<(usize, usize)> = forest
        .iter()
        .enumerate()
        // First node in tree is at index 2
        .map(|(i, tree)| {
            let first_node_idx = tree[0].as_header().first_node_idx();
            (i, tree_max_depth(tree.as_slice(), first_node_idx as usize))
        })
        .collect();
    indexed.sort_unstable_by_key(|(_, depth)| std::cmp::Reverse(*depth));

    let depths: Vec<usize> = indexed.iter().map(|(_, d)| *d).collect();

    // Binary search on the maximum allowed sum S
    let lo = *depths.iter().max().unwrap_or(&0);
    let hi: usize = depths.iter().sum();

    let can_fit = |max_sum: usize| -> bool {
        let mut groups = 1usize;
        let mut current = 0usize;
        for &d in &depths {
            if current + d > max_sum {
                groups += 1;
                current = d;
            } else {
                current += d;
            }
        }
        groups <= num_cells
    };

    let optimal_max = {
        let mut lo = lo;
        let mut hi = hi;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if can_fit(mid) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    };

    // Greedily assign trees to partitions using the optimal max sum
    let mut partitions: Vec<Vec<Vec<Node>>> = vec![Vec::new(); num_cells];
    let mut partition_sums = vec![0usize; num_cells];
    let mut current_partition = 0usize;

    for (orig_idx, depth) in &indexed {
        if current_partition + 1 < num_cells
            && partition_sums[current_partition] + depth > optimal_max
        {
            current_partition += 1;
        }
        partition_sums[current_partition] += depth;
        partitions[current_partition].push(forest[*orig_idx].clone());
    }

    partitions
}

pub(crate) fn tree_max_depth(tree_nodes: &[Node], root_idx: usize) -> usize {
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
