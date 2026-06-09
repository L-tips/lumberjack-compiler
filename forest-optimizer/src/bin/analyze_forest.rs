use std::mem::size_of_val;
use std::path::{Path, PathBuf};

use clap::Parser;
use color_eyre::Result;
use color_eyre::eyre::{Context, eyre};

use embedded_rforest::forest::{Classification, OptimizedForest};
use forest_optimizer::csv_forest::CsvForest;
use forest_optimizer::forest::{Forest, Node};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Input file
    #[arg(short = 'i', long = "input", value_name = "INPUT_FILE")]
    input: PathBuf,

    /// Print forest
    #[arg(long = "print")]
    print: bool,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Cli::parse();
    analyze_classification(args.input, args.print)
}

fn analyze_classification(input: impl AsRef<Path>, print: bool) -> Result<()> {
    let serialized = CsvForest::read(&input).context("Could not read forest definition file.")?;
    let forest = Forest::from_serialized(serialized)?;

    let mut branch_cnt = 0;
    let mut leaf_cnt = 0;
    for n in forest.nodes() {
        if matches!(n, Node::Branch(_)) {
            branch_cnt += 1;
        } else {
            leaf_cnt += 1;
        }
    }

    println!("Forest is a CLASSIFICATION problem.\n\n");

    let forest_len = forest.nodes().len();
    println!(
        "--- Unoptimized forest ---\nTotal length: {} | Branches: {} , leaves: {} | Size: {} bytes\n--------------------------\n\n",
        forest_len,
        branch_cnt,
        leaf_cnt,
        size_of_val(forest.nodes())
    );

    if print {
        println!("Forest: {:?}", forest)
    };

    let optimized_nodes = forest.optimize_nodes();
    let optimized = OptimizedForest::new(
        forest.num_trees().try_into().unwrap(),
        &optimized_nodes,
        forest.num_features().try_into().unwrap(),
        Classification::new(forest.num_targets().try_into().unwrap()).unwrap(),
    )
    .map_err(|_| eyre!("Malformed forest"))?;

    let optimized_len = optimized.nodes().len();

    let serialized = optimized.to_bytes();
    let ptr = serialized.as_ptr();
    assert!((ptr as usize).is_multiple_of(align_of::<OptimizedForest>()));

    println!(
        "--- Optimized forest ---\nTotal length: {} | Branches: {} , leaves: {} | Size: {}\n--------------------------\n\n",
        optimized_len,
        optimized_len,
        0,
        serialized.len()
    );

    let pruned = (forest_len as f32 - optimized_len as f32) / (forest_len as f32);
    println!(
        "--- Analysis results ---\nPruned {:.2}%, Kept {:.2}%\n--------------------------\n\n",
        pruned * 100.0,
        (1.0 - pruned) * 100.0,
    );

    let _deserialized = OptimizedForest::<Classification>::deserialize(&serialized);

    Ok(())
}
