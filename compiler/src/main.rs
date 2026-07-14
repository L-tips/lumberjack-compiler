use clap::{Parser, Subcommand};
use color_eyre::{
    Result,
    eyre::{Context, eyre},
};
use lumberjack_compiler::{
    PlacementStrategy, compiled_model,
    compiler::PartitionStrategy,
    csv_source::{compile_from_csv, compile_split_caches_from_csv},
};
use lumberjack_model::Model;

use std::{
    io::{BufReader, Read},
    path::PathBuf,
};

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Compile a .ljmodelfrom a CSV definition
    #[command(arg_required_else_help = true)]
    Build {
        /// Input model to compile (CSV)
        input: PathBuf,

        /// Path where the compiled model will be written to. If not set, no
        /// file will be written.
        #[arg(short = 'o', long = "output", value_name = "OUTPUT_FILE")]
        output: Option<PathBuf>,

        /// Number of tree cells for which to compile the model
        #[arg(short = 'c', long = "num-cells")]
        num_cells: u8,

        /// Also output analysis information about the compiled model to stdout
        #[arg(short = 'a', long = "analyze")]
        analyze: bool,

        /// Node placement strategy (default: execution-aware)
        #[arg(short = 's', long = "placement-strategy", value_name = "STRATEGY")]
        placement_strategy: Option<PlacementStrategy>,

        /// Cell cache partitioning strategy (default: Equal)
        #[arg(short = 'r', long = "partition-strategy", value_name = "STRATEGY")]
        partition_strategy: Option<PartitionStrategy>,
    },

    /// Compile a Lumberjack model from a CSV definition, splitting it into
    /// per-cell cache .lj_data files
    #[command(arg_required_else_help = true)]
    BuildCacheData {
        /// Input model to compile (CSV)
        input: PathBuf,

        /// Directory in which to write the cache data files
        #[arg(short = 'd', long = "output-dir", value_name = "OUT_DIR")]
        dir: PathBuf,

        /// Prefix to name the cache data files
        #[arg(short = 'p', long = "prefix", value_name = "PREFIX")]
        prefix: Option<String>,

        /// Number of tree cells for which to compile the model
        #[arg(short = 'c', long = "num-cells")]
        num_cells: u8,

        /// Also output analysis information about the compiled model to stdout
        #[arg(short = 'a', long = "analyze")]
        analyze: bool,

        /// Node placement strategy (default: execution-aware)
        #[arg(short = 's', long = "placement-strategy", value_name = "STRATEGY")]
        placement_strategy: Option<PlacementStrategy>,

        /// Cell cache partitioning strategy (default: Equal)
        #[arg(short = 'r', long = "partition-strategy", value_name = "STRATEGY")]
        partition_strategy: Option<PartitionStrategy>,
    },

    /// Analyze a compiled .ljmodel model
    #[command(arg_required_else_help = true)]
    Analyze {
        /// Path to a compiled model to analyze
        model: PathBuf,
    },
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Cli::parse();

    match args.command {
        Command::Build {
            input,
            output,
            analyze,
            num_cells,
            placement_strategy,
            partition_strategy,
        } => {
            compile_from_csv(
                input,
                output,
                num_cells,
                analyze,
                placement_strategy.unwrap_or_default(),
                partition_strategy.unwrap_or_default(),
            )?;
        }

        Command::BuildCacheData {
            input,
            dir,
            prefix,
            num_cells,
            analyze,
            placement_strategy,
            partition_strategy,
        } => {
            compile_split_caches_from_csv(
                input,
                dir,
                prefix,
                num_cells,
                analyze,
                placement_strategy.unwrap_or_default(),
                partition_strategy.unwrap_or_default(),
            )?;
        }

        Command::Analyze { model } => {
            let file = std::fs::File::open(&model)
                .context(format!("Could not open file: {}", model.display()))?;
            let mut file = BufReader::new(file);
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            let model = Model::deserialize(&buf)
                .map_err(|e| eyre!("Could not deserialize compiled model: {e:?}"))?;

            compiled_model::analyze(&model);
        }
    }

    Ok(())
}
