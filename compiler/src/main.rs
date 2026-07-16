use clap::{Parser, Subcommand};
use color_eyre::{
    Result,
    eyre::{Context, eyre},
};
use lumberjack_compiler::{
    PlacementStrategy,
    compiled_model::{self},
    compiler::PartitionStrategy,
    csv_source::{
        CsvSource, compile_from_csv, compile_split_caches_from_csv, write_analysis_results,
    },
    feature_vectors::{features_vector_from_csv, write_test_vectors},
};
use lumberjack_model::Model;
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};
use yaml_serde::{Mapping, Value};

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
    /// Compile a .ljmodel from a CSV definition
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

        /// Write model analysis results to file
        #[arg(short = 'y', long = "write-results")]
        write_analysis: Option<PathBuf>,

        /// Node placement strategy (default: execution-aware)
        #[arg(short = 's', long = "placement-strategy", value_name = "STRATEGY")]
        placement_strategy: Option<PlacementStrategy>,

        /// Cell cache partitioning strategy (default: Equal)
        #[arg(short = 'r', long = "partition-strategy", value_name = "STRATEGY")]
        partition_strategy: Option<PartitionStrategy>,
    },

    /// Compile a Lumberjack model from a CSV definition, splitting it into
    /// per-cell cache .ljcache files
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

    /// Compile .ljdata test vectors from a CSV model file, and a CSV test
    /// vectors file
    #[command(arg_required_else_help = true)]
    BuildTestVectors {
        /// Input model to compile (CSV)
        model_path: PathBuf,

        #[arg(short = 'v', long = "vectors", value_name = "VECTORS_INPUT")]
        vectors_path: PathBuf,

        #[arg(short = 'o', long = "output", value_name = "OUTPUT_FILE")]
        output: PathBuf,
    },

    /// Analyze a compiled .ljmodel model
    #[command(arg_required_else_help = true)]
    Analyze {
        /// Path to a compiled model to analyze
        model: PathBuf,

        /// Write model analysis results to file
        #[arg(short = 'y', long = "write-results")]
        write_analysis: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    color_eyre::install()?;
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")))
        .init();

    let args = Cli::parse();

    match args.command {
        Command::Build {
            input,
            output,
            analyze,
            num_cells,
            placement_strategy,
            partition_strategy,
            write_analysis,
        } => {
            compile_from_csv(
                input,
                output,
                num_cells,
                analyze,
                placement_strategy.unwrap_or_default(),
                partition_strategy.unwrap_or_default(),
                write_analysis,
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

        Command::BuildTestVectors {
            model_path,
            vectors_path,
            output,
        } => {
            let serialized = CsvSource::read(model_path)
                .context("Could not read forest definition file (CSV).")?;
            let model = serialized.lower_to_ir()?.quantize_splits();

            let vectors = features_vector_from_csv(model.problem(), vectors_path)
                .context("Could not open feature vectors file")?;

            write_test_vectors(&model, &vectors, output)?;
        }

        Command::Analyze {
            model: model_path,
            write_analysis,
        } => {
            let file = std::fs::File::open(&model_path)
                .context(format!("Could not open file: {}", model_path.display()))?;
            let mut file = BufReader::new(file);
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            let model = Model::deserialize(&buf)
                .map_err(|e| eyre!("Could not deserialize compiled model: {e:?}"))?;

            let analysis_results = compiled_model::analyze(&model);
            if let Some(p) = write_analysis {
                let extra_data = [("model_path", format!("{}", model_path.display()))]
                    .into_iter()
                    .map(|(k, v)| (Value::from(k), Value::from(v)))
                    .collect::<Mapping>();

                let extra_data = yaml_serde::Value::Mapping(extra_data);
                write_analysis_results(p, &analysis_results, Some(extra_data))?;
            }
        }
    }

    Ok(())
}
