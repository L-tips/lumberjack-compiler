use clap::{Parser, ValueEnum};
use color_eyre::Result;
use forest_optimizer::write_forest::{write_classification, write_regression};

use std::path::PathBuf;

/// Modes for the application
#[derive(Debug, Clone, ValueEnum)]
enum ProblemType {
    Classification,
    Regression,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Input file
    #[arg(short = 'i', long = "input", value_name = "INPUT_FILE")]
    input: PathBuf,

    /// Output file
    #[arg(short = 'o', long = "output", value_name = "OUTPUT_FILE")]
    output: PathBuf,

    /// Problem type
    #[arg(short = 'p', long = "problem-type", value_enum)]
    problem_type: ProblemType,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Cli::parse();

    match args.problem_type {
        ProblemType::Classification => write_classification(args.input, args.output),
        ProblemType::Regression => write_regression(args.input, args.output),
    }
}
