use clap::Parser;
use color_eyre::Result;
use lumberjack_compiler::csv_forest::compile_from_csv;

use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Input file
    #[arg(short = 'i', long = "input", value_name = "INPUT_FILE")]
    input: PathBuf,

    /// Output file
    #[arg(short = 'o', long = "output", value_name = "OUTPUT_FILE")]
    output: PathBuf,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Cli::parse();

    compile_from_csv(args.input, args.output)
}
