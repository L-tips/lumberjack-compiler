use clap::{Parser, Subcommand};
use color_eyre::{
    Result,
    eyre::{Context, eyre},
};
use lumberjack_compiler::csv_forest::compile_from_csv;
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
    #[command(arg_required_else_help = true)]
    Build {
        /// Input model to compile (CSV)
        input: PathBuf,

        /// Path where the compiled model will be written to
        #[arg(short = 'o', long = "output", value_name = "OUTPUT_FILE")]
        output: PathBuf,

        /// Also output analysis information about the compiled model to stdout
        #[arg(short = 'a', long = "analyze")]
        analyze: bool,

        /// Also output analysis information about the compiled model to stdout,
        /// with additional information related to the provided number of cells
        /// in the system.
        #[arg(short = 'c', long = "analyze-cells")]
        analyze_cells: Option<usize>,
    },

    #[command(arg_required_else_help = true)]
    Analyze {
        /// Path to a compiled model to analyze
        model: PathBuf,

        /// Print additional information with the given number of cells in the
        /// system
        #[arg(short = 'c', long = "cells")]
        num_cells: Option<usize>,
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
            analyze_cells,
        } => {
            let mut analysis_info = if analyze { Some(None) } else { None };
            if let Some(cell_info) = analyze_cells {
                analysis_info = Some(Some(cell_info));
            }

            compile_from_csv(input, output, analysis_info)?;
        }
        Command::Analyze { model, num_cells } => {
            let file = std::fs::File::open(&model)
                .context(format!("Could not open file: {}", model.display()))?;
            let mut file = BufReader::new(file);
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            let model = Model::deserialize(&buf)
                .map_err(|e| eyre!("Could not deserialize compiled model: {e:?}"))?;

            model.analyze(num_cells);
        }
    }

    Ok(())
}
