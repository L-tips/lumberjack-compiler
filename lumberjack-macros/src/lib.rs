use std::num::NonZeroU16;
use std::path::Path;
use std::path::PathBuf;

use lumberjack_compiler::csv_forest::CsvForest;
use lumberjack_compiler::serialize::to_bytes;
use lumberjack_model::model::{Classification, Model};
use proc_macro::TokenStream;
use proc_macro2::Literal;
use proc_macro2::Span;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use std::fmt::Debug;
use syn::Token;
use syn::parse::Parse;
use syn::parse::ParseStream;
use syn::{LitStr, parse_macro_input};

macro_rules! yeet_syn_err {
    ($exp: expr) => {
        match $exp {
            Ok(t) => t,
            Err(e) => return e.to_compile_error().into(),
        }
    };
}

#[non_exhaustive]
struct MacroInput {
    path: PathBuf,
    section: Option<String>,
}

impl Parse for MacroInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let path: LitStr = input.parse()?;
        let path = resolve_manifest_relative_path(&path)?;

        let section = if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            Some(input.parse::<LitStr>()?)
        } else {
            None
        };

        let section = section.map(|s| s.value());

        Ok(Self { path, section })
    }
}

/// Include an already-built RF moodel in the `.rforest` format as an `&'static
/// [u8]`.
#[proc_macro]
pub fn include_rf_model(input: TokenStream) -> TokenStream {
    let MacroInput { path, section, .. } = parse_macro_input!(input as MacroInput);
    let section = section.as_deref();

    let bytes = yeet_syn_err!(
        std::fs::read(&path).map_syn_err(format!("Could not read model: {}", path.display()))
    );
    let bytes_len = bytes.len();
    let link_section = section.map(|s| {
        let s = syn::LitStr::new(s, Span::call_site());
        quote! { #[unsafe(link_section = #s)] }
    });

    quote::quote! {
        {
            #link_section
            static BUF: ::lumberjack_model::BackingStorage<#bytes_len> =
                ::lumberjack_model::BackingStorage::new(#(#bytes),*);
            BUF.to_slice()
        }
    }
    .into()
}

/// Build a RF moodel from a CSV spec, and include it in the `.rforest` format
/// as an `&'static [u8]`.
#[proc_macro]
pub fn build_rf_model(input: TokenStream) -> TokenStream {
    let MacroInput { path, section, .. } = parse_macro_input!(input as MacroInput);
    let section = section.as_deref();
    yeet_syn_err!(build_rf_model_from_csv(path, section)).into()
}

/// Include feature vectors from a CSV file as a `&'static [&[bf16]]`.
#[proc_macro]
pub fn include_feat_vectors(input: TokenStream) -> TokenStream {
    let MacroInput { path, .. } = parse_macro_input!(input as MacroInput);
    let res = yeet_syn_err!(read_test_vectors(&path));

    res.into()
}

trait MapSynErr<T, E: Debug> {
    fn map_syn_err(self, message: impl AsRef<str>) -> syn::Result<T>;
}

impl<T, E: Debug> MapSynErr<T, E> for Result<T, E> {
    fn map_syn_err(self, message: impl AsRef<str>) -> syn::Result<T> {
        self.map_err(|e| syn::Error::new(Span::call_site(), format!("{}: {e:?}", message.as_ref())))
    }
}

fn build_rf_model_from_csv<P: AsRef<Path> + Clone>(
    model_path: P,
    section: Option<&str>,
) -> syn::Result<TokenStream2> {
    // Read the input file
    let serialized = CsvForest::read(model_path.clone()).map_syn_err(format!(
        "Could not read forest definition file (CSV) at {})",
        model_path.as_ref().display()
    ))?;
    let forest = serialized
        .into_forest_model()
        .map_syn_err("Could not deserialize the CSV forest")?;

    // Optimize the forest
    let nodes = forest.compile();
    let optimized = Model::new(
        forest.num_trees().try_into().unwrap(),
        &nodes,
        NonZeroU16::new(
            forest
                .num_features()
                .try_into()
                .map_syn_err("Number of forest features must fit into a u16")?,
        )
        .ok_or("Zero features")
        .map_syn_err("Number of features must be non-zero.")?,
        Classification::new(forest.num_targets().try_into().unwrap()).unwrap(),
    )
    .map_syn_err("Malformed forest")?;

    let serialized = to_bytes(&optimized);

    let bytes = serialized.iter();
    let bytes_len = bytes.len();

    let link_section = section.map(|s| {
        let s = syn::LitStr::new(s, Span::call_site());
        quote! { #[unsafe(link_section = #s)] }
    });

    Ok(quote::quote! {
        {
            #link_section
            static BUF: ::lumberjack_model::BackingStorage<#bytes_len> =
                ::lumberjack_model::BackingStorage::new([#(#bytes),*]);
            BUF.to_slice()
        }
    })
}

fn read_test_vectors(data_path: impl AsRef<Path>) -> syn::Result<TokenStream2> {
    let mut rdr = csv::ReaderBuilder::new()
        .from_path(data_path.as_ref())
        .map_syn_err(format!(
            "failed to open CSV: {}",
            data_path.as_ref().display()
        ))?;

    let mut rows: Vec<Vec<f32>> = Vec::new();

    for record in rdr.records() {
        let record = record.map_syn_err("bad CSV row")?;
        let row: Vec<f32> = record
            .iter()
            .map(|v| v.parse::<f32>().expect("invalid float"))
            .collect();
        rows.push(row);
    }

    if rows.is_empty() {
        return Err(syn::Error::new(
            Span::call_site(),
            "sample CSV must contain at least one row",
        ));
    }

    let cols = rows[0].len();
    for (idx, r) in rows.iter().enumerate() {
        if r.len() != cols {
            return Err(syn::Error::new(
                Span::call_site(),
                format!(
                    "ragged CSV not supported: row {idx} has {} cols, expected {cols}",
                    r.len()
                ),
            ));
        }
    }

    let row_tokens = rows.iter().map(|row| {
        let vals = row.iter().map(|v| Literal::f32_suffixed(*v));
        quote! {
            &[ #(::ibex_demo_system_hal::half::bf16::from_f32_const(#vals)),* ]
        }
    });

    Ok(quote! {
        &[ #(#row_tokens),* ]
    })
}

fn resolve_manifest_relative_path(lit: &LitStr) -> syn::Result<PathBuf> {
    let value = lit.value();
    let rel = Path::new(&value);

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map_syn_err("CARGO_MANIFEST_DIR is not set in proc-macro expansion environment")?;

    let mut abs = PathBuf::from(manifest_dir);
    abs.push(rel);
    Ok(abs)
}
