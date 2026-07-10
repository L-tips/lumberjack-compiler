use std::path::Path;
use std::path::PathBuf;

use lumberjack_compiler::csv_forest::CsvForest;
use lumberjack_compiler::problem::Map;
use lumberjack_model::model::Model;
use proc_macro::TokenStream;
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
struct CompileInput {
    path: PathBuf,
    section: Option<String>,
}

impl Parse for CompileInput {
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

struct VectorInput {
    vectors_path: PathBuf,
    model_path: PathBuf,
}

impl Parse for VectorInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let vectors_path = input.parse::<LitStr>()?;
        let vectors_path = resolve_manifest_relative_path(&vectors_path)?;
        input.parse::<Token![,]>()?;
        let model_path: LitStr = input.parse::<LitStr>()?;
        let model_path = resolve_manifest_relative_path(&model_path)?;

        Ok(Self {
            vectors_path,
            model_path,
        })
    }
}

trait MapSynErr<T, E: Debug> {
    fn map_syn_err<O>(self, op: O) -> syn::Result<T>
    where
        O: Fn(&E) -> String;
}

impl<T, E: Debug> MapSynErr<T, E> for Result<T, E> {
    fn map_syn_err<O>(self, op: O) -> syn::Result<T>
    where
        O: Fn(&E) -> String,
    {
        self.map_err(|e| syn::Error::new(Span::call_site(), format!("{}: {e:?}", op(&e))))
    }
}

/// Include an already-built RF moodel in the `.rforest` format as an `&'static
/// [u8]`. Note that converting a CSV model to a `.rforest` is lossy, and cannot
/// be used to generate feature and class maps.
#[proc_macro]
pub fn include_rf_model(input: TokenStream) -> TokenStream {
    let CompileInput { path, section, .. } = parse_macro_input!(input as CompileInput);
    let section = section.as_deref();

    let bytes = yeet_syn_err!(
        std::fs::read(&path).map_syn_err(|_| format!("Could not read model: {}", path.display()))
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
pub fn compile_model(input: TokenStream) -> TokenStream {
    let CompileInput { path, section, .. } = parse_macro_input!(input as CompileInput);
    let section = section.as_deref();
    yeet_syn_err!(compile_model_from_csv(path, section, 0)).into()
}

/// Build a a slice of features from a CSV forest spec as a `&'static [&str]`,
/// where each feature name is at its index in the slice.
#[proc_macro]
pub fn features_map(input: TokenStream) -> TokenStream {
    let CompileInput { path, .. } = parse_macro_input!(input as CompileInput);
    yeet_syn_err!(build_map(path, CsvForest::features)).into()
}

/// Build a a slice of targets from a CSV forest spec as a `&'static [&str]`,
/// where each target name is at its index in the slice.
#[proc_macro]
pub fn targets_map(input: TokenStream) -> TokenStream {
    let CompileInput { path, .. } = parse_macro_input!(input as CompileInput);
    yeet_syn_err!(build_map(path, CsvForest::targets)).into()
}

/// Include feature vectors with targets from a CSV file as a `&'static
/// [(&[bf16], u16)]`.
///
/// The CSV must include a column named "prediction", containing the predicted
/// class from the model.
#[proc_macro]
pub fn feat_vectors(input: TokenStream) -> TokenStream {
    let VectorInput {
        vectors_path,
        model_path,
    } = parse_macro_input!(input as VectorInput);

    let res = yeet_syn_err!(build_feat_vectors(&vectors_path, &model_path));

    res.into()
}

fn compile_model_from_csv(
    model_path: impl AsRef<Path>,
    section: Option<&str>,
    num_cells: u8,
) -> syn::Result<TokenStream2> {
    let csv_forest = read_csv_model(model_path)?;
    let forest = csv_forest
        .into_forest_model()
        .map_syn_err(|_| "Could not deserialize the CSV forest".to_owned())?;

    // Optimize the forest
    let nodes = forest
        .compile(num_cells)
        .map_syn_err(|e| format!("Could not compile model: {e:?}"))?;
    let compiled = Model::new(
        forest.num_trees().try_into().unwrap(),
        num_cells,
        &nodes,
        u16::try_from(forest.num_features())
            .unwrap()
            .try_into()
            .unwrap(),
        u16::try_from(forest.num_targets())
            .unwrap()
            .try_into()
            .unwrap(),
    )
    .map_syn_err(|_| "Malformed forest".to_owned())?;

    let serialized = compiled.serialize();

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

fn build_map(
    model_path: impl AsRef<Path>,
    f: impl Fn(&CsvForest) -> &Map,
) -> syn::Result<TokenStream2> {
    let csv_forest = read_csv_model(model_path)?;
    let mut items = f(&csv_forest).iter().collect::<Vec<_>>();
    items.sort_by_key(|(_, idx)| *idx);
    let items = items.iter().map(|f| f.0);

    Ok(quote! {
        &[
            #(#items,)*
        ]
    })
}

fn build_feat_vectors(
    vectors_path: impl AsRef<Path>,
    model_path: impl AsRef<Path>,
) -> syn::Result<TokenStream2> {
    let model = read_csv_model(model_path)?;

    let rows = model
        .problem()
        .features_vector_from_csv(vectors_path)
        .map_syn_err(|e| format!("Cannot read CSV: {e}"))?;

    let row_tokens = rows.iter().map(|data_point| {
        let vals = data_point.features.iter().map(|v| v.to_f32());
        let prediction = data_point.reference_prediction;

        quote! {
            (&[ #(::ibex_demo_system_hal::half::bf16::from_f32_const(#vals)),* ], #prediction)
        }
    });

    Ok(quote! {
        &[ #(#row_tokens),* ]
    })
}
fn resolve_manifest_relative_path(lit: &LitStr) -> syn::Result<PathBuf> {
    let value = lit.value();
    let rel = Path::new(&value);

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").map_syn_err(|_| {
        "CARGO_MANIFEST_DIR is not set in proc-macro expansion environment".to_owned()
    })?;

    let mut abs = PathBuf::from(manifest_dir);
    abs.push(rel);
    Ok(abs)
}

fn read_csv_model(model_path: impl AsRef<Path>) -> syn::Result<CsvForest> {
    CsvForest::read(model_path.as_ref()).map_syn_err(|_| {
        format!(
            "Could not read forest definition file (CSV) at {})",
            model_path.as_ref().display(),
        )
    })
}
