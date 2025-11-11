//! TensorZero macros

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::parse_macro_input;
use syn::punctuated::Punctuated;
use syn::DeriveInput;
use syn::Expr;
use syn::ExprLit;
use syn::Fields;
use syn::Lit;
use syn::Meta;
use syn::Token;

struct TagData {
    tag: String,
    serde_attrs: Vec<syn::Attribute>,
}

/// Finds the attribute `#[serde(tag = "...")]` on the target enum,
/// and collect all *other* `#[serde]` attributes from the enum.
/// We forward everything except `#[serde(tag = "...")]` to the new enum,
/// so that options like `#[serde(default)]` work correctly.
/// Note that `#[serde(tag = "...")]` must be a standalone attribute,
/// and not part of another attribute like `#[serde(rename_all = "snake_case", tag = "...")]`.
fn extract_tag(input: &DeriveInput) -> Result<TagData, syn::Error> {
    // We want to forward Serde attributes to the new enum, so that things like
    // `#[serde(default)]` work correctly.
    let initial_serde_attrs = input
        .attrs
        .clone()
        .into_iter()
        .filter(|attr| attr.path().is_ident("serde"))
        .collect::<Vec<_>>();

    let mut serde_attrs = Vec::with_capacity(initial_serde_attrs.len());

    let mut tag = None;
    for attr in initial_serde_attrs {
        let nested = attr
            .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            .ok();
        if let Some(nested) = nested {
            let nested_tag = nested.iter().find_map(|meta| {
                if let Meta::NameValue(name_val) = meta {
                    if name_val.path.is_ident("tag") {
                        Some(name_val.value.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            });
            if nested_tag.is_some() {
                // We reject code like `#[serde(rename_all = "snake_case", tag = "...")]` to simplify our
                // macro implementation
                if nested.len() != 1 {
                    return Err(syn::Error::new_spanned(
                        attr,
                        "TensorZeroDeserialize: Please split `#[serde(tag = \"...\")]` into its own attribute",
                    ));
                }
                tag = nested_tag;
                // Don't apply the `#[serde(tag = "...")]` attribute to the new enum
                continue;
            }
        }
        serde_attrs.push(attr);
    }
    let tag = tag.ok_or_else(|| {
        syn::Error::new(
            input.ident.span(),
            "TensorZeroDeserialize: Missing #[serde(tag = \"...\")] attribute",
        )
    })?;
    match tag {
        Expr::Lit(ExprLit {
            lit: Lit::Str(tag), ..
        }) => Ok(TagData {
            tag: tag.value(),
            serde_attrs,
        }),
        _ => Err(syn::Error::new_spanned(
            tag,
            "TensorZeroDeserialize: #[serde(tag = \"...\")] attribute must be a string literal",
        )),
    }
}

#[proc_macro_derive(TensorZeroDeserialize, attributes(serde))]
pub fn tensorzero_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let input_span = input.ident.span();

    let syn::Data::Enum(data) = &input.data else {
        return syn::Error::new(input_span, "TensorZeroDeserialize only supports enums")
            .to_compile_error()
            .into();
    };

    let ident = input.ident.clone();
    // We're going to copy the input enum to a new enum, which will use
    // an externally tagged enum representation `{"variant_name": {"my": "fields"}}`,
    // which we'll deserialize into using `serde_path_to_error`. The externally
    // tagged representation allows Serde to avoid buffering the entire value internally,
    // allowing the `serde_path_to_error` deserializer to get passed along and track
    // the path to the error.
    let new_ident = syn::Ident::new(
        &format!("__TensorZeroDerive_{}", input.ident),
        proc_macro2::Span::call_site(),
    );

    let TagData { tag, serde_attrs } = match extract_tag(&input) {
        Ok(tag) => tag,
        Err(e) => return e.to_compile_error().into(),
    };

    // Remove all non-`#[serde]` attributes from the variant - these might reference
    // other derive macros (e.g. `#[strum]`0. We're not applying any derive macros
    // to our new enum, so we need to avoid copying over other attributes to prevent errors.
    let mut stripped_variants = data.variants.clone();
    for variant in &mut stripped_variants {
        variant.attrs.retain(|attr| attr.path().is_ident("serde"));
        for field in &mut variant.fields {
            field.attrs.retain(|attr| attr.path().is_ident("serde"));
        }
    }

    // Build up match arms that looks like:
    // MyEnum::MyVariant(field1, field2) => __TensorZeroDerive_::MyVariant(field1, field2)
    let match_arms = stripped_variants.iter().map(|variant| {
        let variant_ident = &variant.ident;
        match &variant.fields {
            Fields::Named(_) => {
                let field_idents = variant.fields.clone().into_iter().map(|f| f.ident);
                let field_idents_clone = field_idents.clone();
                quote! {
                    #new_ident::#variant_ident { #(#field_idents),* } => Ok(#ident::#variant_ident { #(#field_idents_clone),* })
                }
            }
            Fields::Unnamed(fields) => {
                let field_idents = fields.unnamed.iter().enumerate().map(|(i, field)| {
                    syn::Ident::new(&format!("field_{i}"), field.ident.as_ref().map(syn::Ident::span).unwrap_or_else(Span::call_site))
                });
                let field_idents_clone = field_idents.clone();
                quote! {
                    #new_ident::#variant_ident(#(#field_idents),*) => Ok(#ident::#variant_ident(#(#field_idents_clone),*))
                }
            }
            Fields::Unit => {
                quote! {
                    #new_ident::#variant_ident => Ok(#ident::#variant_ident)
                }
            }
        }
    });

    let tag_err = format!("TensorZeroDerive: missing tag field `{tag}`");

    let res = quote! {
        #[derive(::serde::Deserialize)]
        #(#serde_attrs)*
        enum #new_ident {
            #stripped_variants
        }

        impl<'de> ::serde::Deserialize<'de> for #ident {
            fn deserialize<D>(de: D) -> Result<Self, D::Error>
            where
                D: ::serde::Deserializer<'de>,
            {
                use ::std::error::Error;
                use ::serde_json::{json, Value};
                use ::serde::Deserialize;
                use ::serde::de::Unexpected;

                // First, deserialize into a serde_json::Value, so that we can
                // extract that tag field.
                let mut value: Value = Deserialize::deserialize(de)?;
                let Some(obj) = value.as_object_mut() else {
                    let unexpected = match &value {
                        Value::Null => Unexpected::Unit,
                        Value::Bool(b) => Unexpected::Bool(*b),
                        Value::Number(n) => Unexpected::Other("number"),
                        Value::String(s) => Unexpected::Str(s),
                        Value::Array(_) => Unexpected::Seq,
                        Value::Object(_) => Unexpected::Map,

                    };
                    return Err(serde::de::Error::invalid_type(unexpected, &"object"));
                };
                let tag = obj.remove(#tag).and_then(|v| v.as_str().map(|v| v.to_owned())).ok_or_else(|| {
                    serde::de::Error::custom(#tag_err)
                })?;

                // Now, build an externally-tagged enum, mapping the tag field to the rest of the original map
                // e.g `{"type": "foo", "my_field": "Bar"}` is mapped to `{"foo": {"my_field": "Bar"}}`
                let modified = json!({
                    tag: value
                });
                // Deserialize the modified value into our copied enum type,
                // which uses an externally-tagged representation.
                let val: #new_ident = ::serde_path_to_error::deserialize(modified).map_err(|e| {
                    // On error, extract the path, skipping the first component (the outer tag field).'
                    // Ideally, we would build up a Vec containing all of the error path components,
                    // but this would a custom top-level deserializer and/or some tricky thread-local variables.
                    let path = e.path().iter().skip(1).map(|s| s.to_string()).collect::<Vec<_>>().join(".");
                    if path.is_empty() {
                        serde::de::Error::custom(e.into_inner())
                    } else {
                        serde::de::Error::custom(format!("{}: {}", path, e.into_inner()))
                    }
                })?;
                match val {
                    #(#match_arms),*
                }
            }
        }
    };
    res.into()
}

/// Attribute macro that exports `schemars::JsonSchema` definitions when tests run.
///
/// The macro generates a `#[test]` that serializes the annotated type's schema and
/// writes it to `clients/schemas/<TypeName>.json`, similar to how `ts-rs`'s
/// `#[ts(export)]` produces TypeScript definitions.
///
/// ## Usage
///
/// ```ignore
/// use schemars::JsonSchema;
/// use tensorzero_derive::export_schema;
///
/// #[derive(JsonSchema)]
/// #[export_schema]
/// pub struct MyType {
///     field: String,
/// }
/// ```
///
/// Running `cargo test` will create or update `clients/schemas/MyType.json`.
#[proc_macro_attribute]
pub fn export_schema(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let name = &input.ident;
    let name_str = name.to_string();

    // Generate a unique test name based on the type name
    let test_name = syn::Ident::new(
        &format!("export_schema_{}", name_str.to_lowercase()),
        name.span(),
    );

    // Generate the test function
    let expanded = quote! {
        // Keep the original item
        #input

        // Generate a test that exports the schema
        #[cfg(test)]
        #[test]
        fn #test_name() {
            use schemars::generate::SchemaSettings;
            use std::fs;
            use std::path::{Path, PathBuf};
            use std::borrow::Cow;

            // Get the schema for this type
            let settings = SchemaSettings::default().with_transform(schemars::transform::RemoveRefSiblings::default());
            let generator = settings.into_generator();
            let schema = generator.into_root_schema_for::<#name>();

            // Serialize to pretty JSON
            let json = serde_json::to_string_pretty(&schema)
                .expect(&format!("Failed to serialize schema for {}", #name_str));

            let schema_output_dir = match std::env::var("SCHEMA_RS_EXPORT_DIR") {
                Err(..) => Cow::Borrowed(Path::new("./schemas")),
                Ok(dir) => Cow::Owned(PathBuf::from(dir)),
            };

            fs::create_dir_all(&schema_output_dir)
                .expect(&format!("Failed to create schemas directory: {}", schema_output_dir.display()));

            // Write the schema file
            let file_path = schema_output_dir.join(format!("{}.json", #name_str));
            fs::write(&file_path, json)
                .expect(&format!("Failed to write schema for {}", #name_str));

            println!("✓ Generated schema: {}", file_path.display());
        }
    };

    TokenStream::from(expanded)
}
