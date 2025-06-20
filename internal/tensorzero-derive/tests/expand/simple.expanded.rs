#[serde(tag = "type")]
enum MyEnum {
    UnitVariant,
    StructVariant { field1: i32, field2: bool },
    TupleVariant(i32, bool),
}
enum __TensorZeroDerive_MyEnum {
    UnitVariant,
    StructVariant { field1: i32, field2: bool },
    TupleVariant(i32, bool),
}
#[doc(hidden)]
#[allow(
    non_upper_case_globals,
    unused_attributes,
    unused_qualifications,
    clippy::absolute_paths,
)]
const _: () = {
    #[allow(unused_extern_crates, clippy::useless_attribute)]
    extern crate serde as _serde;
    #[automatically_derived]
    impl<'de> _serde::Deserialize<'de> for __TensorZeroDerive_MyEnum {
        fn deserialize<__D>(
            __deserializer: __D,
        ) -> _serde::__private::Result<Self, __D::Error>
        where
            __D: _serde::Deserializer<'de>,
        {
            #[allow(non_camel_case_types)]
            #[doc(hidden)]
            enum __Field {
                __field0,
                __field1,
                __field2,
            }
            #[doc(hidden)]
            struct __FieldVisitor;
            #[automatically_derived]
            impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                type Value = __Field;
                fn expecting(
                    &self,
                    __formatter: &mut _serde::__private::Formatter,
                ) -> _serde::__private::fmt::Result {
                    _serde::__private::Formatter::write_str(
                        __formatter,
                        "variant identifier",
                    )
                }
                fn visit_u64<__E>(
                    self,
                    __value: u64,
                ) -> _serde::__private::Result<Self::Value, __E>
                where
                    __E: _serde::de::Error,
                {
                    match __value {
                        0u64 => _serde::__private::Ok(__Field::__field0),
                        1u64 => _serde::__private::Ok(__Field::__field1),
                        2u64 => _serde::__private::Ok(__Field::__field2),
                        _ => {
                            _serde::__private::Err(
                                _serde::de::Error::invalid_value(
                                    _serde::de::Unexpected::Unsigned(__value),
                                    &"variant index 0 <= i < 3",
                                ),
                            )
                        }
                    }
                }
                fn visit_str<__E>(
                    self,
                    __value: &str,
                ) -> _serde::__private::Result<Self::Value, __E>
                where
                    __E: _serde::de::Error,
                {
                    match __value {
                        "UnitVariant" => _serde::__private::Ok(__Field::__field0),
                        "StructVariant" => _serde::__private::Ok(__Field::__field1),
                        "TupleVariant" => _serde::__private::Ok(__Field::__field2),
                        _ => {
                            _serde::__private::Err(
                                _serde::de::Error::unknown_variant(__value, VARIANTS),
                            )
                        }
                    }
                }
                fn visit_bytes<__E>(
                    self,
                    __value: &[u8],
                ) -> _serde::__private::Result<Self::Value, __E>
                where
                    __E: _serde::de::Error,
                {
                    match __value {
                        b"UnitVariant" => _serde::__private::Ok(__Field::__field0),
                        b"StructVariant" => _serde::__private::Ok(__Field::__field1),
                        b"TupleVariant" => _serde::__private::Ok(__Field::__field2),
                        _ => {
                            let __value = &_serde::__private::from_utf8_lossy(__value);
                            _serde::__private::Err(
                                _serde::de::Error::unknown_variant(__value, VARIANTS),
                            )
                        }
                    }
                }
            }
            #[automatically_derived]
            impl<'de> _serde::Deserialize<'de> for __Field {
                #[inline]
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    _serde::Deserializer::deserialize_identifier(
                        __deserializer,
                        __FieldVisitor,
                    )
                }
            }
            #[doc(hidden)]
            struct __Visitor<'de> {
                marker: _serde::__private::PhantomData<__TensorZeroDerive_MyEnum>,
                lifetime: _serde::__private::PhantomData<&'de ()>,
            }
            #[automatically_derived]
            impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                type Value = __TensorZeroDerive_MyEnum;
                fn expecting(
                    &self,
                    __formatter: &mut _serde::__private::Formatter,
                ) -> _serde::__private::fmt::Result {
                    _serde::__private::Formatter::write_str(
                        __formatter,
                        "enum __TensorZeroDerive_MyEnum",
                    )
                }
                fn visit_enum<__A>(
                    self,
                    __data: __A,
                ) -> _serde::__private::Result<Self::Value, __A::Error>
                where
                    __A: _serde::de::EnumAccess<'de>,
                {
                    match _serde::de::EnumAccess::variant(__data)? {
                        (__Field::__field0, __variant) => {
                            _serde::de::VariantAccess::unit_variant(__variant)?;
                            _serde::__private::Ok(__TensorZeroDerive_MyEnum::UnitVariant)
                        }
                        (__Field::__field1, __variant) => {
                            #[allow(non_camel_case_types)]
                            #[doc(hidden)]
                            enum __Field {
                                __field0,
                                __field1,
                                __ignore,
                            }
                            #[doc(hidden)]
                            struct __FieldVisitor;
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                type Value = __Field;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private::Formatter,
                                ) -> _serde::__private::fmt::Result {
                                    _serde::__private::Formatter::write_str(
                                        __formatter,
                                        "field identifier",
                                    )
                                }
                                fn visit_u64<__E>(
                                    self,
                                    __value: u64,
                                ) -> _serde::__private::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        0u64 => _serde::__private::Ok(__Field::__field0),
                                        1u64 => _serde::__private::Ok(__Field::__field1),
                                        _ => _serde::__private::Ok(__Field::__ignore),
                                    }
                                }
                                fn visit_str<__E>(
                                    self,
                                    __value: &str,
                                ) -> _serde::__private::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        "field1" => _serde::__private::Ok(__Field::__field0),
                                        "field2" => _serde::__private::Ok(__Field::__field1),
                                        _ => _serde::__private::Ok(__Field::__ignore),
                                    }
                                }
                                fn visit_bytes<__E>(
                                    self,
                                    __value: &[u8],
                                ) -> _serde::__private::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        b"field1" => _serde::__private::Ok(__Field::__field0),
                                        b"field2" => _serde::__private::Ok(__Field::__field1),
                                        _ => _serde::__private::Ok(__Field::__ignore),
                                    }
                                }
                            }
                            #[automatically_derived]
                            impl<'de> _serde::Deserialize<'de> for __Field {
                                #[inline]
                                fn deserialize<__D>(
                                    __deserializer: __D,
                                ) -> _serde::__private::Result<Self, __D::Error>
                                where
                                    __D: _serde::Deserializer<'de>,
                                {
                                    _serde::Deserializer::deserialize_identifier(
                                        __deserializer,
                                        __FieldVisitor,
                                    )
                                }
                            }
                            #[doc(hidden)]
                            struct __Visitor<'de> {
                                marker: _serde::__private::PhantomData<
                                    __TensorZeroDerive_MyEnum,
                                >,
                                lifetime: _serde::__private::PhantomData<&'de ()>,
                            }
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                type Value = __TensorZeroDerive_MyEnum;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private::Formatter,
                                ) -> _serde::__private::fmt::Result {
                                    _serde::__private::Formatter::write_str(
                                        __formatter,
                                        "struct variant __TensorZeroDerive_MyEnum::StructVariant",
                                    )
                                }
                                #[inline]
                                fn visit_seq<__A>(
                                    self,
                                    mut __seq: __A,
                                ) -> _serde::__private::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::SeqAccess<'de>,
                                {
                                    let __field0 = match _serde::de::SeqAccess::next_element::<
                                        i32,
                                    >(&mut __seq)? {
                                        _serde::__private::Some(__value) => __value,
                                        _serde::__private::None => {
                                            return _serde::__private::Err(
                                                _serde::de::Error::invalid_length(
                                                    0usize,
                                                    &"struct variant __TensorZeroDerive_MyEnum::StructVariant with 2 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field1 = match _serde::de::SeqAccess::next_element::<
                                        bool,
                                    >(&mut __seq)? {
                                        _serde::__private::Some(__value) => __value,
                                        _serde::__private::None => {
                                            return _serde::__private::Err(
                                                _serde::de::Error::invalid_length(
                                                    1usize,
                                                    &"struct variant __TensorZeroDerive_MyEnum::StructVariant with 2 elements",
                                                ),
                                            );
                                        }
                                    };
                                    _serde::__private::Ok(__TensorZeroDerive_MyEnum::StructVariant {
                                        field1: __field0,
                                        field2: __field1,
                                    })
                                }
                                #[inline]
                                fn visit_map<__A>(
                                    self,
                                    mut __map: __A,
                                ) -> _serde::__private::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::MapAccess<'de>,
                                {
                                    let mut __field0: _serde::__private::Option<i32> = _serde::__private::None;
                                    let mut __field1: _serde::__private::Option<bool> = _serde::__private::None;
                                    while let _serde::__private::Some(__key) = _serde::de::MapAccess::next_key::<
                                        __Field,
                                    >(&mut __map)? {
                                        match __key {
                                            __Field::__field0 => {
                                                if _serde::__private::Option::is_some(&__field0) {
                                                    return _serde::__private::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("field1"),
                                                    );
                                                }
                                                __field0 = _serde::__private::Some(
                                                    _serde::de::MapAccess::next_value::<i32>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field1 => {
                                                if _serde::__private::Option::is_some(&__field1) {
                                                    return _serde::__private::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("field2"),
                                                    );
                                                }
                                                __field1 = _serde::__private::Some(
                                                    _serde::de::MapAccess::next_value::<bool>(&mut __map)?,
                                                );
                                            }
                                            _ => {
                                                let _ = _serde::de::MapAccess::next_value::<
                                                    _serde::de::IgnoredAny,
                                                >(&mut __map)?;
                                            }
                                        }
                                    }
                                    let __field0 = match __field0 {
                                        _serde::__private::Some(__field0) => __field0,
                                        _serde::__private::None => {
                                            _serde::__private::de::missing_field("field1")?
                                        }
                                    };
                                    let __field1 = match __field1 {
                                        _serde::__private::Some(__field1) => __field1,
                                        _serde::__private::None => {
                                            _serde::__private::de::missing_field("field2")?
                                        }
                                    };
                                    _serde::__private::Ok(__TensorZeroDerive_MyEnum::StructVariant {
                                        field1: __field0,
                                        field2: __field1,
                                    })
                                }
                            }
                            #[doc(hidden)]
                            const FIELDS: &'static [&'static str] = &[
                                "field1",
                                "field2",
                            ];
                            _serde::de::VariantAccess::struct_variant(
                                __variant,
                                FIELDS,
                                __Visitor {
                                    marker: _serde::__private::PhantomData::<
                                        __TensorZeroDerive_MyEnum,
                                    >,
                                    lifetime: _serde::__private::PhantomData,
                                },
                            )
                        }
                        (__Field::__field2, __variant) => {
                            #[doc(hidden)]
                            struct __Visitor<'de> {
                                marker: _serde::__private::PhantomData<
                                    __TensorZeroDerive_MyEnum,
                                >,
                                lifetime: _serde::__private::PhantomData<&'de ()>,
                            }
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                type Value = __TensorZeroDerive_MyEnum;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private::Formatter,
                                ) -> _serde::__private::fmt::Result {
                                    _serde::__private::Formatter::write_str(
                                        __formatter,
                                        "tuple variant __TensorZeroDerive_MyEnum::TupleVariant",
                                    )
                                }
                                #[inline]
                                fn visit_seq<__A>(
                                    self,
                                    mut __seq: __A,
                                ) -> _serde::__private::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::SeqAccess<'de>,
                                {
                                    let __field0 = match _serde::de::SeqAccess::next_element::<
                                        i32,
                                    >(&mut __seq)? {
                                        _serde::__private::Some(__value) => __value,
                                        _serde::__private::None => {
                                            return _serde::__private::Err(
                                                _serde::de::Error::invalid_length(
                                                    0usize,
                                                    &"tuple variant __TensorZeroDerive_MyEnum::TupleVariant with 2 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field1 = match _serde::de::SeqAccess::next_element::<
                                        bool,
                                    >(&mut __seq)? {
                                        _serde::__private::Some(__value) => __value,
                                        _serde::__private::None => {
                                            return _serde::__private::Err(
                                                _serde::de::Error::invalid_length(
                                                    1usize,
                                                    &"tuple variant __TensorZeroDerive_MyEnum::TupleVariant with 2 elements",
                                                ),
                                            );
                                        }
                                    };
                                    _serde::__private::Ok(
                                        __TensorZeroDerive_MyEnum::TupleVariant(__field0, __field1),
                                    )
                                }
                            }
                            _serde::de::VariantAccess::tuple_variant(
                                __variant,
                                2usize,
                                __Visitor {
                                    marker: _serde::__private::PhantomData::<
                                        __TensorZeroDerive_MyEnum,
                                    >,
                                    lifetime: _serde::__private::PhantomData,
                                },
                            )
                        }
                    }
                }
            }
            #[doc(hidden)]
            const VARIANTS: &'static [&'static str] = &[
                "UnitVariant",
                "StructVariant",
                "TupleVariant",
            ];
            _serde::Deserializer::deserialize_enum(
                __deserializer,
                "__TensorZeroDerive_MyEnum",
                VARIANTS,
                __Visitor {
                    marker: _serde::__private::PhantomData::<__TensorZeroDerive_MyEnum>,
                    lifetime: _serde::__private::PhantomData,
                },
            )
        }
    }
};
impl<'de> ::serde::Deserialize<'de> for MyEnum {
    fn deserialize<D>(de: D) -> Result<Self, D::Error>
    where
        D: ::serde::Deserializer<'de>,
    {
        use ::std::error::Error;
        use ::serde_json::{json, Value};
        use ::serde::Deserialize;
        use ::serde::de::Unexpected;
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
        let tag = obj
            .remove("type")
            .and_then(|v| v.as_str().map(|v| v.to_owned()))
            .ok_or_else(|| {
                serde::de::Error::custom("TensorZeroDerive: missing tag field `type`")
            })?;
        let modified = ::serde_json::Value::Object({
            let mut object = ::serde_json::Map::new();
            let _ = object.insert((tag).into(), ::serde_json::to_value(&value).unwrap());
            object
        });
        let val: __TensorZeroDerive_MyEnum = ::serde_path_to_error::deserialize(
                modified.clone(),
            )
            .map_err(|e| {
                let path = e
                    .path()
                    .iter()
                    .skip(1)
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(".");
                if path.is_empty() {
                    serde::de::Error::custom(e.into_inner())
                } else {
                    serde::de::Error::custom(
                        ::alloc::__export::must_use({
                            let res = ::alloc::fmt::format(
                                format_args!("{0}: {1}", path, e.into_inner()),
                            );
                            res
                        }),
                    )
                }
            })?;
        match val {
            __TensorZeroDerive_MyEnum::UnitVariant => Ok(MyEnum::UnitVariant),
            __TensorZeroDerive_MyEnum::StructVariant { field1, field2 } => {
                Ok(MyEnum::StructVariant {
                    field1,
                    field2,
                })
            }
            __TensorZeroDerive_MyEnum::TupleVariant(field_0, field_1) => {
                Ok(MyEnum::TupleVariant(field_0, field_1))
            }
        }
    }
}
