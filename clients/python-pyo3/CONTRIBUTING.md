## Note on naming

There are several different names in use in this client:
* `python-pyo3` - this is *only* used as the name of the top-level directory, to distinguish it from the pure-python implementation
  In the future, we'll delete the pure-python client and rename this to 'python'
* `tensorzero_python` - this is the rust *crate* name, so that we get sensible output from running Cargo
* `tensorzero` - this is the name of the Python package (python code can use `import tensorzero`)
* `tensorzero_rust` - this is the (locally-renamed) Rust client package, which avoids conflicts with pyo3-generated code.