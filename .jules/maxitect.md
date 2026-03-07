## 2024-03-07 - [Architecture Audit and Refactoring of coreml-rs]

Learning: Monolithic files (like `mlmodel.rs`) accumulate massive amounts of duplicated boilerplate over time because developers copy-paste pointer-management and layout-checking logic rather than abstracting it into generic traits on the data layer itself.

Action: When seeing repeated `into_raw_vec_and_offset` checks and zlib compression blocks, extract them into dedicated data-layer traits (like `MLArrayBaseExt`) and shared utilities (`utils::save_buffer_to_disk`) *before* they proliferate further. Separate configuration state (loaders, options) from operational state (the model itself).
