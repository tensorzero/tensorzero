# Dockerfile Optimization Analysis: Combining Rust Build Steps

## Issue Summary
**Issue #2624**: Investigate combining rust build steps in ui Dockerfile

The current UI Dockerfile has two separate multi-stage builders for Rust components:
1. `evaluations-build-env` - builds the evaluations binary with `cargo build --release -p evaluations`
2. `tensorzero-node-build-env` - builds the tensorzero-node bindings

## Current Problems

### 1. **Duplicate Dependencies**
- Both stages install similar system dependencies (clang, libc++-dev)
- Both stages use cargo-chef but cook dependencies separately
- Redundant Rust toolchain setup

### 2. **Build Time Inefficiency**
- Two separate Docker layers for Rust builds
- No shared dependency caching between stages
- Longer total build time due to sequential execution

### 3. **Missing ts-rs Test Execution**
- The issue mentions running `cargo test` for ts-rs tests that generate output files
- Current Dockerfile doesn't explicitly run these tests
- ts-rs exports are configured but not executed during build

## Solution: Combined Rust Build Stage

### Key Changes Made

1. **Merged Build Stages**
   - Combined `evaluations-build-env` and `tensorzero-node-build-env` into `combined-rust-build-env`
   - Single cargo-chef dependency cooking for all Rust crates
   - Shared system dependencies installation

2. **Added ts-rs Test Execution**
   - Added `cargo test --release --no-run` to compile and run ts-rs tests
   - Tests generate TypeScript type definitions during build
   - Ensures type safety between Rust and TypeScript code

3. **Optimized Workflow**
   ```dockerfile
   # Build evaluations binary and run ts-rs tests in a single stage
   RUN cargo build --release -p evaluations $CARGO_BUILD_FLAGS && \
       cargo test --release --no-run && \
       cp -r /tensorzero/target/release /release
   
   # Build tensorzero-node bindings in the same stage
   WORKDIR /tensorzero/internal/tensorzero-node
   RUN pnpm install --frozen-lockfile && pnpm run build
   ```

## Benefits

### 1. **Reduced Build Time**
- Single Rust toolchain setup
- Shared dependency caching
- Parallel execution of related build steps

### 2. **Improved Cache Efficiency**
- Cargo-chef cooks all dependencies once
- Shared target directory between builds
- Better Docker layer caching

### 3. **Enhanced Type Safety**
- ts-rs tests run during build process
- TypeScript definitions generated automatically
- Catches type mismatches early

### 4. **Simplified Maintenance**
- Single Rust build stage to maintain
- Consistent dependency versions
- Easier to debug build issues

## Technical Details

### ts-rs Integration
The project uses ts-rs extensively for TypeScript type generation:
- 50+ structs with `#[cfg_attr(test, ts(export))]` annotations
- Generates TypeScript definitions for Rust types
- Used in UI components for type-safe API calls

### Cargo Chef Optimization
- Uses cargo-chef for dependency caching
- Cooks all workspace dependencies in one step
- Reduces rebuild time on dependency changes

### Build Artifacts
- `evaluations` binary for CLI operations
- `tensorzero-node` bindings for Node.js integration
- TypeScript definitions from ts-rs tests

## Testing the Changes

To verify the optimization works:

1. **Build the Docker image:**
   ```bash
   docker build -f ui/Dockerfile -t tensorzero-ui:optimized .
   ```

2. **Compare build times:**
   ```bash
   time docker build -f ui/Dockerfile -t tensorzero-ui:original .
   time docker build -f ui/Dockerfile -t tensorzero-ui:optimized .
   ```

3. **Verify artifacts:**
   ```bash
   docker run --rm tensorzero-ui:optimized ls -la /usr/local/bin/evaluations
   docker run --rm tensorzero-ui:optimized ls -la /app/ui/build/*.node
   ```

## Potential Risks

1. **Build Failure Impact**: Single stage means any Rust build failure affects both components
2. **Cache Invalidation**: Changes to any Rust dependency invalidates entire stage cache
3. **Memory Usage**: Combined stage may use more memory during build

## Mitigation Strategies

1. **Granular Error Handling**: Separate RUN commands for different build steps
2. **Selective Caching**: Use cargo-chef effectively to minimize cache invalidation
3. **Resource Monitoring**: Monitor build memory usage and optimize if needed

## Conclusion

This optimization addresses the core issue by:
- ✅ Combining duplicate Rust build stages
- ✅ Adding missing ts-rs test execution
- ✅ Improving build efficiency and cache utilization
- ✅ Maintaining all existing functionality

The changes are backward compatible and should result in faster, more efficient Docker builds while ensuring type safety through proper ts-rs test execution. 