# MidasCpp Installation Guide

## Overview

This guide covers installing MidasCpp for vibrational coupled cluster calculations.

## Requirements

- **C++ Compiler**: gcc ≥ 7.2.0 with C++17 support
- **Operating System**: Linux or macOS
- **Disk Space**: ~500 MB for source + build
- **Memory**: 4+ GB RAM recommended

## License

MidasCpp is distributed separately from this repository under the
[GNU Lesser General Public License v2.1](https://source.coderefinery.org/midascpp/midascpp/-/blob/master/LICENSE)
(LGPL-2.1), a different license from this repository's own Apache License
2.0 (see [`LICENSE.txt`](../LICENSE.txt)). Building and using MidasCpp is
subject to its own license terms, not this repository's.

## Installation Steps

### 1. Download MidasCpp

**Option A: Download Release Tarball**
```bash
# Navigate to download directory
cd ~/Downloads

# Download specific release (example: 2020.04.0)
wget https://gitlab.com/midascpp/midascpp/-/archive/2020.04.0/midascpp-2020.04.0.tar.gz

# Extract
tar -xzf midascpp-2020.04.0.tar.gz
cd midascpp-2020.04.0
```

**Option B: Clone Git Repository**
```bash
# Clone repository
git clone git@gitlab.com:midascpp/midascpp.git
cd midascpp

# Checkout specific version (optional)
git checkout 2020.04.0
```

### 2. Check Compiler Version

```bash
# Check gcc version (need ≥ 7.2.0)
gcc --version

# If gcc is too old, try g++
g++ --version

# On macOS, check clang (should support C++17)
clang++ --version
```

**If compiler is too old:**
- Linux: Install newer gcc via package manager or build from source
- macOS: Update Xcode Command Line Tools

### 3. Configure Build

```bash
# Set installation prefix
export MIDASCPP_INSTALL="${HOME}/software/midascpp"

# Configure (basic build without parallelization)
./configure --prefix="${MIDASCPP_INSTALL}"
```

**Configuration Options** (see manual for full list):
- `--prefix=/path/to/install` - Installation directory
- `--enable-mpi` - Enable MPI parallelization
- `--enable-openmp` - Enable OpenMP parallelization
- `--with-blas=/path/to/blas` - Specify BLAS library
- `--with-lapack=/path/to/lapack` - Specify LAPACK library

**Common Issues:**
- Missing BLAS/LAPACK: Install via package manager
  ```bash
  # Ubuntu/Debian
  sudo apt-get install libblas-dev liblapack-dev
  
  # macOS (via Homebrew)
  brew install openblas lapack
  ```

### 4. Compile and Install

```bash
# Compile and install (this may take 10-30 minutes)
make complete

# The 'complete' target compiles, links, and installs in one step
```

**If compilation fails:**
- Check compiler version
- Check for missing dependencies
- Consult MidasCpp manual or GitLab issues

### 5. Add to PATH

```bash
# Add to current session
export PATH="${MIDASCPP_INSTALL}/bin:${PATH}"

# Add to shell profile for persistence
echo 'export MIDASCPP_INSTALL="${HOME}/software/midascpp"' >> ~/.bashrc
echo 'export PATH="${MIDASCPP_INSTALL}/bin:${PATH}"' >> ~/.bashrc

# Reload shell configuration
source ~/.bashrc
```

**For macOS (using zsh):**
```bash
echo 'export MIDASCPP_INSTALL="${HOME}/software/midascpp"' >> ~/.zshrc
echo 'export PATH="${MIDASCPP_INSTALL}/bin:${PATH}"' >> ~/.zshrc
source ~/.zshrc
```

### 6. Verify Installation

```bash
# Check executable is in PATH
which midascpp

# Check version (if supported)
midascpp --version || midascpp --help

# Expected output: path to executable
# Example: /Users/username/software/midascpp/bin/midascpp
```

### 7. Run Test Suite

```bash
# Navigate to test suite
cd "${MIDASCPP_INSTALL}/../midascpp-2020.04.0/test_suite"

# Run short test (few seconds)
./TEST installshort

# Run long test (may take hours - optional)
# ./TEST installlong
```

**Expected output:**
```
Running installshort tests...
Test 1: PASSED
Test 2: PASSED
...
All tests PASSED
```

**If tests fail:**
- Check installation logs
- Verify compiler compatibility
- Consult MidasCpp documentation

## Environment Setup for Python Integration

```bash
# Set environment variable for Python scripts
export MIDASCPP_BIN="${MIDASCPP_INSTALL}/bin/midascpp"

# Add to shell profile
echo 'export MIDASCPP_BIN="${MIDASCPP_INSTALL}/bin/midascpp"' >> ~/.bashrc
source ~/.bashrc
```

## Troubleshooting

### Issue: "configure: error: C++ compiler cannot create executables"

**Solution:**
```bash
# Install/update compiler
# Ubuntu/Debian
sudo apt-get install build-essential g++

# macOS
xcode-select --install
```

### Issue: "undefined reference to BLAS/LAPACK functions"

**Solution:**
```bash
# Install BLAS/LAPACK libraries
# Ubuntu/Debian
sudo apt-get install libblas-dev liblapack-dev

# macOS
brew install openblas lapack

# Reconfigure with explicit paths
./configure --prefix="${MIDASCPP_INSTALL}" \
    --with-blas=/usr/lib/x86_64-linux-gnu \
    --with-lapack=/usr/lib/x86_64-linux-gnu
```

### Issue: "make: *** No rule to make target 'complete'"

**Solution:**
```bash
# Try standard make targets
make
make install
```

### Issue: Test suite fails

**Solution:**
- Check that installation completed successfully
- Verify PATH includes MidasCpp bin directory
- Check test suite README for known issues
- Report to MidasCpp developers if persistent

## Next Steps

After successful installation:
1. Proceed to Phase 5A.0 format discovery
2. Run manual VCCSD calculation for toy system
3. Document input/output format conventions

## References

- MidasCpp GitLab: https://gitlab.com/midascpp/midascpp
- MidasCpp Manual: `${MIDASCPP_INSTALL}/share/doc/midascpp/manual.pdf`
- MidasCpp Paper: Christiansen, O. et al. (check manual for citations)