/**
 * nVDB Node.js Native Bindings - Git Submodule Version
 * 
 * This module loads the native nVDB bindings directly.
 * Build the native module first with: cargo build --release -p nvdb-node
 * 
 * For git submodule workflow:
 * 1. Add nVDB as submodule: git submodule add https://github.com/nvdb/nvdb.git nVDB
 * 2. Build: cd nVDB && cargo build --release -p nvdb-node
 * 3. The loader below will find the .node file (or .dll/.so/.dylib)
 */

const { existsSync } = require('fs');
const { join, dirname } = require('path');

// Determine the correct native binary name based on platform
function getNativeBinaryName() {
  const platform = process.platform;
  const arch = process.arch;
  
  // Map platform/arch to binary name
  const names = {
    'win32': {
      'x64': 'nvdb-node.win32-x64-msvc.node',
      'arm64': 'nvdb-node.win32-arm64-msvc.node'
    },
    'darwin': {
      'x64': 'nvdb-node.darwin-x64.node',
      'arm64': 'nvdb-node.darwin-arm64.node'
    },
    'linux': {
      'x64': 'nvdb-node.linux-x64-gnu.node',
      'arm64': 'nvdb-node.linux-arm64-gnu.node'
    }
  };
  
  const platformNames = names[platform];
  if (!platformNames) {
    throw new Error(`Unsupported platform: ${platform}`);
  }
  
  const binaryName = platformNames[arch];
  if (!binaryName) {
    throw new Error(`Unsupported architecture ${arch} on ${platform}`);
  }
  
  return binaryName;
}

// Find the native binary
function findNativeBinary() {
  const binaryName = getNativeBinaryName();
  const moduleDir = __dirname;
  
  // Search paths in order of preference
  const searchPaths = [
    // 1. Same directory as this file (if copied/renamed)
    join(moduleDir, binaryName),
    // 2. Raw DLL name (Windows dev builds)
    join(moduleDir, 'nvdb_node.dll'),
    // 3. Parent directory (target/release relative to napi folder)
    join(moduleDir, '..', 'target', 'release', 'nvdb_node.dll'),
    join(moduleDir, '..', 'target', 'release', 'libnvdb_node.so'),
    join(moduleDir, '..', 'target', 'release', 'libnvdb_node.dylib'),
    // 4. Direct build output (various platforms)
    join(moduleDir, 'nvdb_node.node'),
    join(moduleDir, 'nvdb_node.dll'),
    join(moduleDir, 'libnvdb_node.so'),
    join(moduleDir, 'libnvdb_node.dylib'),
  ];
  
  for (const path of searchPaths) {
    if (existsSync(path)) {
      return path;
    }
  }
  
  throw new Error(
    `Native binary not found. The native module must be built after cloning.\n\n` +
    `Searched:\n` +
    searchPaths.map(p => `  - ${p}`).join('\n') +
    `\n\nTo build, run:\n` +
    `  cd nVDB/napi && node setup.js\n` +
    `\nOr manually:\n` +
    `  cargo build --release -p nvdb-node\n` +
    `  copy target/release/nvdb_node.dll napi/${binaryName}  (Windows)\n` +
    `  ln -s target/release/libnvdb_node.so napi/${binaryName}  (Linux)\n` +
    `\nYou can also set the environment variable:\n` +
    `  NODE_NVDB_NATIVE_PATH=/path/to/native/binary`
  );
}

// Allow override via environment variable
const nativePath = process.env.NODE_NVDB_NATIVE_PATH || findNativeBinary();

// Load the native module
let nativeBinding;
try {
  nativeBinding = require(nativePath);
} catch (e) {
  throw new Error(`Failed to load native module from ${nativePath}: ${e.message}`);
}

// Export the classes
module.exports.Database = nativeBinding.Database;
module.exports.Collection = nativeBinding.Collection;
module.exports.FilterBuilder = nativeBinding.FilterBuilder;

// Also export the native path for debugging
module.exports.NATIVE_PATH = nativePath;
