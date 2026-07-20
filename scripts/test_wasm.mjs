import fs from "node:fs";
import path from "node:path";

import { loadPyodide } from "pyodide";

async function main() {
  const wheelArg = process.argv[2];
  if (!wheelArg) {
    throw new Error("Usage: node scripts/test_wasm.mjs path/to/dascore.whl");
  }

  const wheelPath = path.resolve(wheelArg);
  const emscriptenPath = `/tmp/${path.basename(wheelPath)}`;
  const pyodide = await loadPyodide();
  pyodide.FS.writeFile(emscriptenPath, fs.readFileSync(wheelPath));

  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  await micropip.install(`emfs:${emscriptenPath}`);
  micropip.destroy();

  await pyodide.runPythonAsync(`
import platform
from pathlib import Path

import h5py
import numpy as np

import dascore as dc
from dascore.io import H5Reader

assert platform.system() == "Emscripten"

patch = dc.get_example_patch(shape=(2, 8))
detrended = patch.detrend("time")
assert detrended.shape == (2, 8)
assert np.isfinite(detrended.data).all()

assert H5Reader.constructor is h5py.File
h5_path = "/tmp/dascore-wasm-smoke.h5"
with h5py.File(h5_path, "w") as handle:
    handle["data"] = patch.data
with H5Reader.get_handle(h5_path) as handle:
    assert np.array_equal(handle["data"][:], patch.data)

spool_path = Path("/tmp/dascore-wasm-spool")
spool_path.mkdir(exist_ok=True)
dc.write(patch, spool_path / "patch.h5", "DASDAE")
spool = dc.spool(spool_path).update()
assert len(spool) == 1
assert spool[0].shape == patch.shape
assert (spool_path / ".dascore_index.sqlite3").exists()
`);

  console.log("DASCore WebAssembly smoke test passed.");
}

main().catch((error) => {
  console.error(error.message ?? error);
  process.exitCode = 1;
});
