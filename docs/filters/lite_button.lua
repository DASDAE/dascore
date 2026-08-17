--[[
Add an "open in JupyterLite" link to the top of each tutorial page.

The link only appears when a notebook was actually built for the page by
scripts/build_notebooks.py, so pages without code cells (and renders that
skipped the notebook build) simply do not get one.
]]

--- Return the file's name without its tutorial directory or extension.
local function tutorial_stem(input)
  return input and input:match("tutorial/([^/]+)%.qmd$")
end

--- Return the docs directory holding a tutorial page.
local function docs_dir(input)
  return input and input:match("^(.*)/tutorial/[^/]+%.qmd$")
end

--- Return true when a file can be opened for reading.
local function file_exists(path)
  local handle = io.open(path, "r")
  if handle == nil then
    return false
  end
  handle:close()
  return true
end

function Pandoc(doc)
  -- Only the html site gets the link; adding raw html to the ipynb render
  -- would put a stray link inside the notebook the link points at.
  if not quarto.doc.is_format("html") then
    return doc
  end
  local input = quarto.doc.input_file
  local stem = tutorial_stem(input)
  local root = docs_dir(input)
  if stem == nil or root == nil then
    return doc
  end
  local notebook = root .. "/lite_contents/tutorial/" .. stem .. ".ipynb"
  if not file_exists(notebook) then
    return doc
  end
  local href = "/lite/lab/index.html?path=tutorial/" .. stem .. ".ipynb"
  local html = table.concat({
    '<a class="lite-launch" href="',
    href,
    '" target="_blank" rel="noopener" title="',
    "Runs DASCore in your browser with WebAssembly. Nothing is installed on ",
    'your computer, and the first load takes a moment.">',
    "Open in JupyterLite",
    "</a>",
  })
  table.insert(doc.blocks, 1, pandoc.RawBlock("html", html))
  return doc
end
