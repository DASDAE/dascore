--[[
Add an "open in marimo" link to the top of each tutorial page.

The link only appears when scripts/build_notebooks.py exported a notebook for
the page, so pages without code cells (and renders that skipped the notebook
build) simply do not get one.
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
  if not file_exists(root .. "/marimo/" .. stem .. ".html") then
    return doc
  end
  -- The logo is one the export ships beside the notebooks; quarto copies the
  -- whole export but never sees assets referenced from raw html.
  local html = table.concat({
    '<a class="marimo-launch" href="/marimo/',
    stem,
    '.html" target="_blank" rel="noopener" title="',
    "Runs DASCore in your browser with WebAssembly. Nothing is installed on ",
    'your computer, and the first load takes a moment.">',
    '<img class="marimo-launch-logo" src="/marimo/apple-touch-icon.png" alt="">',
    "<span>Open in marimo</span>",
    "</a>",
  })
  table.insert(doc.blocks, 1, pandoc.RawBlock("html", html))
  return doc
end
