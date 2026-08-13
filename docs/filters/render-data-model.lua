-- Renders a `render_data_model` div into an interactive class diagram.
--
-- The div names a spec file which `scripts/_inventory_model.py` generates
-- from DASCore's own pydantic models; see docs/tutorial/inventory.qmd for
-- the one use of it. Everything the diagram draws -- nodes, attributes,
-- edges -- is read from that file, so the picture cannot drift from the
-- models it describes.
local needs_runtime = false

local function stringify(value)
  if value == nil then
    return ""
  end
  return pandoc.utils.stringify(value)
end

local function read_file(path)
  local file = io.open(path, "r")
  if file == nil then
    file = io.open("../" .. path, "r")
  end
  if file == nil then
    error(
      "Could not read the data model spec at " .. path ..
      ". It is generated, so run `python scripts/build_api_docs.py` before " ..
      "rendering the docs."
    )
  end
  local content = file:read("*a")
  file:close()
  return content
end

local function parse_yaml(path)
  local content = read_file(path)
  return pandoc.read("---\n" .. content .. "\n---\n", "markdown").meta
end

local function escape_html(value)
  return tostring(value)
    :gsub("&", "&amp;")
    :gsub("<", "&lt;")
    :gsub(">", "&gt;")
end

local function json_escape(value)
  return tostring(value)
    :gsub("\\", "\\\\")
    :gsub('"', '\\"')
    :gsub("\n", "\\n")
    :gsub("\r", "\\r")
    :gsub("\t", "\\t")
end

local function is_array(table_value)
  if type(table_value) ~= "table" then
    return false
  end
  local count = 0
  for key, _ in pairs(table_value) do
    if type(key) ~= "number" then
      return false
    end
    count = count + 1
  end
  return count == #table_value
end

local function json_encode(value)
  if type(value) == "string" then
    return '"' .. json_escape(value) .. '"'
  elseif type(value) == "number" or type(value) == "boolean" then
    return tostring(value)
  elseif type(value) == "table" then
    local parts = {}
    if is_array(value) then
      for _, item in ipairs(value) do
        table.insert(parts, json_encode(item))
      end
      return "[" .. table.concat(parts, ",") .. "]"
    end
    for key, item in pairs(value) do
      table.insert(parts, json_encode(tostring(key)) .. ":" .. json_encode(item))
    end
    table.sort(parts)
    return "{" .. table.concat(parts, ",") .. "}"
  end
  return "null"
end

-- A node's attributes are a list, in the order the model declares them.
local function attrs_from_meta(attrs)
  local out = {}
  for _, attr in ipairs(attrs or {}) do
    table.insert(out, {
      name = stringify(attr.name),
      type = stringify(attr.type),
      description = stringify(attr.description),
    })
  end
  return out
end

-- An edge is [source, target] with an optional third element labelling it.
local function edge_from_meta(edge)
  return stringify(edge[1]), stringify(edge[2]), stringify(edge[3])
end

local function node_ids(nodes)
  local ids = {}
  for id, _ in pairs(nodes) do
    table.insert(ids, id)
  end
  table.sort(ids)
  return ids
end

local function node_title(node, id)
  return stringify(node.label or id)
end

local function node_display_label(node, id)
  return stringify(node.display or node.label or id)
end

-- The palette is stated in the spec itself, one entry per legend group.
local function style_source_from_spec(spec)
  return spec.styles or {}
end

local function styles_from_spec(spec)
  local styles = {}
  for _, style in ipairs(style_source_from_spec(spec)) do
    local id = stringify(style.id)
    if id ~= "" then
      styles[id] = style
    end
  end
  return styles
end

local function used_style_classes(spec)
  local used = {}
  for _, node in pairs(spec.nodes or {}) do
    local style_class = stringify(node.style_class or node.class)
    if style_class ~= "" then
      used[style_class] = true
    end
  end
  return used
end

local function resolved_style(item, styles)
  local style_class = stringify(item.style_class or item.class)
  local inherited = styles[style_class] or {}
  local inline = item.style or {}
  local resolved = {}

  for key, value in pairs(inherited) do
    resolved[key] = value
  end
  for key, value in pairs(inline) do
    resolved[key] = value
  end

  return resolved
end

local function style_value(item, name, styles)
  local style = resolved_style(item, styles or {})
  return stringify(style[name] or item[name])
end

-- The generator writes each node's API page as a site-absolute qmd path,
-- the same form the cross-reference filter reads. That filter never sees
-- these, since the diagram is raw html by the time it is built, so the
-- rewrite to a rendered page happens here: `root` is the div's way of
-- saying how far the page holding it sits below docs/.
local function api_href(node, root)
  local path = stringify(node.reference_href)
  if path == "" then
    return ""
  end
  return root .. path:gsub("^/", ""):gsub("%.qmd$", ".html")
end

local function graph_node_map(spec, styles, root)
  local nodes = {}
  for _, id in ipairs(node_ids(spec.nodes)) do
    local node = spec.nodes[id]
    local style = resolved_style(node, styles)
    local display = node_display_label(node, id)
    nodes[id] = {
      id = id,
      label = display,
      title = node_title(node, id),
      reference_href = api_href(node, root),
      summary = stringify(node.summary),
      attributes = attrs_from_meta(node.attributes),
      children = stringify(node.children or "open"),
      style_class = stringify(node.style_class or node.class),
      fill = stringify(style.fill or "#ffffff"),
      stroke = stringify(style.stroke or "#b9c4cc"),
      color = stringify(style.color or "#243036"),
    }
  end
  return nodes
end

local function build_graph(spec, root)
  local styles = styles_from_spec(spec)
  local nodes = graph_node_map(spec, styles, root)
  local elements = {}

  for _, id in ipairs(node_ids(spec.nodes)) do
    table.insert(elements, { group = "nodes", data = nodes[id] })
  end

  local edge_index = 0
  for _, edge in ipairs(spec.edges or {}) do
    edge_index = edge_index + 1
    local from, to, label = edge_from_meta(edge)
    table.insert(elements, {
      group = "edges",
      data = {
        id = "edge-" .. tostring(edge_index),
        source = from,
        target = to,
        label = label,
        kind = "containment",
      },
    })
  end

  for _, edge in ipairs(spec.references or {}) do
    edge_index = edge_index + 1
    local from, to, label = edge_from_meta(edge)
    if label == "" then
      label = stringify(spec.reference_label or "references")
    end
    table.insert(elements, {
      group = "edges",
      data = {
        id = "edge-" .. tostring(edge_index),
        source = from,
        target = to,
        label = label,
        kind = "reference",
      },
    })
  end

  return {
    id = stringify(spec.id),
    title = stringify(spec.title),
    direction = stringify(spec.direction or "TB"),
    elements = elements,
  }
end

-- Only the groups some node actually wears are worth explaining.
local function legend_items(spec)
  local used = used_style_classes(spec)
  local items = {}
  for _, style in ipairs(style_source_from_spec(spec)) do
    local id = stringify(style.id)
    if used[id] then
      table.insert(items, style)
    end
  end
  return items
end

local function build_legend(spec)
  local source_items = legend_items(spec)
  if source_items == nil then
    return ""
  end

  local styles = styles_from_spec(spec)
  local items = {}
  for _, item in ipairs(source_items) do
    local label = stringify(item.label)
    if label ~= "" then
      local fill = style_value(item, "fill", styles)
      local stroke = style_value(item, "stroke", styles)
      local color = style_value(item, "color", styles)
      local description = stringify(item.description)
      local swatch_style = {}
      if fill ~= "" then
        table.insert(swatch_style, "background:" .. fill)
      end
      if stroke ~= "" then
        table.insert(swatch_style, "border-color:" .. stroke)
      end
      local label_style = ""
      if color ~= "" then
        label_style = string.format(' style="color:%s"', escape_html(color))
      end
      local body = string.format(
        '<span class="data-model-legend-swatch" style="%s"></span><span class="data-model-legend-label"%s>%s</span>',
        escape_html(table.concat(swatch_style, ";")),
        label_style,
        escape_html(label)
      )
      if description ~= "" then
        body = body .. string.format('<span class="data-model-legend-description">%s</span>', escape_html(description))
      end
      table.insert(items, '<li class="data-model-legend-item">' .. body .. '</li>')
    end
  end

  if #items == 0 then
    return ""
  end

  return '<div class="data-model-legend" aria-label="Data model color legend"><div class="data-model-legend-title">Legend</div><ul>' ..
    table.concat(items, "\n") ..
    '</ul></div>'
end

local function build_model_title(spec)
  local title = stringify(spec.title)
  if title == "" then
    return ""
  end
  return string.format('<div class="data-model-title">%s</div>', escape_html(title))
end

function Div(el)
  if not el.classes:includes("render_data_model") then
    return nil
  end

  local spec_path = el.attributes.spec
  if spec_path == nil or spec_path == "" then
    error("render_data_model requires a spec attribute")
  end

  needs_runtime = true
  local spec = parse_yaml(spec_path)
  local graph = build_graph(spec, stringify(el.attributes.root))
  local title = build_model_title(spec)
  local legend = build_legend(spec)
  local graph_json = json_encode(graph):gsub("</", "<\\/")
  local html = string.format(
    '<div class="render-data-model" data-model-id="%s">\n%s\n<div class="render-data-model-toolbar"><button type="button" data-action="expand">Expand all</button><button type="button" data-action="collapse">Collapse all</button></div>\n<div class="render-data-model-graph" role="img" aria-label="%s"></div>\n%s\n<script type="application/json" class="render-data-model-graph-data">%s</script>\n</div>',
    escape_html(stringify(spec.id or spec_path)),
    title,
    escape_html(stringify(spec.title or spec.id or spec_path)),
    legend,
    graph_json
  )
  return pandoc.RawBlock("html", html)
end

function Pandoc(doc)
  if needs_runtime then
    quarto.doc.add_html_dependency({
      name = "cytoscape",
      version = "3.30.4",
      scripts = { "../vendor/cytoscape/cytoscape.min.js" },
    })
    quarto.doc.add_html_dependency({
      name = "elkjs",
      version = "0.10.0",
      scripts = { "../vendor/elk/elk.bundled.js" },
    })
    quarto.doc.add_html_dependency({
      name = "cytoscape-elk",
      version = "2.2.0",
      scripts = { "../vendor/elk/cytoscape-elk.min.js" },
    })
    quarto.doc.add_html_dependency({
      name = "render-data-model",
      version = "1.0.0",
      scripts = { "../js/render-data-model.js" },
    })
  end
  return doc
end
