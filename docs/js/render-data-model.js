(function () {
  function ensureTooltip() {
    let tooltip = document.querySelector(".data-model-tooltip");
    if (!tooltip) {
      tooltip = document.createElement("div");
      tooltip.className = "data-model-tooltip";
      tooltip.setAttribute("role", "tooltip");
      document.body.appendChild(tooltip);
    }
    return tooltip;
  }

  function appendText(parent, className, text) {
    const node = document.createElement("div");
    node.className = className;
    node.textContent = text;
    parent.appendChild(node);
  }

  function appendTooltipTitle(parent, doc) {
    const node = document.createElement("div");
    node.className = "data-model-tooltip-title";
    if (doc.reference_href) {
      const link = document.createElement("a");
      link.href = doc.reference_href;
      link.textContent = doc.title || "";
      node.appendChild(link);
    } else {
      node.textContent = doc.title || "";
    }
    parent.appendChild(node);
  }

  function renderTooltip(tooltip, doc) {
    tooltip.replaceChildren();
    appendTooltipTitle(tooltip, doc);
    appendText(tooltip, "data-model-tooltip-heading", "Summary");
    appendText(tooltip, "data-model-tooltip-rule", "-------");
    appendText(tooltip, "data-model-tooltip-summary", doc.summary || "");
    appendText(tooltip, "data-model-tooltip-heading", "Attributes");
    appendText(tooltip, "data-model-tooltip-rule", "----------");

    for (const attr of doc.attributes || []) {
      const item = document.createElement("div");
      item.className = "data-model-tooltip-attribute";

      const signature = document.createElement("div");
      signature.className = "data-model-tooltip-signature";
      const name = document.createElement("strong");
      name.textContent = attr.name || "";
      signature.appendChild(name);
      signature.appendChild(document.createTextNode(" : " + (attr.type || "")));

      const description = document.createElement("div");
      description.className = "data-model-tooltip-description";
      description.textContent = attr.description || "";

      item.appendChild(signature);
      item.appendChild(description);
      tooltip.appendChild(item);
    }
  }

  function positionTooltip(tooltip, event) {
    const margin = 16;
    let left = event.clientX + margin;
    let top = event.clientY + margin;
    const rect = tooltip.getBoundingClientRect();
    if (left + rect.width > window.innerWidth - margin) {
      left = event.clientX - rect.width - margin;
    }
    if (top + rect.height > window.innerHeight - margin) {
      top = window.innerHeight - rect.height - margin;
    }
    tooltip.style.left = Math.max(margin, left) + "px";
    tooltip.style.top = Math.max(margin, top) + "px";
  }

  function attachTooltips(cy, container) {
    const tooltip = ensureTooltip();
    let isOverNode = false;
    let isOverTooltip = false;
    let hideTimer = null;

    function cancelHide() {
      if (hideTimer) {
        window.clearTimeout(hideTimer);
        hideTimer = null;
      }
    }

    function scheduleHide() {
      cancelHide();
      hideTimer = window.setTimeout(() => {
        if (!isOverNode && !isOverTooltip) {
          tooltip.classList.remove("is-visible");
        }
      }, 120);
    }

    tooltip.addEventListener("mouseenter", () => {
      isOverTooltip = true;
      cancelHide();
    });

    tooltip.addEventListener("mouseleave", () => {
      isOverTooltip = false;
      scheduleHide();
    });

    cy.on("mouseover", "node", (event) => {
      isOverNode = true;
      cancelHide();
      const rendered = event.target.renderedPosition();
      renderTooltip(tooltip, event.target.data());
      tooltip.classList.add("is-visible");
      positionTooltip(tooltip, {
        clientX: container.getBoundingClientRect().left + rendered.x,
        clientY: container.getBoundingClientRect().top + rendered.y,
      });
    });
    cy.on("mousemove", "node", (event) => {
      if (isOverTooltip) {
        return;
      }
      const rendered = event.target.renderedPosition();
      positionTooltip(tooltip, {
        clientX: container.getBoundingClientRect().left + rendered.x,
        clientY: container.getBoundingClientRect().top + rendered.y,
      });
    });
    cy.on("mouseout", "node", () => {
      isOverNode = false;
      scheduleHide();
    });
  }

  function childClosure(node) {
    const cy = node.cy();
    const seen = new Set();
    function visit(parent) {
      parent.outgoers("edge[kind = 'containment']").targets().forEach((child) => {
        if (seen.has(child.id())) {
          return;
        }
        seen.add(child.id());
        visit(child);
      });
    }
    visit(node);
    return cy.collection([...seen].map((id) => cy.getElementById(id)));
  }

  function visibleReferenceEdges(cy) {
    cy.edges("[kind = 'reference']").forEach((edge) => {
      const visible = edge.source().visible() && edge.target().visible();
      edge.style("display", visible ? "element" : "none");
    });
  }

  function containmentIndegree(node) {
    return node.incomers("edge[kind = 'containment']").length;
  }

  function layoutOptions(cy) {
    const direction = (cy.data("direction") || "TB").toUpperCase();
    const elkDirection = direction === "LR" ? "RIGHT" : "DOWN";
    return {
      name: "elk",
      animate: false,
      fit: false,
      padding: 28,
      nodeDimensionsIncludeLabels: true,
      eles: cy.elements(":visible"),
      elk: {
        algorithm: "layered",
        "elk.direction": elkDirection,
        "elk.edgeRouting": "ORTHOGONAL",
        "elk.layered.spacing.nodeNodeBetweenLayers": "80",
        "elk.spacing.nodeNode": "48",
        "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
        "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
        "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
      },
    };
  }

  function fitVisibleGraph(cy) {
    const visibleElements = cy.elements(":visible");
    if (visibleElements.empty()) {
      return;
    }
    cy.fit(visibleElements, 44);
    cy.center(visibleElements);
  }

  function runLayout(cy) {
    const layout = cy.layout(layoutOptions(cy));
    layout.one("layoutstop", () => fitVisibleGraph(cy));
    layout.run();
  }

  function refreshGraph(cy) {
    cy.resize();
    runLayout(cy);
  }

  function observeResize(target, cy) {
    if (!window.ResizeObserver) {
      return;
    }
    let debounce = null;
    let lastWidth = target.clientWidth;
    let lastHeight = target.clientHeight;
    const observer = new ResizeObserver(() => {
      if (target.clientWidth === lastWidth && target.clientHeight === lastHeight) {
        return;
      }
      lastWidth = target.clientWidth;
      lastHeight = target.clientHeight;
      if (debounce) {
        window.clearTimeout(debounce);
      }
      debounce = window.setTimeout(() => {
        cy.resize();
        fitVisibleGraph(cy);
      }, 150);
    });
    observer.observe(target);
  }

  function setCollapsed(cy, node, collapsed, relayout = true) {
    node.data("collapsed", collapsed);
    const descendants = childClosure(node);
    if (collapsed) {
      descendants.hide();
      descendants.connectedEdges().hide();
    } else {
      descendants.show();
      descendants.connectedEdges("[kind = 'containment']").show();
      descendants.forEach((child) => {
        if (child.data("collapsed")) {
          setCollapsed(cy, child, true, false);
        }
      });
    }
    visibleReferenceEdges(cy);
    if (relayout) {
      runLayout(cy);
    }
  }

  function applyInitialCollapse(cy) {
    cy.nodes().forEach((node) => {
      node.data("collapsed", (node.data("children") || "open") === "collapsed");
    });
    cy.nodes().forEach((node) => {
      if (node.data("collapsed")) {
        setCollapsed(cy, node, true, false);
      }
    });
    visibleReferenceEdges(cy);
    runLayout(cy);
  }

  function addToolbar(container, cy) {
    const toolbar = container.querySelector(".render-data-model-toolbar");
    toolbar.querySelector("[data-action='expand']").addEventListener("click", () => {
      cy.nodes().forEach((node) => node.data("collapsed", false));
      cy.elements().show();
      visibleReferenceEdges(cy);
      runLayout(cy);
    });
    toolbar.querySelector("[data-action='collapse']").addEventListener("click", () => {
      cy.nodes().forEach((node) => node.data("collapsed", false));
      cy.nodes().forEach((node) => {
        if (node.outgoers("edge[kind = 'containment']").length > 0 && containmentIndegree(node) > 0) {
          setCollapsed(cy, node, true, false);
        }
      });
      visibleReferenceEdges(cy);
      runLayout(cy);
    });
  }

  function cytoscapeStyles() {
    return [
      {
        selector: "node",
        style: {
          "background-color": "data(fill)",
          "border-color": "data(stroke)",
          "border-width": 2,
          "color": "data(color)",
          "content": "data(label)",
          "font-size": 15,
          "font-weight": 700,
          "height": 54,
          "shape": "round-rectangle",
          "text-halign": "center",
          "text-valign": "center",
          "text-wrap": "wrap",
          "text-max-width": 130,
          "width": 150,
        },
      },
      {
        selector: "node.has-children",
        style: {
          "border-width": 3,
        },
      },
      {
        selector: "edge",
        style: {
          "curve-style": "bezier",
          "line-color": "#60717c",
          "target-arrow-color": "#60717c",
          "target-arrow-shape": "triangle",
          "width": 2,
        },
      },
      {
        // Labelled with the field which makes the reference: two classes
        // may refer to each other through more than one field, and unlabelled
        // arcs between the same pair would be indistinguishable.
        selector: "edge[kind = 'reference']",
        style: {
          "color": "#5f6d78",
          "font-size": 11,
          "label": "data(label)",
          "line-color": "#8a98a3",
          "line-style": "dashed",
          "target-arrow-color": "#8a98a3",
          "text-background-color": "#ffffff",
          "text-background-opacity": 0.85,
          "text-background-padding": 2,
          "text-rotation": "autorotate",
        },
      },
    ];
  }

  function renderDataModels() {
    if (!window.cytoscape) {
      window.setTimeout(renderDataModels, 100);
      return;
    }
    const containers = document.querySelectorAll(".render-data-model:not([data-processed])");
    for (const container of containers) {
      container.dataset.processed = "true";
      const graph = JSON.parse(container.querySelector("script[type='application/json'].render-data-model-graph-data").textContent);
      const target = container.querySelector(".render-data-model-graph");
      const cy = window.cytoscape({
        container: target,
        elements: graph.elements,
        layout: { name: "preset" },
        style: cytoscapeStyles(),
        minZoom: 0.2,
        maxZoom: 1.6,
        userZoomingEnabled: false,
        userPanningEnabled: true,
        wheelSensitivity: 0.1,
      });
      target.renderDataModelCy = cy;
      cy.data("direction", graph.direction || "TB");
      cy.nodes().forEach((node) => {
        if (node.outgoers("edge[kind = 'containment']").length > 0) {
          node.addClass("has-children");
        }
      });
      cy.on("tap", "node", (event) => {
        const node = event.target;
        if (node.outgoers("edge[kind = 'containment']").length > 0) {
          setCollapsed(cy, node, !node.data("collapsed"));
        }
      });
      // Only applyInitialCollapse lays out: elk runs asynchronously, so a
      // second layout started here would race it over a different set of
      // visible nodes and whichever finished last would win.
      applyInitialCollapse(cy);
      attachTooltips(cy, target);
      addToolbar(container, cy);
      observeResize(target, cy);
    }
  }

  document.addEventListener("shown.bs.collapse", (event) => {
    event.target.querySelectorAll(".render-data-model-graph").forEach((target) => {
      const cy = target.renderDataModelCy;
      if (cy) {
        refreshGraph(cy);
      }
    });
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderDataModels);
  } else {
    renderDataModels();
  }
})();
