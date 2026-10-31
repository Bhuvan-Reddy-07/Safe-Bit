import React, { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import cytoscape from 'cytoscape';

/**
 * Interactive Cytoscape.js Canvas for 2-hop Ego-Subgraph Visualization.
 * Exactly matched to the new Safe-Bit design system:
 *  - Legitimate Account: #0e231c fill, #4fd1a1 border (Circle)
 *  - Flagged Mule: #2a1216 fill, #f0596b border (Circle, 2.5px border)
 *  - Device / IP Telemetry: #1c1730 fill, #c4b5fd border (Diamond)
 *  - Selected Node: #a0c3ec halo
 *  - Edges: Wire (#4a4f56), Mule Outflow (#f0596b dashed), Telemetry (#c4b5fd dotted)
 */
const getCytoscapeStyles = (theme = 'dark') => {
  const isLight = theme === 'light';
  return [
    {
      selector: 'node',
      style: {
        'label': 'data(label)',
        'color': isLight ? '#1e293b' : '#dadbdf',
        'font-family': 'Geist Mono, monospace',
        'font-size': '10px',
        'text-valign': 'bottom',
        'text-margin-y': 6,
        'background-color': isLight ? '#f1f5f9' : '#1a1d24',
        'border-width': 1.5,
        'border-color': isLight ? '#cbd5e1' : '#475569',
        'width': 34,
        'height': 34,
        'transition-property': 'background-color, border-color, width, height',
        'transition-duration': '0.15s',
      },
    },
    // Legitimate account / merchant (Green)
    {
      selector: 'node[type = "merchant"], node[type = "account"]',
      style: {
        'background-color': isLight ? '#ecfdf5' : '#0e231c',
        'border-color': isLight ? '#10b981' : '#4fd1a1',
        'border-width': 1.5,
        'width': 34,
        'height': 34,
        'color': isLight ? '#065f46' : '#dadbdf',
      },
    },
    // Flagged Money Mule (Red)
    {
      selector: 'node[type = "mule"]',
      style: {
        'background-color': isLight ? '#fee2e2' : '#2a1216',
        'border-color': isLight ? '#ef4444' : '#f0596b',
        'border-width': 2.5,
        'width': 42,
        'height': 42,
        'color': isLight ? '#b91c1c' : '#ffffff',
      },
    },
    // Shared Device / IP Telemetry Node (Purple Diamond)
    {
      selector: 'node[type = "device"], node[type = "subnet"]',
      style: {
        'shape': 'diamond',
        'background-color': isLight ? '#f3e8ff' : '#1c1730',
        'border-color': isLight ? '#8b5cf6' : '#c4b5fd',
        'border-width': 1.5,
        'width': 42,
        'height': 42,
        'color': isLight ? '#6d28d9' : '#c4b5fd',
      },
    },
    // Selected node highlight (Blue Halo)
    {
      selector: 'node:selected',
      style: {
        'border-color': isLight ? '#0284c7' : '#a0c3ec',
        'border-width': 3,
        'shadow-blur': 14,
        'shadow-color': isLight ? '#38bdf8' : '#a0c3ec',
      },
    },
    // Edges (Base wire)
    {
      selector: 'edge',
      style: {
        'width': 1.3,
        'line-color': isLight ? '#94a3b8' : '#4a4f56',
        'target-arrow-color': isLight ? '#64748b' : '#5a5f66',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'arrow-scale': 1.0,
        'opacity': 0.85,
        'label': 'data(amount_fmt)',
        'color': isLight ? '#64748b' : '#7d8187',
        'font-family': 'Geist Mono, monospace',
        'font-size': '9px',
        'text-rotation': 'autorotate',
        'text-margin-y': -8,
      },
    },
    // Mule outflow transfer (Dashed red)
    {
      selector: 'edge[is_mule = 1], edge[is_mule = true]',
      style: {
        'line-color': isLight ? '#ef4444' : '#f0596b',
        'target-arrow-color': isLight ? '#ef4444' : '#f0596b',
        'width': 1.6,
        'line-style': 'dashed',
        'line-dash-pattern': [6, 4],
        'color': isLight ? '#dc2626' : '#f0596b',
      },
    },
    // Telemetry link (Dotted purple)
    {
      selector: 'edge[channel = "TELEMETRY_LINK"]',
      style: {
        'line-color': isLight ? '#8b5cf6' : '#c4b5fd',
        'target-arrow-shape': 'none',
        'width': 1.2,
        'line-style': 'dotted',
        'line-dash-pattern': [2, 4],
        'label': '',
      },
    },
  ];
};

const SubgraphVisualizer = forwardRef(function SubgraphVisualizer(
  { graphData, onSelectNode, selectedNodeId, theme = 'dark' },
  ref
) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);

  useImperativeHandle(ref, () => ({
    zoomIn: () => {
      if (cyRef.current) {
        cyRef.current.zoom({
          level: cyRef.current.zoom() * 1.25,
          renderedPosition: {
            x: cyRef.current.width() / 2,
            y: cyRef.current.height() / 2,
          },
        });
      }
    },
    zoomOut: () => {
      if (cyRef.current) {
        cyRef.current.zoom({
          level: cyRef.current.zoom() * 0.8,
          renderedPosition: {
            x: cyRef.current.width() / 2,
            y: cyRef.current.height() / 2,
          },
        });
      }
    },
    resetGraph: () => {
      if (cyRef.current) {
        cyRef.current.fit(undefined, 35);
      }
    },
  }));

  useEffect(() => {
    if (!containerRef.current) return;

    cyRef.current = cytoscape({
      container: containerRef.current,
      style: getCytoscapeStyles(theme),
      layout: {
        name: 'cose',
        animate: false,
      },
      userZoomingEnabled: true,
      userPanningEnabled: true,
    });

    cyRef.current.on('tap', 'node', (evt) => {
      const node = evt.target;
      if (onSelectNode) {
        onSelectNode(node.data());
      }
    });

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
      }
    };
  }, []);

  // Update elements when graphData changes
  useEffect(() => {
    if (!cyRef.current || !graphData) return;

    const cy = cyRef.current;
    cy.elements().remove();

    // Prepare formatted amount labels on edges
    const nodes = (graphData.nodes || []).map((n) => ({
      ...n,
      data: {
        ...n.data,
        label: n.data.label || n.data.id,
      },
    }));

    const edges = (graphData.edges || []).map((e) => {
      const amt = e.data.amount;
      const amt_fmt = amt
        ? `$${amt.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
        : '';
      return {
        ...e,
        data: {
          ...e.data,
          amount_fmt: amt_fmt,
        },
      };
    });

    // Safeguard missing endpoint nodes to prevent Cytoscape orphan edge errors
    const nodeIds = new Set(nodes.map((n) => n.data.id));
    edges.forEach((e) => {
      if (e.data.source && !nodeIds.has(e.data.source)) {
        nodes.push({
          data: {
            id: e.data.source,
            label: e.data.source.slice(0, 14),
            type: 'account',
            risk_score: 0.1,
          },
        });
        nodeIds.add(e.data.source);
      }
      if (e.data.target && !nodeIds.has(e.data.target)) {
        nodes.push({
          data: {
            id: e.data.target,
            label: e.data.target.slice(0, 14),
            type: 'account',
            risk_score: 0.1,
          },
        });
        nodeIds.add(e.data.target);
      }
    });

    if (nodes.length > 0) cy.add(nodes);
    if (edges.length > 0) cy.add(edges);

    const layout = cy.layout({
      name: 'cose',
      idealEdgeLength: 85,
      nodeOverlap: 20,
      refresh: 20,
      fit: true,
      padding: 35,
      randomize: false,
      componentSpacing: 95,
      nodeRepulsion: 400000,
      edgeElasticity: 100,
      nestingFactor: 5,
      gravity: 80,
      numIter: 120,
      animate: false,
    });
    layout.run();
  }, [graphData]);

  // Update Cytoscape stylesheet when theme changes
  useEffect(() => {
    if (cyRef.current) {
      cyRef.current.style(getCytoscapeStyles(theme));
    }
  }, [theme]);

  // Handle selected node update
  useEffect(() => {
    if (!cyRef.current || !selectedNodeId) return;
    const cy = cyRef.current;
    cy.nodes().unselect();
    const sel = cy.getElementById(selectedNodeId);
    if (sel && sel.length > 0) {
      sel.select();
    }
  }, [selectedNodeId]);

  return (
    <div className={`relative w-full h-full ${theme === 'light' ? 'bg-[#f8fafc]' : 'bg-[#0a0a0a]'} overflow-hidden transition-colors`}>
      <div ref={containerRef} className="cytoscape-container" />

      {/* Floating Legend from Design */}
      <div className={`absolute left-4 bottom-4 flex items-center gap-4 text-[11.5px] rounded-full px-4 h-8 z-10 backdrop-blur-sm pointer-events-none select-none border transition-colors ${
        theme === 'light'
          ? 'text-[#334155] bg-white/95 border-[#e2e8f0] shadow-xs'
          : 'text-[#dadbdf] bg-[#0a0a0a]/90 border-[#212327]'
      }`}>
        <span className="flex items-center gap-2">
          <span className={`w-2.5 h-2.5 rounded-full border-[1.5px] ${theme === 'light' ? 'border-[#10b981] bg-[#ecfdf5]' : 'border-[#4fd1a1] bg-[#0e231c]'}`}></span>
          Account
        </span>
        <span className="flex items-center gap-2">
          <span className={`w-2.5 h-2.5 rounded-full border-2 ${theme === 'light' ? 'border-[#ef4444] bg-[#fee2e2]' : 'border-[#f0596b] bg-[#2a1216]'}`}></span>
          Mule
        </span>
        <span className="flex items-center gap-2">
          <span className={`w-2.5 h-2.5 rotate-45 border-[1.5px] ${theme === 'light' ? 'border-[#8b5cf6] bg-[#f3e8ff]' : 'border-[#c4b5fd] bg-[#1c1730]'}`}></span>
          Device / IP
        </span>
        <span className={`w-px h-3.5 ${theme === 'light' ? 'bg-[#e2e8f0]' : 'bg-[#212327]'}`}></span>
        <span className="flex items-center gap-2">
          <span className={`w-5 border-t ${theme === 'light' ? 'border-[#94a3b8]' : 'border-[#5a5f66]'}`}></span>
          Wire
        </span>
        <span className="flex items-center gap-2">
          <span className={`w-5 border-t-[1.5px] border-dashed ${theme === 'light' ? 'border-[#ef4444]' : 'border-[#f0596b]'}`}></span>
          Mule outflow
        </span>
        <span className="flex items-center gap-2">
          <span className={`w-5 border-t-2 border-dotted ${theme === 'light' ? 'border-[#8b5cf6]' : 'border-[#c4b5fd]'}`}></span>
          Telemetry
        </span>
      </div>
    </div>
  );
});

export default SubgraphVisualizer;
