// ============================================================
// NetworkDiagram.tsx
// Interactive network topology visualization
// ============================================================

import { useEffect, useRef, useState } from 'react';
import type { PowerFlowCase, BusData, BranchData } from '../types';
import { BUS_TYPE_NAMES } from '../types';

interface Props {
  case: PowerFlowCase;
  onBusClick?: (busId: number) => void;
  onBranchClick?: (fromBus: number, toBus: number) => void;
}

interface Position {
  x: number;
  y: number;
}

export default function NetworkDiagram({ case: powerCase, onBusClick, onBranchClick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [busPositions, setBusPositions] = useState<Map<number, Position>>(new Map());
  const [hoveredBus, setHoveredBus] = useState<number | null>(null);
  const [hoveredBranch, setHoveredBranch] = useState<string | null>(null);

  // Calculate optimal bus positions using force-directed layout
  useEffect(() => {
    const positions = new Map<number, Position>();
    const buses = powerCase.bus;
    const branches = powerCase.branch;

    if (buses.length === 0) {
      setBusPositions(positions);
      return;
    }

    // Initialize positions in a circle
    const centerX = 300;
    const centerY = 250;
    const radius = Math.min(180, 120 + buses.length * 15);

    buses.forEach((bus, index) => {
      const angle = (index / buses.length) * 2 * Math.PI - Math.PI / 2;
      positions.set(bus.bus_id, {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
      });
    });

    // Simple force-directed adjustment
    for (let iteration = 0; iteration < 50; iteration++) {
      const forces = new Map<number, { x: number; y: number }>();

      // Initialize forces
      buses.forEach(bus => {
        forces.set(bus.bus_id, { x: 0, y: 0 });
      });

      // Repulsion between buses
      buses.forEach(bus1 => {
        buses.forEach(bus2 => {
          if (bus1.bus_id === bus2.bus_id) return;

          const pos1 = positions.get(bus1.bus_id)!;
          const pos2 = positions.get(bus2.bus_id)!;

          const dx = pos1.x - pos2.x;
          const dy = pos1.y - pos2.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;

          const force = 2000 / (dist * dist);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;

          const f = forces.get(bus1.bus_id)!;
          f.x += fx;
          f.y += fy;
        });
      });

      // Attraction along branches
      branches.forEach(branch => {
        if (branch.status === 0) return;

        const pos1 = positions.get(branch.f_bus);
        const pos2 = positions.get(branch.t_bus);

        if (!pos1 || !pos2) return;

        const dx = pos2.x - pos1.x;
        const dy = pos2.y - pos1.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;

        const force = (dist - 150) * 0.1;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;

        const f1 = forces.get(branch.f_bus)!;
        const f2 = forces.get(branch.t_bus)!;

        f1.x += fx;
        f1.y += fy;
        f2.x -= fx;
        f2.y -= fy;
      });

      // Apply forces with damping
      const damping = 0.5;
      buses.forEach(bus => {
        const pos = positions.get(bus.bus_id)!;
        const force = forces.get(bus.bus_id)!;

        pos.x += force.x * damping;
        pos.y += force.y * damping;

        // Keep within bounds
        pos.x = Math.max(80, Math.min(520, pos.x));
        pos.y = Math.max(80, Math.min(420, pos.y));
      });
    }

    setBusPositions(positions);
  }, [powerCase.bus, powerCase.branch]);

  // Draw the network
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = '#f0f0f0';
    ctx.lineWidth = 1;
    for (let x = 0; x <= 600; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, 500);
      ctx.stroke();
    }
    for (let y = 0; y <= 500; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(600, y);
      ctx.stroke();
    }

    // Draw branches
    powerCase.branch.forEach(branch => {
      const pos1 = busPositions.get(branch.f_bus);
      const pos2 = busPositions.get(branch.t_bus);

      if (!pos1 || !pos2) return;

      const branchKey = `${branch.f_bus}-${branch.t_bus}`;
      const isHovered = hoveredBranch === branchKey;
      const isActive = branch.status === 1;

      ctx.beginPath();
      ctx.moveTo(pos1.x, pos1.y);
      ctx.lineTo(pos2.x, pos2.y);

      if (!isActive) {
        ctx.strokeStyle = '#d1d5db';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
      } else if (isHovered) {
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 4;
        ctx.setLineDash([]);
      } else {
        ctx.strokeStyle = '#6b7280';
        ctx.lineWidth = 3;
        ctx.setLineDash([]);
      }

      ctx.stroke();
      ctx.setLineDash([]);

      // Draw branch label (middle of line)
      const midX = (pos1.x + pos2.x) / 2;
      const midY = (pos1.y + pos2.y) / 2;

      ctx.fillStyle = '#ffffff';
      ctx.fillRect(midX - 15, midY - 10, 30, 20);

      ctx.fillStyle = isHovered ? '#2563eb' : '#6b7280';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${branch.f_bus}-${branch.t_bus}`, midX, midY);
    });

    // Draw buses
    powerCase.bus.forEach(bus => {
      const pos = busPositions.get(bus.bus_id);
      if (!pos) return;

      const isHovered = hoveredBus === bus.bus_id;

      // Bus circle
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, isHovered ? 28 : 25, 0, 2 * Math.PI);

      // Color by bus type
      if (bus.bus_type === 3) {
        // Slack - green
        ctx.fillStyle = isHovered ? '#10b981' : '#34d399';
      } else if (bus.bus_type === 2) {
        // PV - blue
        ctx.fillStyle = isHovered ? '#3b82f6' : '#60a5fa';
      } else {
        // PQ - gray
        ctx.fillStyle = isHovered ? '#6b7280' : '#9ca3af';
      }

      ctx.fill();

      ctx.strokeStyle = isHovered ? '#1f2937' : '#374151';
      ctx.lineWidth = isHovered ? 3 : 2;
      ctx.stroke();

      // Bus ID
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 16px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(bus.bus_id.toString(), pos.x, pos.y);

      // Bus type label
      ctx.fillStyle = '#1f2937';
      ctx.font = '11px sans-serif';
      ctx.fillText(BUS_TYPE_NAMES[bus.bus_type], pos.x, pos.y + 40);

      // Voltage label
      if (isHovered) {
        ctx.fillStyle = '#374151';
        ctx.font = '10px sans-serif';
        ctx.fillText(`${bus.vm.toFixed(3)} p.u.`, pos.x, pos.y + 52);
      }
    });
  }, [powerCase, busPositions, hoveredBus, hoveredBranch]);

  // Handle mouse move for hover effects
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check if hovering over a bus
    let foundBus = false;
    for (const [busId, pos] of busPositions.entries()) {
      const dist = Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2);
      if (dist <= 25) {
        setHoveredBus(busId);
        setHoveredBranch(null);
        foundBus = true;
        canvas.style.cursor = 'pointer';
        break;
      }
    }

    if (!foundBus) {
      // Check if hovering over a branch
      let foundBranch = false;
      for (const branch of powerCase.branch) {
        const pos1 = busPositions.get(branch.f_bus);
        const pos2 = busPositions.get(branch.t_bus);

        if (!pos1 || !pos2) continue;

        // Distance from point to line segment
        const dx = pos2.x - pos1.x;
        const dy = pos2.y - pos1.y;
        const length = Math.sqrt(dx * dx + dy * dy);
        const t = Math.max(0, Math.min(1, ((x - pos1.x) * dx + (y - pos1.y) * dy) / (length * length)));
        const projX = pos1.x + t * dx;
        const projY = pos1.y + t * dy;
        const dist = Math.sqrt((x - projX) ** 2 + (y - projY) ** 2);

        if (dist <= 10) {
          setHoveredBranch(`${branch.f_bus}-${branch.t_bus}`);
          setHoveredBus(null);
          foundBranch = true;
          canvas.style.cursor = 'pointer';
          break;
        }
      }

      if (!foundBranch) {
        setHoveredBus(null);
        setHoveredBranch(null);
        canvas.style.cursor = 'default';
      }
    }
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (hoveredBus && onBusClick) {
      onBusClick(hoveredBus);
    } else if (hoveredBranch && onBranchClick) {
      const [from, to] = hoveredBranch.split('-').map(Number);
      onBranchClick(from, to);
    }
  };

  return (
    <div className="relative w-full">
      <canvas
        ref={canvasRef}
        width={600}
        height={500}
        onMouseMove={handleMouseMove}
        onClick={handleClick}
        onMouseLeave={() => {
          setHoveredBus(null);
          setHoveredBranch(null);
        }}
        className="border border-gray-300 rounded-lg bg-white w-full"
        style={{ maxWidth: '600px', height: 'auto' }}
      />

      {/* Legend */}
      <div className="absolute top-4 right-4 bg-white/95 backdrop-blur-sm border border-gray-200 rounded-lg p-3 shadow-lg">
        <h4 className="text-xs font-semibold text-gray-700 mb-2">Bus Types</h4>
        <div className="space-y-1 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-green-400 border-2 border-gray-600"></div>
            <span>Slack (REF)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-blue-400 border-2 border-gray-600"></div>
            <span>PV (Generator)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-gray-400 border-2 border-gray-600"></div>
            <span>PQ (Load)</span>
          </div>
        </div>

        <h4 className="text-xs font-semibold text-gray-700 mt-3 mb-2">Branches</h4>
        <div className="space-y-1 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 bg-gray-600"></div>
            <span>Active</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 bg-gray-300 border-dashed"></div>
            <span>Inactive</span>
          </div>
        </div>
      </div>

      {/* Hover tooltip */}
      {hoveredBus && (
        <div className="absolute bottom-4 left-4 bg-gray-900/90 text-white text-xs px-3 py-2 rounded-lg shadow-lg">
          <div className="font-semibold mb-1">Bus {hoveredBus}</div>
          {powerCase.bus.find(b => b.bus_id === hoveredBus) && (
            <>
              <div>Type: {BUS_TYPE_NAMES[powerCase.bus.find(b => b.bus_id === hoveredBus)!.bus_type]}</div>
              <div>Vm: {powerCase.bus.find(b => b.bus_id === hoveredBus)!.vm.toFixed(3)} p.u.</div>
              <div>Va: {powerCase.bus.find(b => b.bus_id === hoveredBus)!.va.toFixed(2)}°</div>
            </>
          )}
        </div>
      )}

      {hoveredBranch && (
        <div className="absolute bottom-4 left-4 bg-gray-900/90 text-white text-xs px-3 py-2 rounded-lg shadow-lg">
          <div className="font-semibold">Branch {hoveredBranch}</div>
          {powerCase.branch.find(b => `${b.f_bus}-${b.t_bus}` === hoveredBranch) && (
            <>
              <div>R: {powerCase.branch.find(b => `${b.f_bus}-${b.t_bus}` === hoveredBranch)!.r.toFixed(5)} p.u.</div>
              <div>X: {powerCase.branch.find(b => `${b.f_bus}-${b.t_bus}` === hoveredBranch)!.x.toFixed(5)} p.u.</div>
              <div>Status: {powerCase.branch.find(b => `${b.f_bus}-${b.t_bus}` === hoveredBranch)!.status ? 'Active' : 'Inactive'}</div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
