// ============================================================
// NetworkEditor.tsx
// Network editing component with bus and branch tables
// ============================================================

import { useState } from 'react';
import { Plus, Trash2, Network } from 'lucide-react';
import type { PowerFlowCase, BusData, BranchData, GenData, Limits } from '../types';
import { BUS_TYPE_NAMES } from '../types';
import NetworkDiagram from './NetworkDiagram';

interface Props {
  case: PowerFlowCase;
  limits: Limits;
  onChange: (newCase: PowerFlowCase) => void;
}

export default function NetworkEditor({ case: powerCase, limits, onChange }: Props) {
  const [activeTab, setActiveTab] = useState<'bus' | 'branch' | 'gen'>('bus');

  const handleBusChange = (index: number, field: keyof BusData, value: number) => {
    const newBus = [...powerCase.bus];
    newBus[index] = { ...newBus[index], [field]: value };
    onChange({ ...powerCase, bus: newBus });
  };

  const handleAddBus = () => {
    if (powerCase.bus.length >= limits.max_buses) return;

    const maxId = Math.max(...powerCase.bus.map(b => b.bus_id), 0);
    const newBus: BusData = {
      bus_id: maxId + 1,
      bus_type: 1, // PQ
      pd: 0,
      qd: 0,
      gs: 0,
      bs: 0,
      vm: 1.0,
      va: 0,
    };
    onChange({ ...powerCase, bus: [...powerCase.bus, newBus] });
  };

  const handleRemoveBus = (index: number) => {
    if (powerCase.bus.length <= limits.min_buses) return;

    const removedBusId = powerCase.bus[index].bus_id;
    const newBus = powerCase.bus.filter((_, i) => i !== index);

    // Remove branches connected to this bus
    const newBranch = powerCase.branch.filter(
      br => br.f_bus !== removedBusId && br.t_bus !== removedBusId
    );

    // Remove generators on this bus
    const newGen = powerCase.gen.filter(g => g.gen_bus !== removedBusId);

    onChange({ ...powerCase, bus: newBus, branch: newBranch, gen: newGen });
  };

  const handleBranchChange = (index: number, field: keyof BranchData, value: number) => {
    const newBranch = [...powerCase.branch];
    newBranch[index] = { ...newBranch[index], [field]: value };
    onChange({ ...powerCase, branch: newBranch });
  };

  const handleAddBranch = () => {
    if (powerCase.bus.length < 2) return;

    const newBranch: BranchData = {
      f_bus: powerCase.bus[0].bus_id,
      t_bus: powerCase.bus[1].bus_id,
      r: 0.01,
      x: 0.05,
      b: 0,
      tap: 0,
      shift: 0,
      status: 1,
    };
    onChange({ ...powerCase, branch: [...powerCase.branch, newBranch] });
  };

  const handleRemoveBranch = (index: number) => {
    const newBranch = powerCase.branch.filter((_, i) => i !== index);
    onChange({ ...powerCase, branch: newBranch });
  };

  const handleGenChange = (index: number, field: keyof GenData, value: number) => {
    const newGen = [...powerCase.gen];
    newGen[index] = { ...newGen[index], [field]: value };
    onChange({ ...powerCase, gen: newGen });
  };

  const handleAddGen = () => {
    if (powerCase.bus.length === 0) return;

    const newGen: GenData = {
      gen_bus: powerCase.bus[0].bus_id,
      pg: 0,
      qg: 0,
      vg: 1.0,
    };
    onChange({ ...powerCase, gen: [...powerCase.gen, newGen] });
  };

  const handleRemoveGen = (index: number) => {
    const newGen = powerCase.gen.filter((_, i) => i !== index);
    onChange({ ...powerCase, gen: newGen });
  };

  const handleBaseMvaChange = (value: number) => {
    onChange({ ...powerCase, base_mva: value });
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-900">Network Configuration</h2>
        <div className="flex items-center gap-4">
          <label className="text-sm font-medium text-gray-700">
            Base MVA:
            <input
              type="number"
              value={powerCase.base_mva}
              onChange={(e) => handleBaseMvaChange(parseFloat(e.target.value))}
              className="input ml-2 w-24"
              min="1"
              step="1"
            />
          </label>
        </div>
      </div>

      {/* Split Layout: Editor (Left) and Diagram (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Editor Tables */}
        <div className="flex flex-col">
          {/* Tabs */}
          <div className="border-b border-gray-200 mb-4">
            <nav className="-mb-px flex space-x-8">
              {(['bus', 'branch', 'gen'] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`
                    py-2 px-1 border-b-2 font-medium text-sm capitalize
                    ${activeTab === tab
                      ? 'border-primary-600 text-primary-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }
                  `}
                >
                  {tab === 'bus' && `Buses (${powerCase.bus.length})`}
                  {tab === 'branch' && `Branches (${powerCase.branch.length})`}
                  {tab === 'gen' && `Generators (${powerCase.gen.length})`}
                </button>
              ))}
            </nav>
          </div>

      {/* Bus Table */}
      {activeTab === 'bus' && (
        <div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase">ID</th>
                  <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                  <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase">Pd<br/>(MW)</th>
                  <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase">Qd<br/>(MVAr)</th>
                  <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase">Vm<br/>(p.u.)</th>
                  <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase">Va<br/>(deg)</th>
                  <th className="px-2 py-2 text-right text-xs font-medium text-gray-500 uppercase w-12"></th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {powerCase.bus.map((bus, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-2 py-2 text-sm font-medium text-gray-900">{bus.bus_id}</td>
                    <td className="px-2 py-2">
                      <select
                        value={bus.bus_type}
                        onChange={(e) => handleBusChange(index, 'bus_type', parseInt(e.target.value))}
                        className="input text-xs py-1 px-2 w-20"
                      >
                        <option value={1}>PQ</option>
                        <option value={2}>PV</option>
                        <option value={3}>Slack</option>
                      </select>
                    </td>
                    <td className="px-2 py-2">
                      <input
                        type="number"
                        value={bus.pd}
                        onChange={(e) => handleBusChange(index, 'pd', parseFloat(e.target.value))}
                        className="input text-xs w-16 py-1 px-1"
                        step="0.1"
                      />
                    </td>
                    <td className="px-2 py-2">
                      <input
                        type="number"
                        value={bus.qd}
                        onChange={(e) => handleBusChange(index, 'qd', parseFloat(e.target.value))}
                        className="input text-xs w-16 py-1 px-1"
                        step="0.1"
                      />
                    </td>
                    <td className="px-2 py-2">
                      <input
                        type="number"
                        value={bus.vm}
                        onChange={(e) => handleBusChange(index, 'vm', parseFloat(e.target.value))}
                        className="input text-xs w-16 py-1 px-1"
                        step="0.001"
                        min="0.01"
                      />
                    </td>
                    <td className="px-2 py-2">
                      <input
                        type="number"
                        value={bus.va}
                        onChange={(e) => handleBusChange(index, 'va', parseFloat(e.target.value))}
                        className="input text-xs w-16 py-1 px-1"
                        step="0.1"
                      />
                    </td>
                    <td className="px-2 py-2 text-right">
                      <button
                        onClick={() => handleRemoveBus(index)}
                        disabled={powerCase.bus.length <= limits.min_buses}
                        className="text-red-600 hover:text-red-700 disabled:text-gray-300 disabled:cursor-not-allowed"
                        title="Remove bus"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Collapsible Advanced Options */}
          <details className="mt-3 mb-2">
            <summary className="text-xs text-gray-600 cursor-pointer hover:text-gray-800 font-medium">
              Advanced: Shunts (GS, BS)
            </summary>
            <div className="mt-2 overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 text-xs">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-2 py-1 text-left text-xs font-medium text-gray-500">Bus</th>
                    <th className="px-2 py-1 text-left text-xs font-medium text-gray-500">GS (MW)</th>
                    <th className="px-2 py-1 text-left text-xs font-medium text-gray-500">BS (MVAr)</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {powerCase.bus.map((bus, index) => (
                    <tr key={index}>
                      <td className="px-2 py-1 font-medium">{bus.bus_id}</td>
                      <td className="px-2 py-1">
                        <input
                          type="number"
                          value={bus.gs}
                          onChange={(e) => handleBusChange(index, 'gs', parseFloat(e.target.value))}
                          className="input text-xs w-16 py-1 px-1"
                          step="0.01"
                        />
                      </td>
                      <td className="px-2 py-1">
                        <input
                          type="number"
                          value={bus.bs}
                          onChange={(e) => handleBusChange(index, 'bs', parseFloat(e.target.value))}
                          className="input text-xs w-16 py-1 px-1"
                          step="0.01"
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>

          <button
            onClick={handleAddBus}
            disabled={powerCase.bus.length >= limits.max_buses}
            className="btn btn-secondary mt-2 text-sm flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Add Bus
          </button>
        </div>
      )}

      {/* Branch Table */}
      {activeTab === 'branch' && (
        <div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase">From</th>
                  <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase">To</th>
                  <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase">R<br/>(p.u.)</th>
                  <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase">X<br/>(p.u.)</th>
                  <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-2 py-2 text-right text-xs font-medium text-gray-500 uppercase w-12"></th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {powerCase.branch.map((branch, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-2 py-2">
                      <select
                        value={branch.f_bus}
                        onChange={(e) => handleBranchChange(index, 'f_bus', parseInt(e.target.value))}
                        className="input text-xs py-1 px-2 w-16"
                      >
                        {powerCase.bus.map(b => (
                          <option key={b.bus_id} value={b.bus_id}>{b.bus_id}</option>
                        ))}
                      </select>
                    </td>
                    <td className="px-2 py-2">
                      <select
                        value={branch.t_bus}
                        onChange={(e) => handleBranchChange(index, 't_bus', parseInt(e.target.value))}
                        className="input text-xs py-1 px-2 w-16"
                      >
                        {powerCase.bus.map(b => (
                          <option key={b.bus_id} value={b.bus_id}>{b.bus_id}</option>
                        ))}
                      </select>
                    </td>
                    <td className="px-2 py-2">
                      <input
                        type="number"
                        value={branch.r}
                        onChange={(e) => handleBranchChange(index, 'r', parseFloat(e.target.value))}
                        className="input text-xs w-20 py-1 px-1"
                        step="0.0001"
                      />
                    </td>
                    <td className="px-2 py-2">
                      <input
                        type="number"
                        value={branch.x}
                        onChange={(e) => handleBranchChange(index, 'x', parseFloat(e.target.value))}
                        className="input text-xs w-20 py-1 px-1"
                        step="0.0001"
                      />
                    </td>
                    <td className="px-2 py-2">
                      <select
                        value={branch.status}
                        onChange={(e) => handleBranchChange(index, 'status', parseInt(e.target.value))}
                        className="input text-xs py-1 px-2 w-20"
                      >
                        <option value={1}>On</option>
                        <option value={0}>Off</option>
                      </select>
                    </td>
                    <td className="px-2 py-2 text-right">
                      <button
                        onClick={() => handleRemoveBranch(index)}
                        className="text-red-600 hover:text-red-700"
                        title="Remove branch"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Collapsible Advanced Options */}
          <details className="mt-3 mb-2">
            <summary className="text-xs text-gray-600 cursor-pointer hover:text-gray-800 font-medium">
              Advanced: B, Tap, Shift
            </summary>
            <div className="mt-2 overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 text-xs">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-2 py-1 text-left text-xs font-medium text-gray-500">Branch</th>
                    <th className="px-2 py-1 text-left text-xs font-medium text-gray-500">B (p.u.)</th>
                    <th className="px-2 py-1 text-left text-xs font-medium text-gray-500">Tap</th>
                    <th className="px-2 py-1 text-left text-xs font-medium text-gray-500">Shift</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {powerCase.branch.map((branch, index) => (
                    <tr key={index}>
                      <td className="px-2 py-1 font-medium">{branch.f_bus}-{branch.t_bus}</td>
                      <td className="px-2 py-1">
                        <input
                          type="number"
                          value={branch.b}
                          onChange={(e) => handleBranchChange(index, 'b', parseFloat(e.target.value))}
                          className="input text-xs w-20 py-1 px-1"
                          step="0.0001"
                        />
                      </td>
                      <td className="px-2 py-1">
                        <input
                          type="number"
                          value={branch.tap}
                          onChange={(e) => handleBranchChange(index, 'tap', parseFloat(e.target.value))}
                          className="input text-xs w-16 py-1 px-1"
                          step="0.01"
                        />
                      </td>
                      <td className="px-2 py-1">
                        <input
                          type="number"
                          value={branch.shift}
                          onChange={(e) => handleBranchChange(index, 'shift', parseFloat(e.target.value))}
                          className="input text-xs w-16 py-1 px-1"
                          step="0.01"
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>

          <button
            onClick={handleAddBranch}
            disabled={powerCase.bus.length < 2}
            className="btn btn-secondary mt-2 text-sm flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Add Branch
          </button>
        </div>
      )}

      {/* Gen Table */}
      {activeTab === 'gen' && (
        <div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase">Bus</th>
                  <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase">Pg<br/>(MW)</th>
                  <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase">Qg<br/>(MVAr)</th>
                  <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase">Vg<br/>(p.u.)</th>
                  <th className="px-2 py-2 text-right text-xs font-medium text-gray-500 uppercase w-12"></th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {powerCase.gen.map((gen, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-2 py-2">
                      <select
                        value={gen.gen_bus}
                        onChange={(e) => handleGenChange(index, 'gen_bus', parseInt(e.target.value))}
                        className="input text-xs py-1 px-2 w-16"
                      >
                        {powerCase.bus.map(b => (
                          <option key={b.bus_id} value={b.bus_id}>{b.bus_id}</option>
                        ))}
                      </select>
                    </td>
                    <td className="px-2 py-2">
                      <input
                        type="number"
                        value={gen.pg}
                        onChange={(e) => handleGenChange(index, 'pg', parseFloat(e.target.value))}
                        className="input text-xs w-20 py-1 px-1"
                        step="1"
                      />
                    </td>
                    <td className="px-2 py-2">
                      <input
                        type="number"
                        value={gen.qg}
                        onChange={(e) => handleGenChange(index, 'qg', parseFloat(e.target.value))}
                        className="input text-xs w-20 py-1 px-1"
                        step="1"
                      />
                    </td>
                    <td className="px-2 py-2">
                      <input
                        type="number"
                        value={gen.vg}
                        onChange={(e) => handleGenChange(index, 'vg', parseFloat(e.target.value))}
                        className="input text-xs w-16 py-1 px-1"
                        step="0.001"
                        min="0.01"
                      />
                    </td>
                    <td className="px-2 py-2 text-right">
                      <button
                        onClick={() => handleRemoveGen(index)}
                        className="text-red-600 hover:text-red-700"
                        title="Remove generator"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <button
            onClick={handleAddGen}
            disabled={powerCase.bus.length === 0}
            className="btn btn-secondary mt-2 text-sm flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Add Generator
          </button>
        </div>
      )}
        </div>

        {/* Right: Network Diagram */}
        <div className="flex flex-col items-center">
          <div className="mb-2">
            <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
              <Network className="w-4 h-4" />
              Network Topology
            </h3>
          </div>
          <NetworkDiagram case={powerCase} />
          <p className="text-xs text-gray-500 mt-3 text-center">
            Interactive visualization • Hover over buses and branches for details
          </p>
        </div>
      </div>
    </div>
  );
}
