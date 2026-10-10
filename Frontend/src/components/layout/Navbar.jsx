import React, { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { Layers, Activity, Satellite, GitCompare, BookOpen, CheckCircle, AlertCircle } from 'lucide-react';
import { api } from '../../services/api';
import Badge from '../common/Badge';

export default function Navbar() {
  const [health, setHealth] = useState(null);
  const [mambaReady, setMambaReady] = useState(true);

  useEffect(() => {
    api.getHealth()
      .then(data => setHealth(data))
      .catch(() => setHealth({ status: 'offline', device: 'Disconnected' }));

    api.getModels()
      .then(data => {
        const mamba = data.models.find(m => m.id === 'unet_mamba');
        if (mamba) setMambaReady(mamba.is_available);
      })
      .catch(() => {});
  }, []);

  return (
    <header className="navbar">
      <NavLink to="/" className="nav-brand">
        <div className="nav-logo-icon">
          <Layers size={20} />
        </div>
        <div>
          <span className="nav-brand-title">CAVE-Flood</span>
          <span className="nav-brand-tag" style={{ marginLeft: '6px' }}>v2.4-Scientific</span>
        </div>
      </NavLink>

      <nav className="nav-links">
        <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} end>
          <Satellite size={16} />
          <span>Overview</span>
        </NavLink>
        <NavLink to="/analyze" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <Activity size={16} />
          <span>Satellite Analysis</span>
        </NavLink>
        <NavLink to="/compare" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <GitCompare size={16} />
          <span>Model Comparison</span>
        </NavLink>
        <NavLink to="/models" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <BookOpen size={16} />
          <span>Architectures</span>
        </NavLink>
      </nav>

      <div className="nav-telemetry">
        {mambaReady ? (
          <Badge variant="ready" title="UNet-Mamba Checkpoint 171 MB Loaded">
            <CheckCircle size={12} />
            <span>UNet-Mamba: Ready</span>
          </Badge>
        ) : (
          <Badge variant="alert">
            <AlertCircle size={12} />
            <span>UNet-Mamba: Missing</span>
          </Badge>
        )}

        <Badge variant={health?.status === 'healthy' ? 'ready' : 'alert'} title={health?.device || 'Checking system'}>
          <span style={{
            display: 'inline-block',
            width: '6px',
            height: '6px',
            borderRadius: '50%',
            backgroundColor: health?.status === 'healthy' ? '#10b981' : '#ef4444'
          }} />
          <span>{health?.device?.split(' ')[0] || 'FastAPI'}</span>
        </Badge>
      </div>
    </header>
  );
}
