import React from 'react';
import { Link } from 'react-router-dom';
import { Compass, ArrowLeft } from 'lucide-react';

export default function NotFoundPage() {
  return (
    <div className="card" style={{ padding: '4rem 2rem', textAlign: 'center', margin: '2rem auto', maxWidth: '540px' }}>
      <Compass size={48} color="#2563eb" style={{ margin: '0 auto 1rem auto' }} />
      <h1 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#102a43', marginBottom: '0.5rem' }}>
        404 &bull; Viewport Coordinate Not Found
      </h1>
      <p style={{ fontSize: '0.9rem', color: '#64748b', marginBottom: '1.5rem', lineHeight: 1.5 }}>
        The requested routing endpoint does not correspond to an active earth observation workspace.
      </p>
      <Link to="/" className="btn btn-primary">
        <ArrowLeft size={16} />
        <span>Return to Platform Overview</span>
      </Link>
    </div>
  );
}
