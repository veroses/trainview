// App.jsx
import React, { useState } from 'react';
import api from './components/api';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ChartArea from './components/ChartArea';
import './App.css';
import ChartView from './components/ChartView';

export default function App() {
  const [params, setParams] = useState({
    epochs: 10,
    learning_rate: 0.01,
    batch_size: 32,
    optimizer: 'adam',
  });

  function handleChange(e) {
    const { name, value } = e.target;
    setParams(prev => ({
      ...prev,
      [name]: name === 'optimizer' ? value : Number(value),
    }));
  }

  function handleStart() {
    console.log("Sending:", params);
    setData([])
    api.post('/start-training', params)
      .then(res => console.log('Training started:', res.data))
      .catch(err => console.error('Training error:', err));
  }

  return (
    <div className="app-layout">
      <Sidebar params={params} onChange={handleChange} onStart={handleStart} />
      <div className="main-content">
        <Header />
        <ChartArea />

      </div>
      
    </div>
  );
}