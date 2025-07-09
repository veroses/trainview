// App.jsx
import React, { useState } from 'react';
import api from './components/api';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ChartArea from './components/ChartArea';
import './App.css';

export default function App() {
  const [params, setParams] = useState({
    epochs: 10,
    learning_rate: 0.01,
    batch_size: 32,
    optimizer: 'adam',
  });

  const [resetTrigger, setResetTrigger] = useState(0);
  const [isTraining, setIsTraining] = useState(false);

  function handleChange(e) {
    const { name, value } = e.target;
    setParams(prev => ({
      ...prev,
      [name]: name === 'optimizer' ? value : Number(value),
    }));
  }

  function handleStart() {
    console.log("Sending:", params);

    setIsTraining(false);               // 1. stop polling
    setResetTrigger(prev => prev + 1);  // 2. clear chart on next render

    api.post('/start-training', params) // 3. restart training
      .then(() => {
        setIsTraining(true);            // 4. start polling again
      })
      .catch(err => console.error('Training error:', err));
  }

  return (
    <div className="app-layout">
      <Sidebar params={params} onChange={handleChange} onStart={handleStart} />
      <div className="main-content">
        <Header />
        <ChartArea reset={resetTrigger} active={isTraining} />


      </div>
      
    </div>
  );
}