import { useEffect, useState } from 'react';
import api from './api';
import LossChart from './LossChart';
import AccuracyChart from './AccuracyChart';

const ChartArea = ({ reset, active }) => {
  const [data, setData] = useState([]);

  useEffect(() => {
    setData([]);
  }, [reset]);

 useEffect(() => {
  if (!active) return;

  let currentEpoch = -1;

  const interval = setInterval(() => {
    api.get('/training-status')
      .then(res => {
        const { epoch, loss, accuracy } = res.data;

        if (epoch > currentEpoch) {
          setData(prev => [...prev, { epoch, loss, accuracy }]);
          currentEpoch = epoch;
        }
      })
      .catch(err => console.error('Polling error:', err));
  }, 1000);

  return () => clearInterval(interval);
}, [active]);


  return (
    <div className="chart-grid">
      <div className="chart-block">
        <LossChart data={data} />
      </div>
      <div className="chart-block">
        <AccuracyChart data={data} />
      </div>
    </div>
  );
};

export default ChartArea;
