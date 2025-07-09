import { useEffect, useState } from 'react';
import api from './api';
import LossChart from './LossChart';
import AccuracyChart from './AccuracyChart';

const ChartArea = () => {
  const [data, setData] = useState([]);
  const [lastEpoch, setLastEpoch] = useState(-1);

  useEffect(() => {
    const interval = setInterval(() => {
      api.get('/metrics')
        .then(res => {
          const { epoch, loss, accuracy } = res.data;
          if (epoch > lastEpoch) {
            setData(prev => [...prev, { epoch, loss, accuracy }]);
            setLastEpoch(epoch);
          }
        })
        .catch(err => console.error('Polling error:', err));
    }, 1000);

    return () => clearInterval(interval);
  }, [lastEpoch]);

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
