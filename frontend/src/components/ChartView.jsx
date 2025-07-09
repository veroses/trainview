import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts';
import { useEffect, useState } from 'react';
import api from './api';

const ChartView = () => {
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
    <LineChart width={600} height={300} data={data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="epoch" />
      <YAxis />
      <Tooltip />
      <Legend />
      <Line type="monotone" dataKey="loss" stroke="#8884d8" />
      <Line type="monotone" dataKey="accuracy" stroke="#82ca9d" />
    </LineChart>
  );
};

export default ChartView;