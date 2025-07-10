import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const AccuracyChart = ({ data }) => (
<div style={{ textAlign: 'center', color: 'black'}}>
    <h3 style={{ marginBottom: '0.5rem' }}>Accuracy</h3>
<LineChart width={480} height={300} data={data}>
    <CartesianGrid strokeDasharray="3 3" />
    <XAxis dataKey="epoch" />
    <YAxis />
    <Tooltip />
    <Line type="monotone" dataKey="accuracy" stroke="#82ca9d" dot={false} />
  </LineChart>
  </div>
);

export default AccuracyChart;