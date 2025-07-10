import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const LossChart = ({ data }) => (
    <div style={{ textAlign: 'center', color: 'black'}}>
    <h3 style={{ marginBottom: '0.5rem' }}>Loss</h3>
  <LineChart width={400} height={250} data={data}>
    <CartesianGrid strokeDasharray="3 3" />
    <XAxis dataKey="epoch" />
    <YAxis />
    <Tooltip />
    <Line type="monotone" dataKey="loss" stroke="#8884d8" dot={false} />
  </LineChart>
  </div>
);

export default LossChart;